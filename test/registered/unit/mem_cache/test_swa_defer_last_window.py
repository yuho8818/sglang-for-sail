# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deferred last SWA window: properties, not pinned numbers.

`free_swa_out_of_window_slots` normally frees every request-owned SWA slot below
the trim frontier. That leaves the request's most recent window unreachable to
the tree, so a later request whose shared prefix ends inside that span matches a
tombstone and pays a full recompute (the first-touch defect). With the defer
enabled the trim holds one `max(window, page)` range back so the following insert
can capture it, and the range is clamped to `cache_protected_len` so it survives
only until the tree owns it.

The assertions below state the behaviour the fix must have rather than the exact
frontier an implementation happens to compute:

  * Deferring never frees *more* than the original boundary, and never moves the
    frontier backwards.
  * The retained span is at least one full sliding window, which is what makes
    the window capturable at all.
  * The retained span is page-aligned, because the insert path asserts its
    `evicted_seqlen` is page-aligned.
  * The retained span never extends past `cache_protected_len`, so a slot the
    tree already owns is never handed back to the free list.
  * Once the tree owns everything (`cache_protected_len` reaches the frontier)
    nothing is retained, so repeated trims cannot accumulate a growing ring.
  * Both `window % page != 0` and divisible geometries are covered. An earlier
    round of tests for this area used only divisible geometries (`8/1`, `4/4`)
    and so never exercised the alignment boundary.
"""

import unittest
from types import SimpleNamespace

import torch
from unittest.mock import Mock

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Pool:
    """Records which half-open token ranges the trim hands to the free list."""

    def __init__(self, n):
        self.req_to_token = torch.arange(n).reshape(1, -1)
        self.freed: list[tuple[int, int]] = []


def _run_trim(
    *,
    pre_len,
    window,
    page,
    cache_protected_len,
    evicted_seqlen,
    defer,
    switch=True,
    eviction_interval=1,
):
    pool = _Pool(max(pre_len, 1) + 4096)
    # `holds_kv` is `req_pool_idx is not None`, so setting the row registers it.
    req = SimpleNamespace(
        kv=ReqKvInfo(
            req_pool_idx=0,
            cache_protected_len=cache_protected_len,
            swa_evict_floor=0,
        ),
        kv_committed_len=pre_len,
    )
    req.kv.set_evicted_seqlen(ComponentType.SWA, evicted_seqlen)

    freed_ranges: list[tuple[int, int]] = []

    def free_segment(slots, start_pos):
        freed_ranges.append((start_pos, start_pos + len(slots)))

    allocator = Mock(spec=["free_swa_segment"])
    allocator.free_swa_segment = free_segment

    with envs.SGLANG_OPT_SWA_DEFER_LAST_WINDOW.override(switch):
        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=window,
            page_size=page,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            supports_prefix_sharing=True,
            retain_floor=None,  # the production value on this deployment
            eviction_interval=eviction_interval,
            defer_last_window=defer,
        )
    return freed_ranges, req.kv.get_evicted_seqlen(ComponentType.SWA)


class TestDeferredLastWindowProperties(unittest.TestCase):
    # 128/256 is the production geometry: window < page, so window % page != 0.
    GEOMETRIES = [(128, 256), (512, 288), (384, 256), (128, 128), (8, 1), (4, 4)]

    def test_window_mod_page_nonzero_is_covered(self):
        covered = [g for g in self.GEOMETRIES if g[0] % g[1] != 0]
        self.assertTrue(
            covered, "the geometry matrix must exercise window % page != 0"
        )

    def test_defer_retains_at_least_one_window(self):
        """The whole point: a full window survives above the frontier."""
        for window, page in self.GEOMETRIES:
            with self.subTest(window=window, page=page):
                pre_len = window * 8
                on_ranges, on_frontier = _run_trim(
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    cache_protected_len=0,
                    evicted_seqlen=0,
                    defer=True,
                )
                off_ranges, off_frontier = _run_trim(
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    cache_protected_len=0,
                    evicted_seqlen=0,
                    defer=False,
                )
                self.assertTrue(
                    off_ranges, "the baseline trim must free something to compare"
                )
                # The retained span is measured from the highest freed position
                # up to the frontier the baseline trim would have decided on.
                on_end = on_ranges[0][1] if on_ranges else 0
                baseline_end = off_ranges[0][1]
                self.assertLessEqual(
                    on_end,
                    baseline_end,
                    "deferring must never free past the original boundary",
                )
                retained = baseline_end - on_end
                self.assertGreaterEqual(
                    retained,
                    window,
                    "at least one sliding window must stay above the frontier",
                )
                # The deferred frontier sits at or below the baseline one: the
                # trim releases less, it never advances further.
                self.assertLessEqual(on_frontier, off_frontier)

    def test_retained_frontier_is_page_aligned(self):
        """`_clip_to_window` asserts its window_start is page-aligned."""
        for window, page in self.GEOMETRIES:
            with self.subTest(window=window, page=page):
                _, on_frontier = _run_trim(
                    pre_len=window * 8,
                    window=window,
                    page=page,
                    cache_protected_len=0,
                    evicted_seqlen=0,
                    defer=True,
                )
                self.assertEqual(on_frontier % page, 0)

    def test_frontier_never_moves_backwards(self):
        for window, page in self.GEOMETRIES:
            for evicted in (0, page, page * 2):
                with self.subTest(window=window, page=page, evicted=evicted):
                    _, on_frontier = _run_trim(
                        pre_len=window * 12,
                        window=window,
                        page=page,
                        cache_protected_len=0,
                        evicted_seqlen=evicted,
                        defer=True,
                    )
                    self.assertGreaterEqual(on_frontier, evicted)

    def test_retained_span_never_frees_below_cache_protected_len(self):
        """A slot the tree already owns must never be returned to the free list.

        `cache_protected_len` is the tree's ownership boundary, so nothing below
        it may be freed. The deferred range is clamped to it, which is what lets
        the retained window survive until the tree has taken it over.
        """
        for window, page in self.GEOMETRIES:
            for protected in (0, page, page * 3, page * 5):
                with self.subTest(window=window, page=page, protected=protected):
                    ranges, _ = _run_trim(
                        pre_len=window * 10,
                        window=window,
                        page=page,
                        cache_protected_len=protected,
                        evicted_seqlen=0,
                        defer=True,
                    )
                    for start, _ in ranges:
                        self.assertGreaterEqual(
                            start,
                            protected,
                            "the trim must not free a tree-owned slot",
                        )

    def test_no_retention_once_the_tree_owns_the_frontier(self):
        """Repeated trims must not accumulate a growing ring.

        Once the tree owns everything the baseline trim would have freed
        (`cache_protected_len` reaches that boundary) the deferred trim has
        nothing left to hold back, so the frontier does not regress.
        """
        for window, page in self.GEOMETRIES:
            with self.subTest(window=window, page=page):
                pre_len = window * 8
                _, frontier_without = _run_trim(
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    cache_protected_len=0,
                    evicted_seqlen=0,
                    defer=False,
                )
                ranges, frontier = _run_trim(
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    cache_protected_len=frontier_without,
                    evicted_seqlen=0,
                    defer=True,
                )
                # Nothing below the tree's ownership boundary is freed, and the
                # deferred frontier never retreats behind what the tree owns.
                for start, _ in ranges:
                    self.assertGreaterEqual(start, frontier_without)
                self.assertGreaterEqual(frontier, frontier_without)

    def test_switch_off_matches_the_baseline_boundary(self):
        """Switch off = the original boundary, for every geometry."""
        for window, page in self.GEOMETRIES:
            with self.subTest(window=window, page=page):
                _, off_switch = _run_trim(
                    pre_len=window * 8,
                    window=window,
                    page=page,
                    cache_protected_len=0,
                    evicted_seqlen=0,
                    defer=True,
                    switch=False,
                )
                _, no_defer = _run_trim(
                    pre_len=window*8,
                    window=window,
                    page=page,
                    cache_protected_len=0,
                    evicted_seqlen=0,
                    defer=False,
                )
                self.assertEqual(off_switch, no_defer)

    def test_defer_flag_absent_matches_baseline(self):
        """Passing defer=False must be byte-identical to the old behaviour."""
        for window, page in self.GEOMETRIES:
            with self.subTest(window=window, page=page):
                ranges, frontier = _run_trim(
                    pre_len=window * 8,
                    window=window,
                    page=page,
                    cache_protected_len=0,
                    evicted_seqlen=0,
                    defer=False,
                )
                self.assertTrue(ranges, "baseline trim frees the out-of-window body")
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(frontier, ranges[0][1])


if __name__ == "__main__":
    unittest.main()