"""The trim before an insert must keep the window a later hold-back match needs.

With decoder SWA bounded replay a match stops at page_floor(len - window) of the
requesting prompt. A request that resubmits a cached key (or extends it by fewer
tokens than the hold-back) therefore needs live SWA for the window just below
page_floor(key_len - hold_back). The pre-insert trim used to keep only
max(window, page) below the key end, which leaves that window a tombstone for
about half of all key lengths, so such a request missed entirely on first touch.

Invariants over the same geometry sweep as test_swa_retain_unpublished.py:
  1. With a hold-back, the window below the hold-back match point stays live.
  2. Without one, the frontier is exactly what it was before.
  3. The extra SWA kept is bounded by hold_back + window + page.
  4. The frontier stays below the insert boundary; nothing below
     cache_protected_len is freed; repeated trims never free a slot twice.
"""

import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots, page_align_floor
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GEOMETRIES = [(128, 256), (512, 288), (384, 256), (512, 256)]


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free_swa_segment(self, indices, *, start_pos):
        self.freed.append((start_pos, start_pos + len(indices)))


def _make_req(*, protected, allocated):
    pool = SimpleNamespace(req_to_token=torch.arange(16384).reshape(1, -1))
    kv = ReqKvInfo(
        req_pool_idx=0,
        cache_protected_len=protected,
        kv_allocated_len=allocated,
        swa_evict_floor=0,
    )
    kv.set_evicted_seqlen(ComponentType.SWA, 0)
    return SimpleNamespace(kv=kv), pool


def _frontier(*, key_len, window, page, hold_back, protected=0, retain=False):
    req, pool = _make_req(protected=protected, allocated=key_len)
    allocator = _FakeAllocator()
    free_swa_out_of_window_slots(
        req,
        key_len - 1,
        sliding_window_size=window,
        page_size=page,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        retain_unpublished=retain,
        match_hold_back=hold_back,
    )
    return req.kv.get_evicted_seqlen(ComponentType.SWA), allocator


def _key_lengths(window, page):
    lo = 4 * max(window, page)
    return range(lo, lo + 2 * page + 1)


class TestSwaHoldBackTrim(unittest.TestCase):
    def test_window_below_the_hold_back_match_stays_live(self):
        for window, page in GEOMETRIES:
            hold_back = window
            for key_len in _key_lengths(window, page):
                with self.subTest(window=window, page=page, key_len=key_len):
                    frontier, _ = _frontier(
                        key_len=key_len, window=window, page=page, hold_back=hold_back
                    )
                    match_end = page_align_floor(key_len - hold_back, page)
                    self.assertLessEqual(frontier, match_end - window)

    def test_old_trim_tombstones_the_hold_back_window(self):
        # The measured case: a 3848-token key, window 128, page 256. A resubmission
        # matches to 3584 and needs [3456, 3584); the old trim freed up to 3584.
        old, _ = _frontier(key_len=3848, window=128, page=256, hold_back=0)
        new, _ = _frontier(key_len=3848, window=128, page=256, hold_back=128)
        self.assertEqual(old, 3584)
        self.assertGreater(old, 3584 - 128)
        self.assertEqual(new, 3328)

    def test_no_hold_back_keeps_the_old_frontier(self):
        for window, page in GEOMETRIES:
            for key_len in _key_lengths(window, page):
                with self.subTest(window=window, page=page, key_len=key_len):
                    frontier, _ = _frontier(
                        key_len=key_len, window=window, page=page, hold_back=0
                    )
                    old = page_align_floor(
                        key_len - 1 - max(window, page), page
                    )
                    self.assertEqual(frontier, old)

    def test_extra_swa_kept_is_bounded(self):
        for window, page in GEOMETRIES:
            hold_back = window
            for key_len in _key_lengths(window, page):
                with self.subTest(window=window, page=page, key_len=key_len):
                    old, _ = _frontier(
                        key_len=key_len, window=window, page=page, hold_back=0
                    )
                    new, _ = _frontier(
                        key_len=key_len, window=window, page=page, hold_back=hold_back
                    )
                    self.assertLessEqual(new, old)
                    self.assertLessEqual(old - new, hold_back + window + page)

    def test_frontier_stays_below_the_insert_boundary(self):
        for window, page in GEOMETRIES:
            for key_len in _key_lengths(window, page):
                with self.subTest(window=window, page=page, key_len=key_len):
                    frontier, _ = _frontier(
                        key_len=key_len, window=window, page=page, hold_back=window
                    )
                    self.assertLess(frontier, page_align_floor(key_len, page))

    def test_nothing_below_cache_protected_len_is_freed(self):
        for (window, page), retain in product(GEOMETRIES, [False, True]):
            with self.subTest(window=window, page=page, retain=retain):
                protected = page * 3
                frontier, allocator = _frontier(
                    key_len=page * 12,
                    window=window,
                    page=page,
                    hold_back=window,
                    protected=protected,
                    retain=retain,
                )
                for start, _end in allocator.freed:
                    self.assertGreaterEqual(start, protected)
                self.assertGreaterEqual(frontier, protected)

    def test_retain_still_holds_at_what_the_tree_owns(self):
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                frontier, allocator = _frontier(
                    key_len=page * 12,
                    window=window,
                    page=page,
                    hold_back=window,
                    protected=page * 2,
                    retain=True,
                )
                self.assertEqual(frontier, page * 2)
                self.assertEqual(allocator.freed, [])

    def test_repeated_trims_never_free_a_slot_twice(self):
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                allocated = page * 20
                req, pool = _make_req(protected=0, allocated=allocated)
                allocator = _FakeAllocator()
                for step in range(page, allocated, page // 2):
                    free_swa_out_of_window_slots(
                        req,
                        step,
                        sliding_window_size=window,
                        page_size=page,
                        req_to_token_pool=pool,
                        token_to_kv_pool_allocator=allocator,
                        match_hold_back=window,
                    )
                covered = [
                    i for start, end in allocator.freed for i in range(start, end)
                ]
                self.assertEqual(len(covered), len(set(covered)))


class TestSwaComponentPassesHoldBack(unittest.TestCase):
    def _run(self, hold_back):
        calls = []

        def record(req, pre_len, *, retain_unpublished=False, match_hold_back=0):
            calls.append(match_hold_back)

        comp = SimpleNamespace(
            cache=SimpleNamespace(
                token_to_kv_pool_allocator=None,
                swa_evictable_size=lambda: 0,
                swa_reprefill_tail_tokens=lambda req=None: hold_back,
            ),
            component_type=ComponentType.SWA,
            _free_out_of_window_slots=record,
        )
        req = SimpleNamespace(kv=SimpleNamespace(get_evicted_seqlen=lambda ct: 0))
        params = SimpleNamespace(set_evicted_seqlen=lambda ct, value: None)
        with envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(False):
            SWAComponent.free_out_of_window_slots(comp, req, 3847, params)
        self.assertEqual(len(calls), 1)
        return calls[0]

    def test_pre_insert_trim_passes_the_cache_hold_back(self):
        self.assertEqual(self._run(128), 128)

    def test_no_hold_back_without_bounded_replay(self):
        self.assertEqual(self._run(0), 0)


if __name__ == "__main__":
    unittest.main()
