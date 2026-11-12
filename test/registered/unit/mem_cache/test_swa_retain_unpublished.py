"""Property tests for SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.

The switch keeps a request's not-yet-published SWA alive across the trim that
runs immediately before an insert. Without it that trim frees SWA the insert is
about to adopt, the new node keeps only a tombstone, and a first-touch request
whose shared prefix ends before the cached key's end reports cached_tokens = 0.

These are property tests: they assert invariants over a geometry sweep, not the
numbers one particular implementation happens to produce. They cover both
`window % page != 0` (128/256, 512/288, 384/256) and the divisible case
(512/256), because the frontier arithmetic floors to a page and the two classes
take different branches there.

Invariants (task contract, implementation requirement 4):
  1. Nothing below cache_protected_len is ever freed.
  2. The retained span is the whole not-yet-published range, not one window,
     and the frontier stays strictly below the insert boundary so the node the
     insert publishes is not an all-tombstone leaf.
  3. No slot leak: if the request's own SWA is freed afterwards, the pool returns
     to its starting level.
Plus the pressure fallback: retain only while the pool's free ratio is at or
above the threshold, and fall back to the original trimming otherwise.
"""

import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import (
    SWA_RETAIN_TRACE,
    free_swa_out_of_window_slots,
    page_align_floor,
    swa_available_ratio,
    swa_retain_decision,
    swa_retain_trace_summary,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (window, page) pairs: the first three are `window % page != 0`, the last divides.
GEOMETRIES = [(128, 256), (512, 288), (384, 256), (512, 256)]

# A pool large enough that the default threshold never trips, so the retain path
# is what these geometry tests exercise.
POOL_CAPACITY = 1_000_000
POOL_FREE = 900_000


class _FakeAllocator:
    """Minimal stand-in exposing what the retain guard reads.

    Attribute names mirror the real hybrid-SWA allocator
    (`swa_available_size()` / `current_swa_capacity`), not invented ones: an
    earlier version of the guard read a name the real pool does not have, which
    made every call fail closed and silently disabled the whole switch.
    """

    def __init__(self, free, capacity, *, capacity_attr="current_swa_capacity"):
        self._free = free
        setattr(self, capacity_attr, capacity)
        self.freed = []

    def swa_available_size(self):
        return self._free

    def free_swa_segment(self, indices, *, start_pos):
        # Record the exact span the caller released, and book it as a leak check.
        self.freed.append((start_pos, start_pos + len(indices)))
        self._free += len(indices)


def _make_req(*, protected, allocated, evicted=0, evict_floor=0, row=None):
    pool = SimpleNamespace(
        req_to_token=row if row is not None else torch.arange(4096).reshape(1, -1)
    )
    kv = ReqKvInfo(
        req_pool_idx=0,
        cache_protected_len=protected,
        kv_allocated_len=allocated,
        swa_evict_floor=evict_floor,
    )
    kv.set_evicted_seqlen(ComponentType.SWA, evicted)
    return SimpleNamespace(kv=kv), pool


def _call(
    req,
    pool,
    allocator,
    *,
    pre_len,
    window,
    page,
    retain,
    ratio=None,
):
    """Invoke the real entry point with the given switch/guard situation."""
    with envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(retain):
        if ratio is None:
            ratio = allocator.swa_available_size() / allocator.current_swa_capacity
        with envs.SGLANG_SWA_RETAIN_MIN_FREE_RATIO.override(ratio):
            free_swa_out_of_window_slots(
                req,
                pre_len,
                sliding_window_size=window,
                page_size=page,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                retain_unpublished=retain,
                retain_site="pre_insert:test",
            )


class TestSwaRetainUnpublishedInvariants(unittest.TestCase):
    def test_nothing_below_cache_protected_len_is_freed(self):
        """Invariant 1, both geometries, retain on and off."""
        for (window, page), retain in product(GEOMETRIES, [False, True]):
            with self.subTest(window=window, page=page, retain=retain):
                protected = page * 3
                allocated = page * 10
                req, pool = _make_req(protected=protected, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _call(
                    req,
                    pool,
                    allocator,
                    pre_len=allocated - 1,
                    window=window,
                    page=page,
                    retain=retain,
                )
                for start, _end in allocator.freed:
                    self.assertGreaterEqual(
                        start,
                        protected,
                        "the trim freed a slot below cache_protected_len",
                    )
                # The request's own frontier must never sit below the protected line.
                self.assertGreaterEqual(
                    req.kv.get_evicted_seqlen(ComponentType.SWA), protected
                )

    def test_retain_keeps_the_whole_unpublished_span_not_one_window(self):
        """Invariant 2: the frontier is held at the tree-owned boundary.

        With the switch on, the pre-insert trim must not advance the frontier into
        `[cache_protected_len, pre_len]`, the span the insert is about to adopt.
        That span is what a later first-touch match on the new node needs, and it
        is independent of `sliding_window_size` -- which is why every D in the
        grid hits rather than only the tail window.
        """
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                protected = 0
                allocated = page * 16
                pre_len = allocated - 1
                req, pool = _make_req(protected=protected, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _call(
                    req,
                    pool,
                    allocator,
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    retain=True,
                )
                frontier = req.kv.get_evicted_seqlen(ComponentType.SWA)
                old_threshold = page_align_floor(
                    max(0, pre_len - max(window, page)), page
                )
                self.assertEqual(
                    frontier,
                    protected,
                    "retained frontier is not held at the tree-owned boundary",
                )
                self.assertLess(
                    frontier,
                    old_threshold,
                    "retain did not actually hold anything back",
                )
                # Page alignment is a hard contract of the frontier.
                self.assertEqual(frontier % page, 0)
                # The cap must not depend on the window: a wider window would
                # otherwise retain a wider tail, which is the "one window"
                # behaviour this invariant rules out.
                req_w, pool_w = _make_req(protected=protected, allocated=allocated)
                alloc_w = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _call(
                    req_w,
                    pool_w,
                    alloc_w,
                    pre_len=pre_len,
                    window=window * 4,
                    page=page,
                    retain=True,
                )
                self.assertEqual(
                    req_w.kv.get_evicted_seqlen(ComponentType.SWA),
                    frontier,
                    "retained span moves with the window size => only one window kept",
                )
                # Contrast: without retain the old frontier is reached, which is
                # exactly the tombstone source.
                req2, pool2 = _make_req(protected=protected, allocated=allocated)
                alloc2 = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _call(
                    req2,
                    pool2,
                    alloc2,
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    retain=False,
                )
                self.assertGreaterEqual(
                    req2.kv.get_evicted_seqlen(ComponentType.SWA), old_threshold
                )

    def test_retain_holds_only_up_to_what_the_tree_owns(self):
        """A request that already matched a prefix keeps its own protected span.

        The cap is the tree-owned boundary, so a request whose prefix is already
        published retains nothing extra -- the retained span is the newly written
        tail, not the whole sequence.
        """
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                protected = page * 4
                allocated = page * 16
                req, pool = _make_req(protected=protected, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _call(
                    req,
                    pool,
                    allocator,
                    pre_len=allocated - 1,
                    window=window,
                    page=page,
                    retain=True,
                )
                self.assertEqual(
                    req.kv.get_evicted_seqlen(ComponentType.SWA),
                    protected,
                    "retain moved the frontier off the tree-owned boundary",
                )

    def test_retained_frontier_leaves_a_live_leaf(self):
        """Invariant 2 (leaf half): the adopted span keeps live SWA.

        The insert publishes up to page_floor(pre_len). A leaf is all-tombstone
        when the frontier has reached that boundary, so the retained frontier must
        stay strictly below it, leaving the adopted span's SWA live.
        """
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                allocated = page * 12
                pre_len = allocated - 1
                req, pool = _make_req(protected=0, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _call(
                    req,
                    pool,
                    allocator,
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    retain=True,
                )
                frontier = req.kv.get_evicted_seqlen(ComponentType.SWA)
                insert_boundary = page_align_floor(max(0, pre_len), page)
                self.assertLess(
                    frontier,
                    insert_boundary,
                    "frontier reached the insert boundary => all-tombstone leaf",
                )
                # The first-touch request needs a window ending at the adopted
                # span's end; its start must sit at or above the frontier.
                adopted_end = insert_boundary
                need_lo = adopted_end - window
                self.assertGreaterEqual(
                    need_lo,
                    frontier,
                    "the adopted span's own trailing window is already tombstoned",
                )

    def test_no_slot_leak_when_the_request_finishes(self):
        """Invariant 3: freeing afterwards returns the pool to its start."""
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                allocated = page * 16
                start_free = POOL_FREE
                req, pool = _make_req(protected=0, allocated=allocated)
                allocator = _FakeAllocator(start_free, POOL_CAPACITY)
                _call(
                    req,
                    pool,
                    allocator,
                    pre_len=allocated - 1,
                    window=window,
                    page=page,
                    retain=True,
                )
                frontier = req.kv.get_evicted_seqlen(ComponentType.SWA)
                # Whatever the trim held back, the request still owns
                # [frontier, allocated); releasing exactly that span must restore
                # the pool, so nothing is stranded by retaining.
                span = allocated - frontier
                if span > 0:
                    allocator.free_swa_segment(
                        torch.arange(frontier, allocated), start_pos=frontier
                    )
                self.assertEqual(
                    allocator.swa_available_size() - start_free,
                    span,
                    "retaining stranded slots that the request could not release",
                )

    def test_freed_spans_never_overlap(self):
        """A slot must not be handed back twice along the page-floored frontier."""
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                allocated = page * 20
                req, pool = _make_req(protected=0, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                # Trim at successive seqlen values, as decode steps would.
                for step in range(page, allocated, page):
                    _call(
                        req,
                        pool,
                        allocator,
                        pre_len=step,
                        window=window,
                        page=page,
                        retain=True,
                    )
                covered = []
                for start, end in allocator.freed:
                    covered.extend(range(start, end))
                self.assertEqual(
                    len(covered),
                    len(set(covered)),
                    "the same SWA slot was freed more than once",
                )


class TestSwaRetainPressureFallback(unittest.TestCase):
    def test_falls_back_when_the_pool_is_below_the_threshold(self):
        """A tight pool takes the original trimming path, not the retain path."""
        window, page = 128, 256
        allocated = page * 12
        pre_len = allocated - 1

        for free, capacity, should_retain in [
            (900_000, 1_000_000, True),  # 0.90 free
            (50_000, 1_000_000, False),  # 0.05 free, below a 0.10 threshold
        ]:
            with self.subTest(free=free):
                req, pool = _make_req(protected=0, allocated=allocated)
                allocator = _FakeAllocator(free, capacity)
                with (
                    envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(True),
                    envs.SGLANG_SWA_RETAIN_MIN_FREE_RATIO.override(0.10),
                ):
                    retain, _reason = swa_retain_decision(
                        allocator, site="pre_insert:test"
                    )
                    self.assertEqual(retain, should_retain)
                    _call(
                        req,
                        pool,
                        allocator,
                        pre_len=pre_len,
                        window=window,
                        page=page,
                        retain=retain,
                        ratio=0.10,
                    )
                frontier = req.kv.get_evicted_seqlen(ComponentType.SWA)
                old_threshold = page_align_floor(
                    max(0, pre_len - max(window, page)), page
                )
                if should_retain:
                    self.assertLess(frontier, old_threshold)
                else:
                    self.assertGreaterEqual(
                        frontier,
                        old_threshold,
                        "pressure fallback did not restore the original trim",
                    )

    def test_unreadable_pool_reading_falls_back(self):
        """Fail-closed: if free space cannot be read, do not retain."""

        class _Unreadable:
            current_swa_capacity = 1_000_000

            def swa_available_size(self):
                raise RuntimeError("pool reporting is unavailable")

        with envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(True):
            ratio = swa_available_ratio(_Unreadable())
            self.assertIsNone(ratio)
            retain, reason = swa_retain_decision(_Unreadable(), site="pre_insert:test")
        self.assertFalse(retain)
        self.assertEqual(reason, "fallback_unreadable")

    def test_ratio_reads_the_real_allocator_attribute_names(self):
        """The guard must read the names the real pool actually exposes.

        Regression guard for a shipped bug: the first version read
        `swa_max_total_num_tokens`, which the hybrid-SWA allocator does not
        define, so `swa_available_ratio` returned None on every call and the
        switch silently fell back to the old trim on every request. A fake
        allocator that mirrors the real attribute names is what catches this.
        """
        allocator = _FakeAllocator(750_000, 1_000_000)
        self.assertAlmostEqual(swa_available_ratio(allocator), 0.75)
        # `size_swa` is the fallback when the live capacity is unavailable.
        fallback = _FakeAllocator(250_000, 1_000_000, capacity_attr="size_swa")
        self.assertAlmostEqual(swa_available_ratio(fallback), 0.25)
        # Neither name present => unreadable, not a crash and not a blind retain.
        bare = SimpleNamespace(swa_available_size=lambda: 10)
        self.assertIsNone(swa_available_ratio(bare))
        # A zero capacity is not a usable denominator.
        empty = _FakeAllocator(0, 0)
        self.assertIsNone(swa_available_ratio(empty))

    def test_switch_off_never_retains(self):
        with envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(False):
            retain, reason = swa_retain_decision(
                _FakeAllocator(POOL_FREE, POOL_CAPACITY), site="pre_insert:test"
            )
        self.assertFalse(retain)
        self.assertEqual(reason, "switch_off")

    def test_zero_threshold_retains_when_any_space_remains(self):
        with envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(True):
            with envs.SGLANG_SWA_RETAIN_MIN_FREE_RATIO.override(0.0):
                retain, _ = swa_retain_decision(
                    _FakeAllocator(1, POOL_CAPACITY), site="pre_insert:test"
                )
        self.assertTrue(retain)


class TestSwaRetainTrace(unittest.TestCase):
    def test_trace_counts_both_branches_per_site(self):
        """The TRACE must distinguish retained from fell-back, per call site."""
        SWA_RETAIN_TRACE.clear()
        window, page = 128, 256
        allocated = page * 12

        with envs.SGLANG_DEBUG_SWA_RETAIN_TRACE.override(True):
            with envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(True):
                with envs.SGLANG_SWA_RETAIN_MIN_FREE_RATIO.override(0.10):
                    healthy = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                    retain, reason = swa_retain_decision(
                        healthy, site="pre_insert:checkpoint_into_tree"
                    )
                    self.assertTrue(retain)
                    self.assertEqual(reason, "retained")

                    drained = _FakeAllocator(0, POOL_CAPACITY)
                    retain2, reason2 = swa_retain_decision(
                        drained, site="pre_insert:advance_unpublished_req"
                    )
                    self.assertFalse(retain2)
                    self.assertEqual(reason2, "fallback_pressure")

        summary = swa_retain_trace_summary()
        self.assertEqual(summary.get("pre_insert:checkpoint_into_tree/retained"), 1)
        self.assertEqual(
            summary.get("pre_insert:advance_unpublished_req/fallback_pressure"), 1
        )
        self.assertNotIn("decode/retained", summary)
        SWA_RETAIN_TRACE.clear()


if __name__ == "__main__":
    unittest.main()