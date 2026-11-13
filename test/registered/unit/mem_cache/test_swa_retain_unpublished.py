"""Property tests for SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.

The switch keeps a request's not-yet-published SWA alive across the trim that
runs immediately before an insert. Without it that trim frees SWA the insert is
about to adopt, the new node keeps only a tombstone, and a first-touch request
whose shared prefix ends before the cached key's end reports cached_tokens = 0.

The tests assert invariants over a geometry sweep covering `window % page != 0`
(128/256, 512/288, 384/256) and the divisible case (512/256):
  1. Nothing below cache_protected_len is ever freed.
  2. The retained span is the whole not-yet-published range, not one window,
     and the frontier stays strictly below the insert boundary.
  3. No slot leak and no double free.
Plus the pool guard: free plus tree-evictable SWA decides whether to retain.
"""

import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import (
    SWA_RETAIN_MIN_RECLAIMABLE_RATIO,
    free_swa_out_of_window_slots,
    page_align_floor,
    swa_pool_can_retain,
    swa_reclaimable_ratio,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (window, page) pairs: the first three are `window % page != 0`, the last divides.
GEOMETRIES = [(128, 256), (512, 288), (384, 256), (512, 256)]

POOL_CAPACITY = 1_000_000
POOL_FREE = 900_000


class _FakeAllocator:
    """Exposes the names the real hybrid-SWA allocator has
    (`swa_available_size()` / `current_swa_capacity`)."""

    def __init__(self, free, capacity, *, capacity_attr="current_swa_capacity"):
        self._free = free
        setattr(self, capacity_attr, capacity)
        self.freed = []

    def swa_available_size(self):
        return self._free

    def free_swa_segment(self, indices, *, start_pos):
        self.freed.append((start_pos, start_pos + len(indices)))
        self._free += len(indices)


def _make_req(*, protected, allocated, evicted=0, evict_floor=0):
    pool = SimpleNamespace(req_to_token=torch.arange(8192).reshape(1, -1))
    kv = ReqKvInfo(
        req_pool_idx=0,
        cache_protected_len=protected,
        kv_allocated_len=allocated,
        swa_evict_floor=evict_floor,
    )
    kv.set_evicted_seqlen(ComponentType.SWA, evicted)
    return SimpleNamespace(kv=kv), pool


def _trim(req, pool, allocator, *, pre_len, window, page, retain):
    free_swa_out_of_window_slots(
        req,
        pre_len,
        sliding_window_size=window,
        page_size=page,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        retain_unpublished=retain,
    )


def _old_threshold(pre_len, window, page):
    return page_align_floor(max(0, pre_len - max(window, page)), page)


class TestSwaRetainUnpublishedInvariants(unittest.TestCase):
    def test_nothing_below_cache_protected_len_is_freed(self):
        for (window, page), retain in product(GEOMETRIES, [False, True]):
            with self.subTest(window=window, page=page, retain=retain):
                protected = page * 3
                allocated = page * 10
                req, pool = _make_req(protected=protected, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _trim(
                    req,
                    pool,
                    allocator,
                    pre_len=allocated - 1,
                    window=window,
                    page=page,
                    retain=retain,
                )
                for start, _end in allocator.freed:
                    self.assertGreaterEqual(start, protected)
                self.assertGreaterEqual(
                    req.kv.get_evicted_seqlen(ComponentType.SWA), protected
                )

    def test_retain_keeps_the_whole_unpublished_span_not_one_window(self):
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                allocated = page * 16
                pre_len = allocated - 1

                frontiers = {}
                for w in (window, window * 4):
                    req, pool = _make_req(protected=0, allocated=allocated)
                    allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                    _trim(
                        req,
                        pool,
                        allocator,
                        pre_len=pre_len,
                        window=w,
                        page=page,
                        retain=True,
                    )
                    frontiers[w] = req.kv.get_evicted_seqlen(ComponentType.SWA)
                    self.assertEqual(allocator.freed, [])

                self.assertEqual(frontiers[window], 0)
                # A one-window rule would retain more for a wider window.
                self.assertEqual(frontiers[window * 4], frontiers[window])

                req, pool = _make_req(protected=0, allocated=allocated)
                _trim(
                    req,
                    pool,
                    _FakeAllocator(POOL_FREE, POOL_CAPACITY),
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    retain=False,
                )
                self.assertGreaterEqual(
                    req.kv.get_evicted_seqlen(ComponentType.SWA),
                    _old_threshold(pre_len, window, page),
                )

    def test_retain_holds_only_up_to_what_the_tree_owns(self):
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                protected = page * 4
                allocated = page * 16
                req, pool = _make_req(protected=protected, allocated=allocated)
                _trim(
                    req,
                    pool,
                    _FakeAllocator(POOL_FREE, POOL_CAPACITY),
                    pre_len=allocated - 1,
                    window=window,
                    page=page,
                    retain=True,
                )
                self.assertEqual(
                    req.kv.get_evicted_seqlen(ComponentType.SWA), protected
                )

    def test_retained_frontier_leaves_a_live_leaf(self):
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                allocated = page * 12
                pre_len = allocated - 1
                req, pool = _make_req(protected=0, allocated=allocated)
                _trim(
                    req,
                    pool,
                    _FakeAllocator(POOL_FREE, POOL_CAPACITY),
                    pre_len=pre_len,
                    window=window,
                    page=page,
                    retain=True,
                )
                frontier = req.kv.get_evicted_seqlen(ComponentType.SWA)
                insert_boundary = page_align_floor(pre_len, page)
                self.assertLess(frontier, insert_boundary)
                self.assertGreaterEqual(insert_boundary - window, frontier)

    def test_no_slot_leak_when_the_request_finishes(self):
        for window, page in GEOMETRIES:
            with self.subTest(window=window, page=page):
                allocated = page * 16
                req, pool = _make_req(protected=0, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                _trim(
                    req,
                    pool,
                    allocator,
                    pre_len=allocated - 1,
                    window=window,
                    page=page,
                    retain=True,
                )
                frontier = req.kv.get_evicted_seqlen(ComponentType.SWA)
                span = allocated - frontier
                allocator.free_swa_segment(
                    torch.arange(frontier, allocated), start_pos=frontier
                )
                self.assertEqual(allocator.swa_available_size() - POOL_FREE, span)

    def test_freed_spans_never_overlap(self):
        for (window, page), retain in product(GEOMETRIES, [False, True]):
            with self.subTest(window=window, page=page, retain=retain):
                allocated = page * 20
                req, pool = _make_req(protected=0, allocated=allocated)
                allocator = _FakeAllocator(POOL_FREE, POOL_CAPACITY)
                for step in range(page, allocated, page):
                    _trim(
                        req,
                        pool,
                        allocator,
                        pre_len=step,
                        window=window,
                        page=page,
                        retain=retain,
                    )
                covered = [
                    i for start, end in allocator.freed for i in range(start, end)
                ]
                self.assertEqual(len(covered), len(set(covered)))


class TestSwaRetainPoolGuard(unittest.TestCase):
    def test_evictable_tree_swa_counts_as_room(self):
        allocator = _FakeAllocator(50_000, POOL_CAPACITY)
        self.assertFalse(swa_pool_can_retain(allocator, evictable_tokens=0))
        self.assertTrue(swa_pool_can_retain(allocator, evictable_tokens=500_000))

    def test_threshold_boundary(self):
        at = int(POOL_CAPACITY * SWA_RETAIN_MIN_RECLAIMABLE_RATIO)
        self.assertTrue(swa_pool_can_retain(_FakeAllocator(at, POOL_CAPACITY), 0))
        self.assertFalse(
            swa_pool_can_retain(_FakeAllocator(at - 1, POOL_CAPACITY), 0)
        )

    def test_ratio_reads_the_real_allocator_attribute_names(self):
        self.assertAlmostEqual(
            swa_reclaimable_ratio(_FakeAllocator(250_000, POOL_CAPACITY), 500_000),
            0.75,
        )
        fallback = _FakeAllocator(250_000, POOL_CAPACITY, capacity_attr="size_swa")
        self.assertAlmostEqual(swa_reclaimable_ratio(fallback, 0), 0.25)
        bare = SimpleNamespace(swa_available_size=lambda: 10)
        self.assertIsNone(swa_reclaimable_ratio(bare, 0))
        self.assertIsNone(swa_reclaimable_ratio(_FakeAllocator(0, 0), 0))

    def test_ratio_is_capped_at_one(self):
        self.assertEqual(
            swa_reclaimable_ratio(
                _FakeAllocator(POOL_FREE, POOL_CAPACITY), POOL_FREE
            ),
            1.0,
        )

    def test_unreadable_pool_does_not_retain(self):
        class _Unreadable:
            current_swa_capacity = POOL_CAPACITY

            def swa_available_size(self):
                raise RuntimeError("pool reporting is unavailable")

        self.assertIsNone(swa_reclaimable_ratio(_Unreadable(), 0))
        self.assertFalse(swa_pool_can_retain(_Unreadable(), POOL_CAPACITY))


class TestSwaComponentPreInsertTrim(unittest.TestCase):
    """SWAComponent.free_out_of_window_slots is the only trim that may retain."""

    def _run(self, *, switch, free, evictable):
        calls = []

        def record(req, pre_len, *, retain_unpublished=False, match_hold_back=0):
            calls.append(retain_unpublished)

        comp = SimpleNamespace(
            cache=SimpleNamespace(
                token_to_kv_pool_allocator=_FakeAllocator(free, POOL_CAPACITY),
                swa_evictable_size=lambda: evictable,
                swa_reprefill_tail_tokens=lambda req=None: 0,
            ),
            component_type=ComponentType.SWA,
            _free_out_of_window_slots=record,
        )
        req = SimpleNamespace(kv=SimpleNamespace(get_evicted_seqlen=lambda ct: 0))
        params = SimpleNamespace(set_evicted_seqlen=lambda ct, value: None)
        with envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.override(switch):
            SWAComponent.free_out_of_window_slots(comp, req, 1023, params)
        self.assertEqual(len(calls), 1)
        return calls[0]

    def test_switch_off_never_retains(self):
        self.assertFalse(self._run(switch=False, free=POOL_FREE, evictable=0))

    def test_switch_on_retains_with_room(self):
        self.assertTrue(self._run(switch=True, free=POOL_FREE, evictable=0))
        self.assertTrue(self._run(switch=True, free=10_000, evictable=200_000))

    def test_switch_on_falls_back_when_pinned_rows_fill_the_pool(self):
        self.assertFalse(self._run(switch=True, free=10_000, evictable=10_000))


if __name__ == "__main__":
    unittest.main()
