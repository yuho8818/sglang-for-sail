"""First-touch rescue unit tests for defer_last_window (TH-S6, DSV41-S5.13 port).

The pre-insert trim (cache_unfinished_req) and the decode-step trim keep one
max(window, page) range above the eviction frontier alive so the radix insert
that follows can capture it as tree-owned SWA; the next trim releases it if
no insert captured it, so nothing leaks.

Ported-from: 47fe6a08c2 / d8196160f6 (unit semantics rewritten for the
0.5.21+rc1 component-generic free_swa_out_of_window_slots).
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self):
        self.frees: list[tuple[int, int]] = []  # (start_pos, num_slots)

    def free_swa_segment(self, slots, start_pos: int = 0):
        self.frees.append((start_pos, int(len(slots))))


class _FakeKv:
    holds_kv = True
    cache_protected_len = 0
    req_pool_idx = 0

    def __init__(self, evicted: int = 0):
        self._evicted = evicted

    def get_evicted_seqlen(self, component_type) -> int:
        return self._evicted

    def set_evicted_seqlen(self, component_type, value: int) -> None:
        self._evicted = value

    def swa_dead_lo(self, page_size: int) -> int:
        return 0


def _make_req(req_pool_idx: int = 0, evicted: int = 0):
    return SimpleNamespace(kv=_FakeKv(evicted), req_pool_idx=req_pool_idx)


def _make_pool(num_tokens: int = 512):
    return SimpleNamespace(
        req_to_token=torch.zeros((1, num_tokens), dtype=torch.int64)
    )


def _trim(req, pre_len, *, window, page, defer, chunk=False, evicted_start=0):
    alloc = _RecordingAllocator()
    free_swa_out_of_window_slots(
        req,
        pre_len,
        sliding_window_size=window,
        page_size=page,
        req_to_token_pool=_make_pool(),
        token_to_kv_pool_allocator=alloc,
        is_chunk_cache=chunk,
        defer_last_window=defer,
    )
    return alloc


def test_pre_insert_trim_keeps_one_window():
    # window=8, page=1: radix threshold = 100-8 = 92; defer keeps
    # keep_from = 92-8 = 84, so the frontier stops one window short.
    req = _make_req()
    alloc = _trim(req, 100, window=8, page=1, defer=True)
    assert req.kv._evicted == 84
    assert alloc.frees == [(0, 84)]


def test_decode_trim_deferred_too():
    # Same math as the pre-insert site (ScheduleBatch._evict_swa path).
    req = _make_req()
    alloc = _trim(req, 100, window=8, page=1, defer=True)
    assert req.kv._evicted == 84


def test_no_defer_frees_to_threshold():
    # Without defer (old behavior, e.g. cleanup_after_caching_req), the trim
    # advances all the way to the radix threshold.
    req = _make_req()
    alloc = _trim(req, 100, window=8, page=1, defer=False)
    assert req.kv._evicted == 92
    assert alloc.frees == [(0, 92)]


def test_chunk_cache_never_defers():
    # Chunk cache builds no radix tree; defer must not apply even if asked.
    req = _make_req()
    alloc = _trim(req, 100, window=8, page=1, defer=True, chunk=True)
    assert req.kv._evicted == 92  # pre_len - window, chunk branch


def test_deferred_window_is_released_by_later_trim():
    # No-leak: after a deferred trim to 84, a later trim at a larger pre_len
    # must release the kept range past the original threshold (92).
    req = _make_req()
    _trim(req, 100, window=8, page=1, defer=True)
    assert req.kv._evicted == 84
    _trim(req, 120, window=8, page=1, defer=True)
    # threshold = 120-8 = 112; keep_from = 112-8 = 104 >= original 92.
    assert req.kv._evicted == 104


def test_page_aligned_defer():
    # window=4, page=4: threshold = 100-4 = 96; keep_from = 96-4 = 92
    # (already page aligned).
    req = _make_req()
    _trim(req, 100, window=4, page=4, defer=True)
    assert req.kv._evicted == 92


def test_defer_floor_at_current_evicted():
    # A short pre_len keeps the frontier where it is instead of going
    # backwards: keep_from is clamped to the current evicted length.
    req = _make_req(evicted=80)
    _trim(req, 90, window=8, page=1, defer=True)
    # threshold = 90-8 = 82; keep_from = 82-8 = 74 -> clamped to 80.
    assert req.kv._evicted == 80


def test_hold_kv_guard():
    req = _make_req()
    req.kv.holds_kv = False
    alloc = _trim(req, 100, window=8, page=1, defer=True)
    assert req.kv._evicted == 0
    assert alloc.frees == []


def test_component_type_passthrough():
    # The generic setter must be used with the SWA component type.
    req = _make_req()
    seen = {}

    class _Kv(_FakeKv):
        def set_evicted_seqlen(self, component_type, value):
            seen["component_type"] = component_type
            super().set_evicted_seqlen(component_type, value)

    req.kv = _Kv()
    _trim(req, 100, window=8, page=1, defer=True)
    assert seen["component_type"] == ComponentType.SWA


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
