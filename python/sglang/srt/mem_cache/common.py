from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, cast

import numpy as np
import torch

from sglang.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_serving, get_spec
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class RetractionBackup(NamedTuple):
    cpu_tensors: Any = None
    host_indices: Optional[torch.Tensor] = None
    pool_transfers: Optional[list[PoolTransfer]] = None
    # Set when the KV pool leaves the recurrent state to the caller.
    mamba_cpu: Any = None


def kv_to_page_indices(kv_indices: torch.Tensor, page_size: int) -> np.ndarray:
    return (kv_indices[::page_size] // page_size).cpu().numpy()


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


# ---- SWA retain-unpublished diagnostics (one-shot, env-gated) ----
#
# Counters are per call site so a single run proves which path actually retained:
#   "pre_insert:<caller>"  -- the trim that runs immediately before an insert
#                             (UnifiedRadixCache.checkpoint_into_tree /
#                              UnifiedRadixCache.advance_unpublished_req /
#                              SWAComponent.cleanup_after_caching_req)
#   "decode"               -- the periodic decode-side eviction
# The retained count proves the new branch executed; the fallback counters prove
# the pressure guard was reached instead of silently retaining through a full pool.
SWA_RETAIN_TRACE: dict[str, int] = {}
_SWA_RETAIN_TRACE_LOGGED = False


def swa_retain_trace_enabled() -> bool:
    return envs.SGLANG_DEBUG_SWA_RETAIN_TRACE.get()


def _swa_retain_trace_record(site: str, branch: str) -> None:
    if not swa_retain_trace_enabled():
        return
    key = f"{site}/{branch}"
    SWA_RETAIN_TRACE[key] = SWA_RETAIN_TRACE.get(key, 0) + 1
    logger.info("[SWA-RETAIN-TRACE] %s -> %s (total=%d)", site, branch, SWA_RETAIN_TRACE[key])


def _swa_retain_detail(
    site: str,
    *,
    pre_len: int,
    protected: int,
    dead_lo: int,
    old_threshold: int,
    new_frontier: int,
    window: int,
    page: int,
) -> None:
    """One line per trim carrying the numbers the verdict depends on.

    The counters above say which branch ran; this says what it computed, so a
    frontier that refuses to move can be read off the log instead of inferred.
    """
    if not swa_retain_trace_enabled():
        return
    logger.info(
        "[SWA-RETAIN-DETAIL] site=%s pre_len=%d protected=%d dead_lo=%d "
        "old_threshold=%d new_frontier=%d window=%d page=%d",
        site,
        pre_len,
        protected,
        dead_lo,
        old_threshold,
        new_frontier,
        window,
        page,
    )


def swa_retain_trace_summary() -> dict[str, int]:
    """Snapshot of the counters; also usable after a run for the evidence file."""
    return dict(SWA_RETAIN_TRACE)


def _read_swa_free_tokens(token_to_kv_pool_allocator: Any) -> Optional[int]:
    available = getattr(token_to_kv_pool_allocator, "swa_available_size", None)
    if available is None:
        return None
    try:
        return int(available() if callable(available) else available)
    except Exception:  # pragma: no cover - reporting must not break eviction
        return None


def _read_swa_capacity_tokens(token_to_kv_pool_allocator: Any) -> Optional[int]:
    """Live SWA capacity in tokens.

    The hybrid-SWA allocators expose `current_swa_capacity` (free + allocated),
    which is the same denominator `swa_capacity_and_available` reports and so the
    one the pool-usage metric is computed against. `size_swa` is the configured
    bound; it is a fallback, not the first choice, because the byte-coordinated
    live capacity can sit below it.
    """
    for name in ("current_swa_capacity", "size_swa"):
        value = getattr(token_to_kv_pool_allocator, name, None)
        if value is None:
            continue
        try:
            capacity = int(value() if callable(value) else value)
        except Exception:  # pragma: no cover
            continue
        if capacity > 0:
            return capacity
    return None


def swa_available_ratio(token_to_kv_pool_allocator: Any) -> Optional[float]:
    """Free ratio of the SWA pool, or None when the pool cannot report it.

    Returns None (rather than raising) for a non-hybrid-SWA allocator, an
    unreadable counter, or a zero capacity; the caller treats that as fail-closed.
    """
    free_tokens = _read_swa_free_tokens(token_to_kv_pool_allocator)
    if free_tokens is None:
        return None
    capacity = _read_swa_capacity_tokens(token_to_kv_pool_allocator)
    if capacity is None or capacity <= 0:
        return None
    return free_tokens / capacity


def swa_retain_decision(
    token_to_kv_pool_allocator: Any, *, site: str
) -> tuple[bool, str]:
    """Decide whether to retain, applying the pressure fallback.

    Returns ``(retain, reason)``. Never raises: an unreadable pool reading falls
    back to the original trimming behavior (fail-closed, hits are lost rather
    than slots exhausted).

    This is the single place that records the TRACE counter, so a fallback is
    counted once rather than once per caller.
    """
    if not envs.SGLANG_OPT_SWA_RETAIN_UNPUBLISHED.get():
        return False, "switch_off"
    min_ratio = envs.SGLANG_SWA_RETAIN_MIN_FREE_RATIO.get()
    ratio = swa_available_ratio(token_to_kv_pool_allocator)
    if ratio is None:
        _swa_retain_trace_record(site, "fallback_unreadable")
        return False, "fallback_unreadable"
    if ratio < min_ratio:
        _swa_retain_trace_record(site, "fallback_pressure")
        return False, "fallback_pressure"
    _swa_retain_trace_record(site, "retained")
    return True, "retained"


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    supports_prefix_sharing: bool = True,
    retain_floor: int | None = None,
    component_type: ComponentType = ComponentType.SWA,
    free_segment: Callable[..., None] | None = None,
    eviction_interval: int = 1,
    retain_unpublished: bool = False,
    retain_site: str = "unknown",
) -> None:
    if not req.kv.holds_kv:
        return

    # For SWA-capable tree caches, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert req.kv.cache_protected_len % page_size == 0, (
        "cache_protected_len must be page aligned"
    )
    # Protected rows limit what can be freed, not where the interval starts.
    evicted_seqlen = req.kv.get_evicted_seqlen(component_type)
    if pre_len - sliding_window_size < evicted_seqlen + eviction_interval:
        return
    dead_lo = (
        req.kv.swa_dead_lo(page_size)
        if component_type == ComponentType.SWA
        else req.kv.cache_protected_len
    )
    evicted_seqlen = max(evicted_seqlen, dead_lo)
    req.kv.set_evicted_seqlen(component_type, evicted_seqlen)

    if not supports_prefix_sharing:
        # Nothing is inserted into a tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Prefix-sharing cache: keep max(window, page). The trailing floor page-aligns the
        # frontier, and subtracting at least one page keeps it below the insert
        # boundary (page_floor(seq_len)) so the last leaf is never all-tombstone.
        # No extra page margin is needed.
        evict_threshold = pre_len - max(sliding_window_size, page_size)
    if retain_floor is not None and supports_prefix_sharing:
        # The caller owns where the floor is (see BasePrefixCache.swa_retain_floor);
        # this only promises not to free past it. Without prefix sharing a retained
        # checkpoint could never be matched, so holding it is pure cost.
        evict_threshold = min(evict_threshold, retain_floor)

    if retain_unpublished and supports_prefix_sharing:
        # This trim runs immediately before the caller's insert. The insert adopts
        # `[cache_protected_len, pre_len]` and stamps that node's SWA window start
        # from the frontier this function leaves behind. If the frontier lands
        # inside the span being adopted, the node is born with its SWA already
        # clipped: a later first-touch request whose shared prefix ends before the
        # cached key's end asks for a window starting below that frontier,
        # `_clip_to_window` returns None, and the match is rejected =>
        # `cached_tokens = 0`.
        #
        # So do not advance the frontier past what the tree already owns. The
        # request keeps holding the not-yet-published span, the insert publishes
        # it, and the tree's SWA LRU reclaims it afterwards like any other
        # tree-owned SWA. This is a whole-span cap, not a one-window one: it is
        # independent of `sliding_window_size`, which is what makes every D in the
        # first-touch grid hit rather than only the tail window.
        #
        # `dead_lo` already bounds the frontier from below, so clamping here can
        # only hold slots back, never release more.
        retain_cap = dead_lo
        evict_threshold = min(evict_threshold, retain_cap)

    new_evicted_seqlen = max(evicted_seqlen, evict_threshold)

    if page_size > 1:
        new_evicted_seqlen = (new_evicted_seqlen // page_size) * page_size

    _swa_retain_detail(
        retain_site,
        pre_len=pre_len,
        protected=req.kv.cache_protected_len,
        dead_lo=dead_lo,
        old_threshold=evict_threshold,
        new_frontier=new_evicted_seqlen,
        window=sliding_window_size,
        page=page_size,
    )

    if new_evicted_seqlen > evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, evicted_seqlen:new_evicted_seqlen
        ]
        if free_segment is None:
            assert component_type == ComponentType.SWA
            free_segment = token_to_kv_pool_allocator.free_swa_segment
        free_segment(free_slots, start_pos=evicted_seqlen)
        req.kv.set_evicted_seqlen(component_type, new_evicted_seqlen)


def coalesce_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge adjacent half-open ranges so a split that falls mid-page frees that page once."""
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def free_kv_row_segments(
    allocator: BaseTokenToKVPoolAllocator,
    segments: list[tuple[torch.Tensor, int]],
    *,
    swa_evicted_seqlen: int,
    swa_dead_lo: int = 0,
) -> None:
    """Free ascending disjoint ``(kv_indices, start_pos)`` segments of one
    request's kv row; ``[swa_dead_lo, swa_evicted_seqlen)`` goes back full-side only."""
    dead_lo, dead_hi = swa_dead_lo, max(swa_evicted_seqlen, swa_dead_lo)
    swa_dead: list[tuple[torch.Tensor, int]] = []
    swa_alive: list[tuple[torch.Tensor, int]] = []
    for kv_indices, start_pos in segments:
        end_pos = start_pos + kv_indices.numel()
        lo = min(max(dead_lo, start_pos), end_pos)
        hi = min(max(dead_hi, start_pos), end_pos)
        # start_pos <= lo <= hi <= end_pos
        # inside [lo, hi) is the swa dead segment
        if hi <= lo:
            swa_alive.append((kv_indices, start_pos))
            continue
        swa_dead.append((kv_indices[lo - start_pos : hi - start_pos], lo))

        if lo > start_pos:
            swa_alive.append((kv_indices[: lo - start_pos], start_pos))
        if end_pos > hi:
            swa_alive.append((kv_indices[hi - start_pos :], hi))

    if swa_dead and swa_alive:
        # The two sides are separate calls, so neither one's page-disjointness
        # check sees a boundary that splits a page between them.
        assert swa_evicted_seqlen % allocator.page_size == 0, (
            f"SWA eviction cursor {swa_evicted_seqlen} splits a page "
            f"(page_size {allocator.page_size})"
        )
    if swa_dead:
        allocator.free_full_segments(swa_dead)
    if swa_alive:
        allocator.free_segments(swa_alive)


def _req_radix_key(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: array[int],
    max_prefix_len: Optional[int] = None,
) -> RadixKey:
    # unified_kv SWA lives in a per-request ring the tree never stores, so a reused
    # prefix carries stale SWA; cap the match so the window is re-prefilled.
    reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None
    if max_prefix_len is not None:
        key_limit = (
            max_prefix_len if key_limit is None else min(key_limit, max_prefix_len)
        )
    return RadixKey(
        token_ids=token_ids,
        extra_key=req.extra_key,
        limit=key_limit,
        cache_salt=req.cache_salt,
    )


def touch_waiting_prefix(req: Req, tree_cache: BasePrefixCache) -> None:
    tree_cache.touch_prefix(
        _req_radix_key(tree_cache, req, req.origin_input_ids + req.output_ids)
    )


def match_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    token_ids: Optional[array] = None,
    *,
    cow_mamba: bool = False,
    max_prefix_len: Optional[int] = None,
) -> MatchResult:
    """Match token_ids against tree_cache and adopt the hit as req's prefix."""
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=_req_radix_key(tree_cache, req, token_ids, max_prefix_len),
            cow_mamba=cow_mamba,
            req=req,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(
            tree_cache, match_result, extra_key=req.extra_key
        )
    req.prefix_len = match_result.device_prefix_len
    (
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(req.prefix_len + req.host_hit_length, max_len)
    req.swa_branching_seqlen = match_result.swa_branching_seqlen
    # A probe match keeps what it did not report; a new round resets both.
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.kv.cache_protected_len = match_result.cache_protected_len
    return match_result


def checkpoint_kv_cache(req: Req, tree_cache: BasePrefixCache) -> None:
    """Publish what the running request has computed so far, unless it is
    barred from the tree."""
    # The tree reads req.finished() to tell a checkpoint from the final
    # insert; a finished request belongs in release_kv_cache.
    assert not req.finished(), f"checkpointing finished request {req.rid}"
    if not req.skip_radix_cache_insert:
        tree_cache.checkpoint(req, up_to=req.extend_end)
    # The next extend resumes after this one, published or not.
    req.prefix_len = req.extend_end


def evict_from_tree_cache(
    tree_cache: BasePrefixCache | None, num_tokens: int
) -> bool | None:
    if tree_cache is not None and tree_cache.supports_prefix_sharing():
        return tree_cache.token_to_kv_pool_allocator.evict_to_free_tokens(
            tree_cache, num_tokens
        )


def _evict_until_allocatable(
    tree_cache: BasePrefixCache, allocator, num_tokens: int
) -> None:
    """Keep evicting the shortfall until `num_tokens` are allocatable.

    Under classed page sharding available_size() reports the MIN-CLASS
    capacity floor, so a single evict() sized in tokens can raise that floor by
    less than the number of tokens it freed: the evicted pages spread across
    all owner classes. Looping is deterministic, so it stays mirrored across
    the ranks of a shard group. Stock allocators need no extra pass.
    """
    if page_interleave_shard_size(allocator) <= 1:
        return
    while True:
        available_size = allocator.available_size()
        if available_size >= num_tokens:
            return
        shortfall = num_tokens - available_size
        result = tree_cache.evict(
            EvictParams(num_tokens=max(shortfall, allocator.page_size))
        )
        if result.num_tokens_evicted == 0:
            return


def dsv4_needs_rebootstrap(
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
) -> bool:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    pool = token_to_kv_pool_allocator.get_kvcache()
    # No DeepSeek-V4 pool keeps a host copy of a request's KV; the prefill recomputes it.
    return isinstance(pool, DeepSeekV4TokenToKVPool)


def backup_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    backend: str,
) -> bool:
    """Returns False when the request can be neither backed up nor recomputed;
    the caller aborts it."""
    if backend == "none":
        return False
    if dsv4_needs_rebootstrap(token_to_kv_pool_allocator):
        if req.multimodal_inputs is not None:
            # The recompute payload carries token ids only; images would be lost.
            return False
        # Drain the in-flight forward before its slots can receive recomputed KV.
        device = token_to_kv_pool_allocator.get_kvcache().device
        torch.get_device_module(device).synchronize(device)
        return True
    if backend == "cpu_tensor":
        req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
        return True
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.seqlen <= 1:
        return True

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    req.kv.retraction_backup = unified_cache.backup_kv_cache(req)
    return req.kv.retraction_backup is not None


def restore_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    backend: str,
) -> None:
    if backend == "cpu_tensor":
        req.load_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.seqlen <= 1:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    assert req.kv.retraction_backup is not None
    unified_cache.restore_kv_cache(req, req.kv.retraction_backup)
    req.kv.retraction_backup = None


def discard_kv_cache_backup(
    req: Req, tree_cache: BasePrefixCache, backend: str
) -> None:
    if backend == "cpu_tensor":
        req.kv.retraction_backup = None
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.kv.retraction_backup is None:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    unified_cache.discard_kv_cache_backup(req.kv.retraction_backup)
    req.kv.retraction_backup = None


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, *, checkpoint: bool):
    """Give the request's kv row back; with ``checkpoint`` the tree first keeps
    what it can key."""
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    # A mamba-capable cache may alloc mamba state before alloc KV cache
    if not req.kv.holds_kv:
        assert tree_cache.supports_mamba(), (
            "Only a mamba-capable tree cache allows freeing before alloc"
        )
        # TODO (csy, hanming): clean up this early allocation logic
        if req.kv.holds_mamba:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None
        return
    if tree_cache.claim_kv_row(req):
        # A streaming session detached the kv record to keep the row.
        assert not req.kv.holds_kv
        return

    owned_kv_len = req.owned_kv_len()
    checkpoint = checkpoint and not req.skip_radix_cache_insert
    if checkpoint:
        # A tree that takes over component state (mamba) must see the request
        # finished, or the insert forks the state and the slot leaks.
        assert req.finished() or not tree_cache.supports_mamba(), (
            f"releasing unfinished request {req.rid} into a mamba tree"
        )
        # The fill-id array lags output_ids until the next prepare_for_decode.
        req.refresh_fill_ids()
        tree_cache.checkpoint(req, up_to=owned_kv_len)
    # The protected prefix is not this req's to free.
    tree_cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, owned_kv_len)])
    tree_cache.unlock(req.lock)
    req.lock = None
    _release_overallocated_kv_indices(
        req, owned_kv_len, req.kv.kv_allocated_len, tree_cache
    )
    tree_cache.on_release(req, checkpointed=checkpoint)

    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_kv_released()


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
) -> None:
    allocator = tree_cache.token_to_kv_pool_allocator
    page_size = allocator.page_size
    spec_algo = get_spec().speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not get_serving().strip_thinking_cache:
        # A stop landing before the last committed token does the same, via
        # effective_kv_committed_len().
        assert start_p == end_p or (
            req.finished_len is not None
            and len(req.origin_input_ids) + req.finished_len < req.kv.kv_committed_len
        ), (
            f"Unexpected overallocated KV cache, {req.kv.kv_committed_len=}, {req.kv.kv_allocated_len=}"
        )

    # Align to the ALLOCATOR's page, which under DCP is wider than the kernel
    # page: paged free() releases the whole page containing any freed index, so
    # a boundary aligned only to the kernel page could free a widened page whose
    # head rows are still live.
    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        # start_p is aligned to the allocator's page above, so it never shares a
        # page with the tail free_kv_row in this group.
        tree_cache.free_kv_row(req.kv, [(start_p, end_p)])


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
