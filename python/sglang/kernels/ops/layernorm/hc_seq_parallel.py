"""Sequence-parallel mHC coefficient statistics over the TP group (PPU).

The V4.1 mHC mixing weights (``[2*hc_mult + hc_mult^2, hc_mult*hidden]`` fp32)
are not TP-sharded, and the hc residual at every sublayer boundary is
replicated across the attention-TP group (attention TP all-reduces its output;
the EP MoE either all-reduces (a2a=none) or splits/gathers its rows
(``tensor_split`` + ``attn_tp_all_gather``)). Every rank therefore computes
identical coefficient statistics for the whole batch: at c128 decode that is
the dominant ``_hc_mix_stats_partial`` GEMM, 8x redundant.

This module splits the *row* dimension across the TP group, computes
``hc_mix_stats_sinkhorn`` for the local rows only, and all-gathers the packed
``(pre | post | comb)`` coefficients back to full width with one collective
per boundary:

- Row-level arithmetic is unchanged: ``hc_mix_stats_sinkhorn`` is batch
  invariant per row (the K slicing, BLOCK_K tiles and reduction order are
  independent of the row count, and BLOCK_M only re-tiles rows), so each row
  is *bit-identical* to the replicated path. The all-gather only moves bytes.
- ``M`` is padded up to ``world * ceil(M / world)``; padding rows hold
  garbage that is never read (the returned views stop at ``M``).
- The coefficients are tiny (``24`` fp32 per row), so the gathered volume is
  negligible next to the sharded statistics GEMM: c128 decode (M=768) gathers
  ~9 KB per boundary, ~0.8 MB per decode step.

The all-gather is issued on the caller's (main) compute stream, in layer
order, on every rank — M and the split bounds are identical across ranks — so
the collective ordering matches the existing all-reduces and the path is safe
under CUDA graph capture (chunk/lo/hi are capture-time constants).
"""

import torch

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs


def _seq_parallel_rows_active(m: int) -> bool:
    """Whether the statistics of an [m, hc_dim] input should be row-sharded.

    Only when the input is truly replicated across the TP group: DP attention
    and CP shard rows (each rank would then hold different rows and the gather
    would reassemble the wrong thing), and TP1 has nothing to shard.
    """
    if not envs.SGLANG_PPU_HC_SEQ_PARALLEL.get():
        return False
    if m < envs.SGLANG_PPU_HC_SEQ_PARALLEL_MIN_TOKENS.get():
        return False
    if not torch.distributed.is_initialized():
        return False
    group = get_tp_group()
    if group.world_size <= 1:
        return False
    from sglang.srt.runtime_context import get_forward, get_parallel

    parallel = get_parallel()
    return (
        parallel.attn_dp_size == 1
        and parallel.attn_cp_size <= 1
        and not get_forward().sp_active
    )


def hc_mix_stats_sinkhorn_seq_parallel(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
):
    """Row-sharded + all-gathered ``hc_mix_stats_sinkhorn``.

    Same contract and dtypes as
    :func:`sglang.kernels.ops.layernorm.mhc.hc_mix_stats_sinkhorn`; the
    returned ``pre``/``post``/``comb`` cover all rows. ``pre`` may be a strided
    view of the packed gather buffer (stride(1) == 1) — consumers must honor
    strides (``hc_combine`` does). ``post``/``comb`` are contiguous.
    """
    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn

    m = x_flat.shape[0]
    if not _seq_parallel_rows_active(m):
        return hc_mix_stats_sinkhorn(
            x_flat, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps
        )

    group = get_tp_group()
    world, rank = group.world_size, group.rank_in_group
    chunk = (m + world - 1) // world
    lo = rank * chunk
    hi = min(lo + chunk, m)

    mix = 2 * hc_mult + hc_mult * hc_mult
    send = torch.empty((chunk, mix), dtype=torch.float32, device=x_flat.device)
    if hi > lo:
        pre_s, post_s, comb_s = hc_mix_stats_sinkhorn(
            x_flat[lo:hi],
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_iters,
            rms_eps,
            hc_eps,
        )
        rows = hi - lo
        send[:rows, :hc_mult].copy_(pre_s)
        send[:rows, hc_mult : 2 * hc_mult].copy_(post_s)
        send[:rows, 2 * hc_mult :].copy_(comb_s.reshape(rows, hc_mult * hc_mult))

    full = torch.empty(
        (world * chunk, mix), dtype=torch.float32, device=x_flat.device
    )
    group.all_gather_into_tensor(full, send)

    # pre keeps the packed row stride (hc_combine reads through strides);
    # post/comb are materialized contiguous for the tilelang hc_post kernel.
    pre = full[:m, :hc_mult]
    post = full[:m, hc_mult : 2 * hc_mult].contiguous()
    comb = (
        full[:m, 2 * hc_mult :]
        .reshape(m, hc_mult, hc_mult)
        .contiguous()
    )
    return pre, post, comb
