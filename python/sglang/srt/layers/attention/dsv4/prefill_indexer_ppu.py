"""Native PPU prefill path for the DeepSeek-V4.1 low-compression-ratio indexer
(DSV41-S5.11 port).

Replaces the per-request inner loop of ``FullTopKIndexer._torch_prefill`` when
``SGLANG_OPT_PPU_DSV41_PREFILL_INDEXER_NATIVE`` is on (PPU only, default off):

* scoring: ``deep_gemm.bf16_mqa_logits`` instead of the torch
  ``einsum + relu + mul + sum`` chain — per-head dots, relu and the weighted
  head reduction run in fp32 inside the kernel, so the scores differ from the
  bf16 torch reference only by rounding;
* top-k: the existing ``top_k_per_row_prefill_bf16`` kernel with per-row
  causal windows (``row_starts`` 0, ``row_ends`` the request's compressed
  length), a per-row view of the request's compressed slots as the page table
  and ``page_size=1``.

Every request that the native kernels do not admit (short key window,
non-contiguous rows, non-bf16 inputs) runs the reference torch loop
verbatim, so the switch only changes the arithmetic, never the metadata
contracts. (The internal port also carried candidate source/consumer
handling; this checkpoint family has no candidate layers, so the plain
full-top-k contract is the only one replayed here.)
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsv4.topk import top_k_per_row_prefill_bf16
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    _score_chunks,
    write_prefill,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs

# DeepGEMM's bf16 mqa-logits scheduler needs a key window of at least 2 rows;
# below this the torch loop is microseconds anyway.
_NATIVE_MIN_KEYS = 16

# Same temporary-output budget the torch path uses, but the native scores are
# [rows, lc] (bf16) instead of the reference's [rows, heads, lc] intermediate,
# so 16x more rows fit per chunk.
_SCORES_BUDGET_BYTES = 1 << 30

_ZEROS_ROWS: dict = {}


def _zeros_rows(rows: int, device: torch.device) -> torch.Tensor:
    """Cached int32 zero vector (>= rows), the kernel's row_starts."""
    buf = _ZEROS_ROWS.get(device)
    if buf is None or buf.numel() < rows:
        size = max(8192, -(-rows // 8192) * 8192)
        buf = _ZEROS_ROWS[device] = torch.zeros(
            size, dtype=torch.int32, device=device
        )
    return buf[:rows]


def _can_use_native(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    lc: int,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Admission for the native score + top-k on one request."""
    return (
        q.is_cuda
        and lc >= _NATIVE_MIN_KEYS
        and q.ndim == 3
        and q.shape[1:] == (num_heads, head_dim)
        and index_k.ndim == 2
        and index_k.shape[1] == head_dim
        and weights.shape == q.shape[:2]
        and q.dtype == index_k.dtype == weights.dtype == torch.bfloat16
    )


def topk_prefill_ppu(
    *,
    inputs: PrefillInputs,
    out,
    token_to_kv_pool,
    req_to_token: torch.Tensor,
) -> None:
    """PPU-native full-topk prefill; see the module docstring."""
    from deep_gemm import bf16_mqa_logits

    pool = token_to_kv_pool
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    assert req is not None, "the torch indexer needs the batch's rows"
    out.reset()
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    compress_lens = (pos + 1) // inputs.compress_ratio
    topk = indexer.index_topk
    for r in torch.unique_consecutive(req).tolist():
        tok = (req == r).nonzero().squeeze(1)
        lens = compress_lens[tok]
        lc = int(lens.max().item())
        if lc == 0:
            continue
        j = torch.arange(lc, device=pos.device)
        slots = req_to_token[r, j * inputs.compress_ratio].to(torch.int64) // inputs.compress_ratio
        index_k = pool.get_low_ratio_index_k_dequant(inputs.layer_id, slots)
        # The native top-k kernel writes page_indices[row * kTopK + i] and so
        # needs a contiguous row range; prefill lays requests out that way
        # (torch.unique_consecutive above), fall back when it does not.
        tok_first, tok_last = int(tok[0].item()), int(tok[-1].item())
        contiguous_rows = tok_last - tok_first + 1 == tok.numel()
        if not (
            contiguous_rows
            and _can_use_native(
                q, weights, index_k, lc, indexer.n_heads, indexer.index_head_dim
            )
        ):
            _torch_request(
                indexer, out, q, weights, tok, lens, lc, j, slots, index_k, topk
            )
            continue
        # --- native: bf16 logits + kernel top-k over contiguous rows ---
        num_rows = tok.numel()
        rows_per_chunk = max(1, _SCORES_BUDGET_BYTES // (lc * 2))
        # Per-row page table: every row of this request reads the same
        # compressed slots, page_size=1 makes the kernel's translation the
        # plain slot lookup.
        slots_table = slots.to(torch.int32).unsqueeze(0).expand(num_rows, -1)
        for start in range(0, num_rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, num_rows)
            rows = slice(tok_first + start, tok_first + stop)
            lens_c = lens[start:stop]
            s = bf16_mqa_logits(
                q[rows].contiguous(),
                index_k,
                weights[rows].contiguous(),
                _zeros_rows(stop - start, q.device),
                lens_c.to(torch.int32),
                clean_logits=False,
                logits_dtype=torch.bfloat16,
            )
            top_k_per_row_prefill_bf16(
                s,
                _zeros_rows(stop - start, q.device),  # row_starts: col 0
                lens_c.to(torch.int32),  # row_ends: causal length
                slots_table[start:stop],
                out.page_indices[rows],
                1,  # page_size: table holds the slots themselves
                None if out.raw_indices is None else out.raw_indices[rows],
            )


def _torch_request(
    indexer, out, q, weights, tok, lens, lc, j, slots, index_k, topk
) -> None:
    """The reference per-request loop, verbatim, for requests the native
    kernels do not admit."""
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import RequestScores

    k = min(topk, lc)
    request = RequestScores(
        lc=lc, k=k, columns=j, tok=tok, lens=lens, slots=slots
    )
    for chunk in _score_chunks(
        indexer=indexer, q=q, weights=weights, index_k=index_k, request=request
    ):
        idx = chunk.scores.topk(k, dim=-1, sorted=False).indices
        write_prefill(out, request, chunk, idx)
