"""PPU native FP8 sparse prefill attention for DeepSeek-V4 (DSV41-S5.1 port).

The native FP8 sparse kernels keep Q, K and V in E4M3 and run the whole QK/PV
chain on the FP8 tensor path, unlike the bf16 sparse kernel, which dequantises
the cache during the gather and still multiplies in bf16. This module adapts
the FlashMLA ``flash_mla_sparse_quantized_fwd`` contract to what
``DeepseekV4AttnBackend._forward_prefill_sparse`` already builds: a bf16 Q row
block and one flat KV workspace per layer.

Gating
------
``SGLANG_OPT_PPU_DSV4_NATIVE_FP8_PREFILL`` (default off, ``srt/environ.py``).
Off = the bf16 ``flash_mla_sparse_fwd`` path, untouched. On = layers 2..35 of
the layer stack ask this module first; :func:`native_fp8_prefill_applies`
rejects every configuration the native kernel cannot mirror (head count, head
dim, topk width, wheel capability), so an unsupported layer silently keeps the
bf16 kernel instead of failing the request. The caller keeps its own fallback.

Precision
---------
The wheel exposes one E4M3 backend: single-representation probabilities with
FP32 accumulation. It has no two-term (high/residual) probability mode, so the
result differs from the bf16 reference by the FP8 probability quantisation as
well as the FP8 QK quantisation. That is why the path is opt-in and must be
accepted by the accuracy gate before it is enabled in a deployment.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

_flashmla = None
_import_error = None
try:
    import flash_mla as _flashmla
except Exception as _e:  # pragma: no cover - depends on the installed wheel
    _import_error = _e

_IMPORT_ERROR = ImportError(
    "Failed to import flash_mla for PPU. "
    "Ensure the flash_mla package is installed in your PPU environment."
)

# Layer range of the native path. Kept identical to the internal S39 split so
# that a measurement on either side refers to the same layers; layers outside
# it (the first two and the MTP tail / final layers) keep the bf16 kernel.
NATIVE_FP8_PREFILL_LAYERS = (2, 36)

# The quantized kernel takes a multiple of 64 key slots up to this width. The
# backend's combined top-k + SWA index width is already a multiple of 128.
_MAX_TOPK = 2048

# Value dimension the quantized kernels produce.
_D_V = 512


def _native_fp8_enabled() -> bool:
    return envs.SGLANG_OPT_PPU_DSV4_NATIVE_FP8_PREFILL.get()


@functools.lru_cache(maxsize=None)
def _probe() -> Tuple[bool, str]:
    """Return ``(supported, reason)`` for the installed wheel.

    Probed once: the wheel cannot change inside a process.
    """
    if _import_error is not None:
        return False, f"flash_mla unavailable: {_import_error!r}"
    for name in ("flash_mla_sparse_quantized_fwd", "QuantizedAttentionConfig"):
        if not hasattr(_flashmla, name):
            return False, f"installed flash_mla lacks {name}"
    return True, "ok"


def native_fp8_prefill_supported() -> bool:
    """Whether the installed FlashMLA can serve the native FP8 sparse prefill."""
    return _probe()[0]


def native_fp8_prefill_applies(
    layer_id: int,
    q: torch.Tensor,
    head_dim_v: int,
    indices: torch.Tensor,
) -> bool:
    """Cheap admission test; the caller falls back to bf16 when it fails.

    ``q`` is the squeezed ``(s_q, h_q, d_qk)`` query block and ``indices`` the
    combined ``(s_q, 1, topk)`` key list the sparse prefill already built.
    """
    if not _native_fp8_enabled():
        return False
    if not (NATIVE_FP8_PREFILL_LAYERS[0] <= layer_id < NATIVE_FP8_PREFILL_LAYERS[1]):
        return False
    if q.dim() != 3 or q.dtype != torch.bfloat16:
        return False
    s_q, h_q, d_qk = q.shape
    if s_q <= 0 or h_q <= 0 or h_q % 8 != 0:
        return False
    if d_qk != _D_V or head_dim_v != _D_V:
        return False
    if indices.ndim != 3 or indices.shape[0] != s_q or indices.shape[1] != 1:
        return False
    topk = indices.shape[-1]
    if topk <= 0 or topk % 64 != 0 or topk > _MAX_TOPK:
        return False
    return _probe()[0]


def flash_mla_sparse_quantized_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = _D_V,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse FP8 attention over a flat KV workspace.

    Args:
        q: ``(s_q, h_q, d_qk)`` bf16. The kernel quantises Q to E4M3 while
            loading its shared-memory tile, so no Q tensor is materialised.
        kv: ``(s_kv, 1, d_qk)`` e4m3, the per-layer gathered workspace. A bf16
            workspace is converted once inside the kernel to a temporary E4M3
            tensor.
        indices: ``(s_q, 1, topk)`` int32, workspace-relative key slots. ``-1``
            entries are masked exactly.
        sm_scale: softmax scale.
        d_v: value dim, fixed at 512.
        attn_sink: ``(h_q,)`` float32 attention sink (natural-log space).
        topk_length: ``(s_q,)`` int32 valid prefix per row.
        out: optional preallocated ``(s_q, h_q, d_v)`` bf16 output.

    Returns:
        ``(output, max_logits, lse)``; ``max_logits`` is the ceil-quantised
        running maximum, not the true maximum.
    """
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error

    return _flashmla.flash_mla_sparse_quantized_fwd(
        q,
        kv,
        indices,
        sm_scale,
        quantization=_flashmla.QuantizedAttentionConfig(q_scale=1.0, kv_scale=1.0),
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
        # The backend consumes only the output row block.
        compute_aux=False,
        # The combined index list carries arbitrary -1 entries, so the kernel
        # must mask them instead of trusting a dense prefix.
        trusted_full_topk=False,
    )