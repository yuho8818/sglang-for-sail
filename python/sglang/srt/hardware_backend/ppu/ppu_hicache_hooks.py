"""PPU HiCache page-transfer hooks.

Replaces the ``sgl_kernel.kvcacheio`` CUDA entry points used by the
unified host pool's page transfers with Python tensor copies that run on
the PPU stack. NOTE (unconfirmed workaround, carried from the internal
stack): copies are completed synchronously one page at a time —
``write_through`` submits one D2H op per radix insert and a sustained
load (e.g. long prefill fills) would otherwise queue thousands of
per-page async copies on the device_to_host stream; unbounded queued
async copies wedge the PPU copy path and stall concurrent collectives.
Completing each copy before submitting the next bounds the stream queue
to one op at a time. Revisit once the PPU copy path grows a proper
batched/bounded-queue transfer.

Ported-from: 19eabf6af5 + 43a4bf5530 + 2e8b4b4b82 (DSV41 TH-S6),
adapted to the open-source ``transfer_kv_direct`` /
``transfer_kv_per_layer_mla`` call sites.
"""

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _sync_copy_pages(src: torch.Tensor, dst: torch.Tensor,
                     src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
    """Copy whole pages one at a time, each completed before the next is
    submitted (synchronous semantics; see the module docstring)."""
    if src.shape[1:] != dst.shape[1:]:
        raise ValueError(
            f"HiCache page payload mismatch: src {tuple(src.shape)} vs dst {tuple(dst.shape)}"
        )
    if len(src_indices) != len(dst_indices):
        raise ValueError("HiCache source and destination page counts differ")
    for s_idx, d_idx in zip(src_indices.tolist(), dst_indices.tolist()):
        if not (0 <= s_idx < src.shape[0] and 0 <= d_idx < dst.shape[0]):
            raise ValueError(
                f"HiCache page index out of range: src {s_idx}/{src.shape[0]}, "
                f"dst {d_idx}/{dst.shape[0]}"
            )
        # non_blocking=False: complete this copy before submitting the next.
        dst[d_idx].copy_(src[s_idx], non_blocking=False)


@plugin_hook(
    "sgl_kernel.kvcacheio.transfer_kv_per_layer_mla",
    type=HookType.REPLACE,
)
def _ppu_transfer_kv_per_layer_mla(
    src, dst, src_indices, dst_indices, item_size, **kwargs
):
    _sync_copy_pages(src, dst, src_indices, dst_indices)


@plugin_hook(
    "sgl_kernel.kvcacheio.transfer_kv_direct",
    type=HookType.REPLACE,
)
def _ppu_transfer_kv_direct(
    src_layers, dst_layers, src_indices, dst_indices, page_size, **kwargs
):
    if len(src_layers) != len(dst_layers):
        raise ValueError("HiCache layer buffer counts differ")
    for src, dst in zip(src_layers, dst_layers):
        _sync_copy_pages(src, dst, src_indices, dst_indices)
