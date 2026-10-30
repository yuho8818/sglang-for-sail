"""PPU DeepEP hooks.

Uses ``@plugin_hook`` to inject PPU-specific behavior into the DeepEP token
dispatcher without modifying the community source code.

Registered hooks
~~~~~~~~~~~~~~~~

1. ``_is_mnnvl_fabric_supported`` (REPLACE) — Reads
   ``SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED`` so PPU deployments can explicitly
   enable or disable MNNVL FABRIC handles.
"""

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.moe.token_dispatcher.deepep._is_mnnvl_fabric_supported",
    type=HookType.REPLACE,
)
def _ppu_is_mnnvl_fabric_supported(group_size: int = 0) -> bool:
    # Respect an explicit env override first.
    if envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.is_set():
        return envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.get()
    # On PPU, cross-machine EP groups (group_size > device_count) require the
    # ICN/IMEX fabric; single-machine EP keeps P2P.
    if group_size <= 0:
        return True
    import torch  # local import to avoid a top-level torch dep at plugin load

    return group_size > torch.cuda.device_count()
