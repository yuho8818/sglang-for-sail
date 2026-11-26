"""UltraEP placement and replica weights for the PPU MegaMoE prefill path.

Master weights stay peer-visible for the lifetime of each layer. One local
replica pool per expert geometry is refilled before each layer, including graph
replays. MegaMoE retains token dispatch, GEMMs and combine.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_MANAGERS: dict = {}
_HIT_LOGGED = False


@lru_cache(maxsize=2)
def require_runtime(mma_type: str = "fp4xfp4"):
    # Keep the wheel optional when the feature is disabled, including CPU CI.
    try:
        installed_version = metadata.version("ultra-ep")
    except metadata.PackageNotFoundError as exc:
        raise ValueError(
            "SGLANG_OPT_ULTRA_EP_PPU=1 requires the PPU ultra_ep wheel. "
            "Install it with: python3 -m pip install /path/to/ultra_ep-*.whl"
        ) from exc
    try:
        from ultra_ep import ppu
    except ImportError as exc:
        raise ValueError(
            f"ultra_ep {installed_version} is installed but its PPU backend "
            "cannot be imported; install the PPU ultra_ep wheel"
        ) from exc
    try:
        import deep_gemm
    except ImportError as exc:
        raise ValueError("PPU UltraEP requires the PPU DeepGEMM build") from exc
    name = {"fp4xfp4": "fp4_fp4_mega_moe", "bf16xfp4": "bf16_fp4_mega_moe"}[mma_type]
    kernel = getattr(deep_gemm, name, None)
    params = inspect.signature(kernel).parameters if callable(kernel) else {}
    if not {"l1_weights_replicas", "l2_weights_replicas"} <= params.keys():
        raise ValueError(
            f"PPU UltraEP requires DeepGEMM.{name} with "
            "l1_weights_replicas/l2_weights_replicas; upgrade PPU DeepGEMM"
        )
    if mma_type == "fp4xfp4":
        workspace = getattr(deep_gemm, "SymmBuffer", None)
        params = inspect.signature(workspace).parameters if callable(workspace) else {}
        if "base" not in params:
            raise ValueError(
                "PPU UltraEP W4A4 requires DeepGEMM.SymmBuffer(base=...) "
                "for shared native/UltraEP workspace views"
            )
    logger.info("UltraEP PPU wheel detected: ultra_ep %s", installed_version)
    return ppu


def validate_config(cfg, platform) -> None:
    # P/D launchers often export one common environment. Decode must keep its
    # original weights/routing and must not acquire an UltraEP dependency.
    if cfg.disaggregation_mode == "decode":
        logger.info(
            "UltraEP PPU is prefill-only; the PD decode server keeps ordinary MegaMoE"
        )
        return
    if not platform.is_ppu or platform.device_sm != 89:
        raise ValueError("PPU UltraEP requires ppu0015 (ZW-M890P, SM89)")
    if cfg.moe_a2a_backend != "megamoe" or cfg.disaggregation_mode not in (
        "null",
        "prefill",
    ):
        raise ValueError(
            "SGLANG_OPT_ULTRA_EP_PPU requires --moe-a2a-backend megamoe "
            "on an ordinary or prefill server"
        )
    if (
        cfg.enable_eplb
        or cfg.ep_dispatch_algorithm is not None
        or cfg.ep_num_redundant_experts
    ):
        raise ValueError(
            "UltraEP owns expert placement; disable --enable-eplb, "
            "--ep-dispatch-algorithm and --ep-num-redundant-experts"
        )
    if envs.SGLANG_OPT_ULTRA_EP_PPU_REDUNDANT_PER_RANK.get() <= 0:
        raise ValueError("SGLANG_OPT_ULTRA_EP_PPU_REDUNDANT_PER_RANK must be positive")
    if envs.SGLANG_OPT_ULTRA_EP_PPU_MIN_TOKENS_PER_RANK.get() < 0:
        raise ValueError(
            "SGLANG_OPT_ULTRA_EP_PPU_MIN_TOKENS_PER_RANK must be nonnegative"
        )


def is_decode_server() -> bool:
    from sglang.srt.runtime_context import get_disagg

    return get_disagg().disaggregation_mode == "decode"


def use_for_batch(forward_batch) -> bool:
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch

    if is_decode_server():
        return False
    # The flag is shared across DP ranks, so idle EP ranks join a peer's
    # prefill. Graph capture clears it; a prefill capture still has EXTEND mode.
    prefill = get_is_extend_in_batch() or (
        forward_batch is not None
        and forward_batch.forward_mode.is_extend_without_speculative()
    )
    if not prefill:
        return False
    cp = getattr(forward_batch, "attn_cp_metadata", None)
    minimum = envs.SGLANG_OPT_ULTRA_EP_PPU_MIN_TOKENS_PER_RANK.get()
    if cp is not None and minimum > 0:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        # The CPU list is shared by the entire CP group and is replaced when
        # bounded replay enters the compact SWA tail. Never threshold a local
        # row count: an idle/short EP peer must make the same collective choice.
        # Without one CP group covering EP, retain the original participation.
        if (
            parallel.attn_dp_size == 1
            and parallel.attn_cp_size
            == parallel.moe_ep_group.world_size
            == parallel.tp_group.world_size
        ):
            counts = getattr(cp, "per_rank_actual_token", None)
            if counts is not None and len(counts) == parallel.attn_cp_size:
                return max(counts, default=0) >= minimum
    return True


def register_weights(experts) -> None:
    """Register prepared MXFP4 pairs before warmup, workspace sizing and capture."""
    if is_decode_server() or getattr(experts, "_ultra_ep_manager", None) is not None:
        return
    if not getattr(experts, "_mega_moe_weights_built", False):
        raise RuntimeError("prepare MegaMoE weights before UltraEP registration")
    if experts._mega_moe_mma_type not in ("fp4xfp4", "bf16xfp4"):
        raise ValueError("PPU UltraEP supports MXFP4 W4A4/W4A16 expert weights")

    from sglang.srt.runtime_context import get_parallel

    require_runtime(experts._mega_moe_mma_type)
    import ultra_ep

    group = get_parallel().moe_ep_group.device_group
    templates = (*experts.mega_l1_weights, *experts.mega_l2_weights)
    masters = int(templates[0].shape[0])
    replicas = envs.SGLANG_OPT_ULTRA_EP_PPU_REDUNDANT_PER_RANK.get()
    if masters * group.size() != experts.num_experts:
        raise ValueError(
            "UltraEP needs contiguous, evenly sharded logical master experts"
        )
    # Share replicas only across identical layouts. Target/draft models may
    # reuse layer IDs with different expert counts or packed scale strides.
    key = (
        id(group),
        masters,
        replicas,
        experts._mega_moe_mma_type,
        tuple((t.shape[1:], t.stride()[1:], t.dtype, t.device) for t in templates),
    )
    manager = _MANAGERS.get(key)
    if manager is None:
        manager = ultra_ep.ppu.Manager(group, masters, replicas)
        _MANAGERS[key] = manager

    # Re-home already transformed weights, preserving their inner strides:
    # packed N-major scales must never be reinterpreted as row-major scales.
    allocation = ultra_ep.ppu.allocate_symmetric_expert_tensors(group, templates, 0)
    w1, s1, w2, s2 = allocation.tensors
    experts.w13_weight.data = w1
    experts.w13_weight_scale.data = s1
    experts.w2_weight.data = w2
    experts.w2_weight_scale.data = s2
    experts.mega_l1_weights = (w1, s1)
    experts.mega_l2_weights = (w2, s2)
    manager.register_physical_weight_pool(
        experts.layer_id,
        ultra_ep.ppu.PhysicalWeightPool(
            w1=w1,
            w1_scale=s1,
            w1_peer_base_ptrs=allocation.peer_base_ptrs,
            w2=w2,
            w2_scale=s2,
            w2_peer_base_ptrs=allocation.peer_base_ptrs,
            keepalive=(allocation,),
        ),
    )
    if manager.replica_weight_pool() is None:
        replica_alloc = ultra_ep.ppu.allocate_local_expert_tensors(
            (w1, s1, w2, s2), replicas
        )
        manager.register_replica_weight_pool(
            ultra_ep.ppu.ReplicaWeightPool(
                *replica_alloc.tensors, keepalive=(replica_alloc,)
            )
        )
        logger.info(
            "UltraEP PPU prefill enabled: EP=%d, masters/rank=%d, replicas/rank=%d",
            group.size(),
            masters,
            replicas,
        )
    experts._ultra_ep_manager = manager


@dataclass
class Preparation:
    topk_ids: torch.Tensor
    num_experts: int
    weight_event: object
    kernel_kwargs: dict


def prepare(experts, logical_ids: torch.Tensor) -> Preparation:
    global _HIT_LOGGED
    manager = getattr(experts, "_ultra_ep_manager", None)
    if manager is None:
        raise RuntimeError(
            "UltraEP weights were not registered before the first forward"
        )
    layer_id = int(experts.layer_id)
    # reroute mutates its input. Keep recorder/capturer references to logical
    # routes intact, and satisfy the PPU kernel's contiguous int64 contract.
    logical_ids = logical_ids.to(
        dtype=torch.int64, copy=True, memory_format=torch.contiguous_format
    )
    # No local-row threshold: an empty rank still joins the metadata collective
    # and may execute experts on behalf of the other CP/EP ranks.
    manager.update_placement(layer_id, logical_ids)
    event = manager.weight_sync(layer_id, async_finish=True)
    physical_ids = manager.reroute(layer_id, logical_ids)
    replica = manager.replica_weight_pool()
    if replica is None or event is None:
        raise RuntimeError("UltraEP replica pool or weight completion event is missing")
    if not _HIT_LOGGED:
        logger.info(
            "ULTRA_EP_PPU_HIT: prefill routing uses %d physical experts",
            manager.num_global_physical_experts,
        )
        _HIT_LOGGED = True
    return Preparation(
        physical_ids,
        manager.num_global_physical_experts,
        event,
        {
            "l1_weights_replicas": (replica.w1, replica.w1_scale),
            "l2_weights_replicas": (replica.w2, replica.w2_scale),
        },
    )
