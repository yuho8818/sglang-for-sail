# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4."""

from __future__ import annotations

import functools
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.moe.dsv4 import mega_moe_pre_dispatch
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.mega_moe_sm90 import (
    is_sm90_fp8_mega_moe_available,
    run_sm90_mega_routed,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_hip, is_ppu

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


_MEGA_MOE_SYMM_BUFFER: dict = {}
_is_hip = is_hip()
_is_ppu = is_ppu()


def _use_amd_flydsl_mega_moe() -> bool:
    # aiter MegaMoEv2 exists only on ROCm; the platform check keeps the env from
    # diverting a CUDA run into a path whose kernels it does not have.
    return _is_hip and envs.SGLANG_AMD_USE_FLYDSL_MEGA_MOE.get()


def _mega_moe_mma_type(experts=None) -> str:
    if experts is not None and experts._mega_moe_nvfp4:
        return "nvfp4xnvfp4"
    if _use_ppu_mega_moe():
        return "fp4xfp4"
    return "mxf4xmxf4" if get_exec().moe.enable_w4a4_mxfp4_megamoe else "fp8xfp4"


def _use_ppu_mega_moe() -> bool:
    """PPU1.5 runs MegaMoE with DeepGEMM's complete W4A4 call.

    PPU's kernel family is MXFP4 x MXFP4 (``fp4_fp4_mega_moe``) with one
    cross-rank call covering dispatch, GEMM1 + GLU + requantization, GEMM2 and
    combine, instead of the SM100 host-side pipeline. Gated on the switch so
    the default PPU MoE path keeps its numerics bit-for-bit.
    """
    return bool(envs.SGLANG_OPT_PPU_DSV41_MEGAMOE.get() and _is_ppu)


@functools.lru_cache(maxsize=1)
def _mega_moe_max_num_sms() -> Optional[int]:
    if _device_sm < 100:
        # The SM90 MegaMoE implementation does not use the whole-grid clustered
        # launch that needs a residency margin.
        return None

    # Physical count, not deep_gemm.get_num_sms(): two-batch overlap and the DSA
    # indexer reconfigure that process-wide, so reserving on top would compound.
    num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    reserved_num_sms = max(envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.get(), 0)
    return max(2, num_sms - reserved_num_sms)


@contextmanager
def _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
    max_num_sms = _mega_moe_max_num_sms()
    if max_num_sms is None:
        yield
        return

    current_num_sms = deep_gemm.get_num_sms()
    # Stay under an outer context's budget instead of claiming SMs back from it.
    target_num_sms = min(max_num_sms, current_num_sms)
    # Round down: the clustered launch needs an even CTA count.
    target_num_sms -= target_num_sms % 2
    if target_num_sms == current_num_sms:
        yield
        return

    deep_gemm.set_num_sms(target_num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(current_num_sms)


def check_mega_moe_shapes(hidden: int, intermediate: int, mma_type: str) -> None:
    # DeepGEMM keeps one scale row per token and needs 16-byte TMA alignment
    # on it (layout/mega_moe.cuh), so both dims must be multiples of 16 * group.
    scale_group = 16 if mma_type == "nvfp4xnvfp4" else 32
    align = 16 * scale_group
    if hidden % align != 0 or intermediate % align != 0:
        raise ValueError(
            f"DeepGEMM MegaMoE ({mma_type}) needs hidden_size and "
            f"moe_intermediate_size to be multiples of {align}; got "
            f"hidden_size={hidden}, moe_intermediate_size={intermediate}. "
            "Use another --moe-a2a-backend for this model."
        )


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    mma_type: Optional[str] = None,
) -> SymmBuffer:
    import deep_gemm

    if mma_type is None:
        mma_type = _mega_moe_mma_type()
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        key = (
            id(group),
            num_max_tokens_per_rank,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
            mma_type,
            deep_gemm.get_num_sms(),
        )
        buf = _MEGA_MOE_SYMM_BUFFER.get(key)
        if buf is None:
            buf = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                mma_type=mma_type,
                activation="swiglu",
            )
            _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def _get_ppu_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBuffer:
    """PPU workspace: one W4A4 allocation shared by every layer.

    PPU has no shared-expert slot, so ``num_shared_experts`` stays 0, and the
    dispatch packs FP4 activations, so ``use_fp8_dispatch`` is off. Reusing one
    buffer keeps the cross-rank handshake state single-owner, which the kernel
    requires when layers run back-to-back on one stream.
    """
    import deep_gemm

    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        "fp4xfp4",
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts=0,
            use_fp8_dispatch=False,
            mma_type="fp4xfp4",
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def is_mega_moe_experts_ready(experts) -> bool:
    if not experts._mega_moe_weights_built:
        return False
    if _device_sm == 90:
        return is_sm90_fp8_mega_moe_available(experts)
    if _use_ppu_mega_moe():
        return getattr(experts, "ppu_mega_moe_weights_built", False)
    # The SM100 mega kernels exist for compute capability 10.x only.
    return _device_sm // 10 == 10


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if _use_amd_flydsl_mega_moe():
        from sglang.srt.layers.moe.mega_moe_flydsl import (
            should_use_mega_moe as should_use_flydsl_mega_moe,
        )

        return should_use_flydsl_mega_moe(moe, hidden_states)

    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not is_mega_moe_experts_ready(moe.experts):
        return False
    if get_is_capture_mode():
        return True

    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens and not is_dsa_enable_prefill_cp():
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _use_amd_flydsl_mega_moe():
        from sglang.srt.layers.moe.mega_moe_flydsl import (
            forward_mega_moe as forward_flydsl_mega_moe,
        )

        return forward_flydsl_mega_moe(
            moe, hidden_states, forward_batch, input_ids_global
        )

    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    hidden_size = moe.config.hidden_size

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.moe_num_token_non_padded()
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    return run_mega_routed_experts(
        moe.experts,
        hidden_states,
        topk_ids,
        topk_weights,
        hidden_size=hidden_size,
        intermediate_size=moe.config.moe_intermediate_size,
        top_k=moe.config.num_experts_per_tok + moe.num_fused_shared_experts,
        num_tokens=num_tokens,
        activation_clamp=moe.experts.moe_runner_config.swiglu_limit,
        routed_scaling_factor=(
            1.0
            if moe.experts.should_fuse_routed_scaling_factor_in_topk
            else float(moe.routed_scaling_factor)
        ),
    )


def run_mega_routed_experts(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    activation_clamp: Optional[float] = None,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    # Rows are this rank's tokens; the returned rows are fully combined.
    import deep_gemm

    from sglang.srt.runtime_context import get_parallel

    ep_group = get_parallel().moe_ep_group.device_group
    num_experts = experts.num_experts
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or lower "
        f"--cuda-graph-max-bs-decode / --chunked-prefill-size accordingly"
    )

    mma_type = _mega_moe_mma_type(experts)
    if _use_ppu_mega_moe():
        return run_ppu_mega_routed_experts(
            experts,
            hidden_states,
            topk_ids,
            topk_weights,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=top_k,
            num_tokens=num_tokens,
            activation_clamp=activation_clamp,
            routed_scaling_factor=routed_scaling_factor,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
        )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
        mma_type=mma_type,
    )

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    if _device_sm == 90:
        return run_sm90_mega_routed(
            experts,
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf,
            num_tokens,
            hidden_size=hidden_size,
            activation_clamp=activation_clamp,
            routed_scaling_factor=routed_scaling_factor,
        )

    mega_kwargs = {"recipe": (1, 1, 32)}
    if mma_type == "nvfp4xnvfp4":
        # Per-token outer scales go to buf.x_scales; the kernel folds them with
        # the per-expert alphas in the L1 / L2 epilogues.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=16,
            mma_type=mma_type,
            buf_x_scales=buf.x_scales,
        )
        mega_kwargs = {
            "recipe": (1, 1, 16),
            "use_x_scales": True,
            "l1_alphas": experts.mega_l1_alphas,
            "l2_alphas": experts.mega_l2_alphas,
            "l2_act_scales": experts.mega_l2_act_scales,
        }
    elif mma_type == "mxf4xmxf4":
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            mma_type=mma_type,
        )
    else:
        mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
        )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        deep_gemm.fp8_fp4_mega_moe(
            y,
            experts.mega_l1_weights,
            experts.mega_l2_weights,
            buf,
            activation="swiglu",
            activation_clamp=activation_clamp,
            fast_math=True,
            **mega_kwargs,
        )
    y = y[:num_tokens]

    if routed_scaling_factor != 1.0:
        y.mul_(routed_scaling_factor)
    return y


def run_ppu_mega_routed_experts(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    activation_clamp: Optional[float] = None,
    routed_scaling_factor: float = 1.0,
    num_max_tokens_per_rank: Optional[int] = None,
) -> torch.Tensor:
    """PPU1.5 W4A4 MegaMoE: one complete cross-rank call per layer.

    ``fp4_fp4_mega_moe`` consumes the packed MXFP4 activations staged by
    ``mega_moe_pre_dispatch`` into the symmetric workspace and writes the
    combined BF16 rows, so unlike the SM100 path there is no host-side
    dispatch, grouped GEMM or combine to drive here. Inputs are this rank's
    tokens; the returned rows are fully reduced.
    """
    import deep_gemm

    from sglang.srt.runtime_context import get_parallel

    if num_max_tokens_per_rank is None:
        num_max_tokens_per_rank = (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
        )
    # PPU keeps the full intermediate width: there is no shared-expert slot in
    # the workspace, and the fused call derives GEMM1's 2I from the L1 weights.
    assert intermediate_size == experts.mega_l1_weights[0].shape[1] // 2

    ep_group = get_parallel().moe_ep_group.device_group
    buf = _get_ppu_mega_moe_symm_buffer(
        ep_group,
        num_experts=experts.num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
    )

    # The PPU kernels validate int64 routing indices and reject duplicates
    # among the valid physical expert IDs within one token's top-k; masked
    # slots (-1) may repeat. The router already emits that shape, so this only
    # fixes the dtype/layout the kernel demands.
    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int64).contiguous()
        topk_weights_in = topk_weights.to(torch.float32).contiguous()
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int64)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    # Activation clamp is a SwiGLU gate/up bound; 0.0 means unbounded to the
    # kernel, so an unfused config maps to "no clamp".
    clamp = 0.0 if activation_clamp is None else float(activation_clamp)
    _trace_ppu_mega_moe_call(
        num_tokens=num_tokens,
        num_experts=experts.num_experts,
        top_k=top_k,
        hidden=hidden_size,
        intermediate=intermediate_size,
    )

    deep_gemm.mega_moe_pre_dispatch(
        hidden_states,
        topk_ids_in,
        topk_weights_in,
        buf,
        num_tokens=num_tokens,
    )
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    deep_gemm.fp4_fp4_mega_moe(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        activation="swiglu",
        activation_clamp=clamp,
        fast_math=True,
    )
    y = y[:num_tokens]

    if routed_scaling_factor != 1.0:
        y.mul_(routed_scaling_factor)
    return y


_PPU_MEGA_MOE_TRACED = False


def _trace_ppu_mega_moe_call(
    *, num_tokens: int, num_experts: int, top_k: int, hidden: int, intermediate: int
) -> None:
    """Announce the first PPU MegaMoE call so a run can be told apart from the
    grouped-GEMM default in the server log."""
    global _PPU_MEGA_MOE_TRACED
    if _PPU_MEGA_MOE_TRACED:
        return
    _PPU_MEGA_MOE_TRACED = True
    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001 - tracing must never break the forward
        rank = 0
    from sglang.srt.utils.common import log_info_on_rank0

    log_info_on_rank0(
        f"[TH-S13][mega-moe] PPU W4A4 MegaMoE engaged: fp4_fp4_mega_moe rank={rank} "
        f"num_tokens={num_tokens} experts={num_experts} top_k={top_k} "
        f"hidden={hidden} intermediate={intermediate}"
    )


def _interleave_mega_moe_gate_up(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # Match DeepGEMM's L1 gate/up layouts. FP8 activations use contiguous
    # gran-8 chunks; packed MXFP4 activations use even/odd gran-16 chunks.
    num_groups, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = t[:, half:].reshape(num_groups, half // gran, gran, *rest)
    if gran == 16:
        result = torch.cat(
            [gate[:, :, 0::2], up[:, :, 0::2], gate[:, :, 1::2], up[:, :, 1::2]],
            dim=2,
        ).reshape(num_groups, n, *rest)
    else:
        result = torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)
    return torch.empty_like(t).copy_(result)


def _interleave_mega_moe_l1_weights(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    mma_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    gran = 16 if mma_type == "mxf4xmxf4" else 8
    return (
        _interleave_mega_moe_gate_up(l1_weights[0], gran=gran),
        _interleave_mega_moe_gate_up(l1_weights[1], gran=gran),
    )


def _transpose_mega_moe_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(result)


def build_mega_moe_experts_weights(experts) -> None:
    if _use_amd_flydsl_mega_moe():
        from sglang.srt.layers.moe.mega_moe_flydsl import (
            build_mega_moe_experts_weights as build_flydsl_mega_moe_weights,
        )

        build_flydsl_mega_moe_weights(experts)
        return

    if _use_ppu_mega_moe():
        build_ppu_mega_moe_experts_weights(experts)
        return

    from deep_gemm import (
        transform_sf_into_required_layout,
    )

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    mma_type = _mega_moe_mma_type()
    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    check_mega_moe_shapes(hidden=k1, intermediate=n1 // 2, mma_type=mma_type)
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )

    # Build the interleaved L1 weight + scale once; share the weight buffer
    # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
    # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
    # the deep-ep path consumes the non-transposed interleaved scale and a
    # swizzle-aware activation kernel. L2 weight is untouched by the mega
    # transform, so the existing `w2_weight.data` is shared directly.
    w13_interleaved, w13_sf_interleaved = _interleave_mega_moe_l1_weights(
        (w13, w13_sf), mma_type
    )
    w13_sf_utccp = _transpose_mega_moe_sf_for_utccp(w13_sf_interleaved)
    w2_sf_utccp = _transpose_mega_moe_sf_for_utccp(w2_sf)

    experts.w13_weight.data = w13_interleaved
    experts.w13_weight_scale_inv.data = w13_sf_interleaved
    experts.w2_weight_scale_inv.data = w2_sf
    experts.w13_weight_scale_inv.format_ue8m0 = True
    experts.w2_weight_scale_inv.format_ue8m0 = True

    experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
    experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)

    experts._mega_moe_weights_built = True


def _data(t):
    """The tensor behind a module attribute (Parameter/tensor), or None."""
    return getattr(t, "data", t) if torch.is_tensor(t) else None


def _as_uint8(t):
    """Reinterpret a packed nibble/byte tensor as raw uint8, the only spelling
    ``transform_weights_for_mega_moe`` accepts. Both MXFP4 e2m1 weights and
    UE8M0 scales carry the same bits whichever view the loader left them in."""
    return _data(t).view(torch.uint8).contiguous()


def build_ppu_mega_moe_experts_weights(experts) -> None:
    """Pack the checkpoint's MXFP4 experts into the PPU MegaMoE layouts.

    ``transform_weights_for_mega_moe`` returns the pair the complete call
    takes: the L1 gate/up weight interleaved for the fused G1 kernel and the
    UE8M0 scales in the layout both GEMMs expect. The layer keeps its
    ``w13_weight`` used by the grouped paths; the packed pairs get their own
    parameters, so unloading the checkpoint tensors frees the originals before
    the next layer converts.
    """
    if getattr(experts, "ppu_mega_moe_weights_built", False):
        return

    import torch.nn as nn
    from deep_gemm import transform_weights_for_mega_moe

    w13 = getattr(experts, "w13_weight", None)
    w2 = getattr(experts, "w2_weight", None)
    w13_sf = getattr(experts, "w13_weight_scale", None) or getattr(
        experts, "w13_weight_scale_inv", None
    )
    w2_sf = getattr(experts, "w2_weight_scale", None) or getattr(
        experts, "w2_weight_scale_inv", None
    )
    if not all(torch.is_tensor(t) for t in (w13, w2, w13_sf, w2_sf)):
        raise ValueError(
            "PPU MegaMoE needs packed MXFP4 experts (e2m1 weights + UE8M0 "
            "group-32 scales); got "
            f"w13={getattr(w13, 'dtype', None)}, w2={getattr(w2, 'dtype', None)}, "
            f"w13_scale={getattr(w13_sf, 'dtype', None)}, "
            f"w2_scale={getattr(w2_sf, 'dtype', None)}"
        )
    if _device_sm != 89:
        raise RuntimeError("PPU MegaMoE requires PPU1.5 (compute capability 8.9)")

    l1_pair, l2_pair = transform_weights_for_mega_moe(
        (_as_uint8(w13), _as_uint8(w13_sf)),
        (_as_uint8(w2), _as_uint8(w2_sf)),
        activation="swiglu",
    )

    for name, value in (("mega_l1_weights", l1_pair), ("mega_l2_weights", l2_pair)):
        for index, tensor in enumerate(value):
            param = nn.Parameter(tensor, requires_grad=False)
            setattr(experts, f"{name}_{index}", param)
        # Tuple views the forward reads; rebuilt on demand if a loader swaps
        # the parameters (see the aggregate below).
        setattr(experts, name, value)

    experts.ppu_mega_moe_weights_built = True
    experts._mega_moe_weights_built = True
