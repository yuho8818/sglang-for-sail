import inspect
import logging

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
    is_ppu,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()
_is_ppu = is_ppu()


def _sm120_deep_gemm_apis_available() -> bool:
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError):
        return False
    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "fp8_einsum",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "transform_sf_into_required_layout",
        )
    )


def _compute_enable_deep_gemm():
    if not (_is_cuda or _is_musa):
        return False
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        return False

    sm_version = get_device_sm()
    if not _is_ppu and (
        (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31)
    ):
        return False
    elif _is_ppu and sm_version < 80:
        return False
    # SM120/SM121 support (mma.sync block-scale, no TMEM) landed in DeepGEMM#324;
    # probe every API used by the SM120 DSV4 paths since installed builds may
    # expose fp8_einsum but still predate the SM120 kernels.
    if sm_version in (120, 121) and not _sm120_deep_gemm_apis_available():
        return False

    try:
        import deep_gemm  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False

    return True


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and get_platform().is_sm100
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (120, 121)
)
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa or _is_ppu)


def _supports_paged_sparse_mqa_logits() -> bool:
    """Whether this DeepGEMM build indexes K in pages wider than 64 slots.

    The indexer pool page length follows this flag (128 when set, else 64), so
    a build that exports the sparse entry points but whose scheduler still
    asserts ``block_kv == 64`` must not report support: the pool would then be
    allocated at 128 and every paged indexer metadata build would raise.
    PPU builds before the sparse scheduler landed carry the symbol but keep the
    assertion, so require the symbol *and* a scheduler that takes its block
    width from the caller.
    """
    if _is_ppu:
        if not ENABLE_JIT_DEEPGEMM or get_device_sm() < 89:
            return False
    elif not DEEPGEMM_BLACKWELL:
        return False
    import deep_gemm

    if not all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        )
    ):
        return False

    # The scheduler that hard-codes 64 slots has no sparse-block parameter; the
    # one that accepts caller page widths does.
    metadata_fn = getattr(deep_gemm, "get_paged_sparse_mqa_logits_metadata", None)
    try:
        params = inspect.signature(metadata_fn).parameters
    except (TypeError, ValueError):
        return False
    return "sparse_block_kv" in params


DEEPGEMM_PAGED_SPARSE_MQA_LOGITS = _supports_paged_sparse_mqa_logits()
