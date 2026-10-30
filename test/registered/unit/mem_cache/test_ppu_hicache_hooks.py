"""Unit tests for the PPU HiCache page-transfer hooks (TH-S6 port).

The hooks replace ``sgl_kernel.kvcacheio`` entry points with synchronous
per-page Python copies; these tests pin the copy semantics and the
validation guards on CPU tensors (no PPU stack required).
"""

import pytest
import torch

from sglang.srt.hardware_backend.ppu.ppu_hicache_hooks import _sync_copy_pages
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pages(num_pages: int, channels: int = 2):
    return torch.arange(num_pages * channels, dtype=torch.float32).reshape(
        num_pages, channels
    )


def test_copies_pages_in_given_order():
    src, dst = _pages(4), torch.zeros(4, 2)
    _sync_copy_pages(
        src, dst,
        torch.tensor([2, 0, 3], dtype=torch.int64),
        torch.tensor([1, 3, 0], dtype=torch.int64),
    )
    assert torch.equal(dst[1], src[2])
    assert torch.equal(dst[3], src[0])
    assert torch.equal(dst[0], src[3])
    assert torch.equal(dst[2], torch.zeros(2))  # untouched page


def test_count_mismatch_rejected():
    src, dst = _pages(4), torch.zeros(4, 2)
    with pytest.raises(ValueError, match="counts differ"):
        _sync_copy_pages(
            src, dst,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )


def test_out_of_range_rejected():
    src, dst = _pages(2), torch.zeros(2, 2)
    with pytest.raises(ValueError, match="out of range"):
        _sync_copy_pages(
            src, dst,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([5], dtype=torch.int64),
        )
    # Nothing was written before the rejection.
    assert torch.equal(dst, torch.zeros(2, 2))


def test_payload_shape_mismatch_rejected():
    src, dst = _pages(2), torch.zeros(2, 4)
    with pytest.raises(ValueError, match="payload mismatch"):
        _sync_copy_pages(
            src, dst,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )


def test_empty_transfer_is_noop():
    src, dst = _pages(2), torch.zeros(2, 2)
    _sync_copy_pages(
        src, dst,
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
    )
    assert torch.equal(dst, torch.zeros(2, 2))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
