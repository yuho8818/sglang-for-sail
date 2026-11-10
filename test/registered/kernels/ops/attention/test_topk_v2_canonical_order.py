"""Unit tests for the canonical (ascending-index) topk_v2 emission order.

The dsv4 top-k select implementations (``TopKRegister`` / ``TopKStreaming`` /
``radix_tie_select``) claim their output slots with warp-arrival atomics, so the
same selected *set* reached the caller in a different *permutation* on every run.
Sparse attention accumulates in emission order, which makes that permutation
visible in the logits, and identical greedy requests then decode differently run
to run.

``SGLANG_TOPK_V2_DETERMINISTIC_EMIT`` (default on) makes every staged path sort
its selection into ascending index order before the page transform, so the
output order is a fixed function of the selected set. These tests pin:

  - the emitted live indices are strictly ascending (the new invariant);
  - the -1 padding is contiguous at the tail, which is what makes the sort a
    single contiguous span and what the consumers already expect;
  - the selected *set* is unchanged, so the fix is order-only;
  - the truncation behaviour at ``k`` is unchanged;
  - switching the flag off restores the arrival-order emission (the escape
    hatch), which is also the discriminating control: with it off the output is
    generally NOT sorted.

The kernel is JIT-compiled for the local accelerator, so this module is skipped
where no accelerator is present.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "topk_v2 is a JIT kernel; needs an accelerator", allow_module_level=True
    )

from sglang.kernels.ops.attention.dsv4.topk import (  # noqa: E402
    plan_topk_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci  # noqa: E402

# A JIT top-k kernel on a real accelerator; sits with the rest of the topk_v2
# coverage in the kernel unit stage.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

PAGE_SIZE = 64
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1

# (batch, seq, k) chosen so every staged implementation is reached:
#   Register2  seq <= 8192
#   Register4  8192 < seq <= 16384
#   Streaming  seq > 16384
# and so the tie path (``radix_tie_select``) is exercised by the duplicate-score
# cases below.
SHAPES = [
    (4, 4096, 512),
    (4, 8192, 512),
    (2, 12288, 512),
    (2, 32768, 512),
    (2, 32768, 2048),
]


def _plan(seq_lens):
    metadata = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()
    return metadata


def _run_paged(scores, seq_lens, k):
    """Raw (untransformed) selection per row, with the -1 tail preserved."""
    batch = scores.shape[0]
    page_table = (
        torch.arange(scores.shape[1], dtype=torch.int32, device=scores.device)
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    topk_transform_paged_v2(
        scores, seq_lens, page_table, out, PAGE_SIZE, _plan(seq_lens)
    )
    torch.cuda.synchronize()
    return out.cpu()


def _live(row):
    return [v for v in row if v != -1]


def _assert_sorted_with_contiguous_pad(row, k, label):
    live = _live(row)
    assert live == sorted(live), f"{label}: not ascending: {live[:16]}"
    assert len(set(live)) == len(live), f"{label}: duplicate indices"
    # Padding is a single contiguous tail: everything after the live prefix
    # is -1, which is what the canonicalizer relies on.
    assert row[len(live) :] == [-1] * (k - len(live)), f"{label}: non-tail padding"


class TestCanonicalEmissionSink(unittest.TestCase):
    @pytest.mark.parametrize("batch,seq,k", SHAPES)
    def test_emitted_indices_are_ascending(self, batch, seq, k):
        torch.manual_seed(batch * 100003 + seq * 7 + k)
        scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
        out = _run_paged(scores, seq_lens, k)
        for row in range(batch):
            _assert_sorted_with_contiguous_pad(
                out[row].tolist(), k, f"b={row} seq={seq} k={k}"
            )

    @pytest.mark.parametrize("batch,seq,k", SHAPES)
    def test_selected_set_is_unchanged(self, batch, seq, k):
        """The fix is order-only: the same top-k values, not a different set."""
        torch.manual_seed(batch * 100003 + seq * 7 + k + 1)
        scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
        out = _run_paged(scores, seq_lens, k)
        for row in range(batch):
            idx = torch.tensor(_live(out[row].tolist()), dtype=torch.long)
            got = scores[row][idx].sort().values
            want = scores[row].topk(k).values.sort().values
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @pytest.mark.parametrize("batch,seq,k", SHAPES)
    def test_live_count_is_exactly_k(self, batch, seq, k):
        torch.manual_seed(batch * 100003 + seq * 7 + k + 2)
        scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
        out = _run_paged(scores, seq_lens, k)
        for row in range(batch):
            assert len(_live(out[row].tolist())) == k

    @pytest.mark.parametrize("k", [512, 2048])
    def test_duplicate_scores_stay_ascending_in_the_tie_path(self, k):
        """The tie path (radix_tie_select) assigns slots the same way.

        Few distinct values put a wide tie set on the threshold bin, which is
        where ``radix_tie_select`` runs and where its arrival-order
        ``counter`` / ``counter_final`` atomics used to decide the permutation.
        """
        torch.manual_seed(31337 + k)
        batch, seq, nvals = 2, k * 8, 64
        g = torch.Generator(device="cuda").manual_seed(nvals * 7 + seq)
        scores = torch.randint(
            0, nvals, (batch, seq), generator=g, device="cuda"
        ).float()
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
        out = _run_paged(scores, seq_lens, k)
        for row in range(batch):
            _assert_sorted_with_contiguous_pad(
                out[row].tolist(), k, f"ties b={row} k={k}"
            )

    @pytest.mark.parametrize("k", [512, 2048])
    def test_ragged_emission_is_ascending(self, k):
        """The ragged (extend) entry point stages and canonicalizes too."""
        torch.manual_seed(2024 + k)
        rows = [(0, 900), (0, 5000), (0, 20000), (0, 70000)]
        width = ((max(s + n for s, n in rows)) + 3) & ~3
        scores = torch.randn(len(rows), width, dtype=torch.float32, device="cuda")
        starts = torch.tensor([s for s, _ in rows], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([n for _, n in rows], dtype=torch.int32, device="cuda")
        offsets = starts + 1024
        out = torch.empty((len(rows), k), dtype=torch.int32, device="cuda")
        topk_transform_ragged_v2(
            scores, lengths, out_offsets=offsets, out_indices=out, row_starts=starts
        )
        torch.cuda.synchronize()
        for row in range(len(rows)):
            # The emitted values are window-relative positions plus the offset;
            # the offset is a constant per row, so ascending survives it.
            _assert_sorted_with_contiguous_pad(
                out[row].cpu().tolist(), k, f"ragged row={row} k={k}"
            )

    @pytest.mark.parametrize("k", [512])
    def test_trivial_path_is_ascending(self, k):
        """`seq_len <= k` bypasses the sort (it emits 0..seq_len-1 plus -1)."""
        batch, seq = 4, k // 4
        torch.manual_seed(99 + k)
        scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
        out = _run_paged(scores, seq_lens, k)
        for row in range(batch):
            assert out[row].tolist() == list(range(seq)) + [-1] * (k - seq)


class TestDeterministicEmitSwitch(unittest.TestCase):
    """The off state is the discriminating control for the ascending invariant.

    The flag is a compile-time define resolved when the JIT module is first
    built, so flipping it needs a fresh process; ``subprocess`` re-runs this
    module in a child with the flag off and only asserts that the child built a
    non-canonical output. Where the child is skipped (no accelerator) the parent
    skips too.
    """

    def test_flag_defaults_on(self):
        from sglang.srt.environ import envs

        self.assertTrue(envs.SGLANG_TOPK_V2_DETERMINISTIC_EMIT.get())

    def test_flag_off_produces_a_non_canonical_order(self):
        import subprocess
        import textwrap

        child = textwrap.dedent(
            """
            import torch
            from sglang.kernels.ops.attention.dsv4.topk import (
                plan_topk_v2, topk_transform_paged_v2)
            if not torch.cuda.is_available():
                raise SystemExit(77)
            PAGE_SIZE = 64
            k, seq, batch = 512, 32768, 2
            torch.manual_seed(12345)
            scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
            seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
            pt = torch.arange(seq, dtype=torch.int32, device="cuda")
            pt = pt.unsqueeze(0).expand(batch, -1).contiguous()
            out = torch.full((batch, k), -1, dtype=torch.int32, device="cuda")
            md = plan_topk_v2(seq_lens)
            torch.cuda.synchronize()
            topk_transform_paged_v2(scores, seq_lens, pt, out, PAGE_SIZE, md)
            torch.cuda.synchronize()
            unsorted = 0
            for row in out.cpu().tolist():
                live = [v for v in row if v != -1]
                if live != sorted(live):
                    unsorted += 1
            # The arrival-order emission is generally NOT ascending; that is the
            # behaviour the flag exists to remove.
            raise SystemExit(0 if unsorted else 1)
            """
        )
        env = dict(os.environ, SGLANG_TOPK_V2_DETERMINISTIC_EMIT="0")
        proc = subprocess.run(
            [sys.executable, "-c", child], env=env, capture_output=True, text=True
        )
        if proc.returncode == 77:
            self.skipTest("no accelerator in the child process")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))