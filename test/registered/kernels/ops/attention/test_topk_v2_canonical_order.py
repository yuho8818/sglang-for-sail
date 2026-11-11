"""Unit tests for the canonical topk_v2 emission order (V522-S11 fix 1).

The dsv4 top-k select implementations (``TopKRegister`` / ``TopKStreaming`` /
``radix_tie_select``) claim their output slots with warp-arrival atomics, so the
same selected *set* reached the caller in a different *permutation* on every run.
Sparse attention accumulates in emission order, which makes that permutation
visible in the logits, and identical greedy requests then decode differently run
to run.

``SGLANG_TOPK_V2_DETERMINISTIC_EMIT`` (default on) sorts the staged selection into
ascending index order before the page transform, so the output order is a fixed
function of the selected set. These tests pin:

  - the emitted live indices are strictly ascending (the new invariant);
  - the -1 padding is contiguous at the tail, which the consumers already expect;
  - **every one of the ``k`` output slots holds the selected element**, i.e. the
    canonicalizer sorts the whole buffer rather than only the prefix below some
    arrival-order counter. This is the property that catches a mis-derived live
    prefix: masking from ``count_gt`` leaves the threshold bin out and produces a
    short live prefix plus a -1 tail on every row;
  - the selected *set* is unchanged, so the fix is order-only, including under
    heavy ties where the value-then-index tie rule must reproduce ``torch.topk``;
  - ``candidate_block_ids`` (the ``topk_ragged_v2`` consumer feeding the
    two-level indexer and, downstream, ``topk_candidate_kernel``) emits no -1
    inside its valid slots;
  - switching the flag off restores the arrival-order emission (the escape
    hatch), which is also the discriminating control: with it off the output is
    generally NOT sorted.

The kernel is JIT-compiled for the local accelerator, so this module is skipped
where no accelerator is present.
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "topk_v2 is a JIT kernel; needs an accelerator", allow_module_level=True
    )

from sglang.kernels.ops.attention.dsv4.topk import (  # noqa: E402
    plan_topk_v2,
    topk_transform_candidates,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci  # noqa: E402

# A JIT top-k kernel on a real accelerator; sits with the rest of the topk_v2
# coverage in the kernel unit stage.
register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=90, stage="jit-kernel-unit", runner_config="amd")

PAGE_SIZE = 64
PAGE_BITS = PAGE_SIZE.bit_length() - 1

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

# Shapes whose scores are drawn from a tiny pool of distinct values, so the
# threshold bin holds far more than one element and every ``handle_tie`` branch
# (emit all, pad the rest, the rank-based warp paths, ``radix_tie_select``) is
# reached. ``nvals`` is small enough that num_ties exceeds the ``k`` budget.
TIE_SHAPES = [
    (2, 4096, 512, 16),
    (2, 8192, 512, 64),
    (2, 32768, 512, 64),
    (2, 32768, 2048, 128),
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
    # Padding is a single contiguous tail.
    assert row[len(live) :] == [-1] * (k - len(live)), f"{label}: non-tail padding"


def _assert_slots_are_all_live(row, k, label):
    """No -1 inside ``[0, k)`` when ``seq_len > k``: nothing was dropped.

    ``seq_len > k`` guarantees more than ``k`` candidates exist, so the select
    must fill every slot. A live prefix derived from the arrival-order
    ``count_gt`` (the "strictly above" count the tie path continues past)
    instead drops the whole threshold bin and leaves a -1 tail here.
    """
    assert len(_live(row)) == k, (
        f"{label}: only {len(_live(row))} of {k} slots were filled; the sort "
        "covered less than the full staging buffer"
    )
    assert -1 not in row, f"{label}: -1 left inside the budget"


class TestCanonicalEmissionSink:
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
    def test_all_slots_are_filled_when_seq_exceeds_k(self, batch, seq, k):
        torch.manual_seed(batch * 100003 + seq * 7 + k + 5)
        scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
        out = _run_paged(scores, seq_lens, k)
        for row in range(batch):
            _assert_slots_are_all_live(out[row].tolist(), k, f"b={row} seq={seq} k={k}")

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

    @pytest.mark.parametrize("batch,seq,k,nvals", TIE_SHAPES)
    def test_duplicate_scores_stay_ascending_in_the_tie_path(
        self, batch, seq, k, nvals
    ):
        """The tie path (radix_tie_select) assigns slots the same way.

        Few distinct values put a wide tie set on the threshold bin, which is
        where ``radix_tie_select`` runs and where its arrival-order ``counter`` /
        ``counter_final`` atomics used to decide the permutation. The tie rule is
        value descending, then index ascending, so the selected *set* is exactly
        ``torch.topk``'s.
        """
        g = torch.Generator(device="cuda").manual_seed(nvals * 7 + seq + k)
        scores = torch.randint(
            0, nvals, (batch, seq), generator=g, device="cuda"
        ).float()
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
        out = _run_paged(scores, seq_lens, k)
        for row in range(batch):
            vals = out[row].tolist()
            _assert_sorted_with_contiguous_pad(vals, k, f"ties b={row} k={k}")
            _assert_slots_are_all_live(vals, k, f"ties b={row} k={k}")
            # value-descending / index-ascending tie rule == torch.topk on the
            # same scores, so the fix must not change which elements are picked.
            idx = torch.tensor(_live(vals), dtype=torch.long)
            got = scores[row][idx].sort().values
            want = torch.topk(scores[row], k, largest=True, sorted=True).values
            torch.testing.assert_close(got, want.sort().values, rtol=0, atol=0)

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
    def test_candidate_block_ids_are_complete_and_unordered_or_ascending(self, k):
        """The kernel ``topk_candidate_kernel`` consumes must not see -1 early.

        ``candidate_block_ids`` runs ``topk_ragged_v2`` over the block maxima, and
        ``topk_candidate_kernel`` treats the first ``min(num_blocks,
        blocks_stride)`` entries as valid block ids, dereferencing each as
        ``blocks[v >> 1] * 8``. A -1 inside that prefix becomes
        ``uint32_t(-1) * 8`` -- a wild gather, i.e. an illegal memory access --
        so no -1 may appear before the row's own block count.
        """
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_ids,
        )
        from sglang.kernels.ops.attention.dsv4.candidate_table import (
            CANDIDATE_BLOCK_SIZE,
        )

        torch.manual_seed(4242 + k)
        rows, seq = 4, 32 * 1024  # 4096 blocks: far more than topk_blocks
        logits = torch.randn(rows, seq, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((rows,), seq, dtype=torch.int32, device="cuda")
        nblocks = (seq_lens + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        topk_blocks = 128
        blocks = candidate_block_ids(logits, seq_lens, nblocks, topk_blocks)
        torch.cuda.synchronize()
        assert blocks.shape == (rows, topk_blocks)
        for row in range(rows):
            vals = blocks[row].tolist()
            kept = min(int(nblocks[row]), topk_blocks)
            assert -1 not in vals[:kept], (
                f"row {row}: -1 inside the valid block prefix {vals[:kept]}"
            )
            assert all(v >= 0 for v in vals[:kept])
            # Ascending: `candidate_block_ids` stages through `topk_ragged_v2`, so
            # the canonicalizer sorts the block ids it emits. (The block list is
            # re-sorted again downstream by `sort_candidate_blocks`, which also
            # rewrites the padding to INT32_MAX; the assertion here is about this
            # entry point's own output.)
            assert vals[:kept] == sorted(vals[:kept]), f"row {row}: not ascending"
            assert len(set(vals[:kept])) == kept, f"row {row}: duplicate blocks"

    @pytest.mark.parametrize("k", [512])
    def test_candidate_transform_agrees_with_full_topk(self, k):
        """The candidate level's set equals the full top-k over the same scores.

        The emitted order here is the *block* order of the input ``blocks``
        (``topk_transform_candidates`` emits ``block * 8 + j``), and this entry
        point does NOT re-sort its block list -- the production path runs
        ``sort_candidate_blocks`` afterwards, which is what makes the final table
        ascending. So this test pins the set and the -1-freedom, not an ordering
        it would be wrong to expect.
        """
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_ids,
        )
        from sglang.kernels.ops.attention.dsv4.candidate_table import (
            CANDIDATE_BLOCK_SIZE,
        )

        torch.manual_seed(777 + k)
        rows, seq = 2, 16 * 1024
        logits = torch.randn(rows, seq, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((rows,), seq, dtype=torch.int32, device="cuda")
        nblocks = (seq_lens + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        blocks = candidate_block_ids(logits, seq_lens, nblocks, k)
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        topk_transform_candidates(
            logits,
            seq_lens,
            blocks,
            out_offsets=torch.zeros_like(seq_lens),
            out_indices=out,
        )
        torch.cuda.synchronize()
        for row in range(rows):
            vals = out[row].tolist()
            # No -1 inside the budget: this is the property whose violation makes
            # `topk_candidate_kernel` dereference a wild block id.
            assert -1 not in vals, f"row {row}: -1 inside the candidate budget"
            assert len(set(vals)) == k, f"row {row}: duplicate positions"
            idx = torch.tensor(vals, dtype=torch.long)
            got = logits[row][idx].sort().values
            want = torch.topk(logits[row], k, sorted=True).values.sort().values
            torch.testing.assert_close(got, want, rtol=0, atol=0)

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


class TestDeterministicEmitSwitch:
    """The off state is the discriminating control for the ascending invariant.

    The flag is a compile-time define resolved when the JIT module is first
    built, so flipping it needs a fresh process; ``subprocess`` re-runs the core
    measurement in a child with the flag off and only asserts that the child built
    a non-canonical output. Where the child is skipped (no accelerator) the parent
    skips too.
    """

    def test_flag_defaults_on(self):
        from sglang.srt.environ import envs

        assert envs.SGLANG_TOPK_V2_DETERMINISTIC_EMIT.get()

    def test_flag_off_produces_a_non_canonical_order(self):
        import subprocess

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
            pytest.skip("no accelerator in the child process")
        assert proc.returncode == 0, proc.stdout + proc.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))