"""Unit tests for the decode-batch row-0 replica padding (V522-S3 fix 2).

Decode batches are padded to the attn-TP multiple (or the captured cuda-graph
bs) with dummy rows. With zero/fill padding those rows carry
``req_pool_indices=0`` and ``out_cache_loc=0``, so their attention reads request
slot 0's stale ``req_to_token`` mapping and their KV writes land on cache
location 0. That makes the padding rows a function of whatever the previous
batch left behind, and the noise reaches real rows through batch-spanning ops:
identical greedy requests decode differently run to run.

``ForwardBatch._pad_inputs_to_size`` now replicates row 0 into the dummy rows
for DECODE batches when ``SGLANG_DECODE_PAD_CLONE_FIRST_ROW`` is on (default).
These tests pin the contract of that padding:

  - the padded rows are exact copies of row 0, not zeros / fill values;
  - every per-row scalar that carries a request identity is replicated
    (``input_ids``, ``positions``, ``req_pool_indices``, ``seq_lens``,
    ``out_cache_loc``) including the ``_cpu`` twins;
  - ``seq_lens_sum`` stays exact when the host copy exists and falls back to
    ``None`` (the backend's over-allocation path, see #26738) rather than
    under-reporting when it does not;
  - the switch is honoured, and non-DECODE batches keep the fill semantics.

The padding helper is exercised directly rather than through a running model:
``_pad_inputs_to_size`` needs a real model runner only for the seq-len fill
value, which is a one-method stub here.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

# Pure metadata test: `_pad_inputs_to_size` is exercised directly with a stubbed
# model runner (only the seq-len fill value is read from it), so no accelerator
# and no model weights are needed.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The exact per-row scalars the fix is required to replicate. Anything that
# carries a request identity has to be a row-0 copy; a residual zero here is the
# defect the fix removes.
REPLICATED = (
    "input_ids",
    "positions",
    "req_pool_indices",
    "out_cache_loc",
)


class _StubBackend:
    """Only the seq-len fill value is read from the backend."""

    def __init__(self, seq_len_fill_value: int):
        self._fill = seq_len_fill_value

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return self._fill

    def get_cpu_graph_seq_len_fill_value(self) -> int:
        return self._fill


class _StubModelRunner:
    def __init__(self, seq_len_fill_value: int = 1):
        self.attn_backend = _StubBackend(seq_len_fill_value)


class _DraftSpecInfo:
    """The draft-input surface ``_pad_inputs_to_size`` touches, and nothing else.

    ``pad_batch`` takes the padding helper that the caller passes down, so the
    stub records what it was handed instead of reimplementing it.
    """

    def __init__(self, n_rows: int = 3):
        self.hidden_states = torch.tensor(
            [[float(10 * i + j) for j in range(2)] for i in range(n_rows)]
        )
        self.topk_p = torch.full((n_rows, 4), 0.5)
        self.topk_index = torch.arange(n_rows * 4, dtype=torch.int64).view(n_rows, 4)

    def is_draft_input(self) -> bool:
        return True

    def pad_batch(self, pad_fn, bs):
        """Mirrors the real draft types: `_pad_inputs_to_size` pads `topk_p` /
        `topk_index` / `hidden_states` itself and then hands `pad_batch` the same
        helper for whatever else the concrete type carries. The stub only needs
        to accept the call.
        """
        self.pad_batch_called_with = bs


def _make_batch(n_rows: int, seq_len_fill_value: int = 1) -> ForwardBatch:
    """A decode-shaped batch of `n_rows` real rows, built without a model."""
    batch = ForwardBatch.__new__(ForwardBatch)
    batch.forward_mode = ForwardMode.DECODE
    # Row 0 is deliberately distinguishable from every other row and from the
    # zero/fill padding, so a replica is unmistakable.
    batch.input_ids = torch.tensor([10, 20, 30][:n_rows], dtype=torch.int64)
    batch.positions = torch.tensor([100, 200, 300][:n_rows], dtype=torch.int64)
    batch.req_pool_indices = torch.tensor([7, 8, 9][:n_rows], dtype=torch.int32)
    batch.req_pool_indices_cpu = torch.tensor([7, 8, 9][:n_rows], dtype=torch.int32)
    batch.out_cache_loc = torch.tensor([555, 666, 777][:n_rows], dtype=torch.int32)
    batch.seq_lens = torch.tensor([11, 22, 33][:n_rows], dtype=torch.int32)
    batch.seq_lens_cpu = torch.tensor([11, 22, 33][:n_rows], dtype=torch.int32)
    batch.seq_lens_sum = 11 + 22 + 33
    batch.input_embeds = None
    batch.mm_input_embeds = None
    batch.lora_ids = None
    batch.origin_out_cache_loc = None
    batch.encoder_lens = None
    batch.mamba_track_indices = None
    batch.mamba_prefill_track_mask_cpu = None
    batch.mamba_track_seqlens_cpu = None
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None
    batch.mrope_positions = None
    batch.rids_int = None
    batch.bootstrap_room_ids_int = None
    batch.spec_info = None
    batch.extend_seq_lens = None
    batch.extend_prefix_lens = None
    return batch


def _pad(batch: ForwardBatch, num_tokens: int, bs: int, fill_value: int = 1):
    batch._pad_inputs_to_size(_StubModelRunner(fill_value), num_tokens, bs)
    return batch


class TestDecodePadCloneFirstRow(unittest.TestCase):
    def test_padded_rows_replicate_row_zero(self):
        """Every replicated scalar repeats row 0; none of them is zero/fill."""
        batch = _pad(_make_batch(3), num_tokens=8, bs=8)
        for name in REPLICATED:
            tensor = getattr(batch, name)
            self.assertEqual(tensor.shape[0], 8, name)
            for row in range(3, 8):
                self.assertEqual(
                    int(tensor[row]),
                    int(tensor[0]),
                    f"{name} row {row} is not a replica of row 0: {tensor.tolist()}",
                )
            # The real rows are untouched.
            self.assertEqual(
                tensor[:3].tolist(),
                getattr(_make_batch(3), name).tolist(),
                name,
            )

    def test_cpu_twins_are_replicated_too(self):
        """The `_cpu` twins drive host-side bookkeeping and must agree."""
        batch = _pad(_make_batch(3), num_tokens=8, bs=8)
        for name in ("req_pool_indices_cpu", "seq_lens_cpu"):
            tensor = getattr(batch, name)
            self.assertEqual(tensor.shape[0], 8, name)
            for row in range(3, 8):
                self.assertEqual(int(tensor[row]), int(tensor[0]), f"{name} row {row}")

    def test_seq_lens_use_row_zero_length_not_the_fill_value(self):
        """A zero/fill seq_len would give the replica a different KV window."""
        batch = _pad(_make_batch(3), num_tokens=8, bs=8, fill_value=1)
        self.assertNotEqual(int(batch.seq_lens[0]), 1, "test needs a non-fill row 0")
        for row in range(3, 8):
            self.assertEqual(int(batch.seq_lens[row]), int(batch.seq_lens[0]))

    def test_seq_lens_sum_stays_exact_with_a_host_copy(self):
        """With `seq_lens_cpu` the sum counts row 0's length for each replica."""
        batch = _pad(_make_batch(3), num_tokens=8, bs=8, fill_value=1)
        # 3 real rows (11 + 22 + 33) + 5 replicas of row 0 (11 each).
        self.assertEqual(batch.seq_lens_sum, 11 + 22 + 33 + 5 * 11)

    def test_seq_lens_sum_falls_back_to_none_without_a_host_copy(self):
        """No host copy -> None, so the backend over-allocates instead of
        under-reporting the cloned rows' windows (the #26738 path)."""
        batch = _make_batch(3)
        batch.seq_lens_cpu = None
        _pad(batch, num_tokens=8, bs=8, fill_value=1)
        self.assertIsNone(batch.seq_lens_sum)

    def test_first_seq_len_ignores_a_uniform_fill_host_copy(self):
        """Row 0's own length, read from the host tensor, is what is counted."""
        batch = _make_batch(3)
        batch.seq_lens_cpu = torch.tensor([40, 22, 33], dtype=torch.int32)
        batch.seq_lens_sum = 40 + 22 + 33
        _pad(batch, num_tokens=8, bs=8, fill_value=1)
        self.assertEqual(batch.seq_lens_sum, 40 + 22 + 33 + 5 * 40)

    def test_no_padding_is_a_no_op(self):
        """A batch already at size is returned unchanged (no clone needed)."""
        batch = _pad(_make_batch(3), num_tokens=3, bs=3)
        self.assertEqual(batch.input_ids.tolist(), [10, 20, 30])
        self.assertEqual(batch.seq_lens.tolist(), [11, 22, 33])
        self.assertEqual(batch.seq_lens_sum, 11 + 22 + 33)


class TestDecodePadCloneSwitch(unittest.TestCase):
    def test_switch_off_restores_zero_and_fill_padding(self):
        """`SGLANG_DECODE_PAD_CLONE_FIRST_ROW=0` gives back the old padding."""
        with patch.dict(os.environ, {"SGLANG_DECODE_PAD_CLONE_FIRST_ROW": "0"}):
            self.assertFalse(envs.SGLANG_DECODE_PAD_CLONE_FIRST_ROW.get())
            batch = _pad(_make_batch(3), num_tokens=8, bs=8, fill_value=1)
        # input_ids / positions / out_cache_loc pad with 0, seq_lens with the
        # backend fill value, req_pool_indices with 0.
        for name in ("input_ids", "positions", "out_cache_loc", "req_pool_indices"):
            self.assertEqual(
                getattr(batch, name)[3:].tolist(),
                [0] * 5,
                f"{name} should be zero-padded with the switch off",
            )
        self.assertEqual(batch.seq_lens[3:].tolist(), [1] * 5)
        # The old sum accounting adds the fill value per padded row.
        self.assertEqual(batch.seq_lens_sum, 11 + 22 + 33 + 5 * 1)

    def test_switch_defaults_on(self):
        self.assertTrue(envs.SGLANG_DECODE_PAD_CLONE_FIRST_ROW.get())

    def test_extend_batches_keep_their_fill_semantics(self):
        """Only DECODE is in scope: extend keeps zero/fill padding."""
        batch = _make_batch(3)
        batch.forward_mode = ForwardMode.EXTEND
        _pad(batch, num_tokens=8, bs=8, fill_value=1)
        self.assertEqual(batch.input_ids[3:].tolist(), [0] * 5)
        self.assertEqual(batch.req_pool_indices[3:].tolist(), [0] * 5)

    def test_target_verify_batches_keep_their_fill_semantics(self):
        """TARGET_VERIFY pads inside the spec block with its own semantics."""
        batch = _make_batch(3)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        _pad(batch, num_tokens=8, bs=8, fill_value=1)
        self.assertEqual(batch.input_ids[3:].tolist(), [0] * 5)

    def test_draft_decode_batches_keep_their_fill_semantics(self):
        """A DECODE batch carrying a draft ``spec_info`` must NOT be cloned.

        ``_pad_inputs_to_size`` pads that shape's ``hidden_states`` / ``topk_p``
        / ``topk_index`` with zeros further down, so a cloned row would run row
        0's input ids and positions through a *different* draft state and write
        the result into row 0's own KV slots -- a second, divergent writer of the
        same locations, not an idempotent replica. The batch therefore stays on
        the fill path until those fields are replicated too.
        """
        batch = _make_batch(3)
        batch.spec_info = _DraftSpecInfo()
        _pad(batch, num_tokens=8, bs=8, fill_value=1)
        self.assertEqual(batch.input_ids[3:].tolist(), [0] * 5)
        self.assertEqual(batch.positions[3:].tolist(), [0] * 5)
        self.assertEqual(batch.out_cache_loc[3:].tolist(), [0] * 5)
        self.assertEqual(batch.seq_lens[3:].tolist(), [1] * 5)
        # The draft fields really are zero-filled, which is why the clone is
        # unsafe for this shape: the padding rows carry row 0's input ids and
        # positions but a zero draft state.
        self.assertEqual(
            batch.spec_info.hidden_states[3:].tolist(), [[0.0, 0.0]] * 5
        )
        self.assertEqual(batch.spec_info.topk_p[3:].tolist(), [[0.0] * 4] * 5)

    def test_plain_decode_without_spec_info_is_still_cloned(self):
        """The draft guard must not disable the fix for ordinary decode."""
        batch = _pad(_make_batch(3), num_tokens=8, bs=8)
        self.assertEqual(batch.input_ids[3:].tolist(), [10] * 5)


class TestDecodePadNoActivationAmplification(unittest.TestCase):
    """The replica must not become an extra copy of row 0's *output*.

    A replica repeats row 0's forward pass exactly, so its KV write and its
    request-scoped state writes carry values identical to row 0's own and are
    idempotent. What must not happen is the replica reading or writing a
    *different* request's slot, which is what the zero padding did. These are
    structural checks that the replicated rows carry row 0's identity, so any
    per-row side effect they trigger targets row 0's own state.
    """

    def test_replicas_carry_row_zero_request_identity(self):
        batch = _pad(_make_batch(3), num_tokens=8, bs=8)
        self.assertEqual(batch.req_pool_indices[3:].tolist(), [7] * 5)
        self.assertEqual(batch.req_pool_indices_cpu[3:].tolist(), [7] * 5)

    def test_replicas_write_the_same_kv_location_as_row_zero(self):
        """Same `out_cache_loc` as row 0 -> an idempotent write, not a new one."""
        batch = _pad(_make_batch(3), num_tokens=8, bs=8)
        self.assertEqual(batch.out_cache_loc[3:].tolist(), [555] * 5)
        # And emphatically not the reserved dummy location 0.
        self.assertNotIn(0, batch.out_cache_loc[3:].tolist())

    def test_replicas_read_the_same_positions_as_row_zero(self):
        batch = _pad(_make_batch(3), num_tokens=8, bs=8)
        self.assertEqual(batch.positions[3:].tolist(), [100] * 5)
        self.assertEqual(batch.input_ids[3:].tolist(), [10] * 5)


if __name__ == "__main__":
    sys.exit(unittest.main())