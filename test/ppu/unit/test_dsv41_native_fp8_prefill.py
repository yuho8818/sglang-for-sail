"""CPU-only contract tests for the native FP8 sparse prefill wiring.

Covers the switch contract (off = the bf16 kernel path, no e4m3 workspace) and
the admission rules that decide whether a layer may take the quantized kernel.
Kernel numerics and the real wheel only exist on PPU hardware.
"""

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

ROOT = Path(__file__).resolve().parents[3]
PPU_ATTENTION = ROOT / "python/sglang/srt/hardware_backend/ppu/attention"
BACKEND = ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"


def _function_node(path, name):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


class _Env:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _load_admission():
    """Exec the admission helpers against a stub env facade and a stub wheel."""
    namespace = {"__builtins__": __builtins__}
    source = (PPU_ATTENTION / "dsv4_native_fp8_prefill.py").read_text()
    tree = ast.parse(source)
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    keep = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))
    ]
    module = ast.Module(body=imports + keep, type_ignores=[])
    ast.fix_missing_locations(module)
    # Imports of the real package tree are unnecessary here; the wheel, the env
    # facade and torch are the only dependencies of the code under test.
    namespace.update(torch=torch, functools=__import__("functools"))
    for node in imports:
        if node.module in ("__future__", "typing"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<prelude>", "exec"), namespace)
    exec(compile(module, "<dsv4_native_fp8_prefill>", "exec"), namespace)
    return namespace


class TestNativeFp8Admission(unittest.TestCase):
    def setUp(self):
        self.mod = _load_admission()
        wheel = MagicMock()
        wheel.flash_mla_sparse_quantized_fwd = object()
        wheel.QuantizedAttentionConfig = object()
        self.mod["_flashmla"] = wheel
        self.mod["_probe"].cache_clear()

    def call(self, on=True, layer_id=8, q_shape=(4096, 8, 512), topk=640, head_dim_v=512):
        self.mod["envs"] = MagicMock(SGLANG_OPT_PPU_DSV4_NATIVE_FP8_PREFILL=_Env(on))
        q = torch.zeros(q_shape, dtype=torch.bfloat16)
        indices = torch.zeros((q_shape[0], 1, topk), dtype=torch.int32)
        return self.mod["native_fp8_prefill_applies"](layer_id, q, head_dim_v, indices)

    def test_switch_off_never_applies(self):
        self.assertFalse(self.call(on=False))

    def test_switch_on_admits_supported_shape(self):
        self.assertTrue(self.call(on=True))

    def test_layer_range(self):
        for layer_id in (0, 1, 36, 40):
            self.assertFalse(self.call(on=True, layer_id=layer_id), layer_id)
        self.assertTrue(self.call(on=True, layer_id=2))
        self.assertTrue(self.call(on=True, layer_id=35))

    def test_head_multiple_of_eight(self):
        self.assertTrue(self.call(on=True, q_shape=(4096, 8, 512)))
        self.assertTrue(self.call(on=True, q_shape=(4096, 16, 512)))
        self.assertFalse(self.call(on=True, q_shape=(4096, 7, 512)))

    def test_topk_width(self):
        self.assertTrue(self.call(on=True, topk=640))
        self.assertFalse(self.call(on=True, topk=642))  # not a multiple of 64
        self.assertFalse(self.call(on=True, topk=2176))  # above the quantized cap

    def test_head_dim(self):
        self.assertFalse(self.call(on=True, q_shape=(128, 8, 576), head_dim_v=576))
        self.assertFalse(self.call(on=True, head_dim_v=256))

    def test_query_dtype(self):
        self.mod["envs"] = MagicMock(SGLANG_OPT_PPU_DSV4_NATIVE_FP8_PREFILL=_Env(True))
        q = torch.zeros((128, 8, 512), dtype=torch.float8_e4m3fn)
        indices = torch.zeros((128, 1, 640), dtype=torch.int32)
        self.assertFalse(self.mod["native_fp8_prefill_applies"](8, q, 512, indices))

    def test_missing_wheel_api_degrades(self):
        self.mod["_flashmla"] = object()
        self.mod["_probe"].cache_clear()
        self.assertFalse(self.call(on=True))

    def test_import_error_degrades(self):
        self.mod["_import_error"] = ImportError("no wheel")
        self.mod["_probe"].cache_clear()
        self.assertFalse(self.call(on=True))


class TestBackendSwitchContract(unittest.TestCase):
    """With the switch off the body must keep the bf16 kernel and bf16 workspace."""

    def setUp(self):
        self.source = ast.unparse(_function_node(BACKEND, "_forward_prefill_sparse"))

    def test_gate_guarded_by_import_time_probe(self):
        # The per-layer decision is behind an import-time probe so the bf16
        # path pays nothing when the switch is off.
        self.assertIn("if p_ppu_native_fp8_applies:", self.source)
        self.assertIn(
            "p_ppu_native_fp8_applies",
            (PPU_ATTENTION / "flash_mla.py").read_text(),
        )
        self.assertIn(
            "SGLANG_OPT_PPU_DSV4_NATIVE_FP8_PREFILL",
            (ROOT / "python/sglang/srt/environ.py").read_text(),
        )

    def test_fallback_kernel_preserved(self):
        self.assertIn("native_fn or flash_mla_sparse_fwd", self.source)

    def test_workspace_dtype_defaults_to_bf16(self):
        self.assertIn(
            "kv_dtype = fp8_dtype if native_fp8 else torch.bfloat16", self.source
        )


class TestHookRegistration(unittest.TestCase):
    def test_hook_targets_sparse_prefill(self):
        hooks = (PPU_ATTENTION / "ppu_flashmla_hooks.py").read_text()
        self.assertIn(
            "DeepseekV4AttnBackend._forward_prefill_sparse", hooks
        )


if __name__ == "__main__":
    unittest.main()