from pathlib import Path
import sys
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gguf

from conversion.direct_quant import DirectQuantError
from conversion.glm5next import Glm5NextModel


class TestGlm5NextDirectManifest(unittest.TestCase):
    def make_model(self):
        model = object.__new__(Glm5NextModel)
        model.hparams = {
            "num_key_value_heads": 64,
            "qk_nope_head_dim": 256,
            "v_head_dim": 256,
        }

        def output_name(name):
            if "k_b_proj" in name:
                return "blk.3.attn_k_b.weight"
            if "v_b_proj" in name:
                return "blk.3.attn_v_b.weight"
            raise AssertionError(name)

        model._direct_output_name = output_name
        return model

    def test_kv_b_manifest_uses_configured_geometry(self):
        model = self.make_model()
        outputs = model._direct_kv_b_outputs(
            "model.language_model.layers.3.self_attn.kv_b_proj.weight",
            (32768, 512),
        )

        self.assertEqual(outputs, (
            ("blk.3.attn_k_b.weight", (64, 512, 256)),
            ("blk.3.attn_v_b.weight", (64, 256, 512)),
        ))

    def test_canonical_manifest_contains_both_kv_b_outputs(self):
        model = self.make_model()
        manifest = model._canonical_direct_manifest({
            "model.layers.3.self_attn.kv_b_proj.weight": (
                "model.language_model.layers.3.self_attn.kv_b_proj.weight",
                SimpleNamespace(dtype="BF16", shape=(32768, 512)),
            ),
        }, set(), {})
        self.assertEqual(manifest, {
            "blk.3.attn_k_b.weight": (
                ("model.layers.3.self_attn.kv_b_proj.weight",),
                (64, 512, 256),
                gguf.GGMLQuantizationType.BF16,
            ),
            "blk.3.attn_v_b.weight": (
                ("model.layers.3.self_attn.kv_b_proj.weight",),
                (64, 256, 512),
                gguf.GGMLQuantizationType.BF16,
            ),
        })

    def test_kv_b_manifest_rejects_shape_mismatch(self):
        model = self.make_model()
        with self.assertRaisesRegex(DirectQuantError, "expected"):
            model._direct_kv_b_outputs(
                "model.layers.3.self_attn.kv_b_proj.weight",
                (32767, 512),
            )

    def test_kv_b_manifest_rejects_missing_geometry(self):
        model = self.make_model()
        del model.hparams["v_head_dim"]
        with self.assertRaisesRegex(DirectQuantError, "v_head_dim"):
            model._direct_kv_b_outputs(
                "model.layers.3.self_attn.kv_b_proj.weight",
                (32768, 512),
            )

    def test_mandatory_f32_policy_is_shared_with_model_base(self):
        model = self.make_model()
        self.assertTrue(model._direct_requires_f32("blk.3.indexer.proj.weight", (10240, 4), 3))
        self.assertFalse(model._direct_requires_f32("blk.3.ffn_gate_exps.weight", (640, 2560, 512), 3))

    def test_mixed_recipe_uses_predominant_supported_file_type(self):
        model = self.make_model()
        model.ftype = gguf.LlamaFileType.MOSTLY_BF16
        model._direct_set_file_type((
            SimpleNamespace(target_type=gguf.GGMLQuantizationType.BF16, parameter_count=20),
            SimpleNamespace(target_type=gguf.GGMLQuantizationType.Q3_K, parameter_count=30),
            SimpleNamespace(target_type=gguf.GGMLQuantizationType.Q2_K, parameter_count=60),
        ))
        self.assertEqual(model.ftype, gguf.LlamaFileType.MOSTLY_Q2_K)


if __name__ == "__main__":
    unittest.main()
