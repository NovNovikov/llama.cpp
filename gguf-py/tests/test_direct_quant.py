import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DIRECT_QUANT = ROOT / "conversion" / "direct_quant.py"


def load_direct_quant_module():
    spec = importlib.util.spec_from_file_location("direct_quant_test", DIRECT_QUANT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def local_tensor(path: Path, dtype: str, shape: tuple[int, ...], offset: int, size: int):
    return SimpleNamespace(
        dtype=dtype,
        shape=shape,
        data_range=SimpleNamespace(filename=path, offset=offset, size=size),
    )


class TestFP8ScaledTensor(unittest.TestCase):
    def test_storage_tensor_preserves_bf16_bytes(self):
        direct_quant = load_direct_quant_module()
        bits = np.array([[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.bin"
            source_path.write_bytes(bits.tobytes())
            storage = direct_quant.DirectStorageTensor(
                local_tensor(source_path, "BF16", bits.shape, 0, bits.nbytes))
            lazy = storage.lazy_storage(elements_per_chunk=3)
            restored = np.concatenate([chunk() for chunk in lazy._chunks])

            self.assertEqual(storage.ggml_type, direct_quant.gguf.GGMLQuantizationType.BF16)
            self.assertEqual(lazy.shape, bits.shape)
            self.assertTrue(np.array_equal(restored, bits.reshape(-1)))

    def test_writer_accepts_direct_encoded_shape(self):
        direct_quant = load_direct_quant_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "direct.gguf"
            encoded = direct_quant.gguf.LazyChunkedTensor(
                [lambda: np.zeros((2, 84), dtype=np.uint8)], (2, 84), np.uint8)
            writer = direct_quant.gguf.GGUFWriter(output, "test")
            writer.add_tensor(
                "blk.0.ffn_gate_exps.weight",
                encoded,
                raw_shape=(2, 256),
                raw_dtype=direct_quant.gguf.GGMLQuantizationType.Q2_K,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = direct_quant.gguf.GGUFReader(output)
            tensor = reader.tensors[0]
            self.assertEqual(tensor.tensor_type, direct_quant.gguf.GGMLQuantizationType.Q2_K)
            self.assertEqual(tuple(tensor.shape), (256, 2))
            del tensor
            reader.data._mmap.close()
            del reader

    def test_expands_glm_e8m0_grid(self):
        direct_quant = load_direct_quant_module()
        weights = np.full((4, 256), 56, dtype=np.uint8)  # E4M3FN value 1.0
        scales = np.array([[127, 128]], dtype=np.uint8)  # E8M0 values 1.0, 2.0
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.bin"
            source_path.write_bytes(weights.tobytes() + scales.tobytes())
            source = direct_quant.FP8ScaledTensor(
                local_tensor(source_path, "F8_E4M3FN", weights.shape, 0, weights.nbytes),
                local_tensor(source_path, "F8_E8M0", scales.shape, weights.nbytes, scales.nbytes),
            )

            decoded = next(source.iter_float_rows(rows_per_chunk=128))
            self.assertEqual(decoded.dtype, np.float32)
            self.assertTrue(np.all(decoded[:, :128] == 1.0))
            self.assertTrue(np.all(decoded[:, 128:] == 2.0))

    def test_rejects_incompatible_scale_grid(self):
        direct_quant = load_direct_quant_module()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.bin"
            source_path.write_bytes(bytes(4 * 256 + 1))

            with self.assertRaisesRegex(direct_quant.DirectQuantError, "scale shape"):
                direct_quant.FP8ScaledTensor(
                    local_tensor(source_path, "F8_E4M3FN", (4, 256), 0, 4 * 256),
                    local_tensor(source_path, "F8_E8M0", (1, 1), 4 * 256, 1),
                )

    def test_expert_tensor_keeps_stack_order(self):
        direct_quant = load_direct_quant_module()
        weights = np.full((4, 256), 56, dtype=np.uint8)  # E4M3FN value 1.0
        scales = np.array([[127, 128]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.bin"
            source_path.write_bytes(weights.tobytes() + scales.tobytes())
            weight = local_tensor(source_path, "F8_E4M3FN", weights.shape, 0, weights.nbytes)
            scale = local_tensor(source_path, "F8_E8M0", scales.shape, weights.nbytes, scales.nbytes)
            first = direct_quant.FP8ScaledTensor(weight, scale)
            second = direct_quant.FP8ScaledTensor(weight, scale)
            experts = direct_quant.FP8ExpertTensor([first, second])

            class RecordingQuantizer:
                def __init__(self):
                    self.rows = []

                def quantize_rows(self, rows, qtype):
                    del qtype
                    self.rows.append(rows.copy())
                    return np.zeros((rows.shape[0], 84), dtype=np.uint8)

            quantizer = RecordingQuantizer()
            encoded = experts.lazy_quantized(quantizer, direct_quant.gguf.GGMLQuantizationType.Q2_K)
            for chunk in encoded._chunks:
                chunk()

            self.assertEqual(experts.shape, (2, 4, 256))
            self.assertEqual(len(quantizer.rows), 2)
            for decoded in quantizer.rows:
                self.assertTrue(np.all(decoded[:, :128] == 1.0))
                self.assertTrue(np.all(decoded[:, 128:] == 2.0))
