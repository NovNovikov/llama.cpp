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
