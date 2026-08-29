"""Native, chunked quantization support for the opt-in HF direct path.

The converter deliberately calls ggml_quantize_chunk() instead of implementing
Q/K quantizers in Python.  This keeps the direct path byte-compatible with the
quantizer's block representation for identical float inputs.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

import gguf


class DirectQuantError(RuntimeError):
    """The direct quantization path cannot safely encode the requested tensor."""


class GGMLChunkQuantizer:
    """Thin checked binding to ggml_quantize_chunk()."""

    def __init__(self, library_path: Path):
        library_path = library_path.resolve()
        if not library_path.is_file():
            raise DirectQuantError(f"ggml quantization library does not exist: {library_path}")

        self._dll_directory = None
        if os.name == "nt":
            self._dll_directory = os.add_dll_directory(str(library_path.parent))

        try:
            self._library = ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise DirectQuantError(f"failed to load ggml quantization library {library_path}: {exc}") from exc

        self._quantize_chunk = self._library.ggml_quantize_chunk
        self._quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )
        self._quantize_chunk.restype = ctypes.c_size_t

        self._validate_row_data = self._library.ggml_validate_row_data
        self._validate_row_data.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
        self._validate_row_data.restype = ctypes.c_bool

        self._quantize_requires_imatrix = self._library.ggml_quantize_requires_imatrix
        self._quantize_requires_imatrix.argtypes = (ctypes.c_int,)
        self._quantize_requires_imatrix.restype = ctypes.c_bool

    @staticmethod
    def _output_shape(
        shape: tuple[int, ...],
        qtype: gguf.GGMLQuantizationType,
    ) -> tuple[int, ...]:
        try:
            return gguf.quant_shape_to_byte_shape(shape, qtype)
        except ValueError as exc:
            raise DirectQuantError(str(exc)) from exc

    def quantize_rows(self, rows: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
        """Quantize independent complete rows with the native ggml implementation.

        The caller owns chunking.  Rows must be a C-contiguous F32 array whose
        final axis is a whole number of target quantization blocks.
        """
        if rows.dtype != np.float32:
            raise DirectQuantError(f"native direct quantization requires F32 input, got {rows.dtype}")
        if rows.ndim < 2:
            raise DirectQuantError(f"native direct quantization requires at least 2 dimensions, got {rows.ndim}")
        if not rows.flags.c_contiguous:
            raise DirectQuantError("native direct quantization requires C-contiguous rows")
        if self._quantize_requires_imatrix(int(qtype)):
            raise DirectQuantError(
                f"native direct quantization of {qtype.name} requires an importance matrix")

        out_shape = self._output_shape(tuple(rows.shape), qtype)
        output = np.empty(out_shape, dtype=np.uint8)

        n_per_row = rows.shape[-1]
        n_rows = rows.size // n_per_row
        written = self._quantize_chunk(
            int(qtype),
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(output.ctypes.data),
            0,
            n_rows,
            n_per_row,
            None,
        )

        if written != output.nbytes:
            raise DirectQuantError(
                f"ggml wrote {written} bytes for {qtype.name}, expected {output.nbytes}")
        if not self._validate_row_data(int(qtype), ctypes.c_void_p(output.ctypes.data), written):
            raise DirectQuantError(f"ggml rejected the encoded {qtype.name} payload")

        return output
