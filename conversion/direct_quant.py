"""Native, chunked quantization support for the opt-in HF direct path.

The converter deliberately calls ggml_quantize_chunk() instead of implementing
Q/K quantizers in Python.  This keeps the direct path byte-compatible with the
quantizer's block representation for identical float inputs.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Iterator, TYPE_CHECKING

import numpy as np

import gguf

if TYPE_CHECKING:
    from gguf.utility import LocalTensor


class DirectQuantError(RuntimeError):
    """The direct quantization path cannot safely encode the requested tensor."""


class DirectStorageTensor:
    """Byte-preserving lazy storage for an unmodified local safetensors tensor."""

    _DTYPES = {
        "F16": np.dtype("<f2"),
        "BF16": np.dtype("<u2"),
        "F32": np.dtype("<f4"),
    }
    _GGML_TYPES = {
        "F16": gguf.GGMLQuantizationType.F16,
        "BF16": gguf.GGMLQuantizationType.BF16,
        "F32": gguf.GGMLQuantizationType.F32,
    }

    def __init__(self, tensor: LocalTensor) -> None:
        try:
            self.dtype = self._DTYPES[tensor.dtype]
            self.ggml_type = self._GGML_TYPES[tensor.dtype]
        except KeyError as exc:
            raise DirectQuantError(
                f"direct byte-preserving tensor has unsupported safetensors dtype {tensor.dtype}") from exc
        expected_size = int(np.prod(tensor.shape, dtype=np.int64)) * self.dtype.itemsize
        if tensor.data_range.size != expected_size:
            raise DirectQuantError(
                f"direct tensor byte range has {tensor.data_range.size} bytes, expected {expected_size}")
        self.tensor = tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor.shape

    def lazy_storage(self, *, elements_per_chunk: int = 1 << 20) -> gguf.LazyChunkedTensor:
        if elements_per_chunk <= 0:
            raise DirectQuantError(
                f"direct tensor elements_per_chunk must be positive, got {elements_per_chunk}")
        n_elements = int(np.prod(self.shape, dtype=np.int64))
        chunks: list[Callable[[], np.ndarray]] = []
        for element_start in range(0, n_elements, elements_per_chunk):
            n_elements_chunk = min(elements_per_chunk, n_elements - element_start)

            def load_chunk(start: int = element_start, count: int = n_elements_chunk) -> np.ndarray:
                return np.memmap(
                    self.tensor.data_range.filename,
                    mode="r",
                    offset=self.tensor.data_range.offset + start * self.dtype.itemsize,
                    dtype=self.dtype,
                    shape=(count,),
                )

            chunks.append(load_chunk)
        return gguf.LazyChunkedTensor(chunks, self.shape, self.dtype)

    def lazy_float32(self, *, elements_per_chunk: int = 1 << 20) -> gguf.LazyChunkedTensor:
        """Stream a storage tensor as F32 without materializing it as a whole.

        Some GGUF runtime operations require F32 even when their source
        safetensors weights use BF16 or F16.  The normal converter performs
        that promotion before writing.  Direct conversion must do the same
        while retaining its bounded-memory behavior.
        """
        if self.ggml_type == gguf.GGMLQuantizationType.F32:
            return self.lazy_storage(elements_per_chunk=elements_per_chunk)
        if elements_per_chunk <= 0:
            raise DirectQuantError(
                f"direct tensor elements_per_chunk must be positive, got {elements_per_chunk}")

        n_elements = int(np.prod(self.shape, dtype=np.int64))
        chunks: list[Callable[[], np.ndarray]] = []
        for element_start in range(0, n_elements, elements_per_chunk):
            n_elements_chunk = min(elements_per_chunk, n_elements - element_start)

            def load_chunk(start: int = element_start, count: int = n_elements_chunk) -> np.ndarray:
                raw = np.memmap(
                    self.tensor.data_range.filename,
                    mode="r",
                    offset=self.tensor.data_range.offset + start * self.dtype.itemsize,
                    dtype=self.dtype,
                    shape=(count,),
                )
                if self.ggml_type == gguf.GGMLQuantizationType.BF16:
                    return (raw.astype(np.uint32) << 16).view(np.float32)
                return raw.astype(np.float32)

            chunks.append(load_chunk)
        return gguf.LazyChunkedTensor(chunks, self.shape, np.dtype("<f4"))

    def _load_float_rows(self, row_start: int, row_end: int) -> np.ndarray:
        if len(self.shape) != 2:
            raise DirectQuantError(
                f"direct native quantization requires a matrix, got shape {self.shape}")
        rows, cols = self.shape
        if row_start < 0 or row_start >= row_end or row_end > rows:
            raise DirectQuantError(f"invalid direct tensor row range [{row_start}, {row_end}) for {self.shape}")
        raw = np.memmap(
            self.tensor.data_range.filename,
            mode="r",
            offset=self.tensor.data_range.offset + row_start * cols * self.dtype.itemsize,
            dtype=self.dtype,
            shape=(row_end - row_start, cols),
        )
        if self.ggml_type == gguf.GGMLQuantizationType.BF16:
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32)

    def lazy_quantized(
        self,
        quantizer: GGMLChunkQuantizer,
        qtype: gguf.GGMLQuantizationType,
        *,
        rows_per_chunk: int = 128,
    ) -> gguf.LazyChunkedTensor:
        if len(self.shape) != 2:
            raise DirectQuantError(
                f"direct native quantization requires a matrix, got shape {self.shape}")
        try:
            byte_shape = gguf.quant_shape_to_byte_shape(self.shape, qtype)
        except ValueError as exc:
            raise DirectQuantError(
                f"cannot directly quantize tensor shape {self.shape} as {qtype.name}: {exc}") from exc
        rows, _ = self.shape
        chunks: list[Callable[[], np.ndarray]] = []
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)

            def load_chunk(start: int = row_start, end: int = row_end) -> np.ndarray:
                return quantizer.quantize_rows(self._load_float_rows(start, end), qtype)

            chunks.append(load_chunk)
        return gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8)


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


class FP8ScaledTensor:
    """A row-streaming FP8 safetensors matrix with a rectangular scale grid.

    The source tensor remains on disk.  A chunk contains only a whole number of
    scale-grid row blocks, so expanding its scales never requires the complete
    expert matrix.  ``scale_dtype`` is intentionally explicit: GLM's E8M0
    scale bytes are decoded locally, while conventional F16/F32 scales retain
    their native storage semantics.
    """

    _FP8_TORCH_DTYPES = {
        "F8_E4M3": "float8_e4m3fn",
        "F8_E4M3FN": "float8_e4m3fn",
        "F8_E4M3FNUZ": "float8_e4m3fnuz",
        "F8_E5M2": "float8_e5m2",
        "F8_E5M2FNUZ": "float8_e5m2fnuz",
    }
    _SCALE_DTYPES = {
        "F16": np.dtype("<f2"),
        "F32": np.dtype("<f4"),
        "I8": np.dtype("i1"),
        "U8": np.dtype("u1"),
    }

    def __init__(
        self,
        weight: LocalTensor,
        scale: LocalTensor,
        *,
        block_shape: tuple[int, int] = (128, 128),
    ) -> None:
        if len(weight.shape) != 2:
            raise DirectQuantError(
                f"direct FP8 tensor must be a matrix, got shape {weight.shape}")
        if weight.dtype not in self._FP8_TORCH_DTYPES:
            raise DirectQuantError(
                f"direct FP8 tensor has unsupported safetensors dtype {weight.dtype}")
        if len(scale.shape) != 2:
            raise DirectQuantError(
                f"direct FP8 scale must be a matrix, got shape {scale.shape}")
        if block_shape[0] <= 0 or block_shape[1] <= 0:
            raise DirectQuantError(f"invalid FP8 scale block shape {block_shape}")

        rows, cols = weight.shape
        expected_scale_shape = (
            (rows + block_shape[0] - 1) // block_shape[0],
            (cols + block_shape[1] - 1) // block_shape[1],
        )
        if scale.shape != expected_scale_shape:
            raise DirectQuantError(
                f"FP8 scale shape {scale.shape} does not match weight shape "
                f"{weight.shape} and scale block {block_shape}; expected {expected_scale_shape}")
        if weight.data_range.size != rows * cols:
            raise DirectQuantError(
                f"FP8 weight byte range has {weight.data_range.size} bytes, expected {rows * cols}")

        self.weight = weight
        self.scale = scale
        self.block_shape = block_shape

    @property
    def shape(self) -> tuple[int, int]:
        return self.weight.shape

    @staticmethod
    def _native_scale_dtype(dtype: str) -> np.dtype:
        try:
            return FP8ScaledTensor._SCALE_DTYPES[dtype]
        except KeyError as exc:
            raise DirectQuantError(f"unsupported direct FP8 scale dtype {dtype}") from exc

    @staticmethod
    def _decode_fp8(raw: np.ndarray, dtype: str) -> np.ndarray:
        try:
            import torch
            torch_dtype = getattr(torch, FP8ScaledTensor._FP8_TORCH_DTYPES[dtype])
        except (ImportError, AttributeError, KeyError) as exc:
            raise DirectQuantError(
                f"PyTorch cannot decode direct FP8 safetensors dtype {dtype}") from exc

        # torch owns the FP8 bit-format conversion.  ``copy`` avoids retaining
        # a memmap view for the lifetime of the temporary Torch tensor.
        return torch.from_numpy(raw.copy()).view(torch_dtype).float().numpy()

    @staticmethod
    def _decode_scale(raw: np.ndarray, dtype: str) -> np.ndarray:
        if dtype == "F8_E8M0":
            return np.exp2(raw.astype(np.float32) - 127.0, dtype=np.float32)
        if dtype == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32, copy=False)

    def _load_scale(self) -> np.ndarray:
        if self.scale.dtype == "F8_E8M0":
            dtype = np.dtype("u1")
        elif self.scale.dtype == "BF16":
            dtype = np.dtype("<u2")
        else:
            dtype = self._native_scale_dtype(self.scale.dtype)
        expected_size = int(np.prod(self.scale.shape, dtype=np.int64)) * dtype.itemsize
        if self.scale.data_range.size != expected_size:
            raise DirectQuantError(
                f"FP8 scale byte range has {self.scale.data_range.size} bytes, expected {expected_size}")
        raw = np.memmap(
            self.scale.data_range.filename,
            mode="r",
            offset=self.scale.data_range.offset,
            dtype=dtype,
            shape=self.scale.shape,
        )
        return self._decode_scale(raw, self.scale.dtype)

    def _load_float_rows(self, row_start: int, row_end: int) -> np.ndarray:
        rows, cols = self.shape
        if row_start < 0 or row_start >= row_end or row_end > rows:
            raise DirectQuantError(f"invalid direct FP8 row range [{row_start}, {row_end}) for {self.shape}")
        block_rows, block_cols = self.block_shape
        scales = self._load_scale()
        raw = np.memmap(
            self.weight.data_range.filename,
            mode="r",
            offset=self.weight.data_range.offset + row_start * cols,
            dtype=np.dtype("u1"),
            shape=(row_end - row_start, cols),
        )
        values = self._decode_fp8(raw, self.weight.dtype)
        scale_rows = scales[row_start // block_rows:(row_end + block_rows - 1) // block_rows]
        expanded = np.repeat(scale_rows, block_rows, axis=0)
        expanded_start = row_start % block_rows
        expanded = expanded[expanded_start:expanded_start + row_end - row_start]
        expanded = np.repeat(expanded, block_cols, axis=1)[:, :cols]
        return np.multiply(values, expanded, dtype=np.float32)

    def iter_float_rows(self, *, rows_per_chunk: int) -> Iterator[np.ndarray]:
        """Yield F32 rows, aligned to the source scale-grid rows."""
        rows, _ = self.shape
        block_rows, _ = self.block_shape
        if rows_per_chunk <= 0:
            raise DirectQuantError(f"direct FP8 rows_per_chunk must be positive, got {rows_per_chunk}")
        rows_per_chunk = max(block_rows, rows_per_chunk - rows_per_chunk % block_rows)

        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)
            yield self._load_float_rows(row_start, row_end)

    def lazy_float32(self, *, rows_per_chunk: int = 128) -> gguf.LazyChunkedTensor:
        """Stream the dequantized source as F32 for mandatory runtime tensors."""
        rows, _ = self.shape
        block_rows, _ = self.block_shape
        if rows_per_chunk <= 0:
            raise DirectQuantError(f"direct FP8 rows_per_chunk must be positive, got {rows_per_chunk}")
        rows_per_chunk = max(block_rows, rows_per_chunk - rows_per_chunk % block_rows)

        chunks: list[Callable[[], np.ndarray]] = []
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)

            def load_chunk(start: int = row_start, end: int = row_end) -> np.ndarray:
                return self._load_float_rows(start, end)

            chunks.append(load_chunk)
        return gguf.LazyChunkedTensor(chunks, self.shape, np.float32)

    def lazy_quantized(
        self,
        quantizer: GGMLChunkQuantizer,
        qtype: gguf.GGMLQuantizationType,
        *,
        rows_per_chunk: int = 128,
    ) -> gguf.LazyChunkedTensor:
        """Return raw native-quantized chunks suitable for ``GGUFWriter``.

        The returned lazy tensor has the *encoded* byte shape.  Pass
        ``raw_shape=self.shape`` and ``raw_dtype=qtype`` to ``add_tensor`` so
        GGUF records the original matrix dimensions and requested GGML type.
        """
        try:
            byte_shape = gguf.quant_shape_to_byte_shape(self.shape, qtype)
        except ValueError as exc:
            raise DirectQuantError(
                f"cannot directly quantize FP8 tensor shape {self.shape} as {qtype.name}: {exc}") from exc

        # Materialize neither the source weight nor its F32 form until the
        # writer asks for a chunk.  Each closure creates an independent reader
        # so GGUF can write it exactly once in ordinary tensor order.
        chunks: list[Callable[[], np.ndarray]] = []
        rows, _ = self.shape
        block_rows, _ = self.block_shape
        rows_per_chunk = max(block_rows, rows_per_chunk - rows_per_chunk % block_rows)
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)

            def load_chunk(start: int = row_start, end: int = row_end) -> np.ndarray:
                return quantizer.quantize_rows(self._load_float_rows(start, end), qtype)

            chunks.append(load_chunk)

        return gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8)


class FP8ExpertTensor:
    """Stream a GGUF expert tensor from individual FP8 safetensors experts.

    Standard conversion assembles an entire ``[n_expert, rows, cols]`` tensor
    with ``torch.stack``.  This class preserves that exact outer ordering but
    gives the writer one native-quantized row chunk at a time.
    """

    def __init__(self, experts: list[FP8ScaledTensor]) -> None:
        if not experts:
            raise DirectQuantError("direct FP8 expert tensor has no experts")
        shape = experts[0].shape
        if any(expert.shape != shape for expert in experts[1:]):
            shapes = sorted({expert.shape for expert in experts})
            raise DirectQuantError(f"direct FP8 experts have inconsistent shapes: {shapes}")
        self.experts = experts
        self.expert_shape = shape

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.experts), *self.expert_shape)

    def lazy_float32(self) -> gguf.LazyChunkedTensor:
        chunks: list[Callable[[], np.ndarray]] = []
        for expert in self.experts:
            chunks.extend(expert.lazy_float32()._chunks)
        return gguf.LazyChunkedTensor(chunks, self.shape, np.float32)

    def lazy_quantized(
        self,
        quantizer: GGMLChunkQuantizer,
        qtype: gguf.GGMLQuantizationType,
        *,
        rows_per_chunk: int = 128,
    ) -> gguf.LazyChunkedTensor:
        try:
            byte_shape = gguf.quant_shape_to_byte_shape(self.shape, qtype)
        except ValueError as exc:
            raise DirectQuantError(
                f"cannot directly quantize FP8 expert tensor shape {self.shape} as {qtype.name}: {exc}") from exc

        block_rows = self.experts[0].block_shape[0]
        if rows_per_chunk <= 0:
            raise DirectQuantError(f"direct FP8 rows_per_chunk must be positive, got {rows_per_chunk}")
        rows_per_chunk = max(block_rows, rows_per_chunk - rows_per_chunk % block_rows)
        rows, _ = self.expert_shape
        chunks: list[Callable[[], np.ndarray]] = []

        # The order is exactly torch.stack([expert_0, expert_1, ...], dim=0).
        for expert in self.experts:
            for row_start in range(0, rows, rows_per_chunk):
                row_end = min(rows, row_start + rows_per_chunk)

                def load_chunk(
                    source: FP8ScaledTensor = expert,
                    start: int = row_start,
                    end: int = row_end,
                ) -> np.ndarray:
                    return quantizer.quantize_rows(source._load_float_rows(start, end), qtype)

                chunks.append(load_chunk)

        return gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8)


class DirectStorageExpertTensor:
    """Stack local F16/BF16/F32 experts without materializing the 3D tensor."""

    def __init__(self, experts: list[DirectStorageTensor]) -> None:
        if not experts:
            raise DirectQuantError("direct storage expert tensor has no experts")
        shape = experts[0].shape
        ggml_type = experts[0].ggml_type
        if any(expert.shape != shape or expert.ggml_type != ggml_type for expert in experts[1:]):
            raise DirectQuantError("direct storage experts have inconsistent shapes or source types")
        self.experts = experts
        self.expert_shape = shape
        self.ggml_type = ggml_type

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.experts), *self.expert_shape)

    def lazy_storage(self) -> gguf.LazyChunkedTensor:
        chunks: list[Callable[[], np.ndarray]] = []
        for expert in self.experts:
            chunks.extend(expert.lazy_storage()._chunks)
        return gguf.LazyChunkedTensor(chunks, self.shape, self.experts[0].dtype)

    def lazy_float32(self) -> gguf.LazyChunkedTensor:
        chunks: list[Callable[[], np.ndarray]] = []
        for expert in self.experts:
            chunks.extend(expert.lazy_float32()._chunks)
        return gguf.LazyChunkedTensor(chunks, self.shape, np.float32)

    def lazy_quantized(
        self,
        quantizer: GGMLChunkQuantizer,
        qtype: gguf.GGMLQuantizationType,
        *,
        rows_per_chunk: int = 128,
    ) -> gguf.LazyChunkedTensor:
        if len(self.expert_shape) != 2:
            raise DirectQuantError(
                f"direct native expert quantization requires matrices, got shape {self.expert_shape}")
        try:
            byte_shape = gguf.quant_shape_to_byte_shape(self.shape, qtype)
        except ValueError as exc:
            raise DirectQuantError(
                f"cannot directly quantize expert tensor shape {self.shape} as {qtype.name}: {exc}") from exc
        rows, _ = self.expert_shape
        chunks: list[Callable[[], np.ndarray]] = []
        for expert in self.experts:
            for row_start in range(0, rows, rows_per_chunk):
                row_end = min(rows, row_start + rows_per_chunk)

                def load_chunk(
                    source: DirectStorageTensor = expert,
                    start: int = row_start,
                    end: int = row_end,
                ) -> np.ndarray:
                    return quantizer.quantize_rows(source._load_float_rows(start, end), qtype)

                chunks.append(load_chunk)
        return gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8)
