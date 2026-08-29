"""Native direct-quantization recipe planning for HF -> GGUF conversion.

The recipe syntax is the existing llama-quantize ``--tensor-type-file`` syntax.
Python transports the patterns and mapped GGUF tensor descriptors to the native
planner; it deliberately never reimplements its regex matching or type policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import gguf


class DirectRecipeError(RuntimeError):
    """The opt-in direct quantization recipe is malformed or cannot be planned."""


@dataclass(frozen=True)
class DirectModelDescriptor:
    architecture: str
    n_embd: int
    n_ff: int
    n_layer: int
    n_head: int
    n_head_kv: int
    n_expert: int
    n_embd_head_k: int = 0
    n_embd_head_v: int = 0


@dataclass(frozen=True)
class DirectTensorDescriptor:
    """A mapped GGUF tensor whose target type is resolved by the native planner.

    ``shape`` follows NumPy / HF order.  The native GGML descriptor is generated
    in reversed order, exactly as the GGUF writer stores a tensor's dimensions.
    ``source_type`` is the baseline output type: FP8 source weights use Q8_0,
    while F32/BF16/F16 weights retain their original storage type.
    """

    name: str
    source_dtype: str
    source_type: gguf.GGMLQuantizationType
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DirectTensorPlan:
    name: str
    source_dtype: str
    source_type: gguf.GGMLQuantizationType
    target_type: gguf.GGMLQuantizationType
    shape: tuple[int, ...]
    parameter_count: int
    payload_bytes: int
    padding_bytes: int

    @property
    def storage_bytes(self) -> int:
        return self.payload_bytes + self.padding_bytes


class _CModelDescriptor(ctypes.Structure):
    _fields_ = [
        ("architecture", ctypes.c_char_p),
        ("n_embd", ctypes.c_uint32),
        ("n_ff", ctypes.c_uint32),
        ("n_layer", ctypes.c_uint32),
        ("n_head", ctypes.c_uint32),
        ("n_head_kv", ctypes.c_uint32),
        ("n_expert", ctypes.c_uint32),
        ("n_embd_head_k", ctypes.c_uint32),
        ("n_embd_head_v", ctypes.c_uint32),
    ]


class _CTensorOverride(ctypes.Structure):
    _fields_ = [
        ("pattern", ctypes.c_char_p),
        ("type", ctypes.c_int),
    ]


class _CDirectTensorDescriptor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("source_type", ctypes.c_int),
        ("n_dims", ctypes.c_uint32),
        ("ne", ctypes.c_int64 * 4),
    ]


@dataclass(frozen=True)
class DirectQuantRecipe:
    """Parsed ``llama-quantize --tensor-type-file`` entries.

    Pattern evaluation stays in the native C++ planner.  This parser only
    validates the ``pattern=ggml_type`` transport format.
    """

    overrides: tuple[tuple[str, gguf.GGMLQuantizationType], ...]

    @classmethod
    def from_file(cls, path: Path) -> DirectQuantRecipe:
        if not path.is_file():
            raise DirectRecipeError(f"direct quantization recipe does not exist: {path}")

        type_names = {qtype.name.lower(): qtype for qtype in gguf.GGMLQuantizationType}
        overrides: list[tuple[str, gguf.GGMLQuantizationType]] = []
        for entry in path.read_text(encoding="utf-8").split():
            pattern, separator, type_name = entry.partition("=")
            if not separator or not pattern or not type_name:
                raise DirectRecipeError(
                    f"malformed direct quantization recipe entry {entry!r}; expected pattern=ggml_type")
            qtype = type_names.get(type_name.lower())
            if qtype is None:
                raise DirectRecipeError(
                    f"unknown direct quantization type {type_name!r} in recipe entry {entry!r}")
            overrides.append((pattern, qtype))

        return cls(tuple(overrides))


class NativeDirectRecipePlanner:
    """Checked binding to the native direct recipe planner in ``llama.dll``."""

    def __init__(self, library_path: Path):
        library_path = library_path.resolve()
        if not library_path.is_file():
            raise DirectRecipeError(f"native direct recipe planner does not exist: {library_path}")

        self._dll_directories = []
        if os.name == "nt":
            self._dll_directories.append(os.add_dll_directory(str(library_path.parent)))
            cuda_paths = [os.environ.get("CUDA_PATH")]
            cuda_paths.extend(value for key, value in os.environ.items() if key.startswith("CUDA_PATH_V"))
            for cuda_path in cuda_paths:
                if cuda_path is None:
                    continue
                cuda_bin = Path(cuda_path) / "bin"
                if cuda_bin.is_dir():
                    self._dll_directories.append(os.add_dll_directory(str(cuda_bin)))

        try:
            self._library = ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise DirectRecipeError(f"failed to load native direct recipe planner {library_path}: {exc}") from exc

        try:
            self._plan = self._library.llama_quant_plan_direct_from_overrides
        except AttributeError as exc:
            raise DirectRecipeError(
                f"{library_path} does not contain the direct recipe planner; rebuild from the direct-conversion branch") from exc
        self._plan.argtypes = (
            ctypes.POINTER(_CModelDescriptor),
            ctypes.POINTER(_CTensorOverride),
            ctypes.POINTER(_CDirectTensorDescriptor),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
        )
        self._plan.restype = ctypes.c_uint32

    @staticmethod
    def _byte_size(shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType) -> int:
        try:
            byte_shape = gguf.quant_shape_to_byte_shape(shape, qtype)
        except ValueError as exc:
            raise DirectRecipeError(f"cannot calculate {qtype.name} payload for shape {shape}: {exc}") from exc
        return int(np.prod(byte_shape, dtype=np.int64))

    def plan(
        self,
        model: DirectModelDescriptor,
        recipe: DirectQuantRecipe,
        tensors: Sequence[DirectTensorDescriptor],
        *,
        data_alignment: int = 32,
    ) -> list[DirectTensorPlan]:
        if data_alignment <= 0:
            raise DirectRecipeError(f"invalid GGUF data alignment: {data_alignment}")
        if not tensors:
            return []

        model_name = model.architecture.encode("utf-8")
        c_model = _CModelDescriptor(
            model_name,
            model.n_embd,
            model.n_ff,
            model.n_layer,
            model.n_head,
            model.n_head_kv,
            model.n_expert,
            model.n_embd_head_k,
            model.n_embd_head_v,
        )

        pattern_bytes = [pattern.encode("utf-8") for pattern, _ in recipe.overrides]
        c_overrides = (_CTensorOverride * (len(recipe.overrides) + 1))()
        for index, ((_, qtype), pattern) in enumerate(zip(recipe.overrides, pattern_bytes)):
            c_overrides[index] = _CTensorOverride(pattern, int(qtype))
        c_overrides[len(recipe.overrides)] = _CTensorOverride(None, 0)

        tensor_name_bytes = [tensor.name.encode("utf-8") for tensor in tensors]
        c_tensors = (_CDirectTensorDescriptor * len(tensors))()
        for index, (tensor, name) in enumerate(zip(tensors, tensor_name_bytes)):
            if not tensor.shape or len(tensor.shape) > 4 or any(dimension <= 0 for dimension in tensor.shape):
                raise DirectRecipeError(f"invalid mapped tensor shape for {tensor.name}: {tensor.shape}")
            c_shape = (ctypes.c_int64 * 4)(*reversed(tensor.shape), *([1] * (4 - len(tensor.shape))))
            c_tensors[index] = _CDirectTensorDescriptor(name, int(tensor.source_type), len(tensor.shape), c_shape)

        result_types = (ctypes.c_int * len(tensors))()
        result = self._plan(ctypes.byref(c_model), c_overrides, c_tensors, result_types, len(tensors))
        if result != 0:
            raise DirectRecipeError("native direct recipe planner rejected the mapped tensor recipe")

        plans: list[DirectTensorPlan] = []
        for tensor, result_type in zip(tensors, result_types):
            try:
                target_type = gguf.GGMLQuantizationType(result_type)
            except ValueError as exc:
                raise DirectRecipeError(
                    f"native direct recipe planner returned unknown ggml type {result_type} for {tensor.name}") from exc
            payload_bytes = self._byte_size(tensor.shape, target_type)
            padding_bytes = (-payload_bytes) % data_alignment
            plans.append(DirectTensorPlan(
                name=tensor.name,
                source_dtype=tensor.source_dtype,
                source_type=tensor.source_type,
                target_type=target_type,
                shape=tensor.shape,
                parameter_count=int(np.prod(tensor.shape, dtype=np.int64)),
                payload_bytes=payload_bytes,
                padding_bytes=padding_bytes,
            ))

        return plans


def format_direct_plan(plans: Iterable[DirectTensorPlan]) -> str:
    """Render a deterministic dry-run table without loading tensor payloads."""
    rows = list(plans)
    lines = ["mapped tensor | source | target | parameters | payload | padding"]
    for plan in rows:
        lines.append(
            f"{plan.name} | {plan.source_dtype} | {plan.target_type.name} | "
            f"{plan.parameter_count} | {plan.payload_bytes} | {plan.padding_bytes}")
    lines.append(f"total storage bytes | {sum(plan.storage_bytes for plan in rows)}")
    return "\n".join(lines)
