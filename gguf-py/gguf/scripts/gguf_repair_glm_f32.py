#!/usr/bin/env python3
"""Copy a GLM-5.3 GGUF while restoring its mandatory F32 tensor types.

This repairs direct-conversion output made before the direct path mirrored
ModelBase's F32 policy.  It copies all other tensor payloads byte-for-byte and
widens only the affected BF16 tensors in bounded chunks.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Allow execution directly from the source tree.
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf


def requires_f32(name: str, n_dims: int) -> bool:
    """Match ModelBase's required F32 tensors used by GLM-5.3."""
    if n_dims <= 1 or name.endswith("_norm.weight"):
        return True
    return (
        name.endswith(".ffn_gate_inp.weight")
        or name.endswith(".ffn_gate_inp_shexp.weight")
        or name.endswith(".ssm_conv1d.weight")
        or name.endswith(".ssm_conv1d_q.weight")
        or name.endswith(".ssm_conv1d_k.weight")
        or name.endswith(".ssm_conv1d_v.weight")
        or name.endswith(".indexer.proj.weight")
    )


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        # GGUF.version/tensor_count/kv_count are virtual reader fields, while
        # architecture is injected by GGUFWriter from its constructor.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=subtype)


def bf16_as_f32(
    tensor: gguf.ReaderTensor,
    source: Path,
    *,
    elements_per_chunk: int = 1 << 20,
) -> gguf.LazyChunkedTensor:
    if tensor.tensor_type != gguf.GGMLQuantizationType.BF16:
        raise ValueError(f"expected BF16 tensor, got {tensor.tensor_type.name} for {tensor.name}")
    if elements_per_chunk <= 0:
        raise ValueError("elements_per_chunk must be positive")

    chunks = []
    for start in range(0, tensor.n_elements, elements_per_chunk):
        count = min(elements_per_chunk, tensor.n_elements - start)

        def load_chunk(offset: int = start, length: int = count) -> np.ndarray:
            raw = np.memmap(
                source,
                mode="r",
                offset=tensor.data_offset + offset * 2,
                dtype=np.dtype("<u2"),
                shape=(length,),
            )
            return (raw.astype(np.uint32) << 16).view(np.float32)

        chunks.append(load_chunk)

    # Writer input shape is opposite the on-disk GGUF dimension order.
    return gguf.LazyChunkedTensor(chunks, tuple(reversed(tuple(int(d) for d in tensor.shape))), np.float32)


def raw_tensor_copy(
    tensor: gguf.ReaderTensor,
    source: Path,
    *,
    bytes_per_chunk: int = 256 << 20,
) -> gguf.LazyChunkedTensor:
    """Copy an unchanged tensor without retaining its source mmap pages."""
    if bytes_per_chunk <= 0:
        raise ValueError("bytes_per_chunk must be positive")

    source_dtype = tensor.data.dtype
    chunks = []
    for start in range(0, tensor.n_bytes, bytes_per_chunk):
        count = min(bytes_per_chunk, tensor.n_bytes - start)

        def load_chunk(offset: int = start, length: int = count) -> np.ndarray:
            with source.open("rb") as f:
                f.seek(tensor.data_offset + offset)
                data = f.read(length)
            if len(data) != length:
                raise ValueError(f"short read while copying {tensor.name}")
            return np.frombuffer(data, dtype=source_dtype).copy()

        chunks.append(load_chunk)

    return gguf.LazyChunkedTensor(chunks, tuple(int(d) for d in tensor.data.shape), source_dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="existing direct-converted GLM GGUF")
    parser.add_argument("output", type=Path, help="new corrected GGUF path")
    parser.add_argument("--dry-run", action="store_true", help="report the planned change without writing")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if not source.is_file():
        parser.error(f"input does not exist: {source}")
    if source == output:
        parser.error("output must differ from input")
    if output.exists():
        parser.error(f"output already exists: {output}")
    if partial.exists():
        parser.error(f"partial output already exists: {partial}")
    if not output.parent.is_dir():
        parser.error(f"output directory does not exist: {output.parent}")

    reader = gguf.GGUFReader(source, "r")
    promote = [
        tensor for tensor in reader.tensors
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16 and requires_f32(tensor.name, len(tensor.shape))
    ]
    delta = sum(tensor.n_bytes for tensor in promote)
    expected_size = source.stat().st_size + delta
    print(f"input: {source}")
    print(f"output: {output}")
    print(f"promote BF16 -> F32: {len(promote)} tensors, +{delta / 1024**2:.2f} MiB")
    print(f"estimated output size: {expected_size / 1024**3:.3f} GiB")
    if args.dry_run:
        return

    free = shutil.disk_usage(output.parent).free
    if free < expected_size:
        raise RuntimeError(f"not enough free space for {expected_size} bytes in {output.parent}")

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError("input has no general.architecture")
    writer = gguf.GGUFWriter(partial, arch=arch_field.contents(), endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    copy_metadata(reader, writer)

    promoted_names = {tensor.name for tensor in promote}
    for tensor in reader.tensors:
        if tensor.name in promoted_names:
            writer.add_tensor(
                tensor.name,
                bf16_as_f32(tensor, source),
                raw_shape=tuple(reversed(tuple(int(d) for d in tensor.shape))),
                raw_dtype=gguf.GGMLQuantizationType.F32,
                tensor_endianess=reader.endianess,
            )
        else:
            writer.add_tensor(
                tensor.name,
                raw_tensor_copy(tensor, source),
                raw_dtype=tensor.tensor_type,
                tensor_endianess=reader.endianess,
            )

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        check = gguf.GGUFReader(partial, "r")
        check_tensors = {
            tensor.name: (tensor.tensor_type, tuple(int(d) for d in tensor.shape))
            for tensor in check.tensors
        }
        check_data = check.data
        if len(check_tensors) != len(reader.tensors):
            raise ValueError("output tensor count differs from input")
        for tensor in reader.tensors:
            repaired = check_tensors.get(tensor.name)
            if repaired is None or repaired[1] != tuple(int(d) for d in tensor.shape):
                raise ValueError(f"output tensor layout differs for {tensor.name}")
            expected_type = gguf.GGMLQuantizationType.F32 if tensor.name in promoted_names else tensor.tensor_type
            if repaired[0] != expected_type:
                raise ValueError(f"output tensor type differs for {tensor.name}")

        del check_tensors
        del check
        gc.collect()
        check_data._mmap.close()
        partial.replace(output)
    except BaseException:
        writer.close()
        raise


if __name__ == "__main__":
    main()
