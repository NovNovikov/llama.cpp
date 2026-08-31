from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

import gguf

from .base import LazyTorchTensor, ModelBase, logger
from .direct_quant import (
    DirectQuantError,
    DirectStorageExpertTensor,
    DirectStorageTensor,
    FP8ExpertTensor,
    FP8ScaledTensor,
    GGMLChunkQuantizer,
)
from .direct_recipe import (
    DirectModelDescriptor,
    DirectQuantRecipe,
    DirectTensorDescriptor,
    NativeDirectRecipePlanner,
    format_direct_plan,
)
from .glm import GlmMoeDsaModel


@ModelBase.register("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
@ModelBase.example("zai-org/GLM-5.3-Flash")
class Glm5NextModel(GlmMoeDsaModel):
    """GLM-5.3-Flash.

    Trunk that alternates KDA linear attention (34 layers) with MLA + DSA sparse
    attention (11 layers), wrapped in hyper-connection streams. The pieces are
    already in tree: the KDA tensors follow kimi-linear, the hyper-connection and
    k-pool compressor tensors follow deepseek4, and the MLA/MoE/NextN half is
    inherited from GLM-5.2 (GlmMoeDsaModel).
    """

    model_arch = gguf.MODEL_ARCH.GLM5NEXT
    supports_direct_quant = True

    # Tensors that carry no per-layer index and are named differently from the
    # generic mapping, resolved by suffix (same approach as DeepseekV4Model).
    _direct_map = {
        "hc_attn_fn":   (gguf.MODEL_TENSOR.HC_ATTN_FN,    ""),
        "hc_attn_base": (gguf.MODEL_TENSOR.HC_ATTN_BASE,  ""),
        "hc_attn_scale": (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ""),
        "hc_ffn_fn":    (gguf.MODEL_TENSOR.HC_FFN_FN,     ""),
        "hc_ffn_base":  (gguf.MODEL_TENSOR.HC_FFN_BASE,   ""),
        "hc_ffn_scale": (gguf.MODEL_TENSOR.HC_FFN_SCALE,  ""),
        "self_attn.indexer.index_kpool_compress_ape":
            (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_APE,   ""),
        "self_attn.indexer.index_kpool_compress_gate":
            (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_WGATE, ""),
    }

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # TextModel lifts text_config to the root, but only after this runs -
        # and the parent already needs num_hidden_layers from it here.
        # Skip None values: AutoConfig.to_dict() materialises keys that the JSON
        # omits, so text_config carries architectures=None and would clobber the
        # valid top-level value.
        if "text_config" in self.hparams:
            self.hparams = {
                **self.hparams,
                **{k: v for k, v in self.hparams["text_config"].items() if v is not None},
            }
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item):
        name = item[0]
        # text-only for now: drop the vision tower
        if name.startswith("model.visual.") or name.startswith("visual."):
            return None
        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # hyper-connections (mHC): identical formulation to DeepSeek-V4, so the
        # existing sinkhorn graph applies unchanged.
        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hparams["hc_eps"])

        # KDA linear attention
        linear = hparams["linear_attn_config"]
        self.gguf_writer.add_ssm_conv_kernel(linear["short_conv_kernel_size"])
        self.gguf_writer.add_ssm_inner_size(linear["num_heads"] * linear["head_dim"])
        self.gguf_writer.add_ssm_state_size(linear["head_dim"])
        self.gguf_writer.add_ssm_group_count(linear["num_heads"])

        # k-pool compression inside the DSA indexer
        self.gguf_writer.add_indexer_block_size(hparams["index_kpool"])

        # clamped SwiGLU
        if (limit := hparams.get("swiglu_limit")) is not None:
            self.gguf_writer.add_swiglu_clamp_exp([limit] * self.block_count)
            self.gguf_writer.add_swiglu_clamp_shexp([limit] * self.block_count)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the checkpoint wraps the trunk for the multimodal head
        name = re.sub(r"^model\.language_model\.", "model.", name)

        # KDA decay conventions, same as conversion/kimi_linear.py: the graph
        # expects ssm_a to already hold -exp(A_log), and the time-step bias to
        # be named like a bias so it is not loaded as a MUL_MAT weight.
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch.float())
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        for suffix, (tensor, ext) in self._direct_map.items():
            if name.endswith(suffix) and bid is not None:
                return [(self.format_tensor_name(tensor, bid) + ext, data_torch)]

        return super().modify_tensors(data_torch, name, bid)

    @staticmethod
    def _direct_normalize_name(name: str) -> str:
        return re.sub(r"^model\.language_model\.", "model.", name)

    def _direct_output_name(self, name: str) -> str:
        """Map an unmodified GLM-5.3 source tensor to its final GGUF name."""
        name = self._direct_normalize_name(name)
        if name.endswith(".dt_bias"):
            name = name.removesuffix(".dt_bias") + ".dt_proj.bias"

        bid_match = re.search(r"\.layers\.(\d+)\.", name)
        bid = int(bid_match.group(1)) if bid_match else None
        for suffix, (tensor, ext) in self._direct_map.items():
            if name.endswith(suffix) and bid is not None:
                return self.format_tensor_name(tensor, bid) + ext

        if name == "model.embed_tokens.weight":
            name = "token_embd.weight"
        return self.map_tensor_name(name)

    def _direct_requires_f32(self, name: str, shape: tuple[int, ...], bid: int | None) -> bool:
        """Use the ordinary converter's mandatory storage policy."""
        return self.tensor_requires_f32(name, bid, len(shape))

    def _direct_kv_b_outputs(
        self,
        source_name: str,
        source_shape: tuple[int, ...],
    ) -> tuple[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]]:
        """Describe the canonical MLA ``kv_b_proj`` split without reading data."""
        normalized = self._direct_normalize_name(source_name)
        bid_match = re.search(r"\.layers\.(\d+)\.", normalized)
        if bid_match is None:
            raise DirectQuantError(f"direct MLA kv_b tensor has no layer index: {source_name}")
        if len(source_shape) != 2:
            raise DirectQuantError(
                f"direct MLA kv_b tensor {source_name} must be a matrix, got {source_shape}")

        try:
            n_head_kv = int(self.hparams["num_key_value_heads"])
            qk_nope_head_dim = int(self.hparams["qk_nope_head_dim"])
            v_head_dim = int(self.hparams["v_head_dim"])
        except KeyError as exc:
            raise DirectQuantError(
                f"direct MLA kv_b tensor {source_name} requires {exc.args[0]!r} in model hparams") from exc
        if n_head_kv <= 0 or qk_nope_head_dim <= 0 or v_head_dim <= 0:
            raise DirectQuantError(
                f"direct MLA kv_b tensor {source_name} has invalid hparams: "
                f"num_key_value_heads={n_head_kv}, qk_nope_head_dim={qk_nope_head_dim}, "
                f"v_head_dim={v_head_dim}")

        expected_rows = n_head_kv * (qk_nope_head_dim + v_head_dim)
        if source_shape[0] != expected_rows:
            raise DirectQuantError(
                f"direct MLA kv_b tensor {source_name} has shape {source_shape}; expected "
                f"[{n_head_kv} * ({qk_nope_head_dim} + {v_head_dim}), rank]")

        rank = source_shape[1]
        name_k = self._direct_output_name(normalized.replace("kv_b_proj", "k_b_proj"))
        name_v = self._direct_output_name(normalized.replace("kv_b_proj", "v_b_proj"))
        return (
            (name_k, (n_head_kv, rank, qk_nope_head_dim)),
            (name_v, (n_head_kv, v_head_dim, rank)),
        )

    @staticmethod
    def _direct_numpy_dtype(qtype: gguf.GGMLQuantizationType) -> np.dtype:
        if qtype == gguf.GGMLQuantizationType.F32:
            return np.dtype(np.float32)
        if qtype in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
            return np.dtype(np.uint16)
        return np.dtype(np.uint8)

    def _direct_model_descriptor(self) -> DirectModelDescriptor:
        hparams = self.hparams
        head_dim = int(hparams.get("head_dim", hparams["hidden_size"] // hparams["num_attention_heads"]))
        return DirectModelDescriptor(
            architecture=gguf.MODEL_ARCH_NAMES[self.model_arch],
            n_embd=int(hparams["hidden_size"]),
            n_ff=int(hparams.get("moe_intermediate_size", hparams.get("intermediate_size", 1))),
            n_layer=self.block_count,
            n_head=int(hparams["num_attention_heads"]),
            n_head_kv=int(hparams.get("num_key_value_heads", hparams["num_attention_heads"])),
            n_expert=int(hparams.get("n_routed_experts", 0)),
            n_embd_head_k=int(hparams.get("qk_nope_head_dim", 0)) + int(hparams.get("qk_rope_head_dim", head_dim)),
            n_embd_head_v=int(hparams.get("v_head_dim", head_dim)),
        )

    def _direct_prepare_tensors(self) -> None:
        if self.direct_quant_recipe is None or self.direct_quant_lib is None:
            raise AssertionError("direct quantization requires both recipe and native library directory")
        if self.fuse_gate_up_exps:
            raise ValueError("direct quantization does not support --fuse-gate-up-exps yet")

        library_dir = self.direct_quant_lib.resolve()
        planner_path = library_dir / "llama.dll"
        quantizer_path = library_dir / "ggml-base.dll"
        recipe = DirectQuantRecipe.from_file(self.direct_quant_recipe)
        planner = NativeDirectRecipePlanner(planner_path)
        quantizer = GGMLChunkQuantizer(quantizer_path)

        by_normalized: dict[str, tuple[str, Any]] = {}
        for source_name, local_tensor in self.direct_local_tensors.items():
            normalized = self._direct_normalize_name(source_name)
            if normalized in by_normalized:
                raise ValueError(
                    f"direct quantization found duplicate normalized tensor name {normalized!r}")
            by_normalized[normalized] = (source_name, local_tensor)

        fp8_types = FP8ScaledTensor._FP8_TORCH_DTYPES
        scale_for_weight: dict[str, str] = {}
        for normalized, (_, local_tensor) in by_normalized.items():
            if local_tensor.dtype not in fp8_types:
                continue
            candidates = (
                normalized.removesuffix(".weight") + ".scale",
                normalized + "_scale_inv",
                normalized + "_scale",
            )
            scale_name = next((candidate for candidate in candidates if candidate in by_normalized), None)
            if scale_name is None:
                raise DirectQuantError(f"direct FP8 tensor {normalized!r} has no associated scale tensor")
            scale_for_weight[normalized] = scale_name

        def fp8_source(normalized: str) -> FP8ScaledTensor:
            _, weight = by_normalized[normalized]
            scale_name = scale_for_weight[normalized]
            _, scale = by_normalized[scale_name]
            return FP8ScaledTensor(weight, scale)

        expert_pattern = re.compile(
            r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        experts: dict[tuple[int, str], dict[int, str]] = {}
        for normalized in by_normalized:
            match = expert_pattern.match(normalized)
            if match is None:
                continue
            bid, eid, projection = int(match.group(1)), int(match.group(2)), match.group(3)
            experts.setdefault((bid, projection), {})[eid] = normalized

        records: list[dict[str, Any]] = []
        consumed: set[str] = set(scale_for_weight.values())
        for (bid, projection), entries in sorted(experts.items()):
            n_experts = int(self.hparams["n_routed_experts"])
            expected = set(range(n_experts))
            if set(entries) != expected:
                raise DirectQuantError(
                    f"direct FP8 experts for layer {bid} {projection} are incomplete; "
                    f"expected {n_experts}, found {sorted(entries)}")
            ordered_names = [entries[eid] for eid in range(n_experts)]
            dtypes = {by_normalized[name][1].dtype for name in ordered_names}
            merged_name = f"model.layers.{bid}.mlp.experts.{projection}.weight"
            output_name = self._direct_output_name(merged_name)
            requires_f32 = self._direct_requires_f32(
                output_name,
                (n_experts, *by_normalized[ordered_names[0]][1].shape),
                bid,
            )
            if dtypes <= set(fp8_types):
                records.append({
                    "name": output_name,
                    "sources": tuple(ordered_names),
                    "source_dtype": "FP8",
                    "source_type": gguf.GGMLQuantizationType.F32 if requires_f32 else gguf.GGMLQuantizationType.Q8_0,
                    "shape": (n_experts, *fp8_source(ordered_names[0]).shape),
                    "kind": "experts_f32" if requires_f32 else "experts",
                    "requires_f32": requires_f32,
                    "data": FP8ExpertTensor([fp8_source(name) for name in ordered_names]),
                })
            elif len(dtypes) == 1 and next(iter(dtypes)) in DirectStorageTensor._DTYPES:
                storage_experts = [DirectStorageTensor(by_normalized[name][1]) for name in ordered_names]
                expert_tensor = DirectStorageExpertTensor(storage_experts)
                records.append({
                    "name": output_name,
                    "sources": tuple(ordered_names),
                    "source_dtype": next(iter(dtypes)),
                    "source_type": gguf.GGMLQuantizationType.F32 if requires_f32 else expert_tensor.ggml_type,
                    "shape": expert_tensor.shape,
                    "kind": "storage_experts_f32" if requires_f32 else "storage_experts",
                    "requires_f32": requires_f32,
                    "data": expert_tensor,
                })
            else:
                raise DirectQuantError(
                    f"direct expert layer {bid} {projection} has inconsistent source dtypes {sorted(dtypes)}")
            consumed.update(ordered_names)

        for normalized, (source_name, local_tensor) in by_normalized.items():
            if normalized in consumed:
                continue
            if expert_pattern.match(normalized) is not None:
                continue

            if normalized.endswith(".self_attn.kv_b_proj.weight"):
                (k_name, k_shape), (v_name, v_shape) = self._direct_kv_b_outputs(source_name, local_tensor.shape)
                bid = int(re.search(r"\.layers\.(\d+)\.", normalized).group(1))
                source_type = DirectStorageTensor._GGML_TYPES.get(local_tensor.dtype)
                if source_type is None:
                    raise DirectQuantError(
                        f"direct MLA kv_b tensor {source_name} has unsupported safetensors dtype {local_tensor.dtype}")
                for output_name, output_shape, projection in (
                    (k_name, k_shape, "k"),
                    (v_name, v_shape, "v"),
                ):
                    requires_f32 = self._direct_requires_f32(output_name, output_shape, bid)
                    records.append({
                        "name": output_name,
                        "sources": (normalized,),
                        "source_dtype": local_tensor.dtype,
                        "source_type": gguf.GGMLQuantizationType.F32 if requires_f32 else source_type,
                        "shape": output_shape,
                        "kind": "kv_b",
                        "requires_f32": requires_f32,
                        "data": (source_name, projection),
                    })
                consumed.add(normalized)
                continue

            output_name = self._direct_output_name(normalized)
            bid_match = re.search(r"\.layers\.(\d+)\.", normalized)
            bid = int(bid_match.group(1)) if bid_match is not None else None
            requires_f32 = self._direct_requires_f32(output_name, local_tensor.shape, bid)
            if local_tensor.dtype in fp8_types:
                source = fp8_source(normalized)
                records.append({
                    "name": output_name,
                    "sources": (normalized,),
                    "source_dtype": local_tensor.dtype,
                    "source_type": gguf.GGMLQuantizationType.F32 if requires_f32 else gguf.GGMLQuantizationType.Q8_0,
                    "shape": source.shape,
                    "kind": "fp8_f32" if requires_f32 else "fp8",
                    "requires_f32": requires_f32,
                    "data": source,
                })
                consumed.add(normalized)
                continue

            if normalized.endswith(".A_log"):
                records.append({
                    "name": output_name,
                    "sources": (normalized,),
                    "source_dtype": local_tensor.dtype,
                    "source_type": gguf.GGMLQuantizationType.F32,
                    "shape": local_tensor.shape,
                    "kind": "transformed",
                    "requires_f32": True,
                    "data": source_name,
                })
                consumed.add(normalized)
                continue

            storage = DirectStorageTensor(local_tensor)
            requires_f32 = self._direct_requires_f32(output_name, storage.shape, bid)
            records.append({
                "name": output_name,
                "sources": (normalized,),
                "source_dtype": local_tensor.dtype,
                "source_type": gguf.GGMLQuantizationType.F32 if requires_f32 else storage.ggml_type,
                "shape": storage.shape,
                "kind": "storage_f32" if requires_f32 else "storage",
                "requires_f32": requires_f32,
                "data": storage,
            })
            consumed.add(normalized)

        output_names = [record["name"] for record in records]
        duplicates = sorted(name for name, count in Counter(output_names).items() if count > 1)
        if duplicates:
            raise DirectQuantError(f"direct quantization maps multiple sources to {duplicates}")

        descriptors = tuple(
            DirectTensorDescriptor(
                record["name"],
                record["source_dtype"],
                record["source_type"],
                record["shape"],
            )
            for record in records
        )
        plans = planner.plan(self._direct_model_descriptor(), recipe, descriptors)
        for record, plan in zip(records, plans, strict=True):
            if plan.name != record["name"] or plan.shape != record["shape"]:
                raise DirectQuantError(
                    f"direct structural parity failed for {record['sources']}: canonical "
                    f"{record['name']} {record['shape']}, direct {plan.name} {plan.shape}")
            if record["requires_f32"] and plan.target_type != gguf.GGMLQuantizationType.F32:
                raise DirectQuantError(
                    f"direct structural parity failed for {record['sources']}: canonical "
                    f"{plan.name} requires F32, direct recipe selected {plan.target_type.name}")
            if record["kind"] not in ("fp8", "experts") and plan.target_type != record["source_type"]:
                raise DirectQuantError(
                    f"direct structural parity failed for {record['sources']}: canonical "
                    f"{plan.name} uses {record['source_type'].name}, direct recipe selected {plan.target_type.name}")

        if self.dry_run:
            logger.info("Direct quantization plan:\n%s", format_direct_plan(plans))
            for plan in plans:
                self.gguf_writer.add_tensor_info(
                    plan.name,
                    plan.shape,
                    self._direct_numpy_dtype(plan.target_type),
                    plan.payload_bytes,
                    raw_dtype=plan.target_type,
                    tensor_shape_is_raw=True,
                )
            return

        def materialize_kv_b(source_name: str, projection: str, target_type: gguf.GGMLQuantizationType) -> np.ndarray:
            data = LazyTorchTensor.to_eager(self.model_tensors[source_name]())
            n_head_kv = int(self.hparams["num_key_value_heads"])
            qk_nope_head_dim = int(self.hparams["qk_nope_head_dim"])
            v_head_dim = int(self.hparams["v_head_dim"])
            kv_b = data.view(n_head_kv, qk_nope_head_dim + v_head_dim, data.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            output = k_b.transpose(1, 2).contiguous() if projection == "k" else v_b.contiguous()
            if target_type == gguf.GGMLQuantizationType.F32:
                return output.float().numpy()
            if target_type == gguf.GGMLQuantizationType.BF16:
                return output.view(torch.uint16).numpy()
            return output.numpy()

        for record, plan in zip(records, plans, strict=True):
            if record["kind"] in ("storage", "storage_f32"):
                data = record["data"].lazy_float32() if record["kind"] == "storage_f32" else record["data"].lazy_storage()
                self.gguf_writer.add_tensor(plan.name, data, raw_dtype=plan.target_type)
            elif record["kind"] in ("storage_experts", "storage_experts_f32"):
                data = record["data"].lazy_float32() if record["kind"] == "storage_experts_f32" else record["data"].lazy_storage()
                self.gguf_writer.add_tensor(plan.name, data, raw_dtype=plan.target_type)
            elif record["kind"] == "kv_b":
                source_name, projection = record["data"]
                self.gguf_writer.add_tensor(
                    plan.name,
                    materialize_kv_b(source_name, projection, plan.target_type),
                    raw_dtype=plan.target_type,
                )
            elif record["kind"] == "transformed":
                data = LazyTorchTensor.to_eager(self.model_tensors[record["data"]]())
                self.gguf_writer.add_tensor(plan.name, -torch.exp(data.float()).numpy(), raw_dtype=plan.target_type)
            elif record["kind"] in ("fp8_f32", "experts_f32"):
                self.gguf_writer.add_tensor(plan.name, record["data"].lazy_float32(), raw_dtype=plan.target_type)
            else:
                if not plan.target_type.name.startswith(("Q", "TQ")):
                    raise DirectQuantError(
                        f"direct FP8 tensor {plan.name} requires a native quantized target, got {plan.target_type.name}")
                encoded = record["data"].lazy_quantized(quantizer, plan.target_type)
                self.gguf_writer.add_tensor(
                    plan.name,
                    encoded,
                    raw_shape=record["shape"],
                    raw_dtype=plan.target_type,
                )
            logger.info(
                "%-48s %s --> %s, shape = {%s}",
                plan.name,
                plan.source_dtype,
                plan.target_type.name,
                ", ".join(str(size) for size in reversed(plan.shape)),
            )

    def prepare_tensors(self):
        if self.direct_quant_recipe is None:
            return super().prepare_tensors()
        self._direct_prepare_tensors()
