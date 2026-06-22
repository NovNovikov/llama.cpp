from __future__ import annotations

import re
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .llama import LlamaModel
from .base import ModelBase, TextModel, gguf


@ModelBase.register("LagunaForCausalLM")
class LagunaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LAGUNA

    def set_gguf_parameters(self):
        hparams = self.hparams

        # Handle rope_parameters: can be nested (full_attention / sliding_attention) or flat
        rope_params = hparams.get("rope_parameters", {})
        if "full_attention" in rope_params or "sliding_attention" in rope_params:
            full_rope = rope_params.get("full_attention", {})
            swa_rope = rope_params.get("sliding_attention", {})
            self.rope_parameters["full_attention"] = full_rope
            self.rope_parameters["sliding_attention"] = swa_rope
        else:
            self.rope_parameters["full_attention"] = rope_params

        super().set_gguf_parameters()

        n_layers = hparams["num_hidden_layers"]
        n_head_base = hparams["num_attention_heads"]
        n_kv_base = hparams.get("num_key_value_heads", n_head_base)

        layer_types = hparams.get("layer_types", ["full_attention"] * n_layers)

        # Per-layer head counts: use num_attention_heads_per_layer if available
        heads_per_layer = hparams.get("num_attention_heads_per_layer", None)
        kv_per_layer = hparams.get("num_key_value_heads_per_layer", None)

        head_arr = []
        kv_arr = []
        swa_pat = []
        for i, lt in enumerate(layer_types[:n_layers]):
            if heads_per_layer is not None:
                head_arr.append(heads_per_layer[i])
            else:
                head_arr.append(n_head_base)
            if kv_per_layer is not None:
                kv_arr.append(kv_per_layer[i])
            else:
                kv_arr.append(n_kv_base)
            swa_pat.append(lt == "sliding_attention")

        self.gguf_writer.add_head_count(head_arr)
        self.gguf_writer.add_head_count_kv(kv_arr)

        # SWA window
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(swa_pat)

        # Per-layer value length (head_dim)
        head_dim = hparams.get("head_dim", hparams["hidden_size"] // n_head_base)
        self.gguf_writer.add_value_length(head_dim)

        # Partial rotary is data-driven: write rope.dimension_count for the global
        # (full-attention) layers and rope.dimension_count_swa for sliding layers.
        # Laguna-M uses full rotary (1.0) on every layer; Laguna-XS uses half rotary
        # (0.5) on global layers and full rotary on SWA layers.
        full_rope    = self.rope_parameters.get("full_attention", {})
        partial_full = full_rope.get("partial_rotary_factor", 1.0)
        self.gguf_writer.add_rope_dimension_count(int(head_dim * partial_full))
        swa_rope = self.rope_parameters.get("sliding_attention")
        if swa_rope is not None:
            partial_swa = swa_rope.get("partial_rotary_factor", 1.0)
            self.gguf_writer.add_rope_dimension_count_swa(int(head_dim * partial_swa))

        # Attention output gate mode: per-head (broadcasts across head_dim, Laguna-XS)
        # vs per-element (one gate per (head, head_dim) channel, Laguna-M). Default is
        # per-element to match configuration_laguna.py (gating defaults to True/per-element).
        self.gguf_writer.add_attention_gate_per_head(hparams.get("gating", "per-element") == "per-head")

        # MoE params
        n_experts = hparams["num_experts"]
        n_experts_used = hparams["num_experts_per_tok"]
        self.gguf_writer.add_expert_count(n_experts)
        self.gguf_writer.add_expert_used_count(n_experts_used)
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])

        if (shared_dim := hparams.get("shared_expert_intermediate_size")) is not None and shared_dim > 0:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_dim)
            self.gguf_writer.add_expert_shared_count(1)

        if (routing_scale := hparams.get("moe_routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routing_scale)

        # Laguna always normalizes MoE routing weights before applying routed_scaling_factor
        self.gguf_writer.add_expert_weights_norm(True)

        # Dense lead layers (from mlp_layer_types: first 'dense' layers)
        mlp_types = hparams.get("mlp_layer_types", [])
        leading_dense = 0
        for mt in mlp_types:
            if mt == "dense":
                leading_dense += 1
            else:
                break
        self.gguf_writer.add_leading_dense_block_count(leading_dense)

        # Hard-fail if moe_apply_router_weight_on_input is True
        if hparams.get("moe_apply_router_weight_on_input", False):
            raise ValueError("moe_apply_router_weight_on_input=True is not supported by llama.cpp for Laguna")

    def set_vocab(self) -> None:
        super().set_vocab()
        # tokenizer_config.json stores "{% include 'chat_template.jinja' %}" (HF indirection).
        # SpecialVocab writes that verbatim; overwrite with the resolved file content.
        template_file = self.dir_model / "chat_template.jinja"
        if template_file.is_file():
            self.gguf_writer.add_chat_template(template_file.read_text(encoding="utf-8"))
        # config.json has eos_token_id: [2, 24]; SpecialVocab only registers the first as EOS.
        # Token 24 (</assistant>) is the end-of-turn token — register it as EOT so llama.cpp
        # adds it to special_eog_ids and generation stops at turn boundaries.
        eos_ids = self.hparams.get("eos_token_id", [])
        if isinstance(eos_ids, list):
            for tok_id in eos_ids[1:]:
                self.gguf_writer.add_eot_token_id(tok_id)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Squeeze attention gate tensors
        if name.endswith(".self_attn.g_proj.weight"):
            data_torch = data_torch.squeeze().contiguous()

        # Buffer individual expert weight tensors and stack when all collected for a layer
        if bid is not None and re.match(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", name):
            n_experts = self.find_hparam(["moe_num_experts", "num_experts"])

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # Stack per-expert tensors into merged 3D tensors
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
