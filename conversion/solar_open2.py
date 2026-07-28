from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("SolarOpen2Model", "SolarOpen2ForCausalLM")
class SolarOpen2Model(TextModel):
    """Solar Open 2 (upstage): hybrid GQA + KDA linear-attention MoE.

    The KDA block is the same as Kimi Linear's, so tensor handling below mirrors
    conversion/kimi_linear.py. The differences that matter for conversion:

      * `gqa_layers` is 0-INDEXED. Kimi's `linear_attn_config.full_attn_layers`
        is 1-indexed and its converter compensates with `il + 1 in ...`; doing
        that here would misassign every layer.
      * No MLA, so none of Kimi's q/kv-lora or kv_b splitting applies. The
        softmax layers are plain GQA plus a `g_proj` output gate, which already
        maps to ATTN_GATE.
      * Experts use DeepSeek-style `mlp.experts.{i}.{gate,up,down}_proj` naming.
      * NoPE: `rope_theta` / `partial_rotary_factor` in config.json are
        vestigial (tech report §2.2) and deliberately not written.
    """

    model_arch = gguf.MODEL_ARCH.SOLAR_OPEN2

    _experts: list[dict[str, Tensor]] | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        linear_attn_config = hparams["linear_attn_config"]

        # Per-layer KV head count: 0 marks a KDA (recurrent) layer, which is how
        # llama.cpp tells the two branches apart. gqa_layers is 0-indexed.
        gqa_layers = set(hparams["gqa_layers"])
        n_kv_head = hparams["num_key_value_heads"]
        _num_kv_heads = [
            (n_kv_head if il in gqa_layers else 0)
            for il in range(hparams["num_hidden_layers"])
        ]
        assert len(_num_kv_heads) == hparams["num_hidden_layers"]
        assert any(_num_kv_heads), "no softmax layers found -- gqa_layers indexing is wrong"
        self.gguf_writer.add_head_count_kv(_num_kv_heads)
        logger.info(f"solar-open2: {sum(1 for x in _num_kv_heads if x)} softmax / "
                    f"{sum(1 for x in _num_kv_heads if not x)} KDA layers")

        # NOTE: key_length/value_length come from base via hparams["head_dim"]
        # (128, independent of hidden_size/num_heads), as do expert_count and
        # expert_used_count -- setting them again here only produces
        # "Duplicated key name" warnings.

        if (ssm_d_conv := linear_attn_config.get("short_conv_kernel_size")) is not None:
            self.gguf_writer.add_ssm_conv_kernel(ssm_d_conv)
        if (kda_head_dim := linear_attn_config.get("head_dim")) is not None:
            self.gguf_writer.add_kda_head_dim(kda_head_dim)

        # The KDA path derives d_inner as n_head * kda_head_dim, so the global
        # head count and linear_attn_config.num_heads must agree.
        if (kda_heads := linear_attn_config.get("num_heads")) is not None:
            assert kda_heads == hparams["num_attention_heads"], (
                f"KDA num_heads ({kda_heads}) != num_attention_heads "
                f"({hparams['num_attention_heads']}); llama.cpp derives KDA d_inner from n_head()"
            )

        # MoE
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        # e_score_correction_bias present => aux-loss-free sigmoid routing
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # generation_config lists eos_token_id = [<|endoftext|>, <|im:end|>], but the
        # base vocab path only captures the primary eos (<|endoftext|>). The assistant
        # turn ends with <|im:end|>, so without registering it as the end-of-turn token
        # llama.cpp never stops on it and it leaks into the output. Register it as EOT
        # (llama.cpp treats EOT as end-of-generation).
        import json, os
        gc_path = os.path.join(self.dir_model, "generation_config.json")
        if os.path.exists(gc_path):
            eos_ids = json.load(open(gc_path)).get("eos_token_id")
            if isinstance(eos_ids, list):
                primary = self.hparams.get("eos_token_id")
                for tid in eos_ids:
                    if tid != primary:
                        self.gguf_writer.add_eot_token_id(tid)
                        logger.info(f"solar-open2: registered EOT token id {tid} (<|im:end|>)")
                        break

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # KDA conv1d: HF ships [d_inner, 1, d_conv] (or [d_inner, d_conv]).
        # GGUF reverses the numpy shape on write, so (1, d_inner, 1, d_conv)
        # lands as ggml ne = [d_conv, 1, d_inner, 1] with d_conv fastest-varying,
        # which is what ggml_ssm_conv expects.
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
            elif data_torch.ndim == 3:
                d_inner, _, d_conv = data_torch.shape
            else:
                raise ValueError(f"unexpected conv1d rank {data_torch.ndim} for {name}")
            data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        # decay is stored as log; llama.cpp wants -exp(A_log) precomputed
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)

        # llama.cpp looks this up under the SSM_DT name
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # merge per-expert tensors into stacked 3D tensors
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
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
            return

        yield from super().modify_tensors(data_torch, name, bid)
