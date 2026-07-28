#include "models.h"
#include "llama-memory-recurrent.h"

// Solar Open 2 (upstage/Solar-Open2-250B).
//
// Derived from src/models/kimi-linear.cpp: the linear-attention block is KDA
// (Kimi Delta Attention) and transfers essentially verbatim. Per the Solar
// Open 2 tech report §2.2 there are exactly three architectural deltas:
//
//   1. allow_neg_eigval=True  -> beta = 2*sigmoid(.) in (0,2), widening the
//      state-transition eigenvalues to [-1,1]. Kimi Linear uses False, which
//      is why kimi-linear.cpp stops at a bare sigmoid. Applied uniformly to
//      beta before it enters the delta rule, so the GATED_DELTA_NET kernel
//      itself needs no change.
//   2. The softmax layers are GQA (not MLA) with an ELEMENTWISE sigmoid output
//      gate on the SDPA result, applied before o_proj.
//   3. NoPE everywhere -- the config's rope_theta / partial_rotary_factor are
//      vestigial and deliberately ignored.
//
// Layer order is S-L-L-L (softmax FIRST in each block of four), the opposite of
// Kimi Linear and Qwen3.5's L-L-L-S. That ordering arrives via the per-layer
// n_head_kv array written by the converter, so nothing here infers it.

void llama_model_solar_open2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);

    // KDA layers are marked with n_head_kv == 0 (same convention as Kimi Linear
    // and Jamba); the GQA layers carry the real KV head count.
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);

    // Older Solar Open 2 GGUFs predate the expert_gating_func metadata key.
    // Their router is the auxiliary-loss-free sigmoid router from the HF model.
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // Solar Open 2 is MoE in every layer (first_k_dense_replace = 0).
    GGML_ASSERT(hparams.n_layer_dense_lead == 0 && "solar-open2 expects no leading dense blocks");

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_solar_open2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    const int64_t head_dim_kda = hparams.n_embd_head_kda;   // 128
    const int64_t ssm_d_conv   = hparams.ssm_d_conv;        // 4
    const int64_t d_inner      = head_dim_kda * n_head;     // 64 * 128 = 8192

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.is_recr(i)) {
            // ---- KDA linear-attention layer ----
            // conv1d weights are 4D in the GGUF but quantisation may drop the
            // trailing 1, so accept 3D too (same dance as kimi-linear.cpp).
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_q_conv) {
                layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_k_conv) {
                layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_v_conv) {
                layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }

            // linear_attn_config.num_kv_heads is null => K is full width, like Q/V
            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            // low-rank forget gate (kda_use_full_proj = false)
            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, head_dim_kda}, 0);
            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {head_dim_kda, d_inner}, 0);

            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            // -exp(A_log) is applied during conversion
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head, 1, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_a) {
                layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
            }

            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            // low-rank output gate
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, head_dim_kda}, 0);
            layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", i), {head_dim_kda, d_inner}, 0);

            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim_kda}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            // ---- GQA softmax layer (NoPE + elementwise sigmoid output gate) ----
            const int64_t n_head_kv_l = hparams.n_head_kv(i);

            create_tensor_qkv(layer, i, n_embd,
                    n_head      * n_embd_head_k,
                    n_head_kv_l * n_embd_head_k,
                    n_head_kv_l * n_embd_head_v, 0);

            // g_proj: {n_embd, n_head*head_dim} -- FULL width, one gate value per
            // output element. step35's attn_gate is {n_embd, n_head} instead.
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head * n_embd_head_v}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        const int64_t n_ff_exp = hparams.n_ff_exp;

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

        const int64_t n_ff_shexp = n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, TENSOR_NOT_REQUIRED);

        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_solar_open2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Causal conv1d over Q/K/V. Copied from kimi-linear.cpp -- qkv selects which of
// the three conv states to read/write (0=Q, 1=K, 2=V).
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all,
        ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
        int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs,
        int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner          = head_dim * n_head;
    const int64_t conv_state_size  = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total   = 3 * conv_state_size;   // Q + K + V

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // stash the trailing d_conv-1 columns back into the persistent conv state
    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

llama_model_solar_open2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    // NoPE: no inp_pos, no rope factors, nothing positional anywhere.

    auto * inp_kv      = build_inp_mem_hybrid();
    auto * inp_rs      = inp_kv->get_recr();
    auto * inp_attn_kv = inp_kv->get_attn();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head       = hparams.n_head();
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = n_head * head_dim;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    const int64_t n_embd_head_v = hparams.n_embd_head_v();
    const float   kq_scale      = 1.0f / sqrtf((float) n_embd_head_k);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            // ================= KDA linear-attention layer =================
            const auto * mctx_cur = inp_rs->mctx;
            const auto   kv_head  = mctx_cur->get_head();

            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            cb(g1, "kda_g1", il);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
            beta = ggml_sigmoid(ctx0, beta);

            // *** Solar Open 2 delta vs Kimi Linear ***
            // allow_neg_eigval=True: beta = 2*sigmoid(.) in (0,2), applied
            // identically to the delta rule's erase term (beta*k*k^T*S) and
            // write term (beta*k*v^T), which widens the state-transition
            // eigenvalues to [-1,1] and lets the state self-correct.
            beta = ggml_scale(ctx0, beta, 2.0f);
            cb(beta, "kda_beta_neg_eigval", il);

            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            const float eps_norm = hparams.f_norm_rms_eps;
            Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
            Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output    = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;

            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // output gate g2 = g_b(g_a(x)), then RMSNorm(x) * sigmoid(g2)
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a    = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2     = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur   = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);
        } else {
            // ================= GQA softmax layer (NoPE) =================
            const int64_t n_head_kv_l = hparams.n_head_kv(il);

            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, cur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,      n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv_l, n_tokens);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // NoPE -- deliberately no ggml_rope_ext here.

            // wo passed as null so the gate can be applied before o_proj
            ggml_tensor * attn_out = build_attn(inp_attn_kv,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(attn_out, "attn_out", il);

            // Elementwise sigmoid gate: g_proj is full width {n_embd, n_head*head_dim},
            // so this is a straight elementwise multiply -- no per-head broadcast.
            ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate_sigmoid", il);

            attn_out = ggml_mul(ctx0, attn_out, gate);
            cb(attn_out, "attn_gated", il);

            cur = ggml_mul_mat(ctx0, layer.wo, attn_out);
            cb(cur, "attn_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // every layer is MoE (first_k_dense_replace = 0)
        {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
