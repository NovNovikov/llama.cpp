#include "llama-impl.h"

#include "gguf.h"
#include "llama.h"

#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

struct llama_logger_state {
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_logger_state g_logger_state;

time_meas::time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

time_meas::~time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

void llama_log_get(ggml_log_callback * log_callback, void ** user_data) {
    ggml_log_get(log_callback, user_data);
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    ggml_log_set(log_callback, user_data);
    g_logger_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

bool llama_prefill_profile_enabled() {
    static const bool enabled = getenv("LLAMA_PREFILL_PROFILE") != nullptr;
    return enabled;
}

// Aggregated DeepSeek/DSV4 prompt-rendering stats for debugging prompt
// processing regressions without spamming per-layer logs in the hot path.
struct llama_prefill_profile_graph_state {
    int lid_calls = 0;
    int mask_calls = 0;
    int csa_calls = 0;
    int64_t nt_max = 0;
    int64_t n_lid_total = 0;
    int64_t n_stream_max = 0;
    int64_t n_top_k_total = 0;
    int64_t n_top_k_max = 0;
    int64_t raw_kv_max = 0;
    int64_t csa_kv_max = 0;
    uint64_t dense_mask_bytes_total = 0;
    uint64_t dense_mask_bytes_max = 0;
    uint64_t zero_rows_bytes_total = 0;
    uint64_t zero_rows_bytes_max = 0;
    int64_t mask_n_kv_max = 0;
    int64_t mask_n_batch_max = 0;
    int64_t mask_n_stream_max = 0;
};

static thread_local llama_prefill_profile_graph_state g_prefill_profile_graph_state;

void llama_prefill_profile_append(const std::string & text) {
    if (!llama_prefill_profile_enabled()) {
        return;
    }

    static const std::string path = []() {
        const char * env = getenv("LLAMA_PREFILL_PROFILE_FILE");
        return std::string(env && env[0] ? env : "prefill_profile.txt");
    }();
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    FILE * file = ggml_fopen(path.c_str(), "ab");
    if (!file) {
        return;
    }

    fwrite(text.data(), 1, text.size(), file);
    if (text.empty() || text.back() != '\n') {
        fputc('\n', file);
    }
    fclose(file);
}

void llama_prefill_profile_graph_reset() {
    if (!llama_prefill_profile_enabled()) {
        return;
    }

    g_prefill_profile_graph_state = {};
}

void llama_prefill_profile_graph_note_lid(int64_t nt, int64_t n_lid, int64_t n_stream, uint32_t n_top_k) {
    if (!llama_prefill_profile_enabled()) {
        return;
    }

    auto & s = g_prefill_profile_graph_state;
    s.lid_calls++;
    s.nt_max = std::max(s.nt_max, nt);
    s.n_lid_total += n_lid;
    s.n_stream_max = std::max(s.n_stream_max, n_stream);
    s.n_top_k_total += n_top_k;
    s.n_top_k_max = std::max<int64_t>(s.n_top_k_max, n_top_k);
}

void llama_prefill_profile_graph_note_top_k_mask(const ggml_tensor * kq_mask, const ggml_tensor * top_k, const ggml_tensor * kq_mask_all, const ggml_tensor * zeros) {
    if (!llama_prefill_profile_enabled()) {
        return;
    }

    auto & s = g_prefill_profile_graph_state;
    s.mask_calls++;
    s.dense_mask_bytes_total += ggml_nbytes(kq_mask_all);
    s.dense_mask_bytes_max = std::max<uint64_t>(s.dense_mask_bytes_max, ggml_nbytes(kq_mask_all));
    s.zero_rows_bytes_total += ggml_nbytes(zeros);
    s.zero_rows_bytes_max = std::max<uint64_t>(s.zero_rows_bytes_max, ggml_nbytes(zeros));
    s.mask_n_kv_max = std::max<int64_t>(s.mask_n_kv_max, kq_mask->ne[0]);
    s.mask_n_batch_max = std::max<int64_t>(s.mask_n_batch_max, kq_mask->ne[1]);
    s.mask_n_stream_max = std::max<int64_t>(s.mask_n_stream_max, kq_mask->ne[3]);
    s.n_top_k_total += top_k->ne[0];
    s.n_top_k_max = std::max<int64_t>(s.n_top_k_max, top_k->ne[0]);
}

void llama_prefill_profile_graph_note_csa_lid_attention(int64_t nt, int64_t raw_kv, int64_t csa_kv, int64_t n_stream) {
    if (!llama_prefill_profile_enabled()) {
        return;
    }

    auto & s = g_prefill_profile_graph_state;
    s.csa_calls++;
    s.nt_max = std::max(s.nt_max, nt);
    s.raw_kv_max = std::max(s.raw_kv_max, raw_kv);
    s.csa_kv_max = std::max(s.csa_kv_max, csa_kv);
    s.n_stream_max = std::max(s.n_stream_max, n_stream);
}

std::string llama_prefill_profile_graph_consume(const ggml_cgraph * gf, const char * phase, int64_t n_tokens, bool reused) {
    if (!llama_prefill_profile_enabled()) {
        return {};
    }

    auto s = g_prefill_profile_graph_state;
    g_prefill_profile_graph_state = {};

    if (!gf) {
        return format("prefill_profile_graph: phase=%s, n_tokens=%lld, reused=%s, graph=null\n",
                phase, (long long) n_tokens, reused ? "true" : "false");
    }

    int n_flash_attn_ext = 0;
    int n_set_rows = 0;
    int n_lightning_indexer = 0;
    int n_fill = 0;
    int n_add = 0;

    for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf)); ++i) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(gf), i);
        switch (node->op) {
            case GGML_OP_FLASH_ATTN_EXT:   ++n_flash_attn_ext; break;
            case GGML_OP_SET_ROWS:         ++n_set_rows; break;
            case GGML_OP_LIGHTNING_INDEXER:++n_lightning_indexer; break;
            case GGML_OP_FILL:             ++n_fill; break;
            case GGML_OP_ADD:              ++n_add; break;
            default: break;
        }
    }

    return format(
            "prefill_profile_graph: phase=%s, n_tokens=%lld, reused=%s, nodes=%d, ops={lightning=%d, flash_attn=%d, set_rows=%d, fill=%d, add=%d}, dsv4={lid_calls=%d, csa_calls=%d, mask_calls=%d, nt_max=%lld, n_lid_total=%lld, raw_kv_max=%lld, csa_kv_max=%lld, n_stream_max=%lld, top_k_total=%lld, top_k_max=%lld, mask_n_kv_max=%lld, mask_n_batch_max=%lld, mask_n_stream_max=%lld, dense_mask_total_mib=%.2f, dense_mask_max_mib=%.2f, zero_rows_total_mib=%.2f, zero_rows_max_mib=%.2f}\n",
            phase,
            (long long) n_tokens,
            reused ? "true" : "false",
            ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf)),
            n_lightning_indexer,
            n_flash_attn_ext,
            n_set_rows,
            n_fill,
            n_add,
            s.lid_calls,
            s.csa_calls,
            s.mask_calls,
            (long long) s.nt_max,
            (long long) s.n_lid_total,
            (long long) s.raw_kv_max,
            (long long) s.csa_kv_max,
            (long long) s.n_stream_max,
            (long long) s.n_top_k_total,
            (long long) s.n_top_k_max,
            (long long) s.mask_n_kv_max,
            (long long) s.mask_n_batch_max,
            (long long) s.mask_n_stream_max,
            s.dense_mask_bytes_total / 1024.0 / 1024.0,
            s.dense_mask_bytes_max / 1024.0 / 1024.0,
            s.zero_rows_bytes_total / 1024.0 / 1024.0,
            s.zero_rows_bytes_max / 1024.0 / 1024.0);
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, ne.at(i));
    }
    return buf;
}

std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, t->ne[i]);
    }
    return buf;
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}
