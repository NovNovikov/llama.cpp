#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Populates ggml_moe_cache (see ggml-backend-moe-cache.h).

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_moe_cache_register(const void * owner);

// Surrender enough reconstructible cache VRAM for a mandatory CUDA allocation.
// Returns bytes physically released through the cache allocator.
size_t ggml_moe_cache_reclaim(int device, size_t allocation_bytes, const char * reason);

// Compatibility entry point for callers that explicitly want to discard all cache VRAM.
size_t ggml_moe_cache_trim(int device);

#ifdef __cplusplus
}
#endif
