#include "ggml-lut.h"

#include <unordered_map>
#include <cstring>

// Forward declarations from ggml-lut-quant.cpp / ggml-lut-gemm.cpp
extern void ggml_lut_free_quantized_weights(void);
extern "C" void ggml_lut_free_work_buffers(void);

// Side table for tensor -> config mapping
static std::unordered_map<const struct ggml_tensor *, struct ggml_lut_config> g_tensor_configs;
static bool g_lut_initialized = false;

void ggml_lut_global_init(void) {
    if (g_lut_initialized) {
        return;
    }
    g_lut_initialized = true;
}

void ggml_lut_global_free(void) {
    ggml_lut_free_quantized_weights();  // libera pesos cuantizados + cache LUT
    ggml_lut_free_work_buffers();       // libera buffers de activaciones
    g_tensor_configs.clear();
    g_lut_initialized = false;
}

void ggml_lut_set_tensor_config(
    struct ggml_tensor * tensor,
    const struct ggml_lut_config * config
) {
    GGML_ASSERT(tensor != NULL);
    GGML_ASSERT(config != NULL);

    if (!g_lut_initialized) {
        ggml_lut_global_init();
    }

    g_tensor_configs[tensor] = *config;
}

bool ggml_lut_is_enabled(const struct ggml_tensor * tensor) {
    if (!g_lut_initialized) {
        return false;
    }

    auto it = g_tensor_configs.find(tensor);
    if (it == g_tensor_configs.end()) {
        return false;
    }
    return it->second.enabled;
}
