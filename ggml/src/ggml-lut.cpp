#include "ggml-lut.h"
#include "ggml-backend.h"
#include <unordered_map>
#include <cstring>

// Forward declaration from ggml-lut-quant.cpp
extern void ggml_lut_free_quantized_weights();

// Side table for tensor -> config mapping
static std::unordered_map<const struct ggml_tensor *, struct ggml_lut_config> * g_lut_config_map = nullptr;

void ggml_lut_global_init(void) {
    if (g_lut_config_map == nullptr) {
        g_lut_config_map = new std::unordered_map<const struct ggml_tensor *, struct ggml_lut_config>();
    }
}

void ggml_lut_global_free(void) {
    if (g_lut_config_map != nullptr) {
        delete g_lut_config_map;
        g_lut_config_map = nullptr;
    }
    // Free quantized weight data
    ggml_lut_free_quantized_weights();
}

void ggml_lut_set_tensor_config(
    struct ggml_tensor * tensor,
    const struct ggml_lut_config * config
) {
    if (g_lut_config_map == nullptr) {
        ggml_lut_global_init();
    }
    (*g_lut_config_map)[tensor] = *config;
}

bool ggml_lut_is_enabled(const struct ggml_tensor * tensor) {
    if (g_lut_config_map == nullptr) {
        return false;
    }
    auto it = g_lut_config_map->find(tensor);
    if (it == g_lut_config_map->end()) {
        return false;
    }
    return it->second.enabled;
}