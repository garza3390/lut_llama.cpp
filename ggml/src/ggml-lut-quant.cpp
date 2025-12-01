#include "ggml-lut.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>

// Helper structure to store quantized weight data
struct ggml_lut_weight_data {
    uint8_t * w_q;      // Quantized weights
    float * scales;     // Per-group scales
    int num_groups;
};

// Side table for quantized weight storage
static std::unordered_map<const struct ggml_tensor *, ggml_lut_weight_data> * g_lut_weight_map = nullptr;

static void ensure_weight_map() {
    if (g_lut_weight_map == nullptr) {
        g_lut_weight_map = new std::unordered_map<const struct ggml_tensor *, ggml_lut_weight_data>();
    }
}

void ggml_lut_quantize_weights(
    struct ggml_tensor * weights,
    const struct ggml_lut_config * config
) {
    ensure_weight_map();

    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weights));

    const int64_t ne0 = weights->ne[0];
    const int64_t ne1 = weights->ne[1];
    const int64_t total_elements = ne0 * ne1;

    const float * w_data = (const float *) weights->data;
    const int group_size = config->group_size;
    const int w_bits = config->w_bits;
    const int w_levels = 1 << w_bits;
    const float w_max_val = (float)(w_levels - 1);

    // Calculate number of groups
    const int num_groups = (int)((total_elements + group_size - 1) / group_size);

    // Allocate storage
    ggml_lut_weight_data qdata;
    qdata.w_q = new uint8_t[total_elements];
    qdata.scales = new float[num_groups];
    qdata.num_groups = num_groups;

    // Quantize per-group
    for (int g = 0; g < num_groups; ++g) {
        const int start_idx = g * group_size;
        const int end_idx = std::min(start_idx + group_size, (int)total_elements);

        // Find absmax in group for symmetric quantization
        float absmax = 0.0f;
        for (int i = start_idx; i < end_idx; ++i) {
            absmax = std::max(absmax, std::fabs(w_data[i]));
        }

        // Compute scale
        const float scale = (absmax > 0.0f) ? (absmax / w_max_val) : 1.0f;
        qdata.scales[g] = scale;

        // Quantize elements in this group
        for (int i = start_idx; i < end_idx; ++i) {
            float val = w_data[i] / scale;
            int q_val = (int)std::roundf(val);
            // Clamp to valid range
            q_val = std::max(0, std::min(q_val, w_levels - 1));
            qdata.w_q[i] = (uint8_t)q_val;
        }
    }

    // Store in side table
    (*g_lut_weight_map)[weights] = qdata;
}

void ggml_lut_build_table(
    int32_t * lut2d,
    int w_bits,
    int a_bits,
    float scale_w,
    float scale_x
) {
    const int w_levels = 1 << w_bits;
    const int a_levels = 1 << a_bits;

    for (int w = 0; w < w_levels; ++w) {
        for (int a = 0; a < a_levels; ++a) {
            const float w_val = (float)w * scale_w;
            const float a_val = (float)a * scale_x;
            const float prod = w_val * a_val;
            lut2d[w * a_levels + a] = (int32_t)std::roundf(prod);
        }
    }
}

// Cleanup function (should be called from ggml_lut_global_free if needed)
void ggml_lut_free_quantized_weights() {
    if (g_lut_weight_map != nullptr) {
        for (auto & pair : *g_lut_weight_map) {
            delete[] pair.second.w_q;
            delete[] pair.second.scales;
        }
        delete g_lut_weight_map;
        g_lut_weight_map = nullptr;
    }
}

// Internal helper to retrieve quantized weight data
const ggml_lut_weight_data * ggml_lut_get_weight_data(const struct ggml_tensor * weights) {
    if (g_lut_weight_map == nullptr) {
        return nullptr;
    }
    auto it = g_lut_weight_map->find(weights);
    if (it == g_lut_weight_map->end()) {
        return nullptr;
    }
    return &it->second;
}