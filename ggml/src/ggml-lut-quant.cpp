#include "ggml-lut.h"

#include <cmath>
#include <cstring>
#include <unordered_map>

// Weight quantization metadata

// Internal metadata for quantized weights
struct ggml_lut_weight_data {
    uint8_t * w_q;      // Quantized weights (flattened, contiguous)
    float   * scales;   // Per-group scales
    int       num_groups;
};

// Global side table for quantized weights
// Key: pointer to original ggml_tensor (weights)
// Value: quantized data + scales
static std::unordered_map<const struct ggml_tensor *, ggml_lut_weight_data> g_weight_data;

// Quantization: ggml_lut_quantize_weights

void ggml_lut_quantize_weights(
    struct ggml_tensor * weights,
    const struct ggml_lut_config * config
) {
    GGML_ASSERT(weights != NULL);
    GGML_ASSERT(config  != NULL);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weights));

    const int K          = (int) weights->ne[0];
    const int N          = (int) weights->ne[1];
    const int total_size = K * N;

    const int group_size = config->group_size;
    GGML_ASSERT(group_size > 0);

    const int num_groups = (total_size + group_size - 1) / group_size;
    const int w_bits     = config->w_bits;
    GGML_ASSERT(w_bits > 0 && w_bits <= 8);

    const int   w_levels  = 1 << w_bits;
    const float w_max_val = (float) (w_levels - 1);

    const float * w_data = (const float *) weights->data;

    // Allocate quantized weight data
    ggml_lut_weight_data qdata;
    qdata.w_q       = new uint8_t[total_size];
    qdata.scales    = new float[num_groups];
    qdata.num_groups = num_groups;

    // Symmetric per-group quantization (Phase 1b simplified scheme)
    //
    // This is a SIMPLIFIED quantization for experimentation:
    // - Uses unsigned range [0, 2^w_bits - 1]
    // - No zero-point
    //
    // Production-quality quantization would typically use signed ranges
    // and more advanced calibration.
    for (int g = 0; g < num_groups; ++g) {
        const int start = g * group_size;
        const int end   = (start + group_size < total_size) ? (start + group_size) : total_size;

        // Find absmax in this group
        float absmax = 0.0f;
        for (int i = start; i < end; ++i) {
            float v  = w_data[i];
            float av = v >= 0.0f ? v : -v;
            if (av > absmax) {
                absmax = av;
            }
        }

        // Compute scale
        float scale = absmax > 0.0f ? absmax / w_max_val : 1.0f;
        qdata.scales[g] = scale;

        // Quantize to [0, 2^w_bits - 1]
        for (int i = start; i < end; ++i) {
            float v = w_data[i] / scale;
            int   q = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);

            if (q < 0) {
                q = 0;
            } else if (q > w_levels - 1) {
                q = w_levels - 1;
            }

            qdata.w_q[i] = (uint8_t) q;
        }
    }

    // Store in global side table (overwrite if already present)
    g_weight_data[weights] = qdata;
}

// LUT building: ggml_lut_build_table

void ggml_lut_build_table(
    int32_t * lut2d,
    int w_bits,
    int a_bits,
    float scale_w,
    float scale_x
) {
    GGML_ASSERT(lut2d != NULL);
    GGML_ASSERT(w_bits > 0 && w_bits <= 8);
    GGML_ASSERT(a_bits > 0 && a_bits <= 16);

    const int w_levels = 1 << w_bits;
    const int a_levels = 1 << a_bits;

    // TRUE 2D LUT: lut2d[w_idx * a_levels + a_idx]
    // Entry approximates: (w_idx * scale_w) * (a_idx * scale_x)
    for (int w = 0; w < w_levels; ++w) {
        for (int a = 0; a < a_levels; ++a) {
            float w_val = (float) w * scale_w;
            float a_val = (float) a * scale_x;
            float prod  = w_val * a_val;

            // Deterministic rounding to nearest integer
            int32_t rounded = (int32_t) (prod >= 0.0f ? prod + 0.5f : prod - 0.5f);
            lut2d[w * a_levels + a] = rounded;
        }
    }
}

// Accessors for GEMM and global cleanup

const ggml_lut_weight_data * ggml_lut_get_weight_data(const struct ggml_tensor * tensor) {
    auto it = g_weight_data.find(tensor);
    if (it == g_weight_data.end()) {
        return NULL;
    }
    return &it->second;
}

void ggml_lut_free_quantized_weights(void) {
    for (auto & entry : g_weight_data) {
        delete[] entry.second.w_q;
        delete[] entry.second.scales;
        entry.second.w_q    = nullptr;
        entry.second.scales = nullptr;
        entry.second.num_groups = 0;
    }
    g_weight_data.clear();
}
