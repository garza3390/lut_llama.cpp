#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * LUT configuration for a tensor or operation.
 * This controls weight quantization, activation quantization, and LUT parameters.
 */
struct ggml_lut_config {
    int w_bits;       // Weight bits: 3, 4, 5
    int a_bits;       // Activation bits: 8, 16
    int group_size;   // Quantization group: 32, 64, 128
    int mu;           // Subvector length (for HLS/BRAM tiling)
    bool enabled;     // Enable LUT path for this op/tensor
};

/**
 * Initialize global LUT subsystem.
 * Must be called before using any LUT functionality.
 */
GGML_API void ggml_lut_global_init(void);

/**
 * Free global LUT resources.
 */
GGML_API void ggml_lut_global_free(void);

/**
 * Attach LUT configuration to a tensor.
 * The configuration is stored in a side table and can be queried later.
 */
GGML_API void ggml_lut_set_tensor_config(
    struct ggml_tensor * tensor,
    const struct ggml_lut_config * config
);

/**
 * Check if a tensor has LUT enabled.
 */
GGML_API bool ggml_lut_is_enabled(const struct ggml_tensor * tensor);

/**
 * Quantize weights for LUT-based computation.
 * Uses symmetric per-group quantization: w_q = round(w / scale_w)
 * Stores quantized values and per-group scales in the tensor's extra data.
 */
GGML_API void ggml_lut_quantize_weights(
    struct ggml_tensor * weights,
    const struct ggml_lut_config * config
);

/**
 * Build a 2D LUT table for given bit-widths and scales.
 * Table is indexed as: lut2d[w_idx * (2^a_bits) + a_idx]
 * Result is an integer product that must be scaled by scale_w * scale_x later.
 */
GGML_API void ggml_lut_build_table(
    int32_t * lut2d,
    int w_bits,
    int a_bits,
    float scale_w,
    float scale_x
);

/**
 * Compute GEMM using LUT acceleration: C = A * B
 * A: [K, M] activations
 * B: [K, N] weights (should be quantized via ggml_lut_quantize_weights)
 * C: [N, M] output
 */
GGML_API void ggml_lut_compute_gemm(
    const struct ggml_tensor * A,
    const struct ggml_tensor * B,
    struct ggml_tensor * C,
    const struct ggml_lut_config * config
);

#ifdef __cplusplus
}
#endif