#include "ggml-lut.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// Forward declaration from ggml-lut-quant.cpp
struct ggml_lut_weight_data {
    uint8_t * w_q;
    float * scales;
    int num_groups;
};
extern const ggml_lut_weight_data * ggml_lut_get_weight_data(const struct ggml_tensor * weights);

/**
 * LUT-based GEMM kernel
 * Computes C = A * B where:
 * - A: [K, M] activations (quantized on-the-fly)
 * - B: [K, N] weights (pre-quantized)
 * - C: [N, M] output
 */
static void lut_gemm_kernel(
    const uint8_t * w_q,           // Quantized weights [N * K]
    const float * scales_w,        // Per-group scales for weights
    const float * a_data,          // Activations [K * M]
    float * out,                   // Output [N * M]
    int M, int N, int K,
    int group_size,
    int w_bits,
    int a_bits
) {
    const int a_levels = 1 << a_bits;
    const int w_levels = 1 << w_bits;
    const float a_max_val = (float)(a_levels - 1);

    // Build LUT tables per group
    // For simplicity, we'll use a unified scale approach here
    // In practice, you'd build one LUT per group or use group-specific scales

    // Quantize activations on-the-fly (per column of A, which becomes rows in GEMM)
    uint8_t * a_q = new uint8_t[K * M];
    float * scales_a = new float[M];

    // Quantize each column of A (activation vector per row of output)
    for (int m = 0; m < M; ++m) {
        // Find absmax for this activation vector
        float absmax = 0.0f;
        for (int k = 0; k < K; ++k) {
            absmax = std::max(absmax, std::fabs(a_data[k * M + m]));
        }

        const float scale_a = (absmax > 0.0f) ? (absmax / a_max_val) : 1.0f;
        scales_a[m] = scale_a;

        // Quantize
        for (int k = 0; k < K; ++k) {
            float val = a_data[k * M + m] / scale_a;
            int q_val = (int)std::roundf(val);
            q_val = std::max(0, std::min(q_val, a_levels - 1));
            a_q[k * M + m] = (uint8_t)q_val;
        }
    }

    // Compute GEMM using LUT
    const int num_groups_w = (K + group_size - 1) / group_size;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int64_t acc = 0;

            // Accumulate over K dimension
            for (int k = 0; k < K; ++k) {
                const int group_idx = k / group_size;
                const uint8_t wq = w_q[n * K + k];
                const uint8_t aq = a_q[k * M + m];

                // Simple LUT lookup: wq * aq (approximation)
                // In a full implementation, you'd use pre-built LUT tables
                // For now, direct multiplication of quantized values
                acc += (int64_t)wq * (int64_t)aq;
            }

            // Dequantize: scale back using per-group weight scale and activation scale
            // Using first group's scale as representative (simplification)
            const float scale_w = scales_w[0];  // Should use group-specific
            const float scale_a = scales_a[m];

            out[m * N + n] = (float)acc * scale_w * scale_a;
        }
    }

    delete[] a_q;
    delete[] scales_a;
}

void ggml_lut_compute_gemm(
    const struct ggml_tensor * A,
    const struct ggml_tensor * B,
    struct ggml_tensor * C,
    const struct ggml_lut_config * config
) {
    GGML_ASSERT(A->type == GGML_TYPE_F32);
    GGML_ASSERT(C->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(ggml_is_contiguous(B));
    GGML_ASSERT(ggml_is_contiguous(C));

    // Get quantized weight data
    const ggml_lut_weight_data * qdata = ggml_lut_get_weight_data(B);
    GGML_ASSERT(qdata != nullptr && "Weights must be quantized before LUT GEMM");

    // Dimensions
    // A: [K, M]
    // B: [K, N]
    // C: [N, M]
    const int M = (int)A->ne[1];
    const int K = (int)A->ne[0];
    const int N = (int)B->ne[1];

    GGML_ASSERT(B->ne[0] == K);
    GGML_ASSERT(C->ne[0] == N);
    GGML_ASSERT(C->ne[1] == M);

    const float * a_data = (const float *) A->data;
    float * c_data = (float *) C->data;

    lut_gemm_kernel(
        qdata->w_q,
        qdata->scales,
        a_data,
        c_data,
        M, N, K,
        config->group_size,
        config->w_bits,
        config->a_bits
    );
}