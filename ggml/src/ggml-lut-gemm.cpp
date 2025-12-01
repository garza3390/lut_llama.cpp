#include "ggml-lut.h"

#include <cstdint>
#include <cstring>

// -----------------------------------------------------------------------------
// Internal weight metadata type (must match ggml-lut-quant.cpp)
// -----------------------------------------------------------------------------

struct ggml_lut_weight_data {
    uint8_t * w_q;      // Quantized weights (flattened, contiguous)
    float   * scales;   // Per-group scales
    int       num_groups;
};

// Forward declaration of accessor (implemented in ggml-lut-quant.cpp)
extern const ggml_lut_weight_data *
ggml_lut_get_weight_data(const struct ggml_tensor * tensor);

// -----------------------------------------------------------------------------
// Activation quantization helper (column-wise)
// -----------------------------------------------------------------------------

// A is shaped [K, M] (row-major), quantize each column independently
// No dynamic allocation: all buffers are provided by the caller.
static void ggml_lut_quantize_activations(
    const float * GGML_RESTRICT a_data,   // [K * M]
    uint8_t     * GGML_RESTRICT a_q,      // [K * M], output
    float       * GGML_RESTRICT scales_a, // [M], output
    int K,
    int M,
    int a_bits
) {
    const int   a_levels  = 1 << a_bits;
    const float a_max_val = (float) (a_levels - 1);

    for (int m = 0; m < M; ++m) {
        float absmax = 0.0f;

        // Find absmax for column m
        for (int k = 0; k < K; ++k) {
            float v  = a_data[k * M + m];
            float av = v >= 0.0f ? v : -v;
            if (av > absmax) {
                absmax = av;
            }
        }

        float scale_a = absmax > 0.0f ? absmax / a_max_val : 1.0f;
        scales_a[m] = scale_a;

        // Quantize column m to [0, 2^a_bits - 1]
        for (int k = 0; k < K; ++k) {
            float v = a_data[k * M + m] / scale_a;
            int   q = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);
            if (q < 0) {
                q = 0;
            } else if (q > a_levels - 1) {
                q = a_levels - 1;
            }
            a_q[k * M + m] = (uint8_t) q;
        }
    }
}

// -----------------------------------------------------------------------------
// LUT GEMM kernel using TRUE 2D LUT
// -----------------------------------------------------------------------------

// The kernel is FPGA/HLS-friendly: no dynamic allocation and no STL usage.
// All buffers (w_q, a_q, scales, lut2d, out) are owned by the caller.
static void lut_gemm_kernel(
    const uint8_t * GGML_RESTRICT w_q,      // [N * K]
    const float   * GGML_RESTRICT scales_w, // [num_groups_w]
    const uint8_t * GGML_RESTRICT a_q,      // [K * M]
    const float   * GGML_RESTRICT scales_a, // [M]
    const int32_t * GGML_RESTRICT lut2d,    // [2^w_bits * 2^a_bits]
    float         * GGML_RESTRICT out,      // [N * M]
    int M, int N, int K,
    int group_size,
    int w_bits,
    int a_bits
) {
    (void) group_size; // reserved for future per-group scaling logic
    (void) w_bits;     // not needed directly (already baked into lut2d)

    const int   a_levels = 1 << a_bits;

    // Phase 1b simplificación:
    // usamos scales_w[0] como escala representativa para los pesos.
    const float rep_scale_w = scales_w[0];

    for (int m = 0; m < M; ++m) {
        const float scale_a = scales_a[m];

        for (int n = 0; n < N; ++n) {
            int64_t acc = 0;

            // Acumular sobre K usando la LUT 2D
            for (int k = 0; k < K; ++k) {
                uint8_t wv = w_q[n * K + k];
                uint8_t xv = a_q[k * M + m];

                // Lookup: lut2d[w_idx * a_levels + a_idx]
                int32_t lut_val = lut2d[(int) wv * a_levels + (int) xv];
                acc += (int64_t) lut_val;
            }

            out[m * N + n] = (float) acc * rep_scale_w * scale_a;
        }
    }
}

// -----------------------------------------------------------------------------
// Main LUT GEMM orchestrator
// -----------------------------------------------------------------------------

void ggml_lut_compute_gemm(
    const struct ggml_tensor * A,
    const struct ggml_tensor * B,
    struct ggml_tensor * C,
    const struct ggml_lut_config * config
) {
    GGML_ASSERT(A != NULL && B != NULL && C != NULL);
    GGML_ASSERT(config != NULL);

    GGML_ASSERT(A->type == GGML_TYPE_F32);
    GGML_ASSERT(B->type == GGML_TYPE_F32);
    GGML_ASSERT(C->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(ggml_is_contiguous(B));
    GGML_ASSERT(ggml_is_contiguous(C));

    // A: [K, M], B: [K, N], C: [N, M]
    const int M = (int) A->ne[1];
    const int K = (int) A->ne[0];
    const int N = (int) B->ne[1];

    GGML_ASSERT((int) B->ne[0] == K);
    GGML_ASSERT((int) C->ne[0] == N);
    GGML_ASSERT((int) C->ne[1] == M);

    // Retrieve quantized weight data for B
    const ggml_lut_weight_data * qdata = ggml_lut_get_weight_data(B);
    GGML_ASSERT(qdata != NULL && "Weights must be quantized before LUT GEMM");

    // Allocate temporary buffers (CPU-side; en FPGA serían BRAM/local)
    uint8_t * a_q      = new uint8_t[K * M];
    float   * scales_a = new float[M];

    const int w_levels = 1 << config->w_bits;
    const int a_levels = 1 << config->a_bits;

    int32_t * lut2d = new int32_t[w_levels * a_levels];

    // Quantize activations
    const float * a_data = (const float *) A->data;
    ggml_lut_quantize_activations(
        a_data,
        a_q,
        scales_a,
        K,
        M,
        config->a_bits
    );

    // Escalas representativas para construir la LUT 2D
    const float scale_w = qdata->scales[0];
    const float scale_x = scales_a[0];

    ggml_lut_build_table(
        lut2d,
        config->w_bits,
        config->a_bits,
        scale_w,
        scale_x
    );

    // Ejecutar kernel LUT GEMM
    float * c_data = (float *) C->data;

    lut_gemm_kernel(
        qdata->w_q,
        qdata->scales,
        a_q,
        scales_a,
        lut2d,
        c_data,
        M,
        N,
        K,
        config->group_size,
        config->w_bits,
        config->a_bits
    );

    // Cleanup
    delete[] a_q;
    delete[] scales_a;
    delete[] lut2d;
}
