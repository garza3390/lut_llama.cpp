#include "ggml-lut.h"
#include <cstring>

// Debe coincidir con la definición en ggml-lut-quant.cpp
struct ggml_lut_weight_data {
    uint8_t * w_q;
    float   * scales;
    int       num_groups;
};

extern const ggml_lut_weight_data * ggml_lut_get_weight_data(const struct ggml_tensor * tensor);

// Cuantización de activaciones (SIMÉTRICA SIGNED, columna por columna)
//   A: [K, M]
//   a_q: índices LUT ∈ [0, 2^A − 1]
//   scales_a[m]: escala por columna m
static void ggml_lut_quantize_activations(
    const float * GGML_RESTRICT a_data,   // [K * M]
    uint8_t     * GGML_RESTRICT a_q,      // [K * M]
    float       * GGML_RESTRICT scales_a, // [M]
    int K,
    int M,
    int a_bits
) {
    const int   a_levels = 1 << a_bits;
    const int   qmax     = (a_levels - 1) / 2;
    const float qmax_f   = (float) qmax;

    for (int m = 0; m < M; ++m) {
        float absmax = 0.0f;

        // absmax de la columna m
        for (int k = 0; k < K; ++k) {
            float v  = a_data[k * M + m];
            float av = v >= 0.0f ? v : -v;
            if (av > absmax) {
                absmax = av;
            }
        }

        float scale_a = (absmax > 0.0f) ? (absmax / qmax_f) : 1.0f;
        scales_a[m] = scale_a;

        for (int k = 0; k < K; ++k) {
            float v = a_data[k * M + m] / scale_a;
            int   q_centered = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);

            if (q_centered < -qmax) {
                q_centered = -qmax;
            } else if (q_centered > qmax) {
                q_centered = qmax;
            }

            const int idx = q_centered + qmax;  // índice LUT
            a_q[k * M + m] = (uint8_t) idx;
        }
    }
}

// Kernel GEMM usando LUT 2D y UN SOLO scale_w global:
//
//   LUT contiene: lut2d[w_idx, a_idx] = q_w_centered * q_a_centered
//
//   Para cada salida (n, m):
//       acc_int = Σ_k LUT(q_w[n,k], q_a[k,m])
//       y[n,m]  = acc_int * scale_w_global * scale_a[m]
//
// Esto ignora los grupos de pesos en la dequantización efectiva, pero es
// consistente y suficiente para la fase experimental.
static void lut_gemm_kernel(
    const uint8_t * GGML_RESTRICT w_q,         // [N * K]
    float          scale_w_global,             // escala global de pesos
    const uint8_t * GGML_RESTRICT a_q,         // [K * M]
    const float   * GGML_RESTRICT scales_a,    // [M]
    const int32_t * GGML_RESTRICT lut2d,       // [2^w_bits * 2^a_bits]
    float         * GGML_RESTRICT out,         // [N * M]
    int M, int N, int K,
    int a_bits
) {
    const int a_levels = 1 << a_bits;

    for (int m = 0; m < M; ++m) {
        const float scale_a_m = scales_a[m];

        for (int n = 0; n < N; ++n) {
            int64_t acc_int = 0;

            for (int k = 0; k < K; ++k) {
                const uint8_t w_idx = w_q[n * K + k];
                const uint8_t a_idx = a_q[k * M + m];

                const int32_t lut_val = lut2d[(int) w_idx * a_levels + (int) a_idx];
                acc_int += (int64_t) lut_val;
            }

            out[m * N + n] = (float) acc_int * scale_w_global * scale_a_m;
        }
    }
}

// Orquestador principal LUT GEMM
//   A: [K, M], B: [K, N], C: [N, M]
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

    const int M = (int) A->ne[1];
    const int K = (int) A->ne[0];
    const int N = (int) B->ne[1];

    GGML_ASSERT((int) B->ne[0] == K);
    GGML_ASSERT((int) C->ne[0] == N);
    GGML_ASSERT((int) C->ne[1] == M);

    const ggml_lut_weight_data * qdata = ggml_lut_get_weight_data(B);
    GGML_ASSERT(qdata != NULL && "Weights must be quantized before LUT GEMM");

    // Por ahora usamos un solo scale_w global (el primero del tensor)
    const float scale_w_global = (qdata->num_groups > 0) ? qdata->scales[0] : 1.0f;

    // Buffers temporales (fuera del núcleo “HLS-like”)
    uint8_t * a_q      = new uint8_t[K * M];
    float   * scales_a = new float[M];

    const int w_levels = 1 << config->w_bits;
    const int a_levels = 1 << config->a_bits;
    int32_t * lut2d    = new int32_t[w_levels * a_levels];

    // Cuantizar activaciones
    const float * a_data = (const float *) A->data;
    ggml_lut_quantize_activations(
        a_data,
        a_q,
        scales_a,
        K,
        M,
        config->a_bits
    );

    // Construir LUT 2D: LUT almacena solo q_w_centered * q_a_centered
    // Las escalas reales se aplican fuera (scale_w_global y scale_a[m]).
    ggml_lut_build_table(
        lut2d,
        config->w_bits,
        config->a_bits,
        /*scale_w=*/1.0f,
        /*scale_x=*/1.0f
    );

    float * c_data = (float *) C->data;
    lut_gemm_kernel(
        qdata->w_q,
        scale_w_global,
        a_q,
        scales_a,
        lut2d,
        c_data,
        M,
        N,
        K,
        config->a_bits
    );

    delete[] a_q;
    delete[] scales_a;
    delete[] lut2d;
}
