#include "ggml-lut.h"
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Debe coincidir con la definición en ggml-lut-quant.cpp
struct ggml_lut_weight_data {
    uint8_t * w_q;
    float   * scales;
    int       num_groups;
};

extern const ggml_lut_weight_data * ggml_lut_get_weight_data(const struct ggml_tensor * tensor);
extern const int32_t * ggml_lut_get_or_build_table(int w_bits, int a_bits);

// ---------------------------------------------------------------------------
// Buffers de trabajo globales: grow-only, liberados en ggml_lut_global_free.
// Evitan alloc/free en cada llamada a ggml_lut_compute_gemm.
// ---------------------------------------------------------------------------
static uint8_t * g_a_q       = nullptr;
static float   * g_scales_a  = nullptr;
static size_t    g_a_q_cap   = 0;
static size_t    g_scales_cap = 0;

static void lut_ensure_work_buffers(int K, int M) {
    const size_t need_a_q    = (size_t) K * M;
    const size_t need_scales = (size_t) M;

    if (need_a_q > g_a_q_cap) {
        delete[] g_a_q;
        g_a_q     = new uint8_t[need_a_q];
        g_a_q_cap = need_a_q;
    }
    if (need_scales > g_scales_cap) {
        delete[] g_scales_a;
        g_scales_a    = new float[need_scales];
        g_scales_cap  = need_scales;
    }
}

// Llamado desde ggml_lut_global_free (ggml-lut.cpp)
extern "C" void ggml_lut_free_work_buffers(void) {
    delete[] g_a_q;      g_a_q      = nullptr; g_a_q_cap   = 0;
    delete[] g_scales_a; g_scales_a = nullptr; g_scales_cap = 0;
}

// ---------------------------------------------------------------------------
// Cuantización de activaciones (SIMÉTRICA SIGNED, columna por columna)
//   A: [K, M]
//   a_q: índices LUT ∈ [0, 2^a_bits − 1]
//   scales_a[m]: escala por columna m
// ---------------------------------------------------------------------------
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

        for (int k = 0; k < K; ++k) {
            float v  = a_data[k * M + m];
            float av = v >= 0.0f ? v : -v;
            if (av > absmax) { absmax = av; }
        }

        const float scale_a = (absmax > 0.0f) ? (absmax / qmax_f) : 1.0f;
        scales_a[m] = scale_a;

        for (int k = 0; k < K; ++k) {
            float v = a_data[k * M + m] / scale_a;
            int   q_centered = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);

            if (q_centered < -qmax) { q_centered = -qmax; }
            else if (q_centered >  qmax) { q_centered =  qmax; }

            a_q[k * M + m] = (uint8_t) (q_centered + qmax);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel GEMM usando LUT 2D.
//
//   lut2d[w_idx * a_levels + a_idx] = q_w_centered * q_a_centered (int puro)
//
//   Para cada (n, m):
//       acc_int = Σ_k lut2d[w_q[n,k], a_q[k,m]]
//       out[m,n] = acc_int * scale_w_global * scale_a[m]
//
// Path AVX2: procesa 8 k por iteración con gather de 32 bits.
// Fallback escalar para arquitecturas sin AVX2.
// ---------------------------------------------------------------------------

#if defined(__AVX2__)
// Suma horizontal de un __m256i de 8 enteros de 32 bits → int64_t
static inline int64_t hsum_epi32(__m256i v) {
    __m128i lo  = _mm256_castsi256_si128(v);
    __m128i hi  = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    // sum = [a0+a4, a1+a5, a2+a6, a3+a7]
    __m128i shuf = _mm_shuffle_epi32(sum, 0x4E);       // swap hi/lo 64-bit pairs
    __m128i sums = _mm_add_epi32(sum, shuf);
    // sums = [a+b, a+b, c+d, c+d]
    __m128i shuf2 = _mm_shuffle_epi32(sums, 0xB1);    // swap adjacent 32-bit words
    __m128i sums2 = _mm_add_epi32(sums, shuf2);
    return (int64_t) _mm_cvtsi128_si32(sums2);
}
#endif

// Acumula por bloque aplicando la escala de cuantización del bloque
// correspondiente. Para Q4_0 el bloque es de 32 pesos; para el esquema propio
// el bloque coincide con config->group_size. En ambos casos
// blocks_per_row * block_size == K.
static void lut_gemm_kernel(
    const uint8_t * GGML_RESTRICT w_q,         // [N * K]
    const float   * GGML_RESTRICT scales_w,    // [N * blocks_per_row]
    int             blocks_per_row,
    int             block_size,
    const uint8_t * GGML_RESTRICT a_q,         // [K * M]
    const float   * GGML_RESTRICT scales_a,    // [M]
    const int32_t * GGML_RESTRICT lut2d,       // [2^w_bits * 2^a_bits]
    float         * GGML_RESTRICT out,         // [N * M]  (out[m * N + n])
    int M, int N, int K,
    int a_bits
) {
    const int a_levels = 1 << a_bits;

    for (int m = 0; m < M; ++m) {
        const float scale_a_m = scales_a[m];

        for (int n = 0; n < N; ++n) {
            float acc_float = 0.0f;

            for (int b = 0; b < blocks_per_row; ++b) {
                const int k_start = b * block_size;
                const int k_end   = k_start + block_size;
                const float scale_w_b = scales_w[(size_t) n * blocks_per_row + b];

                int64_t acc_int = 0;

#if defined(__AVX2__)
                {
                    __m256i acc_v = _mm256_setzero_si256();
                    const __m256i a_lev_v = _mm256_set1_epi32(a_levels);
                    int k = k_start;

                    for (; k <= k_end - 8; k += 8) {
                        const __m128i w_byte = _mm_loadl_epi64(
                            (const __m128i *) (w_q + (size_t) n * K + k));
                        const __m128i a_byte = _mm_loadl_epi64(
                            (const __m128i *) (a_q + (size_t) k * M + m));

                        const __m256i w_i32 = _mm256_cvtepu8_epi32(w_byte);
                        const __m256i a_i32 = _mm256_cvtepu8_epi32(a_byte);

                        const __m256i idx = _mm256_add_epi32(
                            _mm256_mullo_epi32(w_i32, a_lev_v),
                            a_i32
                        );

                        const __m256i vals = _mm256_i32gather_epi32(lut2d, idx, 4);
                        acc_v = _mm256_add_epi32(acc_v, vals);
                    }

                    acc_int = hsum_epi32(acc_v);

                    for (; k < k_end; ++k) {
                        const uint8_t w_idx = w_q[(size_t) n * K + k];
                        const uint8_t a_idx = a_q[(size_t) k * M + m];
                        acc_int += (int64_t) lut2d[(int) w_idx * a_levels + (int) a_idx];
                    }
                }
#else
                for (int k = k_start; k < k_end; ++k) {
                    const uint8_t w_idx = w_q[(size_t) n * K + k];
                    const uint8_t a_idx = a_q[(size_t) k * M + m];
                    acc_int += (int64_t) lut2d[(int) w_idx * a_levels + (int) a_idx];
                }
#endif

                acc_float += (float) acc_int * scale_w_b;
            }

            out[m * N + n] = acc_float * scale_a_m;
        }
    }
}

// ---------------------------------------------------------------------------
// Orquestador principal LUT GEMM
//   A: [K, M], B: [K, N], C: [N, M]
// ---------------------------------------------------------------------------
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
    GGML_ASSERT(qdata->num_groups > 0 && (qdata->num_groups % N) == 0 &&
                "num_groups debe ser múltiplo de N (N bloques por fila)");

    // Derivar la topología de bloques. Para Q4_0 esto rinde block_size == 32;
    // para el esquema propio, block_size == config->group_size.
    const int blocks_per_row = qdata->num_groups / N;
    const int block_size     = K / blocks_per_row;
    GGML_ASSERT(blocks_per_row * block_size == K &&
                "K debe ser múltiplo del tamaño de bloque de pesos");

    lut_ensure_work_buffers(K, M);

    const float * a_data = (const float *) A->data;
    ggml_lut_quantize_activations(a_data, g_a_q, g_scales_a, K, M, config->a_bits);

    const int32_t * lut2d = ggml_lut_get_or_build_table(config->w_bits, config->a_bits);

    float * c_data = (float *) C->data;
    lut_gemm_kernel(
        qdata->w_q,
        qdata->scales,
        blocks_per_row,
        block_size,
        g_a_q,
        g_scales_a,
        lut2d,
        c_data,
        M, N, K,
        config->a_bits
    );
}
