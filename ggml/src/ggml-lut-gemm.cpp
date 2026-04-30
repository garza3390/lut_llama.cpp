#include "ggml-lut.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

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

static void lut_ensure_work_buffers(int K, int M, int blocks_per_row_a) {
    const size_t need_a_q    = (size_t) K * M;
    const size_t need_scales = (size_t) M * blocks_per_row_a;

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
// Cuantización de activaciones con convención Q_0 (asimétrica, por bloque).
//   A: [K, M]
//   a_q: índices LUT ∈ [0, 2^a_bits − 1]
//   scales_a[m * blocks_per_row_a + b]: escala del bloque b dentro de la
//   columna m. Cada bloque agrupa block_size_a elementos.
//
// El esquema refleja Q8_0 nativo de ggml: absmax local por bloque. Esto
// evita que un outlier en la columna infle la escala de todos los demás
// elementos, lo que reduce el error de cuantización frente a un esquema
// por columna.
// ---------------------------------------------------------------------------
static void ggml_lut_quantize_activations(
    const float * GGML_RESTRICT a_data,   // [K * M]
    uint8_t     * GGML_RESTRICT a_q,      // [K * M]
    float       * GGML_RESTRICT scales_a, // [M * blocks_per_row_a]
    int K,
    int M,
    int a_bits,
    int blocks_per_row_a,
    int block_size_a
) {
    const int   q_neg_max = 1 << (a_bits - 1);
    const int   q_pos_max = q_neg_max - 1;
    const float q_pos_f   = (float) q_pos_max;

    for (int m = 0; m < M; ++m) {
        for (int b = 0; b < blocks_per_row_a; ++b) {
            const int k_start = b * block_size_a;
            const int k_end   = k_start + block_size_a;

            float absmax = 0.0f;
            for (int k = k_start; k < k_end; ++k) {
                float v  = a_data[(size_t) m * K + k];
                float av = v >= 0.0f ? v : -v;
                if (av > absmax) { absmax = av; }
            }

            const float scale_a = (absmax > 0.0f) ? (absmax / q_pos_f) : 1.0f;
            scales_a[(size_t) m * blocks_per_row_a + b] = scale_a;

            for (int k = k_start; k < k_end; ++k) {
                float v = a_data[(size_t) m * K + k] / scale_a;
                int   q_centered = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);

                if (q_centered < -q_neg_max) { q_centered = -q_neg_max; }
                else if (q_centered >  q_pos_max) { q_centered =  q_pos_max; }

                a_q[(size_t) m * K + k] = (uint8_t) (q_centered + q_neg_max);
            }
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

// Acumula por bloque aplicando las escalas del bloque de pesos y del
// bloque de activaciones correspondientes. Tanto pesos como activaciones
// quedan en bloques de LUT_ACT_BLOCK_SIZE == 32 elementos, alineados
// con QK4_0 y con el esquema Q8_0 de ggml.
static void lut_gemm_kernel(
    const uint8_t * GGML_RESTRICT w_q,         // [N * K]
    const float   * GGML_RESTRICT scales_w,    // [N * blocks_per_row]
    const uint8_t * GGML_RESTRICT a_q,         // [K * M]
    const float   * GGML_RESTRICT scales_a,    // [M * blocks_per_row]
    int             blocks_per_row,
    int             block_size,
    const int32_t * GGML_RESTRICT lut2d,       // [2^w_bits * 2^a_bits]
    float         * GGML_RESTRICT out,         // [N * M]  (out[m * N + n])
    int M, int N, int K,
    int a_bits
) {
    const int a_levels = 1 << a_bits;

    const bool debug = (getenv("LUT_DEBUG") != nullptr);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc_float = 0.0f;

            for (int b = 0; b < blocks_per_row; ++b) {
                const int k_start = b * block_size;
                const int k_end   = k_start + block_size;
                const float scale_w_b = scales_w[(size_t) n * blocks_per_row + b];
                const float scale_a_b = scales_a[(size_t) m * blocks_per_row + b];

                int64_t acc_int = 0;
                const bool force_scalar = debug && m == 0 && n == 0;

#if defined(__AVX2__)
                if (!force_scalar) {
                    __m256i acc_v = _mm256_setzero_si256();
                    const __m256i a_lev_v = _mm256_set1_epi32(a_levels);
                    int k = k_start;

                    for (; k <= k_end - 8; k += 8) {
                        const __m128i w_byte = _mm_loadl_epi64(
                            (const __m128i *) (w_q + (size_t) n * K + k));
                        const __m128i a_byte = _mm_loadl_epi64(
                            (const __m128i *) (a_q + (size_t) m * K + k));

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
                        const uint8_t a_idx = a_q[(size_t) m * K + k];
                        acc_int += (int64_t) lut2d[(int) w_idx * a_levels + (int) a_idx];
                    }
                } else {
                    // Camino escalar para diagnóstico (mismo cómputo que el fallback)
                    fprintf(stderr,
                        "[LUT-DEBUG]   bloque b=%d, k_start=%d k_end=%d:\n",
                        b, k_start, k_end);
                    for (int k = k_start; k < k_end; ++k) {
                        const uint8_t w_idx = w_q[(size_t) n * K + k];
                        const uint8_t a_idx = a_q[(size_t) m * K + k];
                        const int32_t lut_val = lut2d[(int) w_idx * a_levels + (int) a_idx];
                        acc_int += (int64_t) lut_val;
                        if (k < k_start + 8) {
                            fprintf(stderr,
                                "      k=%2d  w_idx=%3u a_idx=%3u  q_w=%4d q_a=%5d  "
                                "lut=%6d\n",
                                k, (unsigned) w_idx, (unsigned) a_idx,
                                (int) w_idx - 8, (int) a_idx - 128,
                                (int) lut_val);
                        }
                    }
                }
#else
                for (int k = k_start; k < k_end; ++k) {
                    const uint8_t w_idx = w_q[(size_t) n * K + k];
                    const uint8_t a_idx = a_q[(size_t) m * K + k];
                    acc_int += (int64_t) lut2d[(int) w_idx * a_levels + (int) a_idx];
                }
#endif

                if (debug && m == 0 && n == 0) {
                    fprintf(stderr,
                        "[LUT-DEBUG] m=0 n=0 b=%d: acc_int=%lld  "
                        "scale_w_b=%.6e  scale_a_b=%.6e  "
                        "block_contrib=%.6f\n",
                        b, (long long) acc_int,
                        (double) scale_w_b, (double) scale_a_b,
                        (double) ((float) acc_int * scale_w_b * scale_a_b));
                }

                acc_float += (float) acc_int * scale_w_b * scale_a_b;
            }

            if (debug && m == 0 && n == 0) {
                fprintf(stderr,
                    "[LUT-DEBUG] m=0 n=0 final out=%.6f  "
                    "(esperado F32=-9.4563)\n",
                    (double) acc_float);
            }

            out[m * N + n] = acc_float;
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

    // Topología de bloques para pesos. Para Q4_0 es block_size == 32.
    const int blocks_per_row = qdata->num_groups / N;
    const int block_size     = K / blocks_per_row;
    GGML_ASSERT(blocks_per_row * block_size == K &&
                "K debe ser múltiplo del tamaño de bloque de pesos");

    // Activaciones usan el mismo tamaño de bloque que los pesos, de modo que
    // cada par (w_block, a_block) comparte el mismo rango de k.
    const int blocks_per_row_a = blocks_per_row;
    const int block_size_a     = block_size;

    lut_ensure_work_buffers(K, M, blocks_per_row_a);

    const float * a_data = (const float *) A->data;
    ggml_lut_quantize_activations(
        a_data, g_a_q, g_scales_a, K, M, config->a_bits,
        blocks_per_row_a, block_size_a);

    const int32_t * lut2d = ggml_lut_get_or_build_table(config->w_bits, config->a_bits);

    float * c_data = (float *) C->data;
    lut_gemm_kernel(
        qdata->w_q,
        qdata->scales,
        g_a_q,
        g_scales_a,
        blocks_per_row,
        block_size,
        lut2d,
        c_data,
        M, N, K,
        config->a_bits
    );
}
