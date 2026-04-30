/**
 * lut-debug.cpp
 * =====================================================================
 * Diagnóstico elemento por elemento del kernel LUT-GEMM.
 *
 * Ejecuta una matriz pequeña conocida (M, N, K configurables) por cuatro
 * caminos: F32 manual, F32 ggml, Q4_0 ggml, Q8_0 ggml y LUT-Q4_0. Imprime
 * todos los outputs y la matriz de diferencias entre caminos. Sirve para
 * aislar dónde diverge el LUT respecto a Q4_0 nativo, que es la referencia
 * natural ya que ambos usan los mismos pesos cuantizados.
 *
 * Uso:
 *   ./build/bin/lut-debug                # M=1 N=4 K=32 (default)
 *   ./build/bin/lut-debug 2 4 64
 * =====================================================================
 */

#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

// Datos deterministas para reproducibilidad.
static float det_a(int k, int m) {
    return std::sin(0.13f * (float) k + 0.27f * (float) m);
}

static float det_b(int k, int n) {
    return std::cos(0.21f * (float) k + 0.41f * (float) n);
}

// F32 manual fuera de ggml: out[n,m] = sum_k B[k,n] * A[k,m]
static void compute_f32_manual(const float * A, const float * B, float * out,
                               int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                s += (double) B[(size_t) n * K + k] * (double) A[(size_t) m * K + k];
            }
            out[(size_t) m * N + n] = (float) s;
        }
    }
}

static void print_matrix(const char * name, const float * data, int M, int N) {
    printf("\n%s [N=%d, M=%d]:\n", name, N, M);
    for (int m = 0; m < M; ++m) {
        printf("  m=%d: ", m);
        for (int n = 0; n < N; ++n) {
            printf("%10.4f ", data[(size_t) m * N + n]);
        }
        printf("\n");
    }
}

static void print_diff(const char * label,
                       const float * a, const float * b, int M, int N) {
    float max_abs = 0.f;
    double sse = 0.0;
    int worst_n = -1, worst_m = -1;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float d = std::fabs(a[(size_t) m * N + n] - b[(size_t) m * N + n]);
            sse += (double) d * d;
            if (d > max_abs) {
                max_abs = d;
                worst_n = n;
                worst_m = m;
            }
        }
    }
    float mse = (float) (sse / (double)(M * N));
    printf("  %-35s  max|err|=%.6e  mse=%.6e   peor=(n=%d,m=%d)\n",
           label, (double) max_abs, (double) mse, worst_n, worst_m);
}

int main(int argc, char ** argv) {
    int M = 1, N = 4, K = 32;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    if (K % QK4_0 != 0) {
        fprintf(stderr, "K (%d) debe ser múltiplo de QK4_0 (%d)\n", K, QK4_0);
        return 1;
    }

    printf("=== lut-debug ===\n");
    printf("M=%d  N=%d  K=%d  (bloques Q4_0 por columna = %d)\n",
           M, N, K, K / QK4_0);

    ggml_lut_global_init();

    struct ggml_init_params params{ (size_t)256 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    ggml_backend_t bk = ggml_backend_init_by_name("CPU", NULL);

    auto * A     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    auto * B     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    auto * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);

    // Llenar con datos deterministas siguiendo la convención de ggml:
    // A[k,m] = data[k + m*K] = data[m*K + k]
    // B[k,n] = data[k + n*K] = data[n*K + k]
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            ((float*)A->data)[(size_t) m * K + k] = det_a(k, m);
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            ((float*)B->data)[(size_t) n * K + k] = det_b(k, n);
        }
    }

    // ───────────────────────────── F32 manual ─────────────────────────────
    std::vector<float> out_f32_manual((size_t) M * N, 0.f);
    compute_f32_manual((const float*) A->data, (const float*) B->data,
                       out_f32_manual.data(), M, N, K);

    // ────────────────────────── F32 vía ggml_mul_mat ──────────────────────
    std::vector<float> out_f32_ggml((size_t) M * N, 0.f);
    {
        auto * gf = ggml_new_graph(ctx);
        auto * Ct = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf, Ct);
        ggml_backend_graph_compute(bk, gf);
        std::memcpy(out_f32_ggml.data(), Ct->data,
                    sizeof(float) * (size_t) M * N);
    }

    // ────────────────────────── Q4_0 nativo ggml ──────────────────────────
    std::vector<float> out_q4((size_t) M * N, 0.f);
    auto * Bq4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    quantize_q4_0((const float*) B->data, Bq4->data, N, K, NULL);
    {
        auto * gf = ggml_new_graph(ctx);
        auto * Ct = ggml_mul_mat(ctx, Bq4, A);
        ggml_build_forward_expand(gf, Ct);
        ggml_backend_graph_compute(bk, gf);
        std::memcpy(out_q4.data(), Ct->data, sizeof(float) * (size_t) M * N);
    }

    // ────────────────────────── Q8_0 nativo ggml ──────────────────────────
    std::vector<float> out_q8((size_t) M * N, 0.f);
    auto * Bq8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    quantize_q8_0((const float*) B->data, Bq8->data, N, K, NULL);
    {
        auto * gf = ggml_new_graph(ctx);
        auto * Ct = ggml_mul_mat(ctx, Bq8, A);
        ggml_build_forward_expand(gf, Ct);
        ggml_backend_graph_compute(bk, gf);
        std::memcpy(out_q8.data(), Ct->data, sizeof(float) * (size_t) M * N);
    }

    // ─────────────────────────────── LUT-Q4_0 ──────────────────────────────
    ggml_lut_config cfg{ 4, 8, 32, 1, true };
    ggml_lut_clear_weights();
    ggml_lut_quantize_weights_q4_0(B, &cfg);
    ggml_lut_compute_gemm(A, B, C_lut, &cfg);
    std::vector<float> out_lut((size_t) M * N, 0.f);
    std::memcpy(out_lut.data(), C_lut->data,
                sizeof(float) * (size_t) M * N);

    // ──────────────────────────── Volcado de outputs ───────────────────────
    print_matrix("F32 manual",   out_f32_manual.data(), M, N);
    print_matrix("F32 ggml",     out_f32_ggml.data(),   M, N);
    print_matrix("Q4_0 ggml",    out_q4.data(),         M, N);
    print_matrix("Q8_0 ggml",    out_q8.data(),         M, N);
    print_matrix("LUT-Q4_0",     out_lut.data(),        M, N);

    // ─────────────────────────── Diferencias clave ─────────────────────────
    printf("\n=== Diferencias entre caminos ===\n");
    print_diff("F32 manual    vs  F32 ggml",   out_f32_manual.data(), out_f32_ggml.data(), M, N);
    print_diff("F32 ggml      vs  Q4_0 ggml",  out_f32_ggml.data(),   out_q4.data(),       M, N);
    print_diff("F32 ggml      vs  Q8_0 ggml",  out_f32_ggml.data(),   out_q8.data(),       M, N);
    print_diff("F32 ggml      vs  LUT-Q4_0",   out_f32_ggml.data(),   out_lut.data(),      M, N);
    print_diff("Q4_0 ggml     vs  LUT-Q4_0",   out_q4.data(),         out_lut.data(),      M, N);
    print_diff("Q8_0 ggml     vs  LUT-Q4_0",   out_q8.data(),         out_lut.data(),      M, N);

    printf("\n=== Interpretación ===\n");
    printf("La fila clave es 'Q4_0 ggml vs LUT-Q4_0'. Si max|err| < 1e-3,\n");
    printf("el LUT es matemáticamente equivalente a Q4_0 nativo y cualquier\n");
    printf("error vs F32 viene de la cuantización Q4_0, no del LUT.\n");
    printf("Si max|err| >> 1e-3, hay un bug en el pipeline LUT.\n");

    ggml_lut_global_free();
    ggml_backend_free(bk);
    ggml_free(ctx);
    return 0;
}
