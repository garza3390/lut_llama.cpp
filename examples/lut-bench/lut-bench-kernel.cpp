#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

// Para cuantización Q4_0 nativa (headers internos de ggml, vía target_include_directories)
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"    // block_q4_0, QK4_0
#include "ggml-quants.h"    // quantize_q4_0

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>

static void fill_random(float * data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        data[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }
}

// Baseline F32 GEMM via ggml_mul_mat
static double benchmark_gemm_baseline(
    ggml_backend_t backend,
    struct ggml_context * ctx,
    struct ggml_tensor * A,
    struct ggml_tensor * B,
    int iterations
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * C  = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf, C);
        ggml_backend_graph_compute(backend, gf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    return dt.count() / (double) iterations;
}

// Q4_0 GEMM via ggml_mul_mat (pesos ya en tensor Q4_0)
static double benchmark_gemm_q4_0(
    ggml_backend_t backend,
    struct ggml_context * ctx,
    struct ggml_tensor * A,
    struct ggml_tensor * B_q4,
    int iterations
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * C  = ggml_mul_mat(ctx, B_q4, A);
        ggml_build_forward_expand(gf, C);
        ggml_backend_graph_compute(backend, gf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    return dt.count() / (double) iterations;
}

// LUT GEMM
static double benchmark_gemm_lut(
    struct ggml_tensor * A,
    struct ggml_tensor * B,
    struct ggml_tensor * C,
    const struct ggml_lut_config * config,
    int iterations
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        ggml_lut_compute_gemm(A, B, C, config);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    return dt.count() / (double) iterations;
}

static float compute_max_error(const float * ref, const float * test, size_t size) {
    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float e = std::fabs(ref[i] - test[i]);
        if (e > max_err) { max_err = e; }
    }
    return max_err;
}

static float compute_mse(const float * ref, const float * test, size_t size) {
    double acc = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double e = (double)(ref[i] - test[i]);
        acc += e * e;
    }
    return (float)(acc / (double) size);
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("LUT GEMM Kernel Benchmark\n");
    printf("Comparacion: F32 baseline | Q4_0 nativo ggml | LUT-Q4_0\n");
    printf("=========================================================\n\n");

    ggml_lut_global_init();

    struct {
        int M, N, K;
        int w_bits, a_bits, group_size;
    } configs[] = {
        {32,  4096, 4096, 4, 8, 32},
        {32,  4096, 4096, 4, 8, 64},
        {64,  2048, 2048, 4, 8, 32},
        {16,  8192, 4096, 4, 8, 32},
    };

    const int iterations = 10;

    printf("op_type,m,n,k,w_bits,a_bits,group_size,"
           "f32_ms,q4_ms,lut_ms,"
           "q4_speedup,lut_speedup,"
           "q4_max_err,lut_max_err,"
           "q4_mse,lut_mse\n");

    for (const auto & cfg : configs) {
        // K debe ser múltiplo de QK4_0 (32) para Q4_0
        if (cfg.K % QK4_0 != 0) {
            fprintf(stderr, "Saltando config K=%d (no múltiplo de %d)\n", cfg.K, QK4_0);
            continue;
        }

        ggml_lut_config lut_cfg;
        lut_cfg.w_bits     = cfg.w_bits;
        lut_cfg.a_bits     = cfg.a_bits;
        lut_cfg.group_size = cfg.group_size;
        lut_cfg.mu         = 1;
        lut_cfg.enabled    = true;

        struct ggml_init_params params;
        params.mem_size   = (size_t) 1024 * 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) { fprintf(stderr, "Failed to init ggml context\n"); continue; }

        ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
        if (!backend) { fprintf(stderr, "Failed to init CPU backend\n"); ggml_free(ctx); continue; }

        // Tensores F32
        struct ggml_tensor * A     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.M);
        struct ggml_tensor * B     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.N);
        struct ggml_tensor * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);

        fill_random((float *) A->data, (size_t) cfg.K * cfg.M, 42);
        fill_random((float *) B->data, (size_t) cfg.K * cfg.N, 123);

        // ----------------------------------------------------------------
        // 1) F32 baseline
        // ----------------------------------------------------------------
        double f32_ms = benchmark_gemm_baseline(backend, ctx, A, B, iterations);

        // Capturar salida F32 de referencia
        struct ggml_cgraph * gf_ref = ggml_new_graph(ctx);
        struct ggml_tensor * C_ref_t = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf_ref, C_ref_t);
        ggml_backend_graph_compute(backend, gf_ref);

        struct ggml_tensor * C_ref = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);
        std::memcpy(C_ref->data, C_ref_t->data, ggml_nbytes(C_ref));

        // ----------------------------------------------------------------
        // 2) Q4_0 nativo ggml
        // Crear un tensor Q4_0 con los mismos pesos cuantizados nativamente
        // ----------------------------------------------------------------
        struct ggml_tensor * B_q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, cfg.K, cfg.N);
        {
            const float * b_f32 = (const float *) B->data;
            quantize_q4_0(b_f32, B_q4->data, cfg.N, cfg.K, NULL);
        }

        double q4_ms = benchmark_gemm_q4_0(backend, ctx, A, B_q4, iterations);

        // Capturar salida Q4_0
        struct ggml_cgraph * gf_q4 = ggml_new_graph(ctx);
        struct ggml_tensor * C_q4_t = ggml_mul_mat(ctx, B_q4, A);
        ggml_build_forward_expand(gf_q4, C_q4_t);
        ggml_backend_graph_compute(backend, gf_q4);

        struct ggml_tensor * C_q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);
        std::memcpy(C_q4->data, C_q4_t->data, ggml_nbytes(C_q4));

        // ----------------------------------------------------------------
        // 3) LUT-Q4_0: cuantización nativa Q4_0 + kernel LUT
        // ----------------------------------------------------------------
        ggml_lut_quantize_weights_q4_0(B, &lut_cfg);
        double lut_ms = benchmark_gemm_lut(A, B, C_lut, &lut_cfg, iterations);

        // ----------------------------------------------------------------
        // Errores
        // ----------------------------------------------------------------
        const size_t out_elems = (size_t) cfg.M * cfg.N;
        const float * ref_ptr = (const float *) C_ref->data;
        const float * q4_ptr  = (const float *) C_q4->data;
        const float * lut_ptr = (const float *) C_lut->data;

        float q4_max_err  = compute_max_error(ref_ptr, q4_ptr,  out_elems);
        float lut_max_err = compute_max_error(ref_ptr, lut_ptr, out_elems);
        float q4_mse      = compute_mse(ref_ptr, q4_ptr,  out_elems);
        float lut_mse     = compute_mse(ref_ptr, lut_ptr, out_elems);

        double q4_speedup  = (q4_ms  > 0.0) ? (f32_ms / q4_ms)  : 0.0;
        double lut_speedup = (lut_ms > 0.0) ? (f32_ms / lut_ms) : 0.0;

        printf("GEMM,%d,%d,%d,%d,%d,%d,"
               "%.4f,%.4f,%.4f,"
               "%.4f,%.4f,"
               "%.6e,%.6e,"
               "%.6e,%.6e\n",
               cfg.M, cfg.N, cfg.K,
               cfg.w_bits, cfg.a_bits, cfg.group_size,
               f32_ms, q4_ms, lut_ms,
               q4_speedup, lut_speedup,
               q4_max_err, lut_max_err,
               q4_mse, lut_mse);

        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    ggml_lut_global_free();
    return 0;
}
