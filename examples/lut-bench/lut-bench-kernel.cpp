#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>

static void fill_random(float * data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        // Rango [-1, 1] para emular pesos/activaciones centrados en 0
        data[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }
}

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

static void compute_max_error(
    const float * ref,
    const float * test,
    size_t size,
    float * max_err_out
) {
    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float e = std::fabs(ref[i] - test[i]);
        if (e > max_err) {
            max_err = e;
        }
    }
    *max_err_out = max_err;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("LUT GEMM Kernel Benchmark (multi-config)\n");
    printf("****************************************\n\n");

    ggml_lut_global_init();

    struct {
        int M, N, K;
        int w_bits, a_bits, group_size;
    } configs[] = {
        // Casos tipo LLaMA-ish
        {32, 4096, 4096, 4, 8, 32},
        {32, 4096, 4096, 4, 8, 64},
        {64, 2048, 2048, 4, 8, 32},
        {16, 8192, 4096, 4, 8, 32},
    };

    const int iterations = 10;

    printf("op_type,m,n,k,w_bits,a_bits,group_size,baseline_ms,lut_ms,speedup,max_abs_error\n");

    for (const auto & cfg : configs) {
        // Config LUT
        ggml_lut_config lut_cfg;
        lut_cfg.w_bits     = cfg.w_bits;
        lut_cfg.a_bits     = cfg.a_bits;
        lut_cfg.group_size = cfg.group_size;
        lut_cfg.mu         = 1;
        lut_cfg.enabled    = true;

        // Inicializar ggml
        struct ggml_init_params params;
        params.mem_size   = 1024 * 1024 * 1024;  // 1 GB
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "Failed to init ggml context\n");
            continue;
        }

        ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
        if (!backend) {
            fprintf(stderr, "Failed to init CPU backend\n");
            ggml_free(ctx);
            continue;
        }

        // Tensores:
        // A: [K, M], B: [K, N], C: [N, M]
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.M);
        struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.N);
        struct ggml_tensor * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);

        fill_random((float *) A->data, (size_t) cfg.K * cfg.M, 42);
        fill_random((float *) B->data, (size_t) cfg.K * cfg.N, 123);

        // 1) Baseline GEMM
        double baseline_ms = benchmark_gemm_baseline(backend, ctx, A, B, iterations);

        // Obtener una salida baseline de referencia
        struct ggml_cgraph * gf_ref = ggml_new_graph(ctx);
        struct ggml_tensor * C_ref_t = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf_ref, C_ref_t);
        ggml_backend_graph_compute(backend, gf_ref);

        struct ggml_tensor * C_ref = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);
        std::memcpy(C_ref->data, C_ref_t->data, ggml_nbytes(C_ref));

        // 2) Quantizar pesos y correr LUT GEMM
        ggml_lut_quantize_weights(B, &lut_cfg);

        double lut_ms = benchmark_gemm_lut(A, B, C_lut, &lut_cfg, iterations);

        // 3) Error
        float max_err = 0.0f;
        compute_max_error(
            (const float *) C_ref->data,
            (const float *) C_lut->data,
            (size_t) cfg.M * cfg.N,
            &max_err
        );

        double speedup = (lut_ms > 0.0) ? (baseline_ms / lut_ms) : 0.0;

        printf("GEMM,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.6e\n",
               cfg.M, cfg.N, cfg.K,
               cfg.w_bits, cfg.a_bits, cfg.group_size,
               baseline_ms, lut_ms, speedup, max_err);

        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    ggml_lut_global_free();
    return 0;
}
