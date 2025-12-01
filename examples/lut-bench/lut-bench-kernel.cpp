#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>

static void fill_random(float * data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

static double benchmark_gemm_baseline(
    ggml_backend_t backend,
    struct ggml_context * ctx,
    struct ggml_tensor * A,
    struct ggml_tensor * B,
    int iterations
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * result = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf, result);
        ggml_backend_graph_compute(backend, gf);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    return elapsed.count() / iterations;
}

static double benchmark_gemm_lut(
    struct ggml_tensor * A,
    struct ggml_tensor * B,
    struct ggml_tensor * C,
    const struct ggml_lut_config * config,
    int iterations
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ggml_lut_compute_gemm(A, B, C, config);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    return elapsed.count() / iterations;
}

static void compute_max_error(const float * baseline, const float * lut, size_t size, float * max_err) {
    *max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        const float diff = std::fabs(baseline[i] - lut[i]);
        *max_err = std::max(*max_err, diff);
    }
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    printf("LUT GEMM Kernel Benchmark\n");
    printf("=========================\n\n");
    
    ggml_lut_global_init();
    
    // Benchmark configurations
    struct {
        int M, N, K;
        int w_bits, a_bits, group_size;
    } configs[] = {
        {32, 4096, 4096, 4, 8, 32},
        {32, 4096, 4096, 4, 8, 64},
        {64, 2048, 2048, 4, 8, 32},
        {16, 8192, 4096, 4, 8, 32},
    };
    
    const int iterations = 10;
    
    printf("op_type,m,n,k,w_bits,a_bits,group_size,baseline_ms,lut_ms,speedup,max_abs_error\n");
    
    for (const auto & cfg : configs) {
        struct ggml_lut_config lut_config;
        lut_config.w_bits = cfg.w_bits;
        lut_config.a_bits = cfg.a_bits;
        lut_config.group_size = cfg.group_size;
        lut_config.mu = 4;
        lut_config.enabled = true;
        
        struct ggml_init_params params;
        params.mem_size   = 1024 * 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        
        struct ggml_context * ctx = ggml_init(params);
        ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
        
        if (backend == NULL) {
            fprintf(stderr, "Failed to initialize CPU backend\n");
            ggml_free(ctx);
            continue;
        }
        
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.M);
        struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.N);
        struct ggml_tensor * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);
        
        fill_random((float *)A->data, cfg.M * cfg.K, 42);
        fill_random((float *)B->data, cfg.N * cfg.K, 123);
        
        const double baseline_ms = benchmark_gemm_baseline(backend, ctx, A, B, iterations);
        
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * result_baseline = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf, result_baseline);
        ggml_backend_graph_compute(backend, gf);
        
        struct ggml_tensor * C_baseline = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);
        memcpy(C_baseline->data, result_baseline->data, ggml_nbytes(C_baseline));
        
        ggml_lut_quantize_weights(B, &lut_config);
        
        const double lut_ms = benchmark_gemm_lut(A, B, C_lut, &lut_config, iterations);
        
        float max_err;
        compute_max_error((float *)C_baseline->data, (float *)C_lut->data,
                         cfg.M * cfg.N, &max_err);
        
        const double speedup = baseline_ms / lut_ms;
        
        printf("GEMM,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.6e\n",
               cfg.M, cfg.N, cfg.K, cfg.w_bits, cfg.a_bits, cfg.group_size,
               baseline_ms, lut_ms, speedup, max_err);
        
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    
    ggml_lut_global_free();
    
    return 0;
}