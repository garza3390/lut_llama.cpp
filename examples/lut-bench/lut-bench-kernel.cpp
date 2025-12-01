#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static void fill_random(float * data, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        data[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
    }
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("LUT GEMM Kernel Benchmark\n");
    printf("=========================\n\n");

    ggml_lut_global_init();

    // Simple single-config benchmark for now
    const int M = 32;
    const int N = 4096;
    const int K = 4096;

    ggml_lut_config config;
    config.w_bits     = 4;
    config.a_bits     = 8;
    config.group_size = 32;
    config.mu         = 1;
    config.enabled    = true;

    // Init ggml context
    struct ggml_init_params params;
    params.mem_size   = 1024 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = false;

    struct ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != NULL);

    ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
    if (backend == NULL) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        ggml_free(ctx);
        ggml_lut_global_free();
        return 1;
    }

    // A: [K, M], B: [K, N], C: [N, M]
    struct ggml_tensor * A      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * B      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * C_ref  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * C_lut  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);

    fill_random((float *) A->data, K * M, 42);
    fill_random((float *) B->data, K * N, 123);

    // Baseline GEMM timing

    auto t_baseline_start = std::chrono::high_resolution_clock::now();

    struct ggml_cgraph * gf_base = ggml_new_graph(ctx);
    struct ggml_tensor * result_baseline = ggml_mul_mat(ctx, B, A);
    ggml_build_forward_expand(gf_base, result_baseline);
    ggml_backend_graph_compute(backend, gf_base);

    auto t_baseline_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> baseline_ms = t_baseline_end - t_baseline_start;
    double t_baseline = baseline_ms.count();

    memcpy(C_ref->data, result_baseline->data, ggml_nbytes(C_ref));

    // LUT GEMM timing

    ggml_lut_quantize_weights(B, &config);
    ggml_lut_set_tensor_config(B, &config);

    // Info block (comment-style, does not break CSV output)
    printf("# LUT Configuration:\n");
    printf("#   w_bits: %d, a_bits: %d, group_size: %d\n",
           config.w_bits, config.a_bits, config.group_size);
    printf("#   Using TRUE 2D LUT [%d x %d]\n",
           1 << config.w_bits, 1 << config.a_bits);
    printf("#\n");

    auto t_lut_start = std::chrono::high_resolution_clock::now();
    ggml_lut_compute_gemm(A, B, C_lut, &config);
    auto t_lut_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> lut_ms = t_lut_end - t_lut_start;
    double t_lut = lut_ms.count();

    // Error analysis

    float * c_ref = (float *) C_ref->data;
    float * c_lut = (float *) C_lut->data;

    double max_err = 0.0;
    for (int i = 0; i < N * M; ++i) {
        double diff = (double) c_lut[i] - (double) c_ref[i];
        double err  = diff >= 0.0 ? diff : -diff;
        if (err > max_err) {
            max_err = err;
        }
    }

    // CSV-like output

    printf("baseline_time_ms,lut_time_ms,speedup,max_error\n");
    printf("%.3f,%.3f,%.3fx,%.6f\n",
           t_baseline,
           t_lut,
           (t_lut > 0.0 ? t_baseline / t_lut : 0.0),
           max_err);

    ggml_backend_free(backend);
    ggml_free(ctx);
    ggml_lut_global_free();

    return 0;
}
