#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void fill_random(float * data, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        data[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
    }
}

int main(void) {
    printf("Testing LUT GEMM kernel...\n");

    ggml_lut_global_init();

    // Matrix sizes
    const int M = 16;
    const int N = 32;
    const int K = 64;

    // Create LUT config
    ggml_lut_config config;
    config.w_bits     = 4;
    config.a_bits     = 8;
    config.group_size = 32;
    config.mu         = 1;
    config.enabled    = true;

    // Initialize ggml context
    struct ggml_init_params params;
    params.mem_size   = 256 * 1024 * 1024;
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
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * C_ref = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);

    float * a_data = (float *) A->data;
    float * b_data = (float *) B->data;

    fill_random(a_data, K * M, 42);
    fill_random(b_data, K * N, 123);

    printf("Matrix dimensions: A[%d,%d] * B[%d,%d] = C[%d,%d]\n", K, M, K, N, N, M);
    printf("Quantization: W%dA%d, group_size=%d\n",
           config.w_bits, config.a_bits, config.group_size);

    // Baseline GEMM (CPU backend)

    struct ggml_cgraph * gf_baseline = ggml_new_graph(ctx);
    struct ggml_tensor * result_baseline = ggml_mul_mat(ctx, B, A);
    ggml_build_forward_expand(gf_baseline, result_baseline);
    ggml_backend_graph_compute(backend, gf_baseline);

    memcpy(C_ref->data, result_baseline->data, ggml_nbytes(C_ref));

    // LUT GEMM

    ggml_lut_quantize_weights(B, &config);
    ggml_lut_set_tensor_config(B, &config);

    ggml_lut_compute_gemm(A, B, C_lut, &config);

    // Error analysis

    float * c_ref = (float *) C_ref->data;
    float * c_lut = (float *) C_lut->data;

    double max_err    = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    for (int i = 0; i < N * M; ++i) {
        double diff = (double) c_lut[i] - (double) c_ref[i];
        double err  = diff >= 0.0 ? diff : -diff;

        if (err > max_err) {
            max_err = err;
        }

        sum_sq_err += diff * diff;
        sum_sq_ref += (double) c_ref[i] * (double) c_ref[i];
    }

    double mse     = sum_sq_err / (double) (N * M);
    double rel_err = (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0;

    printf("Max error: %.6f\n", max_err);
    printf("MSE: %.6e\n", mse);
    printf("Relative error: %.6f\n", rel_err);

    // Phase 1b thresholds:
    // We expect quantization-induced errors in the low single digits,
    // not the huge values we had before (hundreds).
    const double threshold = 5.0;

    int ret = 0;
    if (max_err > threshold) {
        fprintf(stderr,
                "FAIL: max error %.6f exceeds threshold %.6f\n",
                max_err, threshold);
        ret = 1;
    } else {
        printf("PASS: max error %.6f within threshold %.6f\n", max_err, threshold);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    ggml_lut_global_free();

    return ret;
}
