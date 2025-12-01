#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

static void fill_random(float * data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Range [-1, 1]
    }
}

static void compute_errors(const float * baseline, const float * lut, size_t size,
                          float * max_err, float * mse, float * rel_err) {
    *max_err = 0.0f;
    *mse = 0.0f;
    *rel_err = 0.0f;
    
    float baseline_norm = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        const float diff = std::fabs(baseline[i] - lut[i]);
        *max_err = std::max(*max_err, diff);
        *mse += diff * diff;
        baseline_norm += baseline[i] * baseline[i];
    }
    
    *mse /= size;
    
    if (baseline_norm > 0.0f) {
        float error_norm = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            const float diff = baseline[i] - lut[i];
            error_norm += diff * diff;
        }
        *rel_err = std::sqrt(error_norm / baseline_norm);
    }
}

int main() {
    printf("Testing LUT GEMM kernel...\n");
    
    ggml_lut_global_init();
    
    // Test configuration
    const int M = 16;
    const int N = 32;
    const int K = 64;
    
    struct ggml_lut_config config;
    config.w_bits = 4;
    config.a_bits = 8;
    config.group_size = 32;
    config.mu = 4;
    config.enabled = true;
    
    // Initialize ggml
    struct ggml_init_params params;
    params.mem_size   = 256 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    
    struct ggml_context * ctx = ggml_init(params);
    
    // Create tensors
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * C_baseline = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    
    // Fill with deterministic random data
    fill_random((float *)A->data, M * K, 42);
    fill_random((float *)B->data, N * K, 123);
    
    printf("Matrix dimensions: A[%d,%d] * B[%d,%d] = C[%d,%d]\n", K, M, K, N, N, M);
    printf("Quantization: W%dA%d, group_size=%d\n", config.w_bits, config.a_bits, config.group_size);
    
    // Baseline GEMM using ggml
    struct ggml_cgraph * gf_baseline = ggml_new_graph(ctx);
    struct ggml_tensor * result_baseline = ggml_mul_mat(ctx, B, A);
    ggml_build_forward_expand(gf_baseline, result_baseline);
    
    // Use backend to compute
    ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
    if (backend == NULL) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        ggml_free(ctx);
        ggml_lut_global_free();
        return 1;
    }
    ggml_backend_graph_compute(backend, gf_baseline);
    
    // Copy baseline result
    memcpy(C_baseline->data, result_baseline->data, ggml_nbytes(C_baseline));
    
    // LUT GEMM
    // First quantize weights
    ggml_lut_quantize_weights(B, &config);
    ggml_lut_set_tensor_config(B, &config);
    
    // Compute using LUT
    ggml_lut_compute_gemm(A, B, C_lut, &config);
    
    // Compute errors
    float max_err, mse, rel_err;
    compute_errors((float *)C_baseline->data, (float *)C_lut->data, M * N,
                   &max_err, &mse, &rel_err);
    
    printf("\n");
    printf("Error metrics:\n");
    printf("  Max absolute error: %.6e\n", max_err);
    printf("  Mean squared error: %.6e\n", mse);
    printf("  Relative error:     %.6e\n", rel_err);
    
    // Tolerance check
    const float max_err_threshold = 1.0f;     // Relaxed for W4A8
    const float mse_threshold = 0.1f;
    
    bool passed = (max_err < max_err_threshold) && (mse < mse_threshold);
    
    if (passed) {
        printf("\n✓ TEST PASSED\n");
    } else {
        printf("\n✗ TEST FAILED\n");
        printf("  Max error %.6e exceeds threshold %.6e: %s\n",
               max_err, max_err_threshold, max_err >= max_err_threshold ? "YES" : "NO");
        printf("  MSE %.6e exceeds threshold %.6e: %s\n",
               mse, mse_threshold, mse >= mse_threshold ? "YES" : "NO");
    }
    
    // Cleanup
    ggml_backend_free(backend);
    ggml_free(ctx);
    ggml_lut_global_free();
    
    return passed ? 0 : 1;
}