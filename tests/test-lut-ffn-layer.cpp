#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

static void fill_random(float * data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

static void compute_errors(const float * baseline, const float * lut, size_t size,
                          float * max_err, float * mse) {
    *max_err = 0.0f;
    *mse = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        const float diff = std::fabs(baseline[i] - lut[i]);
        *max_err = std::max(*max_err, diff);
        *mse += diff * diff;
    }
    
    *mse /= size;
}

int main() {
    printf("Testing LUT FFN mini-layer...\n");
    
    ggml_lut_global_init();
    
    // FFN dimensions (small for testing)
    const int d_model = 64;
    const int d_hidden = 128;
    const int batch_size = 1;
    
    struct ggml_lut_config config;
    config.w_bits = 4;
    config.a_bits = 8;
    config.group_size = 32;
    config.mu = 4;
    config.enabled = true;
    
    printf("FFN dimensions: d_model=%d, d_hidden=%d\n", d_model, d_hidden);
    printf("Quantization: W%dA%d, group_size=%d\n", config.w_bits, config.a_bits, config.group_size);
    
    // Initialize ggml
    struct ggml_init_params params;
    params.mem_size   = 256 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
    
    // Create tensors
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, batch_size);
    struct ggml_tensor * W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_hidden);
    struct ggml_tensor * W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_hidden, d_model);
    
    // Fill with random data
    fill_random((float *)x->data, d_model * batch_size, 42);
    fill_random((float *)W1->data, d_model * d_hidden, 123);
    fill_random((float *)W2->data, d_hidden * d_model, 456);
    
    // --- Baseline FFN ---
    struct ggml_cgraph * gf_baseline = ggml_new_graph(ctx);
    struct ggml_tensor * h_baseline = ggml_mul_mat(ctx, W1, x);
    h_baseline = ggml_silu(ctx, h_baseline);
    struct ggml_tensor * y_baseline = ggml_mul_mat(ctx, W2, h_baseline);
    ggml_build_forward_expand(gf_baseline, y_baseline);
    ggml_backend_graph_compute(backend, gf_baseline);
    
    // Save baseline output
    struct ggml_tensor * output_baseline = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, batch_size);
    memcpy(output_baseline->data, y_baseline->data, ggml_nbytes(output_baseline));
    
    // --- LUT FFN ---
    ggml_lut_quantize_weights(W1, &config);
    ggml_lut_set_tensor_config(W1, &config);
    
    struct ggml_tensor * h_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_hidden, batch_size);
    ggml_lut_compute_gemm(x, W1, h_lut, &config);
    
    struct ggml_cgraph * gf_lut = ggml_new_graph(ctx);
    struct ggml_tensor * h_lut_act = ggml_silu(ctx, h_lut);
    struct ggml_tensor * y_lut = ggml_mul_mat(ctx, W2, h_lut_act);
    ggml_build_forward_expand(gf_lut, y_lut);
    ggml_backend_graph_compute(backend, gf_lut);
    
    // Compute errors
    float max_err, mse;
    compute_errors((float *)output_baseline->data, (float *)y_lut->data,
                   d_model * batch_size, &max_err, &mse);
    
    printf("\n");
    printf("Error metrics:\n");
    printf("  Max absolute error: %.6e\n", max_err);
    printf("  Mean squared error: %.6e\n", mse);
    
    const float max_err_threshold = 2.0f;
    const float mse_threshold = 0.5f;
    
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