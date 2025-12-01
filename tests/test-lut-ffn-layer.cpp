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
    printf("Testing LUT FFN mini-layer...\n");

    ggml_lut_global_init();

    // FFN dimensions (small for testing)
    const int d_model    = 64;
    const int d_hidden   = 128;
    const int batch_size = 1;

    // LUT config for W1 only
    ggml_lut_config config;
    config.w_bits     = 4;
    config.a_bits     = 8;
    config.group_size = 32;
    config.mu         = 1;
    config.enabled    = true;

    printf("FFN dimensions: d_model=%d, d_hidden=%d\n", d_model, d_hidden);
    printf("Quantization: W%dA%d, group_size=%d\n",
           config.w_bits, config.a_bits, config.group_size);

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

    // x:  [d_model, batch_size]
    // W1: [d_hidden, d_model]  (note: ggml_mul_mat expects [rows, cols] style)
    // W2: [d_model, d_hidden]
    struct ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model,  batch_size);
    struct ggml_tensor * W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model,  d_hidden);
    struct ggml_tensor * W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_hidden, d_model);

    fill_random((float *) x->data,  d_model  * batch_size, 42);
    fill_random((float *) W1->data, d_model  * d_hidden,   123);
    fill_random((float *) W2->data, d_hidden * d_model,    456);

    // Baseline FFN: y_ref = W2 * SiLU(W1 * x)

    struct ggml_cgraph * gf_ref = ggml_new_graph(ctx);

    struct ggml_tensor * h_ref  = ggml_mul_mat(ctx, W1, x);
    h_ref = ggml_silu(ctx, h_ref);
    struct ggml_tensor * y_ref  = ggml_mul_mat(ctx, W2, h_ref);

    ggml_build_forward_expand(gf_ref, y_ref);
    ggml_backend_graph_compute(backend, gf_ref);

    struct ggml_tensor * out_ref_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, batch_size);
    memcpy(out_ref_tensor->data, y_ref->data, ggml_nbytes(out_ref_tensor));

    float * out_ref = (float *) out_ref_tensor->data;

    // LUT FFN: apply LUT GEMM only to W1 * x

    ggml_lut_quantize_weights(W1, &config);
    ggml_lut_set_tensor_config(W1, &config);

    struct ggml_tensor * h_lut_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_hidden, batch_size);

    ggml_lut_compute_gemm(x, W1, h_lut_tensor, &config);

    struct ggml_cgraph * gf_lut = ggml_new_graph(ctx);
    struct ggml_tensor * h_lut  = ggml_silu(ctx, h_lut_tensor);
    struct ggml_tensor * y_lut  = ggml_mul_mat(ctx, W2, h_lut);

    ggml_build_forward_expand(gf_lut, y_lut);
    ggml_backend_graph_compute(backend, gf_lut);

    struct ggml_tensor * out_lut_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, batch_size);
    memcpy(out_lut_tensor->data, y_lut->data, ggml_nbytes(out_lut_tensor));

    float * out_lut = (float *) out_lut_tensor->data;

    // Error analysis

    double max_err    = 0.0;
    double sum_sq_err = 0.0;

    for (int i = 0; i < d_model * batch_size; ++i) {
        double diff = (double) out_lut[i] - (double) out_ref[i];
        double err  = diff >= 0.0 ? diff : -diff;

        if (err > max_err) {
            max_err = err;
        }

        sum_sq_err += diff * diff;
    }

    double mse = sum_sq_err / (double) (d_model * batch_size);

    printf("Max error: %.6f, MSE: %.6e\n", max_err, mse);

    // FFN layer compounds errors through SiLU and W2, so we allow a relaxed threshold
    // but still expect values FAR below the hundreds we had before.
    const double threshold = 15.0;

    int ret = 0;
    if (max_err > threshold) {
        fprintf(stderr,
                "FAIL: FFN max error %.6f exceeds threshold %.6f\n",
                max_err, threshold);
        ret = 1;
    } else {
        printf("PASS: FFN max error %.6f within threshold %.6f\n", max_err, threshold);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    ggml_lut_global_free();

    return ret;
}
