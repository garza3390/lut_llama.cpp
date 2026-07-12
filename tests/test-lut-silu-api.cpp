/**
 * test-lut-silu-api.cpp
 * =====================================================================
 * Test funcional de ggml_lut_silu_apply contra una referencia escalar
 * exacta. Verifica que el error máximo absoluto cumple los umbrales
 * esperados para distintas configuraciones de bits.
 * =====================================================================
 */

#include "ggml.h"
#include "ggml-lut.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

static float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

struct case_spec {
    int   bits;
    float x_range;
    float threshold;   // umbral de error máximo absoluto
};

static int run_case(const case_spec & c, int n) {
    struct ggml_init_params params{ (size_t) 64 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    auto * X = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    auto * Y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    float * x = (float *) X->data;
    srand(42);
    for (int i = 0; i < n; ++i) {
        const float u = (float) rand() / (float) RAND_MAX;
        // Mantener dentro del rango con margen para no saturar
        x[i] = -c.x_range * 0.95f + (1.9f * c.x_range * 0.95f) * u;
    }

    ggml_lut_silu_apply(X, Y, c.bits, c.x_range);

    const float * y = (const float *) Y->data;

    float max_err = 0.0f;
    double sse = 0.0;
    for (int i = 0; i < n; ++i) {
        const float ref = silu_ref(x[i]);
        const float d   = std::fabs(y[i] - ref);
        if (d > max_err) max_err = d;
        sse += (double) d * d;
    }
    const float mse = (float) (sse / (double) n);

    printf("  bits=%2d  x_range=%.1f  n=%d  max_err=%.3e  mse=%.3e  ",
           c.bits, (double) c.x_range, n, (double) max_err, (double) mse);

    const bool ok = max_err < c.threshold;
    printf("%s\n", ok ? "OK" : "FAIL");

    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main() {
    printf("test-lut-silu-api: validación de ggml_lut_silu_apply\n\n");

    ggml_lut_global_init();

    const case_spec cases[] = {
        // bits, x_range, error_threshold
        {  8, 8.0f,  5e-3f  },   // 256 entradas: ~3-4 mdec de error con interp lineal
        { 10, 8.0f,  5e-4f  },   // 1024 entradas
        { 12, 8.0f,  5e-5f  },   // 4096 entradas
        { 16, 8.0f,  1e-6f  },   // 65536 entradas
    };

    int fails = 0;
    for (const auto & c : cases) {
        fails += run_case(c, 8192);
    }

    ggml_lut_global_free();

    if (fails == 0) {
        printf("\nTODOS LOS CASOS APROBARON\n");
        return 0;
    }
    printf("\n%d caso(s) fallaron\n", fails);
    return 1;
}
