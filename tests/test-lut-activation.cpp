#include "ggml.h"
#include "ggml-lut.h"
#include <cstdio>
#include <cmath>
#include <cstring>

// Baseline SiLU implementation
static float silu_baseline(float x) {
    return x / (1.0f + std::exp(-x));
}

// LUT-based SiLU
static float silu_lut(float x, const float * lut_table, int lut_size, float x_min, float x_max) {
    if (x < x_min) return silu_baseline(x_min);
    if (x > x_max) return silu_baseline(x_max);
    
    const float range = x_max - x_min;
    const float normalized = (x - x_min) / range;
    const float idx_f = normalized * (lut_size - 1);
    const int idx = (int)idx_f;
    const float frac = idx_f - idx;
    
    if (idx >= lut_size - 1) {
        return lut_table[lut_size - 1];
    }
    
    // Linear interpolation
    return lut_table[idx] * (1.0f - frac) + lut_table[idx + 1] * frac;
}

int main() {
    printf("Testing LUT-based SiLU activation...\n");
    
    const float x_min = -10.0f;
    const float x_max = 10.0f;
    const int lut_size = 1024;
    const int num_test_points = 10000;
    
    // Build LUT table
    float * lut_table = new float[lut_size];
    for (int i = 0; i < lut_size; ++i) {
        const float x = x_min + (x_max - x_min) * i / (lut_size - 1);
        lut_table[i] = silu_baseline(x);
    }
    
    printf("LUT table size: %d entries\n", lut_size);
    printf("Input range: [%.1f, %.1f]\n", x_min, x_max);
    printf("Test points: %d\n", num_test_points);
    
    // Test on random points
    float max_err = 0.0f;
    float mse = 0.0f;
    
    srand(42);
    for (int i = 0; i < num_test_points; ++i) {
        const float x = x_min + (x_max - x_min) * ((float)rand() / RAND_MAX);
        const float y_baseline = silu_baseline(x);
        const float y_lut = silu_lut(x, lut_table, lut_size, x_min, x_max);
        
        const float err = std::fabs(y_baseline - y_lut);
        max_err = std::max(max_err, err);
        mse += err * err;
    }
    
    mse /= num_test_points;
    
    printf("\n");
    printf("Error metrics:\n");
    printf("  Max absolute error: %.6e\n", max_err);
    printf("  Mean squared error: %.6e\n", mse);
    
    // Tolerance check
    const float max_err_threshold = 1e-3f;
    const float mse_threshold = 1e-6f;
    
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
    
    delete[] lut_table;
    
    return passed ? 0 : 1;
}