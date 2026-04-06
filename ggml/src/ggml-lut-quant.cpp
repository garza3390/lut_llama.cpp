#include "ggml-lut.h"
#include <cmath>
#include <cstring>
#include <unordered_map>

// Estructura de metadatos de cuantización de pesos (debe coincidir con ggml-lut-gemm.cpp)
struct ggml_lut_weight_data {
    uint8_t * w_q;      // Pesos cuantizados: índices LUT en [0, 2^W - 1]
    float   * scales;   // Escalas por grupo (symmetric per-group)
    int       num_groups;
};

// Tabla global tensor -> datos de cuantización
static std::unordered_map<const ggml_tensor *, ggml_lut_weight_data> g_weight_data;

// Cuantización de pesos (SIMÉTRICA SIGNED con zero-offset para LUT 2D)
//   Rango real por grupo: [-absmax_g, +absmax_g]
//   Rango entero centrado: q_centered ∈ [-qmax, +qmax],
//   donde qmax = (2^(w_bits) - 1)/2
//   Índice LUT: idx = q_centered + qmax  ∈ [0, 2^w_bits - 1]
//   w_real ≈ q_centered * scale_g
//   con scale_g = absmax_g / qmax
void ggml_lut_quantize_weights(
    struct ggml_tensor * weights,
    const struct ggml_lut_config * config
) {
    GGML_ASSERT(weights != NULL);
    GGML_ASSERT(config  != NULL);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weights));

    const int K = (int) weights->ne[0];
    const int N = (int) weights->ne[1];
    const int total_size = K * N;

    const int group_size = config->group_size;
    const int num_groups = (total_size + group_size - 1) / group_size;

    const int w_bits   = config->w_bits;
    const int w_levels = 1 << w_bits;               // 2^W
    const int qmax     = (w_levels - 1) / 2;        // p.ej. 7 para W=4

    const float * w_data = (const float *) weights->data;

    // Reservar espacio para datos cuantizados
    ggml_lut_weight_data qd;
    qd.w_q        = new uint8_t[total_size];
    qd.scales     = new float[num_groups];
    qd.num_groups = num_groups;

    for (int g = 0; g < num_groups; ++g) {
        const int start = g * group_size;
        const int end   = (start + group_size < total_size) ? start + group_size : total_size;

        // absmax del grupo
        float absmax = 0.0f;
        for (int i = start; i < end; ++i) {
            float v  = w_data[i];
            float av = v >= 0.0f ? v : -v;
            if (av > absmax) {
                absmax = av;
            }
        }

        // Evitar división por cero si el grupo es todo ceros
        const float scale_g = (absmax > 0.0f) ? (absmax / (float) qmax) : 1.0f;
        qd.scales[g] = scale_g;

        for (int i = start; i < end; ++i) {
            float v = w_data[i] / scale_g;  // valor en espacio entero "ideal"
            int   q_centered = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);

            if (q_centered < -qmax) {
                q_centered = -qmax;
            } else if (q_centered > qmax) {
                q_centered = qmax;
            }

            const int idx = q_centered + qmax; // índice LUT
            qd.w_q[i] = (uint8_t) idx;
        }
    }

    // Guardar en tabla global
    g_weight_data[weights] = qd;
}

// Construcción de LUT 2D
//   LUT almacena SOLO el producto entero:
//       lut2d[w_idx, a_idx] = q_w_centered * q_a_centered
//   donde:
//       q_w_centered = w_idx - w_zero
//       q_a_centered = a_idx - a_zero
//   El reescaleo usando scale_w[g] y scale_a[m] se hace en el kernel.
void ggml_lut_build_table(
    int32_t * lut2d,
    int w_bits,
    int a_bits,
    float /*scale_w*/,
    float /*scale_x*/
) {
    GGML_ASSERT(lut2d != NULL);
    GGML_ASSERT(w_bits > 0 && w_bits <= 8);
    GGML_ASSERT(a_bits > 0 && a_bits <= 16);

    const int w_levels = 1 << w_bits;
    const int a_levels = 1 << a_bits;

    const int w_zero = (w_levels - 1) / 2;
    const int a_zero = (a_levels - 1) / 2;

    for (int wi = 0; wi < w_levels; ++wi) {
        const int q_w = wi - w_zero;

        for (int ai = 0; ai < a_levels; ++ai) {
            const int q_a = ai - a_zero;
            lut2d[wi * a_levels + ai] = q_w * q_a;  // producto entero puro
        }
    }
}

// Acceso a datos de pesos cuantizados
const ggml_lut_weight_data * ggml_lut_get_weight_data(const struct ggml_tensor * tensor) {
    auto it = g_weight_data.find(tensor);
    if (it == g_weight_data.end()) {
        return NULL;
    }
    return &it->second;
}

// Liberar memoria global de cuantización
void ggml_lut_free_quantized_weights(void) {
    for (auto & entry : g_weight_data) {
        delete[] entry.second.w_q;
        delete[] entry.second.scales;
    }
    g_weight_data.clear();
}
