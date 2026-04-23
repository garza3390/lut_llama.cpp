#include "ggml-lut.h"

// Incluir tipos de cuantización nativos de ggml (block_q4_0, QK4_0, etc.)
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"    // block_q4_0, QK4_0 — usar variante C++
#include "ggml-quants.h"    // quantize_row_q4_0_ref, quantize_q4_0
#include "ggml-impl.h"      // GGML_FP16_TO_FP32

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Estructura de metadatos de cuantización de pesos
// ---------------------------------------------------------------------------
struct ggml_lut_weight_data {
    uint8_t * w_q;       // Pesos cuantizados: índices LUT en [0, 2^W - 1]
    float   * scales;    // Escalas por grupo (symmetric per-group)
    int       num_groups;
};

// Tabla global tensor -> datos de cuantización
static std::unordered_map<const ggml_tensor *, ggml_lut_weight_data> g_weight_data;

// ---------------------------------------------------------------------------
// Rec 1: Cache de tablas LUT indexado por (w_bits, a_bits).
// La tabla solo depende de estas dos dimensiones y almacena productos enteros
// puros (q_w_centered * q_a_centered), por lo que se puede reutilizar entre
// todas las llamadas con la misma configuración de bits.
// ---------------------------------------------------------------------------
static std::unordered_map<uint64_t, std::vector<int32_t>> g_lut_cache;

static uint64_t lut_cache_key(int w_bits, int a_bits) {
    return ((uint64_t) w_bits << 32) | (uint64_t) a_bits;
}

// Retorna puntero a la tabla LUT del cache; la construye si no existe.
const int32_t * ggml_lut_get_or_build_table(int w_bits, int a_bits) {
    const uint64_t key = lut_cache_key(w_bits, a_bits);
    auto it = g_lut_cache.find(key);
    if (it != g_lut_cache.end()) {
        return it->second.data();
    }
    const int size = (1 << w_bits) * (1 << a_bits);
    std::vector<int32_t> tbl((size_t) size);
    ggml_lut_build_table(tbl.data(), w_bits, a_bits, 1.0f, 1.0f);
    g_lut_cache.emplace(key, std::move(tbl));
    return g_lut_cache[key].data();
}

// ---------------------------------------------------------------------------
// Cuantización de pesos — esquema propio (convención Q_0, per-group).
//
// Rango entero centrado: q_signed ∈ [-2^(w_bits-1), +2^(w_bits-1) - 1]
// Índice LUT: idx = q_signed + 2^(w_bits-1)  ∈ [0, 2^w_bits - 1]
// Escala: absmax / q_pos_max para preservar el extremo positivo sin clamp.
// ---------------------------------------------------------------------------
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

    const int w_bits    = config->w_bits;
    const int q_neg_max = 1 << (w_bits - 1);
    const int q_pos_max = q_neg_max - 1;

    const float * w_data = (const float *) weights->data;

    ggml_lut_weight_data qd;
    qd.w_q        = new uint8_t[total_size];
    qd.scales     = new float[num_groups];
    qd.num_groups = num_groups;

    for (int g = 0; g < num_groups; ++g) {
        const int start = g * group_size;
        const int end   = (start + group_size < total_size) ? start + group_size : total_size;

        float absmax = 0.0f;
        for (int i = start; i < end; ++i) {
            float av = w_data[i] >= 0.0f ? w_data[i] : -w_data[i];
            if (av > absmax) { absmax = av; }
        }

        const float scale_g = (absmax > 0.0f) ? (absmax / (float) q_pos_max) : 1.0f;
        qd.scales[g] = scale_g;

        for (int i = start; i < end; ++i) {
            float v          = w_data[i] / scale_g;
            int   q_centered = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);
            if (q_centered < -q_neg_max) { q_centered = -q_neg_max; }
            else if (q_centered > q_pos_max) { q_centered = q_pos_max; }
            qd.w_q[i] = (uint8_t) (q_centered + q_neg_max);
        }
    }

    g_weight_data[weights] = qd;
}

// ---------------------------------------------------------------------------
// Rec 4: Cuantización de pesos usando Q4_0 nativo de ggml.
//
// Los nibbles de Q4_0 están en el rango [0, 15] unsigned (almacenados como
// índice = q_signed + 8, donde q_signed ∈ [-8, 7]).  Esos son exactamente
// los mismos índices que el LUT necesita con w_bits=4.
// Así el LUT opera sobre los mismos índices que usa el modelo real.
//
// Restricciones: solo válido para w_bits = 4 y K múltiplo de QK4_0 (32).
// ---------------------------------------------------------------------------
void ggml_lut_quantize_weights_q4_0(
    struct ggml_tensor * weights,
    const struct ggml_lut_config * config
) {
    GGML_ASSERT(weights != NULL);
    GGML_ASSERT(config  != NULL);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(config->w_bits == 4 && "ggml_lut_quantize_weights_q4_0 requiere w_bits=4");

    const int K = (int) weights->ne[0];
    const int N = (int) weights->ne[1];
    GGML_ASSERT(K % QK4_0 == 0 && "K debe ser múltiplo de QK4_0 (32) para Q4_0");

    const float * w_f32 = (const float *) weights->data;

    // Número de bloques Q4_0 por fila
    const int blocks_per_row = K / QK4_0;
    const int num_groups     = N * blocks_per_row;

    // Reservar buffer temporal para los bloques Q4_0
    const size_t q4_bytes = (size_t) N * blocks_per_row * sizeof(block_q4_0);
    block_q4_0 * q4_blocks = (block_q4_0 *) malloc(q4_bytes);
    GGML_ASSERT(q4_blocks != NULL);

    // Cuantizar con Q4_0 nativo (una fila a la vez)
    for (int n = 0; n < N; ++n) {
        quantize_row_q4_0_ref(
            w_f32 + (size_t) n * K,
            q4_blocks + (size_t) n * blocks_per_row,
            K
        );
    }

    // Extraer índices uint8 y escalas desde los bloques Q4_0.
    // Cada block_q4_0:
    //   d    : ggml_half (fp16) — escala del bloque
    //   qs[] : 16 bytes = 32 nibbles, índices unsigned [0,15]
    //          nibble_low  = qs[i] & 0x0F  → elemento 2*i
    //          nibble_high = qs[i] >> 4    → elemento 2*i+1
    ggml_lut_weight_data qd;
    qd.w_q        = new uint8_t[(size_t) N * K];
    qd.scales     = new float[num_groups];
    qd.num_groups = num_groups;

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_0 & blk = q4_blocks[(size_t) n * blocks_per_row + b];
            const float scale_g    = GGML_FP16_TO_FP32(blk.d);
            qd.scales[n * blocks_per_row + b] = scale_g;

            const int base_k = b * QK4_0;
            for (int i = 0; i < QK4_0 / 2; ++i) {
                const uint8_t byte  = blk.qs[i];
                const uint8_t lo    = byte & 0x0F;   // índice [0,15]
                const uint8_t hi    = byte >> 4;      // índice [0,15]
                qd.w_q[(size_t) n * K + base_k + 2 * i    ] = lo;
                qd.w_q[(size_t) n * K + base_k + 2 * i + 1] = hi;
            }
        }
    }

    free(q4_blocks);
    g_weight_data[weights] = qd;
}

// ---------------------------------------------------------------------------
// Construcción de LUT 2D (almacena solo el producto entero centrado).
//
//   lut2d[w_idx * a_levels + a_idx] = q_w_centered * q_a_centered
//   donde q_centered = idx - zero_offset
//
// Convención Q_0 (compatible con Q4_0 / Q8_0 de ggml):
//   zero_offset = 2^(bits-1)
//   q_signed ∈ [-2^(bits-1), +2^(bits-1) - 1]
//
// Para w_bits=4 esto da q_w = idx - 8, exactamente el mapeo que usa Q4_0.
// Las escalas reales se aplican fuera del kernel, por lo que esta función
// las ignora.
// ---------------------------------------------------------------------------
void ggml_lut_build_table(
    int32_t * lut2d,
    int w_bits,
    int a_bits,
    float /*scale_w*/,
    float /*scale_x*/
) {
    GGML_ASSERT(lut2d  != NULL);
    GGML_ASSERT(w_bits > 0 && w_bits <= 8);
    GGML_ASSERT(a_bits > 0 && a_bits <= 16);

    const int w_levels = 1 << w_bits;
    const int a_levels = 1 << a_bits;
    const int w_zero   = 1 << (w_bits - 1);
    const int a_zero   = 1 << (a_bits - 1);

    for (int wi = 0; wi < w_levels; ++wi) {
        const int q_w = wi - w_zero;
        for (int ai = 0; ai < a_levels; ++ai) {
            const int q_a = ai - a_zero;
            lut2d[wi * a_levels + ai] = q_w * q_a;
        }
    }
}

// ---------------------------------------------------------------------------
// Acceso y gestión de datos de cuantización
// ---------------------------------------------------------------------------
const ggml_lut_weight_data * ggml_lut_get_weight_data(const struct ggml_tensor * tensor) {
    auto it = g_weight_data.find(tensor);
    return (it != g_weight_data.end()) ? &it->second : NULL;
}

void ggml_lut_free_quantized_weights(void) {
    for (auto & entry : g_weight_data) {
        delete[] entry.second.w_q;
        delete[] entry.second.scales;
    }
    g_weight_data.clear();
    g_lut_cache.clear();
}

// Limpia solo la tabla de pesos cuantizados, sin tocar el cache LUT ni los
// buffers de activaciones.  Útil entre fases de benchmark cuando los
// contextos ggml se recrean y los punteros de tensores pueden reutilizar
// la misma dirección de memoria.
void ggml_lut_clear_weights(void) {
    for (auto & entry : g_weight_data) {
        delete[] entry.second.w_q;
        delete[] entry.second.scales;
    }
    g_weight_data.clear();
}
