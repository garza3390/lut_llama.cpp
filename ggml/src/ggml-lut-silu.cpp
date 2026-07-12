#include "ggml-lut.h"

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// LUT-based SiLU activation
//
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// La tabla se construye una sola vez por configuración (bits, x_range) y
// queda en caché global. Se aplica saturación a [-x_range, +x_range] y se
// interpola linealmente entre entradas adyacentes para reducir el error de
// muestreo. Esta organización es directa de portar a Vitis HLS, donde la
// tabla cabría en uno o dos BRAM y la interpolación se mapearía a una
// suma con shift.
// ---------------------------------------------------------------------------

namespace {

inline float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

// Clave del caché: combina bits y un cuantizado fijo del rango. El bias
// 1e-3 evita colisiones por floats casi iguales.
struct silu_key {
    int   bits;
    int   range_q;
};

struct silu_key_hash {
    size_t operator()(const silu_key & k) const noexcept {
        return ((size_t) k.bits << 24) ^ (size_t) k.range_q;
    }
};

inline bool operator==(const silu_key & a, const silu_key & b) {
    return a.bits == b.bits && a.range_q == b.range_q;
}

struct silu_table {
    std::vector<float> values;  // 2^bits entradas con silu(x_idx)
    float              x_range;
    int                levels;  // 2^bits
};

static std::unordered_map<silu_key, silu_table, silu_key_hash> g_silu_cache;

const silu_table & get_or_build_silu_table(int bits, float x_range) {
    silu_key key{ bits, (int) std::round(x_range * 1000.0f) };
    auto it = g_silu_cache.find(key);
    if (it != g_silu_cache.end()) {
        return it->second;
    }

    silu_table tbl;
    tbl.levels  = 1 << bits;
    tbl.x_range = x_range;
    tbl.values.resize((size_t) tbl.levels);

    const float step = (2.0f * x_range) / (float) (tbl.levels - 1);
    for (int i = 0; i < tbl.levels; ++i) {
        const float x = -x_range + step * (float) i;
        tbl.values[(size_t) i] = silu_ref(x);
    }

    return g_silu_cache.emplace(key, std::move(tbl)).first->second;
}

}  // namespace

extern "C" void ggml_lut_silu_apply(
    const struct ggml_tensor * in,
    struct ggml_tensor       * out,
    int   bits,
    float x_range
) {
    GGML_ASSERT(in  != NULL);
    GGML_ASSERT(out != NULL);
    GGML_ASSERT(in->type  == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(in));
    GGML_ASSERT(ggml_is_contiguous(out));
    GGML_ASSERT(ggml_nelements(in) == ggml_nelements(out));
    GGML_ASSERT(bits >= 4 && bits <= 16 && "bits fuera de rango");
    GGML_ASSERT(x_range > 0.0f);

    const silu_table & tbl = get_or_build_silu_table(bits, x_range);
    const int    levels = tbl.levels;
    const float  range  = tbl.x_range;
    const float  inv_step = (float) (levels - 1) / (2.0f * range);
    const float * lut   = tbl.values.data();

    const int64_t n      = ggml_nelements(in);
    const float * in_d   = (const float *) in->data;
    float       * out_d  = (float       *) out->data;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        float x = in_d[i];

        // Saturación a [-range, +range]
        if (x < -range) x = -range;
        else if (x >  range) x =  range;

        // Mapeo a índice con interpolación lineal
        const float idx_f = (x + range) * inv_step;
        int idx0 = (int) idx_f;
        if (idx0 < 0)            idx0 = 0;
        else if (idx0 >= levels - 1) idx0 = levels - 2;
        const int   idx1 = idx0 + 1;
        const float frac = idx_f - (float) idx0;

        out_d[i] = lut[idx0] + (lut[idx1] - lut[idx0]) * frac;
    }
}
