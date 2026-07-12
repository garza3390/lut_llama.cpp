/**
 * lut-bench-silu.cpp
 * =====================================================================
 * Benchmark de la función de activación SiLU implementada con LUT,
 * comparada contra una referencia escalar y contra el operador SiLU
 * nativo de ggml.
 *
 * Caminos evaluados sobre el mismo tensor de entrada F32:
 *   1) ref     : silu escalar con std::exp                  (referencia exacta)
 *   2) ggml    : ggml_silu sobre el grafo de cómputo        (camino nativo)
 *   3) lut8    : ggml_lut_silu_apply con tabla de 256 entradas, x_range=8
 *   4) lut12   : ggml_lut_silu_apply con tabla de 4096 entradas, x_range=8
 *
 * Salida: tiempos por iteración, error máximo absoluto contra la referencia
 * escalar y archivo CSV con historial. Una corrida cubre varios tamaños de
 * tensor representativos de capas FFN intermedias.
 * =====================================================================
 */

#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

struct SiluResult {
    int    n_elements;
    int    iters;
    double ref_ms;     // baseline escalar exacta
    double ggml_ms;    // ggml_silu nativo
    double lut8_ms;    // LUT 256 entradas, range=8
    double lut12_ms;   // LUT 4096 entradas, range=8
    float  ggml_max_err;
    float  lut8_max_err;
    float  lut12_max_err;
    float  ggml_mse;
    float  lut8_mse;
    float  lut12_mse;
};

static void fill_random(float * data, size_t size, unsigned int seed,
                        float lo = -8.0f, float hi = 8.0f) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        const float u = (float) rand() / (float) RAND_MAX;
        data[i] = lo + (hi - lo) * u;
    }
}

static float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

static double bench_ref(const float * in, float * out, int n, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < n; ++i) out[i] = silu_ref(in[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static double bench_ggml(ggml_backend_t bk, struct ggml_context * ctx,
                         struct ggml_tensor * X, int iters,
                         float * out_buf, int n) {
    // Warmup
    { auto * gf = ggml_new_graph(ctx);
      auto * Y  = ggml_silu(ctx, X);
      ggml_build_forward_expand(gf, Y);
      ggml_backend_graph_compute(bk, gf);
      std::memcpy(out_buf, Y->data, sizeof(float) * (size_t) n); }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        auto * gf = ggml_new_graph(ctx);
        auto * Y  = ggml_silu(ctx, X);
        ggml_build_forward_expand(gf, Y);
        ggml_backend_graph_compute(bk, gf);
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / iters;
}

static double bench_lut(struct ggml_tensor * X, struct ggml_tensor * Y,
                        int bits, float x_range, int iters) {
    // Warmup
    ggml_lut_silu_apply(X, Y, bits, x_range);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        ggml_lut_silu_apply(X, Y, bits, x_range);
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / iters;
}

static float max_abs_diff(const float * a, const float * b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float mean_sq_err(const float * a, const float * b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        s += d * d;
    }
    return (float) (s / (double) n);
}

static std::string now_iso() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

static const char * SILU_CSV_HEADER =
    "timestamp,label,n_elements,iters,"
    "ref_ms,ggml_ms,lut8_ms,lut12_ms,"
    "ggml_speedup,lut8_speedup,lut12_speedup,"
    "ggml_max_err,lut8_max_err,lut12_max_err,"
    "ggml_mse,lut8_mse,lut12_mse\n";

static void append_csv(const std::string & path,
                       const std::string & ts, const std::string & label,
                       const std::vector<SiluResult> & rows) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const bool needs_hdr = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!f) { fprintf(stderr, "[silu-bench] no se pudo abrir %s\n", path.c_str()); return; }
    if (needs_hdr) f << SILU_CSV_HEADER;

    for (const auto & r : rows) {
        const double ggml_speed  = (r.ggml_ms  > 0.0) ? r.ref_ms / r.ggml_ms  : 0.0;
        const double lut8_speed  = (r.lut8_ms  > 0.0) ? r.ref_ms / r.lut8_ms  : 0.0;
        const double lut12_speed = (r.lut12_ms > 0.0) ? r.ref_ms / r.lut12_ms : 0.0;
        f << ts << "," << label << ","
          << r.n_elements << "," << r.iters << ","
          << r.ref_ms << "," << r.ggml_ms << "," << r.lut8_ms << "," << r.lut12_ms << ","
          << ggml_speed << "," << lut8_speed << "," << lut12_speed << ","
          << r.ggml_max_err << "," << r.lut8_max_err << "," << r.lut12_max_err << ","
          << r.ggml_mse << "," << r.lut8_mse << "," << r.lut12_mse << "\n";
    }
}

int main(int argc, char ** argv) {
    std::string label    = "silu_default";
    std::string data_dir = "llama2LutTestData";
    int iters_default    = 30;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--label"    && i+1 < argc) { label    = argv[++i]; }
        else if (a == "--data-dir" && i+1 < argc) { data_dir = argv[++i]; }
        else if (a == "--iters"    && i+1 < argc) { iters_default = std::atoi(argv[++i]); }
        else if (a == "--help") {
            printf(
                "Uso: lut-bench-silu [--label <s>] [--data-dir <ruta>] [--iters <n>]\n"
                "Compara silu escalar / ggml_silu / LUT-8 / LUT-12 sobre tensores\n"
                "de varios tamaños representativos de capas FFN.\n"
            );
            return 0;
        }
    }

    ggml_lut_global_init();

    // Tamaños representativos (LLaMA 2 7B FFN intermedio = 11008,
    // multiplicado por batch típico)
    const int sizes[] = {
        4096,       // pequeño
        16384,      // medio
        65536,      // grande
        262144,     // muy grande
        1048576,    // ~1M
    };

    const std::string ts = now_iso();
    const std::string csv_path = data_dir + "/silu_bench_log.csv";

    printf("LUT SiLU benchmark\n");
    printf("Label    : %s\n", label.c_str());
    printf("Timestamp: %s\n", ts.c_str());
    printf("Data dir : %s\n", data_dir.c_str());
    printf("Iters    : %d\n\n", iters_default);

    printf("%-10s %-6s %10s %10s %10s %10s | %10s %10s %10s | "
           "%10s %10s %10s\n",
           "n", "iters", "ref_ms", "ggml_ms", "lut8_ms", "lut12_ms",
           "ggml_x", "lut8_x", "lut12_x",
           "ggml_err", "lut8_err", "lut12_err");
    printf("%s\n", std::string(140, '-').c_str());

    std::vector<SiluResult> all_results;

    for (int n : sizes) {
        // Para tensores grandes reducimos iters automáticamente para no
        // perder demasiado tiempo en la referencia escalar.
        int iters = iters_default;
        if (n > 65536)  iters = std::max(5, iters_default / 4);

        struct ggml_init_params params{ (size_t) 256 * 1024 * 1024, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_backend_t bk = ggml_backend_init_by_name("CPU", NULL);

        auto * X = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto * Y_lut = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

        fill_random((float*) X->data, (size_t) n, 42);

        std::vector<float> out_ref((size_t) n);
        std::vector<float> out_ggml((size_t) n);

        // 1) Referencia escalar exacta
        const double ref_ms = bench_ref((const float*) X->data, out_ref.data(), n, iters);

        // 2) ggml_silu (nativo)
        const double ggml_ms = bench_ggml(bk, ctx, X, iters, out_ggml.data(), n);

        // 3) LUT 8 bits, range 8
        const double lut8_ms = bench_lut(X, Y_lut, 8, 8.0f, iters);
        std::vector<float> out_lut8((size_t) n);
        std::memcpy(out_lut8.data(), Y_lut->data, sizeof(float) * (size_t) n);

        // 4) LUT 12 bits (4096 entradas), range 8
        const double lut12_ms = bench_lut(X, Y_lut, 12, 8.0f, iters);
        std::vector<float> out_lut12((size_t) n);
        std::memcpy(out_lut12.data(), Y_lut->data, sizeof(float) * (size_t) n);

        SiluResult r;
        r.n_elements    = n;
        r.iters         = iters;
        r.ref_ms        = ref_ms;
        r.ggml_ms       = ggml_ms;
        r.lut8_ms       = lut8_ms;
        r.lut12_ms      = lut12_ms;
        r.ggml_max_err  = max_abs_diff(out_ref.data(), out_ggml.data(),  n);
        r.lut8_max_err  = max_abs_diff(out_ref.data(), out_lut8.data(),  n);
        r.lut12_max_err = max_abs_diff(out_ref.data(), out_lut12.data(), n);
        r.ggml_mse      = mean_sq_err(out_ref.data(), out_ggml.data(),  n);
        r.lut8_mse      = mean_sq_err(out_ref.data(), out_lut8.data(),  n);
        r.lut12_mse     = mean_sq_err(out_ref.data(), out_lut12.data(), n);

        printf("%-10d %-6d %10.4f %10.4f %10.4f %10.4f | "
               "%10.3f %10.3f %10.3f | %.2e %.2e %.2e\n",
               n, iters,
               ref_ms, ggml_ms, lut8_ms, lut12_ms,
               (ref_ms > 0 ? ref_ms / ggml_ms  : 0.0),
               (ref_ms > 0 ? ref_ms / lut8_ms  : 0.0),
               (ref_ms > 0 ? ref_ms / lut12_ms : 0.0),
               (double) r.ggml_max_err,
               (double) r.lut8_max_err,
               (double) r.lut12_max_err);

        all_results.push_back(r);

        ggml_backend_free(bk);
        ggml_free(ctx);
    }

    append_csv(csv_path, ts, label, all_results);
    printf("\n[silu-bench] Resultados guardados en: %s\n", csv_path.c_str());

    ggml_lut_global_free();
    return 0;
}
