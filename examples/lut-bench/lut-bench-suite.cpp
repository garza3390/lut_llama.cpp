/**
 * lut-bench-suite.cpp
 * =====================================================================
 * Suite de benchmarks exhaustiva para el módulo LUT-GEMM.
 * Alineada con los objetivos del TFG:
 *   OE1/OE2  — medir y caracterizar rendimiento y precisión
 *   OE3      — encontrar la configuración óptima (W/A bits, group_size)
 *   OE5      — comparativa cuantitativa vs Q4_0 nativo y F32
 *   OE6      — identificar la configuración ideal para traspasar a Vitis HLS
 *
 * La suite corre completamente desatendida y acumula resultados en
 * llama2LutTestData/bench_log.csv usando el mismo formato que
 * lut-bench-kernel.
 *
 * Fases:
 *   Phase 1 — Barrido de precisión (W×A×group_size, tamaño medio fijo)
 *   Phase 2 — Dimensiones reales de LLaMA 2 7B (decode, M=1..32)
 *   Phase 3 — Escalado de batch size (M sweep, config óptima)
 *   Phase 4 — Análisis de group_size y footprint de memoria
 *
 * Uso:
 *   ./build/bin/lut-bench-suite
 *   ./build/bin/lut-bench-suite --label "post_avx2" --iters 20
 *   ./build/bin/lut-bench-suite --phases 1,2
 *   ./build/bin/lut-bench-suite --data-dir "/ruta"
 *   ./build/bin/lut-bench-suite --help
 * =====================================================================
 */

#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <set>

// ---------------------------------------------------------------------------
// Estructuras de datos
// ---------------------------------------------------------------------------

struct BenchResult {
    std::string timestamp;
    std::string label;          // phase label
    int M, N, K;
    int w_bits, a_bits, group_size;
    int lut_table_size;         // 2^(w_bits+a_bits)
    int lut_table_bytes;        // lut_table_size * sizeof(int32_t)
    double f32_ms;
    double q4_ms;
    double lut_ms;
    double q4_speedup;
    double lut_speedup;
    float  q4_max_err;
    float  lut_max_err;
    float  q4_mse;
    float  lut_mse;
    // Flags para análisis HLS
    bool   fits_l1_cache;       // table_bytes <= 32 KB
    bool   fits_single_bram36;  // table_bytes <= 32 KB
};

struct MatrixConfig {
    int M, N, K;
    const char * name;  // descripción legible
};

struct QuantConfig {
    int w_bits, a_bits, group_size;
};

// ---------------------------------------------------------------------------
// Utilidades generales
// ---------------------------------------------------------------------------

static void fill_random(float * data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        data[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }
}

static std::string now_iso() {
    auto t  = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

static float max_abs_error(const float * a, const float * b, size_t n) {
    float e = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

static float mse(const float * a, const float * b, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)(a[i] - b[i]);
        acc += d * d;
    }
    return (float)(acc / (double) n);
}

// ---------------------------------------------------------------------------
// Kernels de medición de tiempo
// ---------------------------------------------------------------------------

static double time_f32(ggml_backend_t bk, struct ggml_context * ctx,
                        struct ggml_tensor * A, struct ggml_tensor * B,
                        int iters) {
    // Warmup
    { auto * gf = ggml_new_graph(ctx);
      ggml_build_forward_expand(gf, ggml_mul_mat(ctx, B, A));
      ggml_backend_graph_compute(bk, gf); }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, B, A));
        ggml_backend_graph_compute(bk, gf);
    }
    return std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / iters;
}

static double time_q4(ggml_backend_t bk, struct ggml_context * ctx,
                       struct ggml_tensor * A, struct ggml_tensor * Bq4,
                       int iters) {
    { auto * gf = ggml_new_graph(ctx);
      ggml_build_forward_expand(gf, ggml_mul_mat(ctx, Bq4, A));
      ggml_backend_graph_compute(bk, gf); }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, Bq4, A));
        ggml_backend_graph_compute(bk, gf);
    }
    return std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / iters;
}

static double time_lut(struct ggml_tensor * A, struct ggml_tensor * B,
                        struct ggml_tensor * C, const ggml_lut_config * cfg,
                        int iters) {
    // Warmup (primera llamada construye el LUT si no está en cache)
    ggml_lut_compute_gemm(A, B, C, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_lut_compute_gemm(A, B, C, cfg);
    }
    return std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / iters;
}

// ---------------------------------------------------------------------------
// Función principal de un run: una combinación (matrix, quant_config)
// Devuelve BenchResult o {lut_ms==-1} si la config no es válida.
// ---------------------------------------------------------------------------

static BenchResult run_one(
    const MatrixConfig & mat,
    const QuantConfig  & qcfg,
    const std::string  & ts,
    const std::string  & label,
    int iters
) {
    BenchResult r{};
    r.timestamp  = ts;
    r.label      = label;
    r.M = mat.M; r.N = mat.N; r.K = mat.K;
    r.w_bits     = qcfg.w_bits;
    r.a_bits     = qcfg.a_bits;
    r.group_size = qcfg.group_size;
    r.lut_table_size  = (1 << qcfg.w_bits) * (1 << qcfg.a_bits);
    r.lut_table_bytes = r.lut_table_size * (int)sizeof(int32_t);
    r.fits_l1_cache      = r.lut_table_bytes <= 32 * 1024;
    r.fits_single_bram36 = r.lut_table_bytes <= 32 * 1024;

    // K debe ser múltiplo de QK4_0 (32) y de group_size
    if (mat.K % QK4_0 != 0 || mat.K % qcfg.group_size != 0) {
        r.lut_ms = -1.0;
        return r;
    }

    // ggml context (1 GB para matrices grandes)
    struct ggml_init_params params{ (size_t)1024*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { r.lut_ms = -1.0; return r; }

    ggml_backend_t bk = ggml_backend_init_by_name("CPU", NULL);
    if (!bk) { ggml_free(ctx); r.lut_ms = -1.0; return r; }

    // Tensores F32
    auto * A     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat.K, mat.M);
    auto * B     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat.K, mat.N);
    auto * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat.N, mat.M);

    fill_random((float*)A->data, (size_t)mat.K * mat.M, 42);
    fill_random((float*)B->data, (size_t)mat.K * mat.N, 123);

    // ── F32 baseline ──────────────────────────────────────────────────
    r.f32_ms = time_f32(bk, ctx, A, B, iters);

    // Capturar referencia F32
    { auto * gf = ggml_new_graph(ctx);
      auto * Ct  = ggml_mul_mat(ctx, B, A);
      ggml_build_forward_expand(gf, Ct);
      ggml_backend_graph_compute(bk, gf);
      auto * C_ref = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat.N, mat.M);
      std::memcpy(C_ref->data, Ct->data, ggml_nbytes(C_ref));

      // ── Q4_0 nativo ────────────────────────────────────────────────
      auto * Bq4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, mat.K, mat.N);
      quantize_q4_0((const float*)B->data, Bq4->data, mat.N, mat.K, NULL);
      r.q4_ms = time_q4(bk, ctx, A, Bq4, iters);

      { auto * gfq = ggml_new_graph(ctx);
        auto * Cq  = ggml_mul_mat(ctx, Bq4, A);
        ggml_build_forward_expand(gfq, Cq);
        ggml_backend_graph_compute(bk, gfq);
        auto * C_q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat.N, mat.M);
        std::memcpy(C_q4->data, Cq->data, ggml_nbytes(C_q4));
        r.q4_max_err = max_abs_error((float*)C_ref->data, (float*)C_q4->data,
                                     (size_t)mat.M * mat.N);
        r.q4_mse = mse((float*)C_ref->data, (float*)C_q4->data,
                       (size_t)mat.M * mat.N);
      }

      // ── LUT-Q4_0 ───────────────────────────────────────────────────
      ggml_lut_config lut_cfg{ qcfg.w_bits, qcfg.a_bits, qcfg.group_size, 1, true };
      ggml_lut_clear_weights();              // evitar aliasing con run anterior
      ggml_lut_quantize_weights_q4_0(B, &lut_cfg);
      r.lut_ms = time_lut(A, B, C_lut, &lut_cfg, iters);

      r.lut_max_err = max_abs_error((float*)C_ref->data, (float*)C_lut->data,
                                    (size_t)mat.M * mat.N);
      r.lut_mse = mse((float*)C_ref->data, (float*)C_lut->data,
                      (size_t)mat.M * mat.N);
    }

    r.q4_speedup  = (r.q4_ms  > 0.0) ? r.f32_ms / r.q4_ms  : 0.0;
    r.lut_speedup = (r.lut_ms > 0.0) ? r.f32_ms / r.lut_ms : 0.0;

    ggml_backend_free(bk);
    ggml_free(ctx);
    return r;
}

// ---------------------------------------------------------------------------
// CSV I/O
// ---------------------------------------------------------------------------

static const char * CSV_HEADER =
    "timestamp,label,"
    "M,N,K,w_bits,a_bits,group_size,lut_table_size,lut_table_bytes,"
    "f32_ms,q4_ms,lut_ms,"
    "q4_speedup,lut_speedup,"
    "q4_max_err,lut_max_err,"
    "q4_mse,lut_mse,"
    "fits_l1_cache,fits_single_bram36\n";

static void append_csv(const std::string & path, const std::vector<BenchResult> & rows) {
    bool needs_hdr = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!f) { fprintf(stderr, "[suite] No se pudo abrir %s\n", path.c_str()); return; }
    if (needs_hdr) f << CSV_HEADER;
    for (const auto & r : rows) {
        if (r.lut_ms < 0) continue;  // config inválida, no registrar
        f << r.timestamp << "," << r.label << ","
          << r.M << "," << r.N << "," << r.K << ","
          << r.w_bits << "," << r.a_bits << "," << r.group_size << ","
          << r.lut_table_size << "," << r.lut_table_bytes << ","
          << r.f32_ms << "," << r.q4_ms << "," << r.lut_ms << ","
          << r.q4_speedup << "," << r.lut_speedup << ","
          << r.q4_max_err << "," << r.lut_max_err << ","
          << r.q4_mse << "," << r.lut_mse << ","
          << (r.fits_l1_cache ? 1 : 0) << ","
          << (r.fits_single_bram36 ? 1 : 0) << "\n";
    }
}

// ---------------------------------------------------------------------------
// Impresión en consola de una fila
// ---------------------------------------------------------------------------

static void print_result(const BenchResult & r) {
    if (r.lut_ms < 0) { printf("  [SKIP] K no múltiplo de group_size/QK4_0\n"); return; }
    printf("  M=%-4d N=%-5d K=%-5d W%dA%d g%-3d | "
           "F32=%6.2fms  Q4=%6.2fms(%.2fx)  LUT=%6.2fms(%.2fx) | "
           "err_q4=%.2e err_lut=%.2e | table=%dB L1=%s BRAM=%s\n",
           r.M, r.N, r.K, r.w_bits, r.a_bits, r.group_size,
           r.f32_ms, r.q4_ms, r.q4_speedup, r.lut_ms, r.lut_speedup,
           r.q4_max_err, r.lut_max_err,
           r.lut_table_bytes,
           r.fits_l1_cache ? "SI" : "no",
           r.fits_single_bram36 ? "SI" : "no");
}

// ---------------------------------------------------------------------------
// Progreso global
// ---------------------------------------------------------------------------

static int g_total = 0, g_done = 0;

static void progress(const char * phase_name) {
    printf("\n[%d/%d] %s\n", g_done, g_total, phase_name);
}

// ---------------------------------------------------------------------------
// PHASE 1 — Barrido de precisión
// Objetivo: encontrar qué combinaciones W×A×group_size dan error aceptable
// Configuración fija: M=32, N=4096, K=4096
// ---------------------------------------------------------------------------

static std::vector<BenchResult> phase1(
    const std::string & base_label,
    const std::string & ts,
    int iters
) {
    progress("PHASE 1 — Barrido de precision (W x A x group_size)");

    // w_bits=3 con QK4_0=32: los nibbles Q4_0 son de 4 bits, así que w_bits=3
    // no es compatible con ggml_lut_quantize_weights_q4_0 (requiere w_bits=4).
    // Para w_bits=3 usamos ggml_lut_quantize_weights (esquema propio) como referencia.
    // NOTA: solo w_bits=4 es compatible con Q4_0 nativo; para 3 y 5 se usa el
    // esquema propio de Copilot.  Esto se documenta explícitamente en la label.
    const int w_arr[]  = { 4 };          // solo W4 es compatible con Q4_0 nativo
    const int a_arr[]  = { 4, 8 };
    const int gs_arr[] = { 32, 64, 128 };

    const MatrixConfig mat{ 32, 4096, 4096, "medium_gemm" };

    std::vector<BenchResult> results;

    for (int w : w_arr)
    for (int a : a_arr)
    for (int gs : gs_arr) {
        std::string lbl = base_label + "|phase1_W" + std::to_string(w) +
                          "A" + std::to_string(a) + "_g" + std::to_string(gs);
        printf("  W%dA%d g%d ...", w, a, gs); fflush(stdout);
        auto r = run_one(mat, {w, a, gs}, ts, lbl, iters);
        print_result(r);
        results.push_back(r);
        ++g_done;
    }
    return results;
}

// ---------------------------------------------------------------------------
// PHASE 2 — Dimensiones reales de LLaMA 2 7B
// Objetivo: medir latencia de cada tipo de capa en condiciones de decode
// Batch sizes representativos: M = 1 (decode puro), 4, 8, 16, 32
// ---------------------------------------------------------------------------

// Dimensiones de LLaMA 2 7B:
//   hidden_size = 4096
//   intermediate_size = 11008
//   head_dim = 128, num_heads = 32
// Shapes de pesos [K, N]:
//   Atención Q/K/V/O : K=4096, N=4096
//   FFN gate/up      : K=4096, N=11008
//   FFN down         : K=11008, N=4096

static std::vector<BenchResult> phase2(
    const std::string & base_label,
    const std::string & ts,
    int iters
) {
    progress("PHASE 2 — Capas reales LLaMA 2 7B (decode M=1..32)");

    const MatrixConfig layers[] = {
        { 1,  4096,  4096, "attn_qkv_M1"   },
        { 4,  4096,  4096, "attn_qkv_M4"   },
        { 8,  4096,  4096, "attn_qkv_M8"   },
        { 16, 4096,  4096, "attn_qkv_M16"  },
        { 32, 4096,  4096, "attn_qkv_M32"  },
        { 1,  11008, 4096, "ffn_gate_M1"   },
        { 4,  11008, 4096, "ffn_gate_M4"   },
        { 8,  11008, 4096, "ffn_gate_M8"   },
        { 16, 11008, 4096, "ffn_gate_M16"  },
        { 32, 11008, 4096, "ffn_gate_M32"  },
        { 1,  4096, 11008, "ffn_down_M1"   },
        { 4,  4096, 11008, "ffn_down_M4"   },
        { 8,  4096, 11008, "ffn_down_M8"   },
        { 16, 4096, 11008, "ffn_down_M16"  },
        { 32, 4096, 11008, "ffn_down_M32"  },
    };

    // Config más relevante para HLS: W4A8, group_size=32 (un bloque Q4_0 = 32 pesos)
    const QuantConfig best_cfg{ 4, 8, 32 };

    std::vector<BenchResult> results;

    for (const auto & mat : layers) {
        std::string lbl = base_label + "|phase2_" + mat.name;
        printf("  %-20s ...", mat.name); fflush(stdout);
        auto r = run_one(mat, best_cfg, ts, lbl, iters);
        print_result(r);
        results.push_back(r);
        ++g_done;
    }
    return results;
}

// ---------------------------------------------------------------------------
// PHASE 3 — Escalado de batch size (M sweep)
// Objetivo: ver cómo escala el speedup LUT a medida que M crece
// Relevante para HLS: el KV260 puede procesar lotes pequeños en FPGA
// ---------------------------------------------------------------------------

static std::vector<BenchResult> phase3(
    const std::string & base_label,
    const std::string & ts,
    int iters
) {
    progress("PHASE 3 — Escalado de batch size (M sweep, W4A8 g32)");

    const int M_arr[] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    const QuantConfig cfg{ 4, 8, 32 };

    // Dos tamaños de matriz representativos
    struct { int N, K; const char * tag; } shapes[] = {
        { 4096,  4096, "attn"   },
        { 11008, 4096, "ffn_up" },
    };

    std::vector<BenchResult> results;

    for (const auto & sh : shapes)
    for (int M : M_arr) {
        std::string name = std::string(sh.tag) + "_M" + std::to_string(M);
        MatrixConfig mat{ M, sh.N, sh.K, name.c_str() };
        std::string lbl = base_label + "|phase3_" + name;
        printf("  %-22s ...", name.c_str()); fflush(stdout);
        auto r = run_one(mat, cfg, ts, lbl, iters);
        print_result(r);
        results.push_back(r);
        ++g_done;
    }
    return results;
}

// ---------------------------------------------------------------------------
// PHASE 4 — Análisis de group_size y footprint de memoria
// Objetivo: encontrar el group_size óptimo (precisión vs overhead de escalas)
// y cuantificar el ahorro de memoria de LUT vs F32
// ---------------------------------------------------------------------------

static std::vector<BenchResult> phase4(
    const std::string & base_label,
    const std::string & ts,
    int iters
) {
    progress("PHASE 4 — group_size sweep + footprint de memoria");

    const int gs_arr[] = { 32, 64, 128 };
    const MatrixConfig matrices[] = {
        { 32, 4096,  4096, "medium"  },
        { 32, 11008, 4096, "ffn_up"  },
        { 32, 4096, 11008, "ffn_down"},
    };

    std::vector<BenchResult> results;

    for (const auto & mat : matrices)
    for (int gs : gs_arr) {
        QuantConfig cfg{ 4, 8, gs };
        std::string name = std::string(mat.name) + "_g" + std::to_string(gs);
        std::string lbl  = base_label + "|phase4_" + name;
        printf("  %-28s ...", name.c_str()); fflush(stdout);
        auto r = run_one(mat, cfg, ts, lbl, iters);

        // Imprimir también la comparativa de footprint en memoria
        if (r.lut_ms >= 0) {
            int weights_f32_kb = (mat.K * mat.N * 4) / 1024;
            int num_groups = (mat.K * mat.N + gs - 1) / gs;
            int weights_q4_kb  = (mat.K * mat.N / 2 + num_groups * 2) / 1024;
            printf("\n    Footprint pesos: F32=%dKB  Q4_0=%dKB  ratio=%.1fx",
                   weights_f32_kb, weights_q4_kb,
                   (float)weights_f32_kb / (float)weights_q4_kb);
        }
        print_result(r);
        results.push_back(r);
        ++g_done;
    }
    return results;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string label    = "suite_run";
    std::string data_dir = "llama2LutTestData";
    int iters = 15;
    std::set<int> run_phases = { 1, 2, 3, 4 };  // por defecto todas

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--label"    && i+1 < argc) { label    = argv[++i]; }
        else if (arg == "--data-dir"  && i+1 < argc) { data_dir = argv[++i]; }
        else if (arg == "--iters"     && i+1 < argc) { iters = std::atoi(argv[++i]); }
        else if (arg == "--phases"    && i+1 < argc) {
            run_phases.clear();
            std::istringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) run_phases.insert(std::atoi(tok.c_str()));
        }
        else if (arg == "--help") {
            printf(
                "lut-bench-suite — Suite exhaustiva LUT-GEMM\n\n"
                "  --label <str>     Etiqueta del run (default: suite_run)\n"
                "  --data-dir <ruta> Directorio de datos (default: llama2LutTestData)\n"
                "  --iters <n>       Iteraciones por config (default: 15)\n"
                "  --phases <n,n>    Fases a ejecutar, ej: 1,2  (default: 1,2,3,4)\n\n"
                "Fases:\n"
                "  1 — Barrido W x A x group_size  (18 configs)\n"
                "  2 — Dimensiones LLaMA 2 7B      (15 configs)\n"
                "  3 — M sweep / batch scaling     (16 configs)\n"
                "  4 — group_size + footprint      ( 9 configs)\n\n"
                "Resultados: <data-dir>/bench_log.csv\n"
                "Visualizar: cd <data-dir> && python plot_results.py\n"
            );
            return 0;
        }
    }

    // Contar total de configs
    if (run_phases.count(1)) g_total += 6;   // W4, A={4,8}, gs={32,64,128}
    if (run_phases.count(2)) g_total += 15;
    if (run_phases.count(3)) g_total += 16;
    if (run_phases.count(4)) g_total += 9;

    const std::string ts = now_iso();
    std::filesystem::create_directories(data_dir);
    const std::string csv_path = data_dir + "/bench_log.csv";

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║           LUT-GEMM Benchmark Suite — TFG                ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("Label     : %s\n", label.c_str());
    printf("Data dir  : %s\n", data_dir.c_str());
    printf("CSV       : %s\n", csv_path.c_str());
    printf("Timestamp : %s\n", ts.c_str());
    printf("Iters/cfg : %d\n", iters);
    printf("Fases     : ");
    for (int p : run_phases) printf("%d ", p);
    printf("\nTotal cfg : %d\n\n", g_total);

    printf("Kria KV260 — Capacidad BRAM: 144 × BRAM-36K = 4.6 MB\n");
    printf("  W4A8  tabla = %4d KB (1 BRAM-36K)\n", (16*256*4)/1024);
    printf("  W4A4  tabla = %4d  B\n", 16*16*4);
    printf("  W5A8  tabla = %4d KB (1 BRAM-36K)\n", (32*256*4)/1024);
    printf("  W5A16 tabla = %4d KB (4 BRAM-36K)\n", (32*65536*4)/1024);
    printf("\n");

    ggml_lut_global_init();

    std::vector<BenchResult> all_results;
    auto run_and_save = [&](std::vector<BenchResult> rows) {
        for (const auto & r : rows) all_results.push_back(r);
        append_csv(csv_path, rows);
        printf("  [guardado %zu filas]\n", rows.size());
    };

    if (run_phases.count(1)) run_and_save(phase1(label, ts, iters));
    if (run_phases.count(2)) run_and_save(phase2(label, ts, iters));
    if (run_phases.count(3)) run_and_save(phase3(label, ts, iters));
    if (run_phases.count(4)) run_and_save(phase4(label, ts, iters));

    ggml_lut_global_free();

    // ── Resumen final ────────────────────────────────────────────────
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("RESUMEN  —  %zu configs completadas\n", all_results.size());
    printf("══════════════════════════════════════════════════════════\n");

    // Mejor speedup LUT
    auto best = std::max_element(all_results.begin(), all_results.end(),
        [](const BenchResult & a, const BenchResult & b) {
            return a.lut_speedup < b.lut_speedup;
        });
    if (best != all_results.end() && best->lut_ms >= 0) {
        printf("Mejor speedup LUT : %.2fx  (M=%d N=%d K=%d W%dA%d g%d)\n",
               best->lut_speedup, best->M, best->N, best->K,
               best->w_bits, best->a_bits, best->group_size);
    }

    // Config con menor error LUT que aún supera a Q4_0 en speedup
    BenchResult * best_tradeoff = nullptr;
    for (auto & r : all_results) {
        if (r.lut_ms < 0) continue;
        if (r.lut_speedup > r.q4_speedup) {
            if (!best_tradeoff || r.lut_max_err < best_tradeoff->lut_max_err)
                best_tradeoff = &r;
        }
    }
    if (best_tradeoff) {
        printf("Mejor tradeoff    : err=%.2e  speedup=%.2fx  "
               "(W%dA%d g%d  table=%dB)\n",
               best_tradeoff->lut_max_err, best_tradeoff->lut_speedup,
               best_tradeoff->w_bits, best_tradeoff->a_bits,
               best_tradeoff->group_size, best_tradeoff->lut_table_bytes);
    }

    printf("\nResultados guardados en: %s\n", csv_path.c_str());
    printf("Visualizar           : cd %s && python plot_results.py\n\n", data_dir.c_str());

    return 0;
}
