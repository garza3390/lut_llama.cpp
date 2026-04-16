#include "ggml.h"
#include "ggml-lut.h"
#include "ggml-backend.h"

// Para cuantización Q4_0 nativa (headers internos de ggml, vía target_include_directories)
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"    // block_q4_0, QK4_0
#include "ggml-quants.h"    // quantize_q4_0

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

// ---------------------------------------------------------------------------
// Resultado de un run completo (una config de matriz)
// ---------------------------------------------------------------------------
struct BenchResult {
    int M, N, K;
    int w_bits, a_bits, group_size;
    int lut_table_size;    // 2^(w_bits + a_bits)
    double f32_ms;
    double q4_ms;
    double lut_ms;
    double q4_speedup;
    double lut_speedup;
    float  q4_max_err;
    float  lut_max_err;
    float  q4_mse;
    float  lut_mse;
};

// ---------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------
static void fill_random(float * data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        data[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }
}

static double benchmark_gemm_baseline(
    ggml_backend_t backend, struct ggml_context * ctx,
    struct ggml_tensor * A, struct ggml_tensor * B, int iterations
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * C  = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf, C);
        ggml_backend_graph_compute(backend, gf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
}

static double benchmark_gemm_q4_0(
    ggml_backend_t backend, struct ggml_context * ctx,
    struct ggml_tensor * A, struct ggml_tensor * B_q4, int iterations
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * C  = ggml_mul_mat(ctx, B_q4, A);
        ggml_build_forward_expand(gf, C);
        ggml_backend_graph_compute(backend, gf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
}

static double benchmark_gemm_lut(
    struct ggml_tensor * A, struct ggml_tensor * B, struct ggml_tensor * C,
    const struct ggml_lut_config * config, int iterations
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        ggml_lut_compute_gemm(A, B, C, config);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
}

static float compute_max_error(const float * ref, const float * test, size_t size) {
    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float e = std::fabs(ref[i] - test[i]);
        if (e > max_err) max_err = e;
    }
    return max_err;
}

static float compute_mse(const float * ref, const float * test, size_t size) {
    double acc = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double e = (double)(ref[i] - test[i]);
        acc += e * e;
    }
    return (float)(acc / (double) size);
}

// ---------------------------------------------------------------------------
// Timestamp ISO-8601 para el log
// ---------------------------------------------------------------------------
static std::string get_timestamp() {
    auto now   = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Guardar / agregar resultados al CSV de historial
// ---------------------------------------------------------------------------
static const char * CSV_HEADER =
    "timestamp,label,"
    "M,N,K,w_bits,a_bits,group_size,lut_table_size,"
    "f32_ms,q4_ms,lut_ms,"
    "q4_speedup,lut_speedup,"
    "q4_max_err,lut_max_err,"
    "q4_mse,lut_mse\n";

static void save_results(
    const std::string & data_dir,
    const std::string & label,
    const std::string & timestamp,
    const std::vector<BenchResult> & results
) {
    std::filesystem::create_directories(data_dir);
    const std::string csv_path = data_dir + "/bench_log.csv";

    // Si el archivo no existe, escribir cabecera
    bool needs_header = !std::filesystem::exists(csv_path);
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) {
        fprintf(stderr, "[LUT bench] No se pudo abrir %s para escritura\n", csv_path.c_str());
        return;
    }

    if (needs_header) {
        f << CSV_HEADER;
    }

    for (const auto & r : results) {
        f << timestamp << ","
          << label << ","
          << r.M << "," << r.N << "," << r.K << ","
          << r.w_bits << "," << r.a_bits << "," << r.group_size << ","
          << r.lut_table_size << ","
          << r.f32_ms << "," << r.q4_ms << "," << r.lut_ms << ","
          << r.q4_speedup << "," << r.lut_speedup << ","
          << r.q4_max_err << "," << r.lut_max_err << ","
          << r.q4_mse << "," << r.lut_mse << "\n";
    }
    printf("[LUT bench] Resultados guardados en: %s\n", csv_path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    // Argumentos opcionales: --label <str>  --data-dir <path>  --iters <n>
    std::string label    = "sin_etiqueta";
    std::string data_dir = "../llama2LutTestData";   // relativo al repo root
    int iterations = 10;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--label"    && i + 1 < argc) { label    = argv[++i]; }
        else if (std::string(argv[i]) == "--data-dir"  && i + 1 < argc) { data_dir = argv[++i]; }
        else if (std::string(argv[i]) == "--iters"     && i + 1 < argc) { iterations = std::atoi(argv[++i]); }
        else if (std::string(argv[i]) == "--help") {
            printf("Uso: lut-bench-kernel [--label <str>] [--data-dir <ruta>] [--iters <n>]\n");
            printf("  --label    Etiqueta descriptiva del run (ej: 'avx2_cache_lut')\n");
            printf("  --data-dir Directorio donde guardar bench_log.csv (default: ../llama2LutTestData)\n");
            printf("  --iters    Iteraciones por config (default: 10)\n");
            return 0;
        }
    }

    const std::string timestamp = get_timestamp();

    printf("LUT GEMM Kernel Benchmark\n");
    printf("Label    : %s\n", label.c_str());
    printf("Data dir : %s\n", data_dir.c_str());
    printf("Timestamp: %s\n", timestamp.c_str());
    printf("Iters    : %d\n", iterations);
    printf("Comparacion: F32 | Q4_0 nativo ggml | LUT-Q4_0\n");
    printf("=============================================================\n\n");

    ggml_lut_global_init();

    struct { int M, N, K, w_bits, a_bits, group_size; } configs[] = {
        {32,  4096, 4096, 4, 8, 32},
        {32,  4096, 4096, 4, 8, 64},
        {64,  2048, 2048, 4, 8, 32},
        {16,  8192, 4096, 4, 8, 32},
        // Barrido de w_bits para análisis de tradeoff tabla/error
        {32,  1024, 1024, 3, 8, 32},
        {32,  1024, 1024, 4, 8, 32},
        {32,  1024, 1024, 5, 8, 32},
        // Barrido de a_bits
        {32,  1024, 1024, 4, 4, 32},
        {32,  1024, 1024, 4, 8, 32},
        // Barrido de group_size
        {32,  2048, 2048, 4, 8, 32},
        {32,  2048, 2048, 4, 8, 64},
        {32,  2048, 2048, 4, 8, 128},
    };

    printf("op_type,m,n,k,w_bits,a_bits,group_size,lut_table_size,"
           "f32_ms,q4_ms,lut_ms,q4_speedup,lut_speedup,"
           "q4_max_err,lut_max_err,q4_mse,lut_mse\n");

    std::vector<BenchResult> all_results;

    for (const auto & cfg : configs) {
        if (cfg.K % QK4_0 != 0) {
            fprintf(stderr, "Saltando K=%d (no múltiplo de %d)\n", cfg.K, QK4_0);
            continue;
        }

        ggml_lut_config lut_cfg;
        lut_cfg.w_bits     = cfg.w_bits;
        lut_cfg.a_bits     = cfg.a_bits;
        lut_cfg.group_size = cfg.group_size;
        lut_cfg.mu         = 1;
        lut_cfg.enabled    = true;

        struct ggml_init_params params;
        params.mem_size   = (size_t) 1024 * 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) { fprintf(stderr, "Failed to init ggml context\n"); continue; }

        ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
        if (!backend) { fprintf(stderr, "Failed to init CPU backend\n"); ggml_free(ctx); continue; }

        struct ggml_tensor * A     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.M);
        struct ggml_tensor * B     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.K, cfg.N);
        struct ggml_tensor * C_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);

        fill_random((float *) A->data, (size_t) cfg.K * cfg.M, 42);
        fill_random((float *) B->data, (size_t) cfg.K * cfg.N, 123);

        // 1) F32 baseline
        double f32_ms = benchmark_gemm_baseline(backend, ctx, A, B, iterations);

        struct ggml_cgraph * gf_ref = ggml_new_graph(ctx);
        struct ggml_tensor * C_ref_t = ggml_mul_mat(ctx, B, A);
        ggml_build_forward_expand(gf_ref, C_ref_t);
        ggml_backend_graph_compute(backend, gf_ref);

        struct ggml_tensor * C_ref = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);
        std::memcpy(C_ref->data, C_ref_t->data, ggml_nbytes(C_ref));

        // 2) Q4_0 nativo ggml
        struct ggml_tensor * B_q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, cfg.K, cfg.N);
        quantize_q4_0((const float *) B->data, B_q4->data, cfg.N, cfg.K, NULL);

        double q4_ms = benchmark_gemm_q4_0(backend, ctx, A, B_q4, iterations);

        struct ggml_cgraph * gf_q4 = ggml_new_graph(ctx);
        struct ggml_tensor * C_q4_t = ggml_mul_mat(ctx, B_q4, A);
        ggml_build_forward_expand(gf_q4, C_q4_t);
        ggml_backend_graph_compute(backend, gf_q4);

        struct ggml_tensor * C_q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.N, cfg.M);
        std::memcpy(C_q4->data, C_q4_t->data, ggml_nbytes(C_q4));

        // 3) LUT-Q4_0
        ggml_lut_quantize_weights_q4_0(B, &lut_cfg);
        double lut_ms = benchmark_gemm_lut(A, B, C_lut, &lut_cfg, iterations);

        // Errores
        const size_t out_elems = (size_t) cfg.M * cfg.N;
        const float * ref_ptr = (const float *) C_ref->data;
        const float * q4_ptr  = (const float *) C_q4->data;
        const float * lut_ptr = (const float *) C_lut->data;

        BenchResult r;
        r.M = cfg.M; r.N = cfg.N; r.K = cfg.K;
        r.w_bits = cfg.w_bits; r.a_bits = cfg.a_bits; r.group_size = cfg.group_size;
        r.lut_table_size = (1 << cfg.w_bits) * (1 << cfg.a_bits);
        r.f32_ms     = f32_ms;
        r.q4_ms      = q4_ms;
        r.lut_ms     = lut_ms;
        r.q4_speedup  = (q4_ms  > 0.0) ? f32_ms / q4_ms  : 0.0;
        r.lut_speedup = (lut_ms > 0.0) ? f32_ms / lut_ms : 0.0;
        r.q4_max_err  = compute_max_error(ref_ptr, q4_ptr,  out_elems);
        r.lut_max_err = compute_max_error(ref_ptr, lut_ptr, out_elems);
        r.q4_mse      = compute_mse(ref_ptr, q4_ptr,  out_elems);
        r.lut_mse     = compute_mse(ref_ptr, lut_ptr, out_elems);

        printf("GEMM,%d,%d,%d,%d,%d,%d,%d,"
               "%.4f,%.4f,%.4f,%.4f,%.4f,%.6e,%.6e,%.6e,%.6e\n",
               r.M, r.N, r.K, r.w_bits, r.a_bits, r.group_size, r.lut_table_size,
               r.f32_ms, r.q4_ms, r.lut_ms,
               r.q4_speedup, r.lut_speedup,
               r.q4_max_err, r.lut_max_err,
               r.q4_mse, r.lut_mse);

        all_results.push_back(r);

        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    ggml_lut_global_free();

    // Guardar en historial CSV
    save_results(data_dir, label, timestamp, all_results);

    return 0;
}
