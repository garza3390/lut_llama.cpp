# ============================================================
# LUT-GEMM — Comandos del módulo experimental
# ============================================================

# DELETE bin folder
rm -rf build

# ── Build ────────────────────────────────────────────────────
# Build básico con LUT habilitado
cmake -B build -DGGML_LUT=ON -DGGML_CUDA=OFF
cmake --build build --config Release -j $(nproc)

# Build optimizado (recomendado para benchmarks justos)
cmake -B build -DGGML_LUT=ON -DGGML_CUDA=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build -j $(nproc)

# ── Tests ─────────────────────────────────────────────────────
# Ejecutar todos los tests LUT
ctest --test-dir build -R lut --output-on-failure

# Tests individuales
./build/bin/test-lut-gemm
./build/bin/test-lut-activation
./build/bin/test-lut-ffn-layer

# ── Benchmark rápido (config única, sin registro) ─────────────
# Salida CSV en pantalla únicamente
./build/bin/lut-bench-kernel

# ── Benchmark con registro histórico ─────────────────────────
# Guarda los resultados en llama2LutTestData/bench_log.csv
# --label  : etiqueta descriptiva del estado del código en este run
# --data-dir: directorio donde se acumula el historial (default: llama2LutTestData)
# --iters  : iteraciones por configuración (default: 10)

./build/bin/lut-bench-kernel \
    --label "baseline_inicial" \
    --data-dir "llama2LutTestData" \
    --iters 10

# Ejemplos de etiquetas útiles:
#   "baseline_inicial"      → primer estado del código
#   "cache_lut"             → tras añadir cache de tabla LUT
#   "buffers_preallocados"  → tras eliminar alloc/free por llamada
#   "avx2_vectorizado"      → tras añadir path AVX2
#   "q4_0_nativo"           → tras integrar cuantización Q4_0 de ggml

# ── Suite exhaustiva (deja la PC corriendo) ───────────────────
# Ejecuta 4 fases linealmente (~46 configs, ~30-60 min desatendido)
# Guarda todo en bench_log.csv con timestamp y label

./build/bin/lut-bench-suite \
    --label "post_optimizaciones" \
    --data-dir "llama2LutTestData" \
    --iters 15

# Opciones de la suite:
#   --label    Etiqueta del run (ej: "avx2_q4_nativo")
#   --data-dir Directorio de datos (default: llama2LutTestData)
#   --iters    Iteraciones por config para estabilidad estadística (default: 15)
#   --phases   Fases a correr, separadas por coma (default: 1,2,3,4)
#   --help     Ver descripción completa de cada fase

# Fases:
#   1 — Barrido de precisión W×A×group_size  (6 configs, config fija M=32 N=4096 K=4096)
#   2 — Capas reales de LLaMA 2 7B           (15 configs: Attn + FFN gate/down, M=1..32)
#   3 — Escalado de batch size M sweep       (16 configs: M=1,2,4,8,16,32,64,128)
#   4 — group_size sweep + footprint memoria  (9 configs: gs=32,64,128 × 3 matrices)

# Correr solo una fase (útil para reejecutar sin repetir todo):
./build/bin/lut-bench-suite --label "solo_llama" --phases 2
./build/bin/lut-bench-suite --label "solo_batch"  --phases 3

# ── Visualización de resultados ───────────────────────────────
# Instalar dependencias Python (una sola vez)
pip install pandas matplotlib seaborn

# Generar todas las gráficas (lee bench_log.csv del mismo directorio)
cd llama2LutTestData
python plot_results.py

# Opciones del script
python plot_results.py --csv bench_log.csv   # archivo específico
python plot_results.py --out graficas/       # directorio de salida (default: graficas/)
python plot_results.py --show                # mostrar en pantalla además de guardar

# ── Gráficas generadas ────────────────────────────────────────
# graficas/01_speedup_evolution.png      — evolución del speedup LUT/Q4 por iteración
# graficas/02_error_evolution.png        — evolución del error numérico por iteración
# graficas/03_tradeoff_table_error.png   — tamaño tabla LUT vs error (scatter)
# graficas/04_tradeoff_table_speedup.png — tamaño tabla LUT vs speedup
# graficas/05_heatmap_wbits_abits.png    — heatmap W×A bits: speedup / error / MSE
# graficas/06_speedup_vs_groupsize.png   — speedup vs group_size por tamaño de matriz
# graficas/07_bar_f32_q4_lut.png         — comparativa tiempos F32 / Q4_0 / LUT
# graficas/08_pareto_error_speedup.png   — frente de Pareto speedup ↔ error

# ── Flujo de trabajo versionado (recomendado) ────────────────
# Cada cambio importante en el código se prueba con una corrida versionada.
# El script run_version.sh:
#   - crea una carpeta test_vN_<descripcion> con N auto-incremental
#   - ejecuta kernel + suite con --data-dir apuntando a esa carpeta
#   - genera las 8 gráficas dentro de la carpeta
#   - guarda metadatos del run (git hash, host, timestamp)
#
# Flujo:
# 1. Modificar código y recompilar:
#      cmake --build build -j $(nproc)
# 2. Correr la versión:
#      chmod +x llama2LutTestData/run_version.sh    # solo primera vez
#      ./llama2LutTestData/run_version.sh "descripcion_del_cambio"
# 3. Revisar los resultados:
#      ls llama2LutTestData/test_v*/
#      cat llama2LutTestData/test_vN_descripcion/run_info.txt

# ── Flujo manual (avanzado) ──────────────────────────────────
# Si se requiere control fino sobre labels o iteraciones:
./build/bin/lut-bench-kernel \
    --label "v1_kernel_baseline" \
    --data-dir "llama2LutTestData/test_v1_baseline" \
    --iters 10

./build/bin/lut-bench-suite \
    --label "v1_suite_baseline" \
    --data-dir "llama2LutTestData/test_v1_baseline" \
    --iters 15

python llama2LutTestData/plot_results.py \
    --csv llama2LutTestData/test_v1_baseline/bench_log.csv \
    --out llama2LutTestData/test_v1_baseline/graficas
