#!/usr/bin/env bash
#
# run_version.sh — Ejecuta una corrida versionada de los benchmarks LUT
# y genera las gráficas asociadas dentro de una carpeta única.
#
# Cada invocación crea llama2LutTestData/test_vN_<descripcion>/ con:
#   bench_log.csv         resultados de kernel + suite en este run
#   graficas/             las 8 PNG generadas por plot_results.py
#   run_info.txt          metadatos del run (fecha, git hash, host)
#
# Uso:
#   ./llama2LutTestData/run_version.sh "descripcion_corta"
#
# Requisitos:
#   - build/bin/lut-bench-kernel y lut-bench-suite ya compilados
#   - python con pandas, matplotlib, seaborn (o ~/.venv-lut activable)
#
set -euo pipefail

DESCRIPTION="${1:-unnamed}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-incremento del número de versión (robusto a directorios inexistentes)
N=$(( $(find "$SCRIPT_DIR" -maxdepth 1 -type d -name 'test_v*' 2>/dev/null | wc -l) + 1 ))
VERSION_DIR="$SCRIPT_DIR/test_v${N}_${DESCRIPTION}"

if [ -d "$VERSION_DIR" ]; then
    echo "[run_version] El directorio $VERSION_DIR ya existe; use otro nombre" >&2
    exit 1
fi

mkdir -p "$VERSION_DIR"
echo "[run_version] Nueva versión: test_v${N}_${DESCRIPTION}"
echo "[run_version] Directorio    : $VERSION_DIR"

# Metadatos del run
{
    echo "version        : test_v${N}_${DESCRIPTION}"
    echo "timestamp      : $(date -Iseconds)"
    echo "host           : $(hostname)"
    echo "uname          : $(uname -a)"
    if git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
        echo "git_commit     : $(git -C "$REPO_ROOT" rev-parse HEAD)"
        echo "git_branch     : $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
        echo "git_dirty      : $(git -C "$REPO_ROOT" status --porcelain | wc -l) archivos modificados"
    fi
} > "$VERSION_DIR/run_info.txt"

cat "$VERSION_DIR/run_info.txt"

# Kernel bench
echo ""
echo "[run_version] === Ejecutando lut-bench-kernel ==="
"$REPO_ROOT/build/bin/lut-bench-kernel" \
    --label "v${N}_kernel_${DESCRIPTION}" \
    --data-dir "$VERSION_DIR" \
    --iters 10

# Suite bench
echo ""
echo "[run_version] === Ejecutando lut-bench-suite ==="
"$REPO_ROOT/build/bin/lut-bench-suite" \
    --label "v${N}_suite_${DESCRIPTION}" \
    --data-dir "$VERSION_DIR" \
    --iters 15

# SiLU bench (independiente del GEMM, escribe a silu_bench_log.csv)
if [ -x "$REPO_ROOT/build/bin/lut-bench-silu" ]; then
    echo ""
    echo "[run_version] === Ejecutando lut-bench-silu ==="
    "$REPO_ROOT/build/bin/lut-bench-silu" \
        --label "v${N}_silu_${DESCRIPTION}" \
        --data-dir "$VERSION_DIR" \
        --iters 30
fi

# Activar venv de Python si existe
if [ -f "$HOME/.venv-lut/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.venv-lut/bin/activate"
fi

# Visualización
echo ""
echo "[run_version] === Generando gráficas ==="
python3 "$SCRIPT_DIR/plot_results.py" \
    --csv "$VERSION_DIR/bench_log.csv" \
    --out "$VERSION_DIR/graficas"

echo ""
echo "[run_version] Terminado. Resultados en:"
echo "              $VERSION_DIR"
echo ""
echo "  CSV       : $VERSION_DIR/bench_log.csv"
echo "  Gráficas  : $VERSION_DIR/graficas/"
echo "  Metadatos : $VERSION_DIR/run_info.txt"

# Sincronización opcional con carpeta Windows (workflow WSL).
# Si la ruta existe, copia la carpeta completa de la versión para análisis.
# Puede sobrescribirse con la variable de entorno LUT_SYNC_DIR.
SYNC_DIR="${LUT_SYNC_DIR:-/mnt/c/Users/USUARIO/OneDrive/Escritorio/Llama2/llama2LutTestData}"
if [ -d "$(dirname "$SYNC_DIR")" ]; then
    mkdir -p "$SYNC_DIR"
    dest="$SYNC_DIR/$(basename "$VERSION_DIR")"
    rm -rf "$dest"
    cp -r "$VERSION_DIR" "$dest"
    echo ""
    echo "[run_version] Sincronizado a Windows:"
    echo "              $dest"
fi
