"""
plot_results.py — Visualización del historial de benchmarks LUT-GEMM
=====================================================================

Lee bench_log.csv (generado por lut-bench-kernel) y produce un conjunto
de gráficas que documentan la evolución del rendimiento y la precisión
a lo largo de las iteraciones del proyecto.

Uso:
    python plot_results.py                      # lee ./bench_log.csv
    python plot_results.py --csv otra.csv       # archivo específico
    python plot_results.py --out graficas/      # directorio de salida
    python plot_results.py --show               # mostrar en pantalla además de guardar

Dependencias:
    pip install pandas matplotlib seaborn
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración visual global
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
FIGURE_DPI = 150

# ---------------------------------------------------------------------------
# Carga y preparación del CSV
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el archivo: {csv_path}")
        print("  Ejecuta primero: ./build/bin/lut-bench-kernel --label <nombre>")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Columnas derivadas útiles
    df["config"] = (
        "W" + df["w_bits"].astype(str) +
        "A" + df["a_bits"].astype(str) +
        "_g" + df["group_size"].astype(str)
    )
    df["matrix"] = (
        "M" + df["M"].astype(str) +
        "N" + df["N"].astype(str) +
        "K" + df["K"].astype(str)
    )
    df["lut_table_size_label"] = "2^" + (df["w_bits"] + df["a_bits"]).astype(str) + \
                                  " (" + df["lut_table_size"].astype(str) + ")"
    df["run_idx"] = df.groupby("label", sort=False).ngroup()

    # Orden cronológico de labels según primera aparición
    label_order = df.drop_duplicates("label").sort_values("timestamp")["label"].tolist()
    df["label"] = pd.Categorical(df["label"], categories=label_order, ordered=True)

    print(f"[OK] {len(df)} filas cargadas | {df['label'].nunique()} runs | "
          f"{df['config'].nunique()} configuraciones")
    return df


# ---------------------------------------------------------------------------
# Gráfica 1 — Evolución temporal del speedup LUT vs baseline F32
# ---------------------------------------------------------------------------

def plot_speedup_evolution(df: pd.DataFrame, out_dir: Path):
    """
    Líneas por configuración (W·A·group_size), eje X = label del run.
    Permite ver si cada mejora (cache LUT, AVX2, Q4_0 nativo...) sube el speedup.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, col, title in zip(
        axes,
        ["lut_speedup", "q4_speedup"],
        ["LUT speedup vs F32", "Q4_0 speedup vs F32"]
    ):
        pivot = df.groupby(["label", "config"])[col].mean().reset_index()
        sns.lineplot(
            data=pivot, x="label", y=col, hue="config",
            marker="o", ax=ax, linewidth=1.8
        )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, label="baseline (1×)")
        ax.set_title(title)
        ax.set_xlabel("Run / iteración")
        ax.set_ylabel("Speedup (×)")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Config", fontsize=8, title_fontsize=8)

    fig.suptitle("Evolución del speedup a lo largo de las iteraciones", fontsize=13)
    fig.tight_layout()
    path = out_dir / "01_speedup_evolution.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica 2 — Evolución del error máximo absoluto (LUT vs Q4_0)
# ---------------------------------------------------------------------------

def plot_error_evolution(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, col, title in zip(
        axes,
        ["lut_max_err", "q4_max_err"],
        ["Error máx. absoluto — LUT", "Error máx. absoluto — Q4_0"]
    ):
        pivot = df.groupby(["label", "config"])[col].mean().reset_index()
        sns.lineplot(
            data=pivot, x="label", y=col, hue="config",
            marker="s", ax=ax, linewidth=1.8
        )
        ax.set_title(title)
        ax.set_xlabel("Run / iteración")
        ax.set_ylabel("Max abs error")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Config", fontsize=8, title_fontsize=8)

    fig.suptitle("Evolución del error numérico (escala log)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "02_error_evolution.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica 3 — Tradeoff: tamaño de tabla LUT vs error (scatter Pareto)
# ---------------------------------------------------------------------------

def plot_tradeoff_table_error(df: pd.DataFrame, out_dir: Path):
    """
    Eje X = tamaño de la tabla (entradas = 2^(W+A)), eje Y = error máximo LUT.
    Cada punto es una (config, run), coloreado por la última iteración.
    """
    # Usar solo la última iteración para no saturar el plot
    last_label = df["label"].cat.categories[-1]
    last = df[df["label"] == last_label].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, y_col, y_label in zip(
        axes,
        ["lut_max_err", "lut_mse"],
        ["Max abs error (LUT)", "MSE (LUT)"]
    ):
        sns.scatterplot(
            data=last, x="lut_table_size", y=y_col,
            hue="config", size="group_size", sizes=(60, 200),
            ax=ax, alpha=0.85
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"$2^{{{int(round(x)).bit_length()-1}}}$\n({int(x)})"
        ))
        ax.set_xlabel("Entradas en la tabla LUT  (2^{W+A})")
        ax.set_ylabel(y_label)
        ax.set_title(f"Tradeoff: tamaño tabla vs {y_label}")
        ax.legend(title="Config", fontsize=8, title_fontsize=8)

    fig.suptitle(f"Tradeoff tamaño LUT ↔ precisión  [{last_label}]", fontsize=13)
    fig.tight_layout()
    path = out_dir / "03_tradeoff_table_error.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica 4 — Tradeoff: tamaño de tabla vs speedup (curva de eficiencia)
# ---------------------------------------------------------------------------

def plot_tradeoff_table_speedup(df: pd.DataFrame, out_dir: Path):
    last_label = df["label"].cat.categories[-1]
    last = df[df["label"] == last_label].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=last, x="lut_table_size", y="lut_speedup",
        hue="config", size="group_size", sizes=(60, 200),
        ax=ax, alpha=0.85
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, label="baseline (1×)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Entradas en la tabla LUT  (2^{W+A})")
    ax.set_ylabel("Speedup LUT vs F32 (×)")
    ax.set_title(f"Tradeoff: tamaño de tabla LUT ↔ speedup  [{last_label}]")
    ax.legend(title="Config", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    path = out_dir / "04_tradeoff_table_speedup.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica 5 — Heatmap: w_bits × a_bits → speedup y error (último run)
# ---------------------------------------------------------------------------

def plot_heatmaps_wbits_abits(df: pd.DataFrame, out_dir: Path):
    last_label = df["label"].cat.categories[-1]
    last = df[df["label"] == last_label].copy()

    # Agrupar por (w_bits, a_bits) promediando sobre matrices y group_size
    grp = last.groupby(["w_bits", "a_bits"]).agg(
        lut_speedup=("lut_speedup", "mean"),
        lut_max_err=("lut_max_err", "mean"),
        lut_mse=("lut_mse", "mean"),
    ).reset_index()

    metrics = [
        ("lut_speedup", "Speedup LUT (×)", "Blues"),
        ("lut_max_err", "Max abs error",   "Reds"),
        ("lut_mse",     "MSE",             "Oranges"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for ax, (col, title, cmap) in zip(axes, metrics):
        pivot = grp.pivot(index="w_bits", columns="a_bits", values=col)
        sns.heatmap(
            pivot, annot=True, fmt=".3g", cmap=cmap,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title(title)
        ax.set_xlabel("Bits activaciones (a_bits)")
        ax.set_ylabel("Bits pesos (w_bits)")

    fig.suptitle(f"Heatmap W×A bits  [{last_label}]", fontsize=13)
    fig.tight_layout()
    path = out_dir / "05_heatmap_wbits_abits.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica 6 — Speedup vs group_size para distintos M/N/K (último run)
# ---------------------------------------------------------------------------

def plot_speedup_vs_groupsize(df: pd.DataFrame, out_dir: Path):
    last_label = df["label"].cat.categories[-1]
    last = df[df["label"] == last_label].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=last, x="group_size", y="lut_speedup",
        hue="matrix", marker="o", ax=ax, linewidth=1.8
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("group_size")
    ax.set_ylabel("Speedup LUT vs F32 (×)")
    ax.set_title(f"Speedup LUT vs group_size por tamaño de matriz  [{last_label}]")
    ax.legend(title="Matriz (M×N×K)", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    path = out_dir / "06_speedup_vs_groupsize.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica 7 — Comparativa F32 / Q4_0 / LUT en barras (último run, por matriz)
# ---------------------------------------------------------------------------

def plot_bar_f32_q4_lut(df: pd.DataFrame, out_dir: Path):
    last_label = df["label"].cat.categories[-1]
    last = df[df["label"] == last_label].copy()

    # Solo la config W4A8 g32 como representativa (si existe)
    rep = last[(last["w_bits"] == 4) & (last["a_bits"] == 8) & (last["group_size"] == 32)]
    if rep.empty:
        rep = last  # fallback: todo el último run

    melt = rep[["matrix", "f32_ms", "q4_ms", "lut_ms"]].melt(
        id_vars="matrix", var_name="kernel", value_name="ms"
    )
    melt["kernel"] = melt["kernel"].map({"f32_ms": "F32", "q4_ms": "Q4_0", "lut_ms": "LUT"})

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melt, x="matrix", y="ms", hue="kernel", ax=ax)
    ax.set_xlabel("Tamaño de matriz")
    ax.set_ylabel("Tiempo (ms)")
    ax.set_title(f"Tiempo de ejecución: F32 vs Q4_0 vs LUT  [{last_label}] — W4A8 g32")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = out_dir / "07_bar_f32_q4_lut.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica 8 — Frente de Pareto error vs speedup (último run)
# ---------------------------------------------------------------------------

def plot_pareto_error_speedup(df: pd.DataFrame, out_dir: Path):
    last_label = df["label"].cat.categories[-1]
    last = df[df["label"] == last_label].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        last["lut_speedup"], last["lut_max_err"],
        c=last["lut_table_size"], cmap="viridis",
        s=80, alpha=0.85, edgecolors="k", linewidths=0.4
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Entradas LUT (2^{W+A})")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Speedup LUT vs F32 (×)")
    ax.set_ylabel("Error máx. absoluto LUT")
    ax.set_yscale("log")
    ax.set_title(f"Frente de Pareto: speedup ↔ error  [{last_label}]")

    # Anotar cada punto con su config
    for _, row in last.iterrows():
        ax.annotate(row["config"], (row["lut_speedup"], row["lut_max_err"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points", alpha=0.8)

    fig.tight_layout()
    path = out_dir / "08_pareto_error_speedup.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resumen estadístico en consola
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print("\n=== Resumen por run ===")
    summary = df.groupby("label").agg(
        runs=("M", "count"),
        lut_speedup_mean=("lut_speedup", "mean"),
        lut_speedup_max=("lut_speedup", "max"),
        lut_max_err_mean=("lut_max_err", "mean"),
        q4_speedup_mean=("q4_speedup", "mean"),
    ).round(4)
    print(summary.to_string())
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualización de benchmarks LUT-GEMM")
    parser.add_argument("--csv",  default="bench_log.csv",
                        help="Ruta al CSV de historial (default: bench_log.csv)")
    parser.add_argument("--out",  default="graficas",
                        help="Directorio de salida para las gráficas (default: graficas/)")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar cada gráfica en pantalla además de guardarla")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(str(csv_path))
    print_summary(df)

    print(f"\nGenerando gráficas en: {out_dir}\n")

    plots = [
        plot_speedup_evolution,
        plot_error_evolution,
        plot_tradeoff_table_error,
        plot_tradeoff_table_speedup,
        plot_heatmaps_wbits_abits,
        plot_speedup_vs_groupsize,
        plot_bar_f32_q4_lut,
        plot_pareto_error_speedup,
    ]

    for plot_fn in plots:
        try:
            plot_fn(df, out_dir)
        except Exception as e:
            print(f"  [WARN] {plot_fn.__name__} falló: {e}")

    if args.show:
        plt.show()

    print(f"\n[OK] {len(plots)} gráficas guardadas en {out_dir}/")


if __name__ == "__main__":
    main()
