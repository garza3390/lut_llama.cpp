"""
plot_results.py — Visualización del historial de benchmarks LUT-GEMM.

Lee bench_log.csv y produce ocho gráficas estandarizadas que documentan
la evolución del rendimiento y la precisión a lo largo de las iteraciones.

Uso:
    python plot_results.py
    python plot_results.py --csv ruta/bench_log.csv
    python plot_results.py --out graficas/
    python plot_results.py --show
    python plot_results.py --run "v2_suite_..."        # filtrar un run

Dependencias:
    pip install pandas matplotlib seaborn
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuración visual global
# ---------------------------------------------------------------------------
sns.set_theme(
    style="whitegrid",
    palette="mako",
    context="notebook",
    font_scale=1.05,
)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.alpha": 0.35,
})

FIGSIZE_WIDE = (12, 5.5)
FIGSIZE_SQR  = (8, 5.5)


# ---------------------------------------------------------------------------
# Helpers de etiquetas
# ---------------------------------------------------------------------------
_PHASE_RE   = re.compile(r"^phase\d+_")
_RUN_PREFIX = re.compile(r"^v\d+_(kernel|suite)_")


def short_label(full: str) -> str:
    """Reduce un label largo tipo 'vN_suite_desc|phaseK_XYZ' a 'XYZ'."""
    s = full.split("|", 1)[-1]
    s = _PHASE_RE.sub("", s)
    s = _RUN_PREFIX.sub("", s)
    return s


def run_of(full: str) -> str:
    """Extrae solo el prefijo del run (antes de '|')."""
    return full.split("|", 1)[0]


def matrix_tag(row) -> str:
    return f"M{row['M']}·N{row['N']}·K{row['K']}"


def config_tag(row) -> str:
    return f"W{row['w_bits']}A{row['a_bits']} g{row['group_size']}"


def legend_outside(ax, title=None, ncol=1):
    """Coloca la leyenda a la derecha del axis, fuera del área de ploteo."""
    ax.legend(
        title=title, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0, ncol=ncol,
    )


# ---------------------------------------------------------------------------
# Carga y preparación del CSV
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el archivo: {csv_path}", file=sys.stderr)
        print("  Ejecuta primero ./build/bin/lut-bench-kernel o la suite.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    df["config"]      = df.apply(config_tag, axis=1)
    df["matrix"]      = df.apply(matrix_tag, axis=1)
    df["short_label"] = df["label"].map(short_label)
    df["run"]         = df["label"].map(run_of)

    run_order = (
        df.drop_duplicates("run")
          .sort_values("timestamp")["run"].tolist()
    )
    df["run"] = pd.Categorical(df["run"], categories=run_order, ordered=True)

    print(f"[OK] {len(df)} filas  |  runs: {df['run'].nunique()}  "
          f"|  configs: {df['config'].nunique()}  "
          f"|  matrices: {df['matrix'].nunique()}")
    return df


def latest_run(df: pd.DataFrame):
    """Devuelve (subset, nombre_run) del run más reciente por timestamp."""
    latest = df.sort_values("timestamp")["run"].iloc[-1]
    return df[df["run"] == latest].copy(), str(latest)


# ---------------------------------------------------------------------------
# Gráfica 1 — Evolución del speedup entre iteraciones
# ---------------------------------------------------------------------------

def plot_speedup_evolution(df: pd.DataFrame, out_dir: Path):
    agg = df.groupby(["run", "config"], observed=True).agg(
        lut_speedup=("lut_speedup", "mean"),
        q4_speedup=("q4_speedup", "mean"),
    ).reset_index()

    if agg["run"].nunique() < 2:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQR)
        ax.text(0.5, 0.5,
                "Se requiere más de un run para graficar evolución temporal.\n"
                "Corre otra versión del benchmark y re-ejecuta el script.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#555555")
        ax.axis("off")
        path = out_dir / "01_speedup_evolution.png"
        fig.savefig(path); plt.close(fig)
        print(f"  -> {path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    palette = sns.color_palette("mako", n_colors=max(agg["config"].nunique(), 3))

    for ax, col, title in zip(
        axes, ["lut_speedup", "q4_speedup"],
        ["Speedup LUT vs F32", "Speedup Q4_0 vs F32"]
    ):
        sns.lineplot(data=agg, x="run", y=col, hue="config",
                     marker="o", ax=ax, linewidth=1.6, palette=palette)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9,
                   label="paridad F32")
        ax.set_title(title)
        ax.set_xlabel("Iteración (run)")
        ax.set_ylabel("Speedup (×)")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Config",
                    loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0)

    fig.suptitle("Evolución del speedup a lo largo de iteraciones", y=1.02)
    path = out_dir / "01_speedup_evolution.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Gráfica 2 — Evolución del error numérico entre iteraciones
# ---------------------------------------------------------------------------

def plot_error_evolution(df: pd.DataFrame, out_dir: Path):
    agg = df.groupby(["run", "config"], observed=True).agg(
        lut_max_err=("lut_max_err", "mean"),
        q4_max_err=("q4_max_err", "mean"),
    ).reset_index()

    if agg["run"].nunique() < 2:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQR)
        ax.text(0.5, 0.5,
                "Se requiere más de un run para graficar evolución temporal.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#555555")
        ax.axis("off")
        path = out_dir / "02_error_evolution.png"
        fig.savefig(path); plt.close(fig)
        print(f"  -> {path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    palette = sns.color_palette("flare", n_colors=max(agg["config"].nunique(), 3))

    for ax, col, title in zip(
        axes, ["lut_max_err", "q4_max_err"],
        ["Error máx. absoluto  —  LUT", "Error máx. absoluto  —  Q4_0"]
    ):
        sns.lineplot(data=agg, x="run", y=col, hue="config",
                     marker="s", ax=ax, linewidth=1.6, palette=palette)
        ax.set_title(title)
        ax.set_xlabel("Iteración (run)")
        ax.set_ylabel("Max |error|")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Config",
                    loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0)

    fig.suptitle("Evolución del error numérico (escala log)", y=1.02)
    path = out_dir / "02_error_evolution.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Gráfica 3 — Tamaño de tabla vs error
# ---------------------------------------------------------------------------

def plot_tradeoff_table_error(df: pd.DataFrame, out_dir: Path):
    last, run_name = latest_run(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, ycol, ylabel in zip(
        axes,
        ["lut_max_err", "lut_mse"],
        ["Max |error| LUT", "MSE LUT"]
    ):
        sns.scatterplot(
            data=last, x="lut_table_size", y=ycol,
            hue="config", size="group_size", sizes=(55, 180),
            ax=ax, alpha=0.9, edgecolor="#333",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Entradas LUT ($2^{W+A}$)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Tradeoff tamaño LUT ↔ {ylabel}")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x)}")
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                    fontsize=8, title_fontsize=9)

    fig.suptitle(f"Tamaño de LUT vs precisión  —  run {run_name}", y=1.02)
    path = out_dir / "03_tradeoff_table_error.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Gráfica 4 — Tamaño de tabla vs speedup
# ---------------------------------------------------------------------------

def plot_tradeoff_table_speedup(df: pd.DataFrame, out_dir: Path):
    last, run_name = latest_run(df)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.scatterplot(
        data=last, x="lut_table_size", y="lut_speedup",
        hue="config", size="group_size", sizes=(55, 180),
        ax=ax, alpha=0.9, edgecolor="#333",
    )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9,
               label="paridad F32")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Entradas LUT ($2^{W+A}$)")
    ax.set_ylabel("Speedup LUT vs F32 (×)")
    ax.set_title(f"Tamaño de LUT vs speedup  —  run {run_name}")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              fontsize=8, title_fontsize=9)

    path = out_dir / "04_tradeoff_table_speedup.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Gráfica 5 — Heatmaps W × A
# ---------------------------------------------------------------------------

def plot_heatmaps_wbits_abits(df: pd.DataFrame, out_dir: Path):
    last, run_name = latest_run(df)

    grp = last.groupby(["w_bits", "a_bits"]).agg(
        lut_speedup=("lut_speedup", "mean"),
        lut_max_err=("lut_max_err", "mean"),
        lut_mse=("lut_mse", "mean"),
    ).reset_index()

    metrics = [
        ("lut_speedup", "Speedup LUT (×)",     "crest"),
        ("lut_max_err", "Error máx. absoluto", "rocket_r"),
        ("lut_mse",     "MSE",                 "flare"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    for ax, (col, title, cmap) in zip(axes, metrics):
        pivot = grp.pivot(index="w_bits", columns="a_bits", values=col)
        sns.heatmap(
            pivot, annot=True, fmt=".3g", cmap=cmap,
            linewidths=0.4, linecolor="white", ax=ax,
            cbar_kws={"shrink": 0.8, "pad": 0.02},
            annot_kws={"size": 10},
        )
        ax.set_title(title)
        ax.set_xlabel("Bits activación")
        ax.set_ylabel("Bits peso")

    fig.suptitle(f"Heatmap W×A bits  —  run {run_name}", y=1.03)
    path = out_dir / "05_heatmap_wbits_abits.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Gráfica 6 — Speedup vs group_size por matriz
# ---------------------------------------------------------------------------

def plot_speedup_vs_groupsize(df: pd.DataFrame, out_dir: Path):
    last, run_name = latest_run(df)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(
        data=last, x="group_size", y="lut_speedup",
        hue="matrix", marker="o", ax=ax, linewidth=1.8,
        palette=sns.color_palette("mako",
                                  n_colors=max(last["matrix"].nunique(), 3)),
    )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xlabel("group_size")
    ax.set_ylabel("Speedup LUT vs F32 (×)")
    ax.set_title(f"Speedup LUT según tamaño de grupo  —  run {run_name}")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Matriz (M·N·K)",
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=8, title_fontsize=9)

    path = out_dir / "06_speedup_vs_groupsize.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Gráfica 7 — Barras F32 / Q4_0 / LUT por matriz
# ---------------------------------------------------------------------------

def plot_bar_f32_q4_lut(df: pd.DataFrame, out_dir: Path):
    last, run_name = latest_run(df)

    rep = last[
        (last["w_bits"] == 4) & (last["a_bits"] == 8) & (last["group_size"] == 32)
    ].copy()
    if rep.empty:
        rep = last.copy()

    rep = rep.drop_duplicates(subset="matrix", keep="first")

    rep = rep.assign(
        mat_label=lambda d: d.apply(
            lambda r: f"M={r['M']}\nN={r['N']}\nK={r['K']}", axis=1)
    ).sort_values(["K", "N", "M"])

    melt = rep[["mat_label", "f32_ms", "q4_ms", "lut_ms"]].melt(
        id_vars="mat_label", var_name="kernel", value_name="ms"
    )
    melt["kernel"] = melt["kernel"].map(
        {"f32_ms": "F32", "q4_ms": "Q4_0", "lut_ms": "LUT"}
    )

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(rep)), 6))
    palette = {"F32": "#3e6fb8", "Q4_0": "#e08a3b", "LUT": "#3f9d4a"}
    sns.barplot(data=melt, x="mat_label", y="ms", hue="kernel",
                ax=ax, palette=palette, edgecolor="#222", linewidth=0.4)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=7,
                     padding=2, color="#333")

    ax.set_xlabel("Dimensión (M, N, K)")
    ax.set_ylabel("Tiempo (ms, escala log)")
    ax.set_title(f"F32 vs Q4_0 vs LUT  —  W4A8 g32  —  run {run_name}")
    ax.set_yscale("log")
    ax.legend(title="Kernel", loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    path = out_dir / "07_bar_f32_q4_lut.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Gráfica 8 — Frente de Pareto speedup ↔ error
# ---------------------------------------------------------------------------

def plot_pareto_error_speedup(df: pd.DataFrame, out_dir: Path):
    last, run_name = latest_run(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        last["lut_speedup"], last["lut_max_err"],
        c=last["lut_table_size"], cmap="mako",
        s=70, alpha=0.85, edgecolors="#222", linewidths=0.4,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Entradas LUT ($2^{W+A}$)")
    ax.axvline(1.0, color="#777777", linestyle="--", linewidth=0.9,
               label="paridad F32")
    ax.set_xlabel("Speedup LUT vs F32 (×)")
    ax.set_ylabel("Error máx. absoluto LUT")
    ax.set_yscale("log")
    ax.set_title(f"Frente de Pareto  —  run {run_name}")

    # Solo anotar los puntos de interés para no saturar
    top_speed = last.nlargest(3, "lut_speedup")
    low_err   = last.nsmallest(3, "lut_max_err")
    to_annotate = pd.concat([top_speed, low_err]).drop_duplicates(
        subset=["matrix", "config"]
    )

    for _, row in to_annotate.iterrows():
        label = f"{row['matrix']}\n{row['config']}"
        ax.annotate(
            label,
            (row["lut_speedup"], row["lut_max_err"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, color="#222",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor="#cccccc", alpha=0.85),
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1.0))
    path = out_dir / "08_pareto_error_speedup.png"
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Resumen en consola
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print("\n=== Resumen por run ===")
    summary = df.groupby("run", observed=True).agg(
        filas=("M", "count"),
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
    parser = argparse.ArgumentParser(description="Visualización LUT-GEMM.")
    parser.add_argument("--csv",  default="bench_log.csv",
                        help="Ruta al CSV (default: bench_log.csv)")
    parser.add_argument("--out",  default="graficas",
                        help="Directorio de salida (default: graficas/)")
    parser.add_argument("--run",  default=None,
                        help="Filtrar a un run específico por nombre")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar las gráficas además de guardarlas")
    args = parser.parse_args()

    # Rutas relativas se resuelven contra el CWD (convención Unix estándar).
    csv_path = Path(args.csv).resolve()
    out_dir  = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(str(csv_path))

    if args.run:
        df = df[df["run"] == args.run].copy()
        if df.empty:
            print(f"[ERROR] No hay filas para run='{args.run}'", file=sys.stderr)
            sys.exit(1)
        print(f"[OK] Filtrado a run='{args.run}': {len(df)} filas")

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
