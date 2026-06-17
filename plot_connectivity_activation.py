#!/usr/bin/env python3
"""
plot_connectivity_activation.py

Generate two overlay plots from evolution CSV files:

1. All algebraic-connectivity curves on one plot.
2. All activated-node curves on one separate plot.

Expected columns in each CSV:
    V
    activated_nodes                 (always written by sweep_analysis.sweep)
    algebraic_connectivity          (only populated when the sweep ran with
                                     algebraic_connectivity=True; may be absent
                                     or entirely empty/NaN otherwise)

Robustness notes:
    * The activated-node column is the only hard requirement. The script prefers
      'activated_nodes_sampled_Va' if present, otherwise falls back to
      'activated_nodes'.
    * 'algebraic_connectivity' is OPTIONAL. If a CSV lacks it, or the column is
      present but entirely NaN (e.g. Case R, whose workers run the sweep with
      algebraic_connectivity=False for speed), that file is simply skipped in
      the connectivity plot. If NO file has usable connectivity data, the
      connectivity figure is skipped entirely with a message instead of raising.

Supported filename patterns include:
    evolution_case1_N500_mean8_sigma1_seed41.csv
    evolution_case2a_N500_mean4_sigma1_seed41.csv
    evolution_case4_N200_mean6_sigma3_seed41.csv
    evolution_caseR_N500_mean6_sigma3_seed41_fv0.10.csv

PowerShell examples:

Case 1:
    python plot_connectivity_activation.py evolution_case1_N500_mean8_sigma1_seed41.csv evolution_case1_N500_mean8_sigma3_seed41.csv evolution_case1_N500_mean8_sigma5_seed41.csv evolution_case1_N500_mean8_sigma7_seed41.csv --group case1 --outdir plots_connectactivate

Case 2A:
    python plot_connectivity_activation.py evolution_case2a_N500_mean4_sigma1_seed41.csv evolution_case2a_N500_mean6_sigma1_seed41.csv evolution_case2a_N500_mean8_sigma1_seed41.csv evolution_case2a_N500_mean10_sigma1_seed41.csv --group case2a --outdir plots_connectactivate

Case 4:
    python plot_connectivity_activation.py evolution_case4_N200_mean6_sigma3_seed41.csv evolution_case4_N400_mean6_sigma3_seed41.csv evolution_case4_N600_mean6_sigma3_seed41.csv evolution_case4_N800_mean6_sigma3_seed41.csv --group case4 --outdir plots_connectactivate

Case R:
    python plot_connectivity_activation.py evolution_caseR_N500_mean6_sigma3_seed41_fv0.00.csv evolution_caseR_N500_mean6_sigma3_seed41_fv0.10.csv evolution_caseR_N500_mean6_sigma3_seed41_fv0.15.csv evolution_caseR_N500_mean6_sigma3_seed41_fv0.20.csv --group caseR --outdir plots_connectactivate

Wildcard use is also supported:
    python plot_connectivity_activation.py evolution_*.csv --group case4 --outdir plots_connectactivate

All plots are trimmed to V <= 12 V by default.
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# STYLE SETTINGS — consistent with plot_all_cases.py
STYLE = {
    "font_family": "STIXGeneral",
    "font_size_base": 14,
    "font_size_label": 17,
    "font_size_tick": 15,
    "font_size_legend": 14,

    "line_width": 2.5,
    "marker_size": 8,

    "dpi": 300,
    "fig_format": "png",

    "grid": False,
    "V_MAX_PLOT": 12.0,
    "ymin_zero": True,
    "tight_pad": 0.4,
    "figsize": (6.0, 5.5),

    # When True, activated-node curves are cut at the first point where the
    # curve reaches its final maximum value. This removes the flat saturation
    # plateau region from N_act plots.
    "trim_activation_plateau": True,

    # Also trim algebraic-connectivity curves once they reach the plateau.
    "trim_algebraic_plateau": True,

    # Move the y=0 tick slightly above the bottom frame by allowing a small
    # negative lower limit. This is only a visual padding; data are unchanged.
    "y_zero_pad_frac": 0.06,

    # Fractional margin added above/below the data range when a plot uses a
    # data-tight y-axis (ymin_zero=False), e.g. algebraic connectivity.
    "y_tight_margin_frac": 0.08,

    "colors": [
        "#2B6CB0",
        "purple",
        "darkorange",
        "red",
        "green",
        "brown",
        "teal",
        "magenta",
        "black",
        "gray",
        "olive",
        "cyan",
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
    ],
}

# Candidate activated-node column names, in order of preference.
ACTIVATION_COLUMNS = ("activated_nodes_sampled_Va", "activated_nodes")

def apply_style():
    plt.rcParams.update({
        "font.family": STYLE["font_family"],
        "font.size": STYLE["font_size_base"],
        "mathtext.fontset": "stix",
        "axes.labelsize": STYLE["font_size_label"],
        "xtick.labelsize": STYLE["font_size_tick"],
        "ytick.labelsize": STYLE["font_size_tick"],
        "legend.fontsize": STYLE["font_size_legend"],
        "axes.grid": STYLE["grid"],

        # Ticks: outward, bottom/left only.
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.top": False,
        "ytick.right": False,

        # Keep closed box frame, matching the other plotting script.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.4,
    })

def savefig(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{STYLE['fig_format']}"
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def _extract_float(pattern, text):
    match = re.search(pattern, str(text))
    return float(match.group(1)) if match else None

def _extract_int(pattern, text):
    match = re.search(pattern, str(text))
    return int(match.group(1)) if match else None

def parse_label_from_filename(path):
    """
    Create clean legend labels from filenames.

    Examples:
        sigma=3 V
        <Va>=8 V
        N=400
        fv=0.10
    """
    stem = Path(path).stem

    N = _extract_int(r"N(\d+)", stem)
    mean = _extract_float(r"mean(\d+(?:\.\d+)?)", stem)
    sigma = _extract_float(r"sigma(\d+(?:\.\d+)?)", stem)
    seed = _extract_int(r"seed(\d+)", stem)
    fv = _extract_float(r"fv(\d+(?:\.\d+)?)", stem)

    if "case1" in stem:
        return rf"$\sigma$={sigma:g} V" if sigma is not None else "Case 1"

    if "case2a" in stem:
        return rf"$\langle V_a\rangle$={mean:g} V" if mean is not None else "Case 2A"

    if "case2b" in stem:
        return rf"$\langle V_a\rangle$={mean:g} V" if mean is not None else "Case 2B"

    if "case4" in stem:
        return rf"$N$={N:g}" if N is not None else "Case 4"

    if "caseR" in stem:
        return rf"$f_v$={fv:.2f}" if fv is not None else "Case R"

    parts = []
    if N is not None:
        parts.append(rf"$N$={N:g}")
    if mean is not None:
        parts.append(rf"$\langle V_a\rangle$={mean:g} V")
    if sigma is not None:
        parts.append(rf"$\sigma$={sigma:g} V")
    if fv is not None:
        parts.append(rf"$f_v$={fv:.2f}")
    if seed is not None:
        parts.append(rf"seed={seed}")

    return ", ".join(parts) if parts else stem

def sort_key_from_filename(path):
    """
    Sort curves in a meaningful order:
    Case 1 by sigma, Case 2 by mean, Case 4 by N, Case R by void fraction.
    """
    stem = Path(path).stem

    if "case1" in stem:
        return (1, _extract_float(r"sigma(\d+(?:\.\d+)?)", stem) or 0)

    if "case2a" in stem:
        return (2, _extract_float(r"mean(\d+(?:\.\d+)?)", stem) or 0)

    if "case2b" in stem:
        return (3, _extract_float(r"mean(\d+(?:\.\d+)?)", stem) or 0)

    if "case4" in stem:
        return (4, _extract_int(r"N(\d+)", stem) or 0)

    if "caseR" in stem:
        return (5, _extract_float(r"fv(\d+(?:\.\d+)?)", stem) or 0)

    return (99, stem)

def read_evolution_file(path, vmax=None):
    """
    Read one evolution CSV.

    Hard requirement: 'V' and at least one activated-node column (preferring
    'activated_nodes_sampled_Va', else 'activated_nodes'). The resulting
    activated-node series is exposed as 'activated_nodes_plot'.

    'algebraic_connectivity' is OPTIONAL. If the column is absent it is created
    as all-NaN, so downstream code can uniformly check for finite values and
    skip the curve when none exist (e.g. Case R sweeps run with
    algebraic_connectivity=False).
    """
    df = pd.read_csv(path)

    if "V" not in df.columns:
        raise ValueError(f"{Path(path).name} is missing required column 'V'.")

    act_col = next((c for c in ACTIVATION_COLUMNS if c in df.columns), None)
    if act_col is None:
        raise ValueError(
            f"{Path(path).name} must contain one of {list(ACTIVATION_COLUMNS)}."
        )

    df = df.copy()
    df["V"] = pd.to_numeric(df["V"], errors="coerce")
    df["activated_nodes_plot"] = pd.to_numeric(df[act_col], errors="coerce")

    if "algebraic_connectivity" in df.columns:
        df["algebraic_connectivity"] = pd.to_numeric(
            df["algebraic_connectivity"], errors="coerce"
        )
    else:
        df["algebraic_connectivity"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["V"])

    if vmax is not None:
        df = df[df["V"] <= float(vmax) + 1e-12].copy()

    return df

def _finalise_ax(ax, plotted, vmax, ymin_zero=None):
    """
    Final plot formatting.

    By default the y-axis lower limit is placed slightly below zero so the 0
    tick sits just above the bottom frame (appropriate for counts, which start
    at 0). Pass ymin_zero=False to instead keep a DATA-TIGHT y-axis: the axis
    is autoscaled around the actual data range with a small symmetric margin.
    This is used for algebraic connectivity, whose values sit in a narrow band
    far from zero, so zero-anchoring would crush all curves into a flat line.

    ymin_zero=None falls back to STYLE["ymin_zero"].
    """
    ax.set_xlim(0, vmax)

    use_zero = STYLE.get("ymin_zero", True) if ymin_zero is None else ymin_zero

    if use_zero:
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymax) and ymax > 0:
            pad = STYLE.get("y_zero_pad_frac", 0.06) * ymax
            ax.set_ylim(bottom=-pad)
        else:
            ax.set_ylim(bottom=-0.05)
    else:
        # Data-tight: expand the autoscaled range by a small margin on both
        # ends so the curves fill the panel and separate visibly.
        ax.relim()
        ax.autoscale(enable=True, axis="y")
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            span = ymax - ymin
            margin = STYLE.get("y_tight_margin_frac", 0.08) * span
            ax.set_ylim(ymin - margin, ymax + margin)

    if plotted > 1:
        ax.legend(frameon=False)

    ax.tick_params(top=False, right=False, direction="out")
    ax.grid(False)

def trim_algebraic_plateau(df, ycol="algebraic_connectivity"):
    """
    Remove the flat/near-flat plateau from an algebraic-connectivity curve.

    This keeps the rising transition region and cuts the curve at the first
    point where the algebraic connectivity reaches its final near-maximum
    value. It is intentionally more aggressive than a smoothing method because
    some curves plateau gradually and were not being trimmed.
    """
    if df.empty or ycol not in df.columns:
        return df

    d = df.sort_values("V").copy()
    y = pd.to_numeric(d[ycol], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(y)
    if finite.sum() < 3:
        return d

    yf = y[finite]
    ymax = np.nanmax(yf)

    if not np.isfinite(ymax) or ymax <= 0:
        return d

    # First point where the curve is essentially at its maximum.
    # 97% is used so curves are cut before the long visual plateau.
    target = 0.97 * ymax
    idx_all = np.arange(len(y))
    hit = idx_all[np.isfinite(y) & (y >= target)]

    if len(hit) == 0:
        return d

    cut_idx = int(hit[0])

    # Keep one point at the start of the plateau for visual closure.
    return d.iloc[: cut_idx + 1].copy()

def trim_activation_plateau(df, ycol="activated_nodes_plot"):
    """
    Remove the flat saturation plateau from an activated-node curve.

    The curve is kept up to and including the first voltage where it reaches
    its final maximum activated-node count. Later repeated plateau points are
    removed. This preserves the activation transition while removing the flat
    saturation region.
    """
    if df.empty or ycol not in df.columns:
        return df

    d = df.sort_values("V").copy()
    y = pd.to_numeric(d[ycol], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(y)
    if not finite.any():
        return d

    ymax = np.nanmax(y)
    if not np.isfinite(ymax) or ymax <= 0:
        return d

    hit = np.where(np.isfinite(y) & np.isclose(y, ymax, rtol=0.0, atol=1e-12))[0]
    if len(hit) == 0:
        return d

    cut_idx = int(hit[0])
    return d.iloc[: cut_idx + 1].copy()

def plot_algebraic_connectivity(files, outdir, output_name, vmax):
    """
    Plot all algebraic-connectivity curves on one graph.

    Files whose algebraic_connectivity is absent or entirely NaN are skipped
    with a message. If no file has usable connectivity data, the figure is not
    written (no crash).
    """
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    colors = STYLE["colors"]

    plotted = 0
    for k, path in enumerate(files):
        df = read_evolution_file(path, vmax=vmax)
        df = df.sort_values("V")

        # Algebraic connectivity is NaN before source-drain connectivity, and
        # entirely NaN when the sweep ran with algebraic_connectivity=False.
        d = df[np.isfinite(df["algebraic_connectivity"])]

        if d.empty:
            print(
                f"Skipped algebraic connectivity for {Path(path).name}: "
                "no finite values (column absent, empty, or all NaN)."
            )
            continue

        if STYLE.get("trim_algebraic_plateau", True):
            d = trim_algebraic_plateau(d, ycol="algebraic_connectivity")

        if d.empty:
            print(f"Skipped algebraic connectivity for {Path(path).name}: empty after trim.")
            continue

        ax.plot(
            d["V"],
            d["algebraic_connectivity"],
            "-o",
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
            color=colors[k % len(colors)],
            label=parse_label_from_filename(path),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(
            "No algebraic-connectivity curves were plotted "
            "(no input file had usable connectivity data). "
            "Skipping the connectivity figure."
        )
        return

    ax.set_xlabel(r"Voltage, $V$ [V]")
    ax.set_ylabel(r"Algebraic connectivity, $\lambda_2$ [-]")
    _finalise_ax(ax, plotted, vmax, ymin_zero=False)
    fig.tight_layout(pad=STYLE["tight_pad"])

    savefig(fig, outdir, output_name)

def plot_activated_nodes(files, outdir, output_name, vmax):
    """
    Plot all activated-node curves on one graph.
    """
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    colors = STYLE["colors"]

    plotted = 0
    for k, path in enumerate(files):
        df = read_evolution_file(path, vmax=vmax)
        df = df.sort_values("V")

        d = df[np.isfinite(df["activated_nodes_plot"])]
        if d.empty:
            print(f"Skipped activated nodes for {Path(path).name}: no finite values.")
            continue

        if STYLE.get("trim_activation_plateau", True):
            d = trim_activation_plateau(d, ycol="activated_nodes_plot")

        ax.plot(
            d["V"],
            d["activated_nodes_plot"],
            "-o",
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
            color=colors[k % len(colors)],
            label=parse_label_from_filename(path),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("No activated-node curves were plotted.")
        return

    ax.set_xlabel(r"Voltage, $V$ [V]")
    ax.set_ylabel(r"Activated node count, $N_\mathrm{act}$ [-]")
    _finalise_ax(ax, plotted, vmax)
    fig.tight_layout(pad=STYLE["tight_pad"])

    savefig(fig, outdir, output_name)

def filter_files(files, group):
    """
    Optional filename filter, e.g.:
      --group case1
      --group case2a
      --group case4
      --group caseR
    """
    if group is None:
        return files

    group = group.lower()
    return [f for f in files if group in Path(f).name.lower()]

def expand_file_patterns(patterns):
    """
    Expand shell-style wildcards inside Python.

    This is useful on PowerShell, where evolution_*.csv may be passed to Python
    as a literal string rather than being expanded by the shell.
    """
    if patterns:
        files = []
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                files.extend(matches)
            else:
                files.append(pattern)
    else:
        files = glob.glob("evolution_*.csv")

    return files

def main():
    parser = argparse.ArgumentParser(
        description="Plot algebraic connectivity and activated nodes from evolution CSV files."
    )

    parser.add_argument(
        "files",
        nargs="*",
        default=[],
        help="Evolution CSV files, e.g. evolution_*.csv. If omitted, uses evolution_*.csv.",
    )

    parser.add_argument(
        "--outdir",
        default="plots",
        help="Output directory for generated figures.",
    )

    parser.add_argument(
        "--group",
        default=None,
        help="Optional filename filter, e.g. case1, case2a, case4, or caseR.",
    )

    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix. If omitted, uses group name or 'all'.",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=STYLE["V_MAX_PLOT"],
        help="Maximum voltage to include in plots. Default: 12 V.",
    )

    parser.add_argument(
        "--keep-plateau",
        action="store_true",
        help="Do not trim the flat saturation plateau from activated-node curves.",
    )

    parser.add_argument(
        "--keep-algebraic-plateau",
        action="store_true",
        help="Do not trim the flat saturation plateau from algebraic-connectivity curves.",
    )

    args = parser.parse_args()

    if args.keep_plateau:
        STYLE["trim_activation_plateau"] = False

    if args.keep_algebraic_plateau:
        STYLE["trim_algebraic_plateau"] = False

    apply_style()

    files = expand_file_patterns(args.files)
    files = filter_files(files, args.group)

    if not files:
        raise SystemExit(
            f"No matching evolution CSV files found for group={args.group}"
        )

    files = sorted(files, key=sort_key_from_filename)

    prefix = args.prefix
    if prefix is None:
        prefix = args.group if args.group else "all"

    print("Files used:")
    for f in files:
        print(f"  {Path(f).name}")

    plot_algebraic_connectivity(
        files,
        args.outdir,
        f"{prefix}_algebraic_connectivity_overlay",
        vmax=args.vmax,
    )

    plot_activated_nodes(
        files,
        args.outdir,
        f"{prefix}_activated_nodes_overlay",
        vmax=args.vmax,
    )

    print("\nDone.")

if __name__ == "__main__":
    main()
