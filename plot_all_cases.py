#!/usr/bin/env python3
"""
plot_all_cases.py — generate V_T and zeta plots for every case in
all_cases_VT_zeta.csv.

The CSV pools the fitted threshold voltage (fit_V_T_V) and scaling exponent
(fit_zeta) for all sweep cases. This script splits it by `case` and draws the
natural plot for each:

  case_1  : V_T and zeta vs sigma  (width of the activation-voltage spread)
  case_2a : V_T and zeta vs <V_a>  (mean activation voltage, sigma = 1 V)
  case_2b : V_T and zeta vs <V_a>  (mean activation voltage, sigma = 5 V)
  case_3  : V_T and zeta heatmaps over the (N, <V_a>) grid
  case_4  : V_T and zeta vs N      (junction count)

Where several seeds exist per x-value the points are averaged and drawn with a
mean +/- std error bar; single-seed points are drawn without error bars.

USAGE
  python plot_all_cases.py all_cases_VT_zeta.csv --outdir case_plots
  python plot_all_cases.py all_cases_VT_zeta.csv --only case_4

Edit the STYLE block to restyle (fonts, colours, sizes).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# STYLE  —  Image-1 reference style
#
# Key choices (matching the reference figure):
#   • Larger canvas with generous whitespace via tight_layout pad
#   • STIXGeneral math font so italic variables render like proper LaTeX
#   • Bracket notation for units: [V], [-] rather than (V)
#   • No legend on single-curve panels (legend only for multi-curve overlays)
#   • Y-axis floor set to 0 so the scale is honest
#   • Heavier line (2.5 pt) and larger filled circle markers (10 pt)
#   • Thicker, outward-pointing ticks on bottom and left axes only
#   • Top and right spines kept (closed box) — matching the reference
#   • No grid
# =============================================================================
STYLE = {
    # --- fonts ---
    "font_family":      "STIXGeneral",   # italic math variables match Image 1
    "font_size_base":   14,
    "font_size_label":  17,              # axis-label size
    "font_size_tick":   15,              # tick-label size
    "font_size_legend": 14,
    "font_size_annot":  11,              # heatmap cell text

    # --- line / marker ---
    "line_width":   2.5,                 # heavier than the default 1.5
    "marker_size":  10,                  # large filled circles
    "cap_size":     5,                   # error-bar cap length
    "elinewidth":   2.0,                 # error-bar line width

    # --- colours ---
    "color_VT":     "#2B6CB0",           # strong blue matching Image 1
    "color_zeta":   "#C53030",           # red for zeta panels
    "heatmap_cmap": "viridis",
    "heatmap_text": "white",
    # palette for multi-curve overlays (distributions, I-V by N)
    "overlay_colors": ["#2B6CB0", "purple", "darkorange", "red", "green", "brown"],

    # --- I-V overlay representative seed ---
    # Case 1 / Case 2 I-V CSVs may contain multiple seeds per parameter value.
    # Plotting all of them as one curve causes vertical "teeth", because points
    # from different seeds share the same voltage values. Use one representative
    # seed for clean publication overlays.
    "iv_seed": 41,

    # --- I-V curve trimming ---
    # The sweep now trims each I-V CSV at that run's own connectivity-transition
    # voltage (the nonlinear->linear knee, where algebraic connectivity goes
    # flat), so each curve already ends where it should. Leave this False so the
    # plotter shows the CSV as-is — each curve ending at its OWN transition.
    # Set True to additionally impose a single common upper voltage
    # (common_V_max) across all curves; note that would re-chop any curve whose
    # transition sits above common_V_max, discarding data the sweep kept.
    "trim_to_common_V": False,
    "common_V_max":     12.0,    # V, common cap, only used if trim_to_common_V

    # --- activation-voltage distribution x-range ---
    # The distribution CSVs are now written as a Gaussian CLIPPED and
    # renormalized to this window (see save_distribution_csv in
    # parameter_sweep_cases.py). Bound the distribution plot x-axis to the same
    # window so the figure shows exactly the clipped range (sub-2 V left tail
    # stays visible) instead of autoscaling. Set to None to autoscale.
    "VA_CLIP":      (0.0, 20.0),

    # --- output ---
    "dpi":          200,
    "fig_format":   "png",

    # --- axes ---
    "ymin_zero":    True,   # set y-axis lower bound to 0 (honest scale)
    "tight_pad":    0.4,    # extra whitespace around axes
}


def apply_style():
    plt.rcParams.update({
        "font.family":          STYLE["font_family"],
        "font.size":            STYLE["font_size_base"],
        "mathtext.fontset":     "stix",        # italic math variables
        "axes.labelsize":       STYLE["font_size_label"],
        "xtick.labelsize":      STYLE["font_size_tick"],
        "ytick.labelsize":      STYLE["font_size_tick"],
        "legend.fontsize":      STYLE["font_size_legend"],
        "axes.grid":            False,
        # ticks: outward, bottom and left only — top/right suppressed here,
        # confirmed off per-axes in _finalise_ax / _finalise_overlay
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     6,
        "ytick.major.size":     6,
        "xtick.major.width":    1.4,
        "ytick.major.width":    1.4,
        # closed box (all four spines visible, no ticks on top/right)
        "axes.spines.top":      True,
        "axes.spines.right":    True,
        "axes.linewidth":       1.4,
    })


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.{STYLE['fig_format']}")
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _common_V_mask(V):
    """Boolean mask keeping I-V points at or below the common upper voltage."""
    V = np.asarray(V, dtype=float)
    if not STYLE.get("trim_to_common_V", False):
        return np.ones(len(V), dtype=bool)
    return V <= STYLE["common_V_max"] + 1e-9


def _finalise_ax(ax, multi):
    """
    Apply Image-1 finishing touches to a 1-D axes:
      - ticks on bottom and left only (not top or right)
      - y-axis lower bound = 0 (honest scale)
      - no legend on single-curve panels; frameless legend on multi-curve
    """
    ax.tick_params(top=False, right=False, which="both")
    if STYLE.get("ymin_zero", True):
        ax.set_ylim(bottom=0)
    if multi:
        ax.legend(frameon=False)


def _finalise_overlay(ax):
    """Ticks on bottom/left only — for multi-curve overlay plots."""
    ax.tick_params(top=False, right=False, which="both")


# =============================================================================
# axis-label helpers  (bracket SI notation + descriptive prefix)
# =============================================================================
_YLABEL = {
    "VT":   r"Threshold voltage, $V_\mathrm{T}$ [V]",
    "zeta": r"Scaling exponent, $\zeta$ [-]",
}

_XLABEL = {
    "sigma_Va_target_V":  r"Activation voltage standard deviation, $\sigma$ [V]",
    "mean_Va_target_V":   r"Mean activation voltage, $\langle V_a\rangle$ [V]",
    "N":                  r"Junction count, $N$ [-]",
    "void_fraction":      r"Void fraction, $f_v$ [-]",
}


def _make_xlabel(xcol, fallback=None):
    return _XLABEL.get(xcol, fallback or xcol)


# =============================================================================
# 1-D sweep:  metric vs a single swept variable (with seed averaging)
# =============================================================================
def plot_metric_vs_x(d, xcol, xlabel, case, outdir,
                     metric, ycol, ylabel, fname):
    """Average `ycol` over seeds at each x, plot with mean +/- std error bars."""
    d = d[np.isfinite(d[ycol])]
    xs = sorted(d[xcol].unique())
    mean   = [d[d[xcol] == x][ycol].mean() for x in xs]
    counts = [len(d[d[xcol] == x]) for x in xs]
    std    = [d[d[xcol] == x][ycol].std(ddof=1) if c > 1 else 0.0
              for x, c in zip(xs, counts)]
    multi  = any(c > 1 for c in counts)

    ylabel_final = _YLABEL.get(metric, ylabel)
    xlabel_final = _make_xlabel(xcol, xlabel)
    color = STYLE["color_VT"] if metric == "VT" else STYLE["color_zeta"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.errorbar(xs, mean,
                yerr=(std if multi else None),
                fmt="-o",
                ms=STYLE["marker_size"],
                lw=STYLE["line_width"],
                color=color,
                ecolor=color,
                capsize=STYLE["cap_size"],
                elinewidth=STYLE["elinewidth"],
                label="_nolegend_")
    ax.set_xlabel(xlabel_final)
    ax.set_ylabel(ylabel_final)
    _finalise_ax(ax, multi)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, f"{case}_{fname}")


def do_1d(d, xcol, xlabel, case, outdir):
    plot_metric_vs_x(d, xcol, xlabel, case, outdir,
                     "VT",   "fit_V_T_V", r"$V_T$ [V]",  "VT")
    plot_metric_vs_x(d, xcol, xlabel, case, outdir,
                     "zeta", "fit_zeta",  r"$\zeta$", "zeta")


# =============================================================================
# 2-D grid (case_3):  heatmaps over (N, <V_a>)
# =============================================================================
def plot_heatmap(d, case, outdir, ycol, clabel, fname):
    xs = sorted(d["N"].unique())
    ys = sorted(d["mean_Va_target_V"].unique())
    Z = np.full((len(ys), len(xs)), np.nan)
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            sub = d[(d["N"] == xv) & (d["mean_Va_target_V"] == yv)]
            vals = sub[ycol].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals):
                Z[iy, ix] = vals.mean()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   cmap=STYLE["heatmap_cmap"], interpolation="nearest")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{y:g}" for y in ys])
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            if np.isfinite(Z[iy, ix]):
                ax.text(ix, iy, f"{Z[iy, ix]:.2f}", ha="center", va="center",
                        color=STYLE["heatmap_text"],
                        fontsize=STYLE["font_size_annot"], fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(clabel, rotation=270, labelpad=18)
    ax.set_xlabel(r"Junction count, $N$ [-]")
    ax.set_ylabel(r"Mean activation voltage, $\langle V_a\rangle$ [V]")
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, f"{case}_{fname}")


def do_heatmaps(d, case, outdir):
    plot_heatmap(d, case, outdir, "fit_zeta",
                 r"Scaling exponent, $\zeta$ [-]",         "zeta_heatmap")
    plot_heatmap(d, case, outdir, "fit_V_T_V",
                 r"Threshold voltage, $V_\mathrm{T}$ [V]", "VT_heatmap")


# =============================================================================
# per-case routing
# =============================================================================
def plot_case(case, d, outdir):
    print(f"[{case}]  {len(d)} rows")
    if case == "case_1":
        do_1d(d, "sigma_Va_target_V",
              r"Activation voltage standard deviation, $\sigma$ [V]",
              case, outdir)
    elif case in ("case_2a", "case_2b"):
        do_1d(d, "mean_Va_target_V",
              r"Mean activation voltage, $\langle V_a\rangle$ [V]",
              case, outdir)
    elif case == "case_3":
        do_heatmaps(d, case, outdir)
    elif case == "case_4":
        do_1d(d, "N", r"Junction count, $N$ [-]", case, outdir)
    else:
        for xcol, xlabel in [
            ("N",                 r"Junction count, $N$ [-]"),
            ("mean_Va_target_V",  r"Mean activation voltage, $\langle V_a\rangle$ [V]"),
            ("sigma_Va_target_V", r"Activation voltage standard deviation, $\sigma$ [V]"),
        ]:
            if d[xcol].nunique() > 1:
                do_1d(d, xcol, xlabel, case, outdir)
                break
        else:
            print(f"  [skip] nothing varies in {case}")


def plot_distribution(path, outdir):
    """
    Plot activation-voltage distribution curves from a *_distribution_data.csv
    (V_a column + one density_meanM_sigmaS column per curve).

    If a column name contains "shifted", it is plotted with the same colour as
    its matching sigma curve but with a dashed line. This is useful for showing
    a second curve with the same sigma value but a slightly shifted distribution.
    """
    import re
    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    x = df["V_a"].to_numpy()
    dens_cols = [c for c in df.columns if c.startswith("density")]

    def parse(c):
        m = re.search(r"mean(\d+(?:\.\d+)?)", c)
        s = re.search(r"sigma(\d+(?:\.\d+)?)", c)
        shifted = "shifted" in c.lower()
        return (
            float(m.group(1)) if m else None,
            float(s.group(1)) if s else None,
            shifted,
        )

    parsed = {c: parse(c) for c in dens_cols}
    means = {parsed[c][0] for c in dens_cols}
    sigmas = {parsed[c][1] for c in dens_cols}
    by_sigma = len(sigmas) > 1 and len(means) <= 2

    cols = sorted(
        dens_cols,
        key=lambda c: (
            parsed[c][1] if by_sigma else parsed[c][0],
            1 if parsed[c][2] else 0,
        ),
    )

    colors = STYLE["overlay_colors"]
    if by_sigma:
        unique_sigmas = sorted({parsed[c][1] for c in cols})
        color_map = {sv: colors[i % len(colors)] for i, sv in enumerate(unique_sigmas)}
    else:
        unique_means = sorted({parsed[c][0] for c in cols})
        color_map = {mv: colors[i % len(colors)] for i, mv in enumerate(unique_means)}

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for c in cols:
        mv, sv, shifted = parsed[c]
        if by_sigma:
            if shifted:
                lbl = rf"$\sigma_a$ = {sv:g} V ($\langle V_a\rangle$ = {mv:g} V)"
            else:
                lbl = rf"$\sigma_a$ = {sv:g} V"
            color = color_map[sv]
        else:
            if shifted:
                lbl = rf"$\langle V_a\rangle$ = {mv:g} V"
            else:
                lbl = rf"$\langle V_a\rangle$ = {mv:g} V"
            color = color_map[mv]

        ax.plot(
            x,
            df[c],
            linestyle=(0, (6, 3)) if shifted else "-",
            lw=STYLE["line_width"] + (0.3 if shifted else 0.0),
            color=color,
            label=lbl,
        )

    ax.set_xlabel(r"Activation voltage, $V_a$ [V]")
    ax.set_ylabel("Density [-]")
    clip = STYLE.get("VA_CLIP")
    if clip is not None:
        ax.set_xlim(float(clip[0]), float(clip[1]))
    ax.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    borderaxespad=0.2
)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, f"{stem}")


def plot_iv_overlay(path, outdir, seed=None):
    """
    Overlay I-V curves from a *_VT_zeta_iv_data.csv.

    If a curve label contains "shifted", it is treated as a separate curve even
    when it has the same sigma value as another curve. It is plotted dashed and
    with the same colour as its corresponding sigma curve.
    """
    import re
    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    if seed is None:
        seed = STYLE.get("iv_seed", 41)

    def parse(label):
        label = str(label)
        mn = re.search(r"mean(\d+(?:\.\d+)?)", label)
        sg = re.search(r"sigma(\d+(?:\.\d+)?)", label)
        sd = re.search(r"seed(\d+)", label)
        shifted = "shifted" in label.lower()
        return (
            float(mn.group(1)) if mn else None,
            float(sg.group(1)) if sg else None,
            int(sd.group(1)) if sd else None,
            shifted,
        )

    df = df.copy()
    parsed = df["label"].map(parse)
    df["_mean"] = parsed.map(lambda t: t[0])
    df["_sigma"] = parsed.map(lambda t: t[1])
    df["_seed"] = parsed.map(lambda t: t[2])
    df["_shifted"] = parsed.map(lambda t: t[3])

    if df["_seed"].notna().any():
        sub = df[df["_seed"] == seed].copy()

        if len(sub) == 0:
            available = sorted(int(s) for s in df["_seed"].dropna().unique())
            fallback = available[0]
            print(
                f"  [I-V overlay] seed {seed} not found in {os.path.basename(path)}; "
                f"using seed {fallback} instead."
            )
            sub = df[df["_seed"] == fallback].copy()

        df = sub

    n_mean = df["_mean"].nunique()
    n_sigma = df["_sigma"].nunique()
    by_sigma = n_sigma > 1 and n_mean <= 2
    keycol = "_sigma" if by_sigma else "_mean"

    groups = []
    for (val, shifted), s in df.groupby([keycol, "_shifted"], dropna=True):
        groups.append((float(val), bool(shifted), s.copy()))
    groups.sort(key=lambda item: (item[0], 1 if item[1] else 0))

    colors = STYLE["overlay_colors"]
    unique_vals = sorted({g[0] for g in groups})
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_vals)}
    ycol = "current_nA" if "current_nA" in df.columns else None

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    for val, shifted, s in groups:
        s = s.sort_values("voltage_V")
        V = s["voltage_V"].to_numpy()
        y = (s[ycol] if ycol else s["current_A"] * 1e9).to_numpy()

        m = _common_V_mask(V)
        V, y = V[m], y[m]

        if by_sigma:
            mean_val = float(s["_mean"].dropna().iloc[0]) if s["_mean"].notna().any() else np.nan
            if shifted and np.isfinite(mean_val):
                lbl = rf"$\sigma_a$ = {val:g} V ($\langle V_a\rangle$ = {mean_val:g} V)"
            else:
                lbl = rf"$\sigma_a$ = {val:g} V"
        else:
            lbl = rf"$\langle V_a\rangle$ = {val:g} V"

        ax.plot(
            V,
            y,
            linestyle=(0, (6, 3)) if shifted else "-",
            lw=STYLE["line_width"] + (0.3 if shifted else 0.0),
            color=color_map[val],
            label=lbl,
        )

    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [nA]")
    ax.legend(frameon=False)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, f"{stem}_iv_overlay")


def plot_sampled_va_hist(path, outdir, bins=40):
    """
    Histogram of the ACTUAL sampled (clipped) activation voltages from a
    *_sampled_va.csv file.

    Expected format:
        columns named va_mean{m}_sigma{s}

    This differs from the analytic Gaussian distribution overlay because it
    shows the actual node-level values used in the simulation, including
    clipping at the imposed V_a bounds.
    """
    import re

    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    cols = [c for c in df.columns if c.startswith("va_")]

    if not cols:
        print(f"  [sampled V_a] no va_* columns found in {os.path.basename(path)}")
        return

    def parse(c):
        m = re.search(r"mean(\d+(?:\.\d+)?)", c)
        s = re.search(r"sigma(\d+(?:\.\d+)?)", c)
        return (
            float(m.group(1)) if m else None,
            float(s.group(1)) if s else None,
        )

    parsed  = {c: parse(c) for c in cols}
    means   = {parsed[c][0] for c in cols}
    sigmas  = {parsed[c][1] for c in cols}
    by_sigma = len(sigmas) > 1 and len(means) <= 1

    cols = sorted(cols, key=lambda c: parsed[c][1] if by_sigma else parsed[c][0])

    allvals = np.concatenate([df[c].dropna().to_numpy(dtype=float) for c in cols])
    edges   = np.linspace(allvals.min(), allvals.max(), bins + 1)
    colors  = STYLE["overlay_colors"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for k, c in enumerate(cols):
        vals = df[c].dropna().to_numpy(dtype=float)
        mv, sv = parsed[c]
        lbl = (rf"$\sigma_a$ = {sv:g} V" if by_sigma
               else rf"$\langle V_a\rangle$ = {mv:g} V")
        ax.hist(vals, bins=edges, density=True, histtype="step",
                lw=STYLE["line_width"], color=colors[k % len(colors)], label=lbl)

    ax.set_xlabel(r"Activation voltage, $V_a$ [V]")
    ax.set_ylabel("Density [-]")

    # Extend the y-axis upper limit by 35 % so the legend sits clear of the
    # tallest bar. The lower bound is enforced at 0 explicitly.
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.35)

    ax.legend(frameon=False)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, f"{stem}_hist")


def plot_case4_iv_by_N(path, outdir, seed=41):
    """
    Overlay the case_4 I-V curves for a single seed (default 41), one curve per
    junction count N, from case4_VT_zeta_iv_data.csv
    (columns: label, voltage_V, current_A, current_nA). Labels look like
    'N=200_mean6_sigma3_seed41'.
    """
    import re
    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    def parse_N_seed(label):
        n = re.search(r"N=?(\d+)", str(label))
        s = re.search(r"seed(\d+)", str(label))
        return (int(n.group(1)) if n else None,
                int(s.group(1)) if s else None)

    df       = df.copy()
    NS       = df["label"].map(parse_N_seed)
    df["_N"]    = NS.map(lambda t: t[0])
    df["_seed"] = NS.map(lambda t: t[1])

    sub = df[df["_seed"] == seed]
    if len(sub) == 0:
        print(f"  [case4 I-V] no rows for seed {seed}")
        return
    Ns     = sorted(n for n in sub["_N"].dropna().unique())
    ycol   = "current_nA" if "current_nA" in sub.columns else None
    colors = STYLE["overlay_colors"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for k, N in enumerate(Ns):
        s = sub[sub["_N"] == N].sort_values("voltage_V")
        V = s["voltage_V"].to_numpy()
        y = (s[ycol] if ycol else s["current_A"] * 1e9).to_numpy()
        m = _common_V_mask(V)
        V, y = V[m], y[m]
        ax.plot(V, y, "-", lw=STYLE["line_width"],
                color=colors[k % len(colors)], label=rf"$N$ = {N:g}")
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [nA]")
    ax.legend(frameon=False)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, f"{stem}_seed{seed}_by_N")


def plot_caseR_aggregated(path, outdir):
    """
    Plot V_T and zeta vs void fraction from caseR_VT_zeta_aggregated.csv
    (columns: void_fraction, n_seeds, VT_mean, VT_std, zeta_mean, zeta_std).
    Draws mean +/- std error bars wherever a non-zero std is present.
    """
    df = pd.read_csv(path)
    x  = df["void_fraction"].to_numpy()

    specs = [
        ("VT",   "VT_mean",   "VT_std",
         r"Threshold voltage, $V_\mathrm{T}$ [V]", "VT"),
        ("zeta", "zeta_mean", "zeta_std",
         r"Scaling exponent, $\zeta$",          "zeta"),
    ]
    for metric, mcol, scol, ylabel, fname in specs:
        y     = df[mcol].to_numpy()
        yerr  = df[scol].to_numpy() if scol in df.columns else None
        multi = yerr is not None and np.any(np.isfinite(yerr) & (yerr > 0))
        color = STYLE["color_VT"] if metric == "VT" else STYLE["color_zeta"]
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        ax.errorbar(
                    x, y,
                    yerr=(yerr if multi else None),
                    fmt="-o",
                    ms=STYLE["marker_size"],
                    lw=STYLE["line_width"],
                    color=color,
                    ecolor=color,
                    capsize=STYLE["cap_size"],
                    elinewidth=STYLE["elinewidth"],
                    )
        ax.set_xlabel(r"Void fraction, $f_v$ [-]")

        # Dimensionless metric: label the zeta axis with [-] to match convention.
        if metric == "zeta":
            ax.set_ylabel(r"Scaling exponent, $\zeta$ [-]")
        else:
            ax.set_ylabel(ylabel)

        # Pass False so _finalise_ax does not create the "mean ± std" legend.
        _finalise_ax(ax, False)
        fig.tight_layout(pad=STYLE["tight_pad"])
        _save(fig, outdir, f"caseR_{fname}")


def plot_caseR_iv(path, outdir, void_fractions=(0.00, 0.10, 0.15, 0.20)):
    """
    Overlay I-V curves from caseR_iv_random_voids_data.csv (columns: label,
    voltage_V, current_A, current_nA). Labels look like 'fv0.10'. Only the
    requested void fractions are plotted. Uses current_nA and trims to the
    common upper voltage.
    """
    import re
    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    def parse_fv(label):
        m = re.search(r"fv(\d+(?:\.\d+)?)", str(label))
        return float(m.group(1)) if m else None

    df       = df.copy()
    df["_fv"] = df["label"].map(parse_fv)
    wanted   = sorted(void_fractions)
    ycol     = "current_nA" if "current_nA" in df.columns else None
    colors   = STYLE["overlay_colors"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    plotted = 0
    for k, fv in enumerate(wanted):
        s = df[np.isclose(df["_fv"], fv)].sort_values("voltage_V")
        if len(s) == 0:
            print(f"  [caseR I-V] no rows for void fraction {fv:g}")
            continue
        V = s["voltage_V"].to_numpy()
        y = (s[ycol] if ycol else s["current_A"] * 1e9).to_numpy()
        m = _common_V_mask(V)
        V, y = V[m], y[m]
        ax.plot(V, y, "-", lw=STYLE["line_width"],
                color=colors[k % len(colors)],
                label=rf"$f_v$ = {fv:g}")
        plotted += 1
    if plotted == 0:
        print("  [caseR I-V] nothing to plot")
        plt.close(fig)
        return
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [nA]")
    ax.legend(frameon=False)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, f"{stem}_iv_overlay")


def plot_sweep_table(path, outdir):
    """
    Plot a single-network per-voltage evolution table (the sweep_table_*.csv
    written by run_sweep_analysis.py) as SEPARATE figures:

      <stem>_iv.png                      : total current [nA] vs voltage.
      <stem>_activated_nodes.png         : activated node count vs voltage.
      <stem>_conducting_edges.png        : conducting edges vs voltage.
      <stem>_algebraic_connectivity.png  : algebraic connectivity vs voltage.
      <stem>_conductance.png             : conductance [S] vs voltage.
      <stem>_effective_resistance.png    : effective resistance [Ohm] vs voltage.

    Each metric is its own standalone file (one network, one seed -> single
    series per figure). Missing columns are skipped gracefully.
    """
    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    if "V" not in df.columns:
        print(f"  [sweep table] {stem}: no 'V' column — skipped")
        return
    df = df.sort_values("V")
    V = pd.to_numeric(df["V"], errors="coerce").to_numpy()

    def col(name):
        return (pd.to_numeric(df[name], errors="coerce").to_numpy()
                if name in df.columns else None)

    def single_plot(y, ylabel, fname, color):
        """One standalone metric-vs-voltage figure."""
        mask = np.isfinite(V) & np.isfinite(y)
        if not mask.any():
            print(f"  [sweep table] {stem}: '{fname}' has no finite data — skipped")
            return
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        ax.plot(V[mask], y[mask], "-o", lw=STYLE["line_width"],
                ms=STYLE.get("marker_size", 5), color=color)
        ax.set_xlabel("Voltage, $V$ [V]")
        ax.set_ylabel(ylabel)
        ax.tick_params(top=False, right=False, direction="out")
        fig.tight_layout(pad=STYLE["tight_pad"])
        _save(fig, outdir, f"{stem}_{fname}")

    colors = STYLE["overlay_colors"]

    # ---- I-V curve ----
    cur = col("total_current_chargeconserving_A")
    if cur is None:
        cur = col("total_current_A")
    if cur is not None:
        single_plot(cur * 1e9, "Current [nA]", "iv", colors[0])
    else:
        print(f"  [sweep table] {stem}: no current column — I-V skipped")

    # ---- evolution metrics, each as its own figure ----
    metrics = [
        ("activated_nodes",         r"Activated node count, $N_\mathrm{act}$ [-]", "activated_nodes"),
        ("conducting_edges",        r"Conducting edges [-]",                       "conducting_edges"),
        ("algebraic_connectivity",  r"Algebraic connectivity, $\lambda_2$ [-]",    "algebraic_connectivity"),
        ("conductance_S",           r"Conductance [S]",                            "conductance"),
        ("effective_resistance_ohm", r"Effective resistance [$\Omega$]",           "effective_resistance"),
    ]
    plotted_any = False
    for idx, (c, lab, fname) in enumerate(metrics):
        y = col(c)
        if y is None:
            continue
        single_plot(y, lab, fname, colors[(idx + 1) % len(colors)])
        plotted_any = True
    if not plotted_any:
        print(f"  [sweep table] {stem}: no evolution columns — metrics skipped")


def plot_sweep_iv_multi(paths, outdir, phases=None, out_name="iv_curve_multiseed",
                        spread="std"):
    """
    Multi-seed I-V curve with a spread band, from several sweep_table_*.csv files
    (one per seed, same network condition). Computes the mean current at each
    voltage across seeds and shades the spread, reproducing the line+band style.

    Parameters
    ----------
    paths   : list[str]  sweep_table_*.csv files (>= 2 for a meaningful band)
    outdir  : output directory
    phases  : optional list of voltage breakpoints; dashed vertical dividers are
              drawn at each (e.g. [2.0, 12.0] for Phase 1|2|3).
    out_name: output filename stem
    spread  : "std"  -> band = mean +/- 1 standard deviation across seeds
              "minmax" -> band = min..max envelope across seeds

    Current is taken from total_current_chargeconserving_A if present, else
    total_current_A. Seeds are aligned on the voltages common to ALL files.
    """
    cur_curves = {}   # voltage-keyed dict per file, for alignment
    for p in paths:
        df = pd.read_csv(p)
        if "V" not in df.columns:
            print(f"  [iv multi] {os.path.basename(p)}: no 'V' column — skipped")
            continue
        if "total_current_chargeconserving_A" in df.columns:
            cur = pd.to_numeric(df["total_current_chargeconserving_A"], errors="coerce")
        elif "total_current_A" in df.columns:
            cur = pd.to_numeric(df["total_current_A"], errors="coerce")
        else:
            print(f"  [iv multi] {os.path.basename(p)}: no current column — skipped")
            continue
        V = pd.to_numeric(df["V"], errors="coerce")
        s = pd.Series(cur.to_numpy(), index=np.round(V.to_numpy(), 6))
        cur_curves[p] = s[~s.index.duplicated(keep="first")]

    if len(cur_curves) < 2:
        print(f"  [iv multi] need >=2 valid seed tables, got {len(cur_curves)} "
              "— a spread band requires multiple seeds.")
        return

    # align on voltages present in ALL seeds (intersection)
    common_V = None
    for s in cur_curves.values():
        common_V = set(s.index) if common_V is None else (common_V & set(s.index))
    common_V = np.array(sorted(common_V), dtype=float)
    if len(common_V) == 0:
        print("  [iv multi] seeds share no common voltages — cannot align.")
        return

    # matrix: rows = seeds, cols = voltages (in nA)
    M = np.vstack([cur_curves[p].reindex(common_V).to_numpy() * 1e9
                   for p in cur_curves])
    mean = np.nanmean(M, axis=0)
    if spread == "minmax":
        lo, hi = np.nanmin(M, axis=0), np.nanmax(M, axis=0)
        band_label = "Spread (min–max)"
    else:
        sd = np.nanstd(M, axis=0)
        lo, hi = mean - sd, mean + sd
        band_label = "Spread (±1σ)"

    # ---- aggregated CSV over the FULL voltage union (complete rows) ----
    # Mean/std in nanoamps at every voltage any seed has; where a voltage is
    # missing from some seeds, nanmean/nanstd use the seeds that have it.
    all_V = sorted(set().union(*[set(s.index) for s in cur_curves.values()]))
    all_V = np.array(all_V, dtype=float)
    Mfull = np.vstack([cur_curves[p].reindex(all_V).to_numpy() * 1e9
                       for p in cur_curves])
    csv_mean = np.nanmean(Mfull, axis=0)
    csv_std = np.nanstd(Mfull, axis=0)
    csv_path = os.path.join(outdir, f"{out_name}_data.csv")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({
        "voltage_V": all_V,
        "current_mean_A": csv_mean,   # values are in nA (column name matches request)
        "current_std_A": csv_std,
    }).to_csv(csv_path, index=False)
    print(f"  [iv multi] wrote {csv_path}")

    # apply common-V cap if --vmax was set
    m = _common_V_mask(common_V)
    Vp, meanp, lop, hip = common_V[m], mean[m], lo[m], hi[m]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.fill_between(Vp, lop, hip, color="red", alpha=0.20, linewidth=0,
                    label=band_label)
    ax.plot(Vp, meanp, "-", lw=STYLE["line_width"], color="red", label="Current")

    if phases:
        for vb in phases:
            ax.axvline(float(vb), color="blue", ls="--", lw=1.8)

    ax.set_xlabel("Voltage, $V$ [V]")
    ax.set_ylabel("Current [nA]")
    ax.legend(frameon=False)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, out_name)
    print(f"  [iv multi] {len(cur_curves)} seeds, {len(Vp)} common voltages, "
          f"band = {band_label}")


def plot_iv_aggregated(path, outdir, phases=None, out_name=None):
    """
    Plot an I-V curve with a spread band DIRECTLY from a pre-aggregated CSV
    (the iv_curve_multiseed_data.csv written by --sweep-iv-multi), rather than
    re-computing statistics from per-seed tables.

    Expected columns: voltage_V, current_mean_A, current_std_A. Values are taken
    as-is (the writer stores them in nanoamps), plotted as a mean line with a
    +/-1 std shaded band. Optional dashed phase dividers via `phases`.
    """
    df = pd.read_csv(path)
    need = {"voltage_V", "current_mean_A", "current_std_A"}
    missing = need - set(df.columns)
    if missing:
        print(f"  [iv aggregated] {os.path.basename(path)}: missing columns "
              f"{sorted(missing)} — skipped")
        return
    df = df.sort_values("voltage_V")
    V = pd.to_numeric(df["voltage_V"], errors="coerce").to_numpy()
    mean = pd.to_numeric(df["current_mean_A"], errors="coerce").to_numpy()
    sd = pd.to_numeric(df["current_std_A"], errors="coerce").to_numpy()
    finite = np.isfinite(V) & np.isfinite(mean)
    V, mean, sd = V[finite], mean[finite], np.nan_to_num(sd[finite])

    m = _common_V_mask(V)
    V, mean, sd = V[m], mean[m], sd[m]

    if out_name is None:
        out_name = os.path.splitext(os.path.basename(path))[0]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.fill_between(V, mean - sd, mean + sd, color="red", alpha=0.20,
                    linewidth=0, label="Spread (±1σ)")
    ax.plot(V, mean, "-", lw=STYLE["line_width"], color="red", label="Current")
    if phases:
        for vb in phases:
            ax.axvline(float(vb), color="blue", ls="--", lw=1.8)
    ax.set_xlabel("Voltage, $V$ [V]")
    ax.set_ylabel("Current [nA]")
    ax.legend(frameon=False)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE["tight_pad"])
    _save(fig, outdir, out_name)
    print(f"  [iv aggregated] {len(V)} points from {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot V_T and zeta for each case in all_cases_VT_zeta.csv, "
                    "plus distribution and case-4 I-V overlays.")
    ap.add_argument("csv", nargs="?",
                    help="path to all_cases_VT_zeta.csv (optional)")
    ap.add_argument("--outdir", default="case_plots", help="output directory")
    ap.add_argument("--only", help="plot only this case (e.g. case_4)")
    ap.add_argument("--dist", nargs="*", default=[],
                    help="one or more *_distribution_data.csv files to plot")
    ap.add_argument("--case4-iv", dest="case4_iv",
                    help="case4_VT_zeta_iv_data.csv -> I-V overlay by N")
    ap.add_argument("--seed", type=int, default=41,
                    help="seed to use for the case-4 I-V overlay (default 41)")
    ap.add_argument("--iv", nargs="*", default=[],
                    help="one or more *_VT_zeta_iv_data.csv files -> I-V overlay "
                         "auto-labelled by sigma (case1) or <V_a> (case2a)")
    ap.add_argument("--iv-seed", dest="iv_seed", type=int,
                    default=STYLE["iv_seed"],
                    help="representative seed to use for Case 1 / Case 2 I-V overlays "
                         "(default: 41)")
    ap.add_argument("--sampled", nargs="*", default=[],
                    help="one or more *_sampled_va.csv files -> histogram of the "
                         "actual sampled/clipped activation voltages")
    ap.add_argument("--caseR-agg", dest="caseR_agg",
                    help="caseR_VT_zeta_aggregated.csv -> V_T and zeta vs "
                         "void fraction")
    ap.add_argument("--caseR-iv", dest="caseR_iv",
                    help="caseR_iv_random_voids_data.csv -> I-V overlay "
                         "(void fractions 0.00, 0.10, 0.15, 0.20)")
    ap.add_argument("--sweep-table", dest="sweep_table", nargs="*", default=[],
                    help="one or more sweep_table_*.csv (single-network "
                         "per-voltage evolution tables from run_sweep_analysis.py) "
                         "-> I-V curve + evolution-metrics panel for each")
    ap.add_argument("--sweep-iv-multi", dest="sweep_iv_multi", nargs="*", default=[],
                    help="multiple sweep_table_*.csv (one per seed, same "
                         "condition) -> single I-V curve with a seed-spread band")
    ap.add_argument("--iv-aggregated", dest="iv_aggregated", nargs="*", default=[],
                    help="pre-aggregated iv_curve_multiseed_data.csv "
                         "(voltage_V, current_mean_A, current_std_A) -> I-V curve "
                         "with spread band, plotted directly from the stored stats")
    ap.add_argument("--phases", nargs="*", type=float, default=None,
                    help="voltage breakpoints for dashed phase dividers on the "
                         "multi-seed I-V plot, e.g. --phases 2 12")
    ap.add_argument("--spread", choices=["std", "minmax"], default="std",
                    help="multi-seed I-V band: 'std' (mean +/-1 sigma) or "
                         "'minmax' (envelope). Default: std")
    ap.add_argument("--vmax", type=float, default=None,
                    help="optional common upper voltage [V] applied to all I-V "
                         "overlays (case1/2, case4, caseR). By default each "
                         "curve ends at its own transition voltage as stored in "
                         "the CSV; setting --vmax additionally caps every curve "
                         "at this voltage.")
    args = ap.parse_args()

    apply_style()

    # A command-line --vmax turns on the common-voltage cap used by the I-V
    # overlay helpers (via _common_V_mask). Without it, curves are shown exactly
    # as stored (already trimmed at each run's transition voltage upstream).
    if args.vmax is not None:
        STYLE["trim_to_common_V"] = True
        STYLE["common_V_max"] = float(args.vmax)

    if args.csv:
        df = pd.read_csv(args.csv)
        if "case" not in df.columns:
            ap.error("CSV has no 'case' column")
        cases = [args.only] if args.only else sorted(df["case"].unique())
        for case in cases:
            d = df[df["case"] == case]
            if len(d) == 0:
                print(f"[{case}]  no rows — skipped")
                continue
            plot_case(case, d, args.outdir)

    for p in args.dist:
        print(f"[distribution] {os.path.basename(p)}")
        plot_distribution(p, args.outdir)

    if args.case4_iv:
        print(f"[case4 I-V by N] seed {args.seed}")
        plot_case4_iv_by_N(args.case4_iv, args.outdir, seed=args.seed)

    for p in args.iv:
        print(f"[I-V overlay] {os.path.basename(p)} using seed {args.iv_seed}")
        plot_iv_overlay(p, args.outdir, seed=args.iv_seed)

    for p in args.sampled:
        print(f"[sampled V_a histogram] {os.path.basename(p)}")
        plot_sampled_va_hist(p, args.outdir)

    if args.caseR_agg:
        print(f"[caseR aggregated] {os.path.basename(args.caseR_agg)}")
        plot_caseR_aggregated(args.caseR_agg, args.outdir)

    if args.caseR_iv:
        print(f"[caseR I-V] {os.path.basename(args.caseR_iv)}")
        plot_caseR_iv(args.caseR_iv, args.outdir)

    for p in args.sweep_table:
        print(f"[sweep table] {os.path.basename(p)}")
        plot_sweep_table(p, args.outdir)

    if args.sweep_iv_multi:
        print(f"[iv multi] {len(args.sweep_iv_multi)} seed tables")
        plot_sweep_iv_multi(args.sweep_iv_multi, args.outdir,
                            phases=args.phases, spread=args.spread)

    for p in args.iv_aggregated:
        print(f"[iv aggregated] {os.path.basename(p)}")
        plot_iv_aggregated(p, args.outdir, phases=args.phases)

    if (not args.csv and not args.dist and not args.case4_iv and not args.iv
            and not args.sampled
            and not args.caseR_agg and not args.caseR_iv
            and not args.sweep_table and not args.sweep_iv_multi
            and not args.iv_aggregated):
        ap.error("provide the cases CSV, --dist, --case4-iv, --iv, --sampled, "
                 "--caseR-agg, --caseR-iv, --sweep-table, --sweep-iv-multi, "
                 "and/or --iv-aggregated files")


if __name__ == "__main__":
    main()