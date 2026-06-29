#!/usr/bin/env python3
"""
plot_combined.py — unified plotting for the nanonecklace sweep suite.

This merges two former scripts into one:

  (A) plot_all_cases.py
      V_T / zeta summaries, heatmaps, I-V overlays, distribution overlays,
      sampled-V_a histograms, single-network sweep-table panels, and multiseed
      I-V bands.  (the "non-evolution" family)

  (B) plot_connectivity_activation.py
      per-voltage evolution overlays: algebraic connectivity (lambda_2) and
      activated-node count, read from evolution_*.csv files.
      (the "evolution" family)

STYLE IS PRESERVED PER FAMILY
-----------------------------
The two original scripts used different style constants (e.g. DPI 200 vs 300,
marker size 10 vs 8) and slightly different tick handling. To keep each figure
looking EXACTLY as it did before, this file keeps TWO independent style blocks:

  STYLE_CASES  — drives the non-evolution figures (family A); identical to the
                 old plot_all_cases.py STYLE.
  STYLE_EVO    — drives the evolution figures (family B); identical to the old
                 plot_connectivity_activation.py STYLE.

Each family has its own apply_style() and save helper, so building a figure from
one family does not change the matplotlib rcParams used by the other (apply is
called immediately before each figure's family is drawn).

SHARED X-AXIS ENDPOINT (auto-detected)
--------------------------------------
To keep endpoints consistent between families in a single run:

  * If --vmax is given, that common cap is the endpoint for BOTH families
    (non-evolution I-V overlays already honour it via STYLE_CASES; the evolution
    plots use it directly).
  * If --vmax is NOT given, each non-evolution I-V curve ends at its own stored
    transition voltage, so there is no single cap. In that case the script
    records the MAXIMUM voltage actually drawn across all non-evolution curves
    in this run, and hands that detected endpoint to the evolution plots so they
    span the full range the I-V plots collectively cover.

The non-evolution plotters run FIRST (so the endpoint is known), then the
evolution plotters read the shared endpoint. The detected value is reported in
the console so it is never a silent choice.

USAGE (examples)
  # cases summary + evolution overlays in one run, endpoints auto-shared:
  python plot_combined.py all_cases_VT_zeta.csv \
      --iv case1_VT_zeta_iv_data.csv \
      --evolution evolution_case1_*.csv --evo-group case1 \
      --outdir combined_plots

  # force a common cap on everything:
  python plot_combined.py all_cases_VT_zeta.csv --vmax 12 \
      --evolution evolution_*.csv --evo-group case4

Notes
-----
* Family-A flags are unchanged from plot_all_cases.py.
* Family-B inputs use --evolution (files) and --evo-group / --evo-prefix to mirror
  the old plot_connectivity_activation.py --group / --prefix, kept distinct so
  the two families' file lists never collide in one command.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# SHARED ENDPOINT TRACKER
# -----------------------------------------------------------------------------
# Records the maximum voltage actually plotted by the non-evolution (family A)
# I-V curves during a run. The evolution (family B) plotters read this when the
# user did not pass --vmax, so both families share the same x-axis endpoint.
# =============================================================================
class _Endpoint:
    """Mutable run-wide endpoint state shared between the two plot families."""
    def __init__(self):
        self.user_vmax = None      # set from --vmax if provided
        self.detected_max = None   # running max of voltages actually drawn (A)

    def note_voltages(self, V):
        """Update the detected maximum from an array of plotted voltages."""
        V = np.asarray(V, dtype=float)
        V = V[np.isfinite(V)]
        if V.size == 0:
            return
        vmax = float(np.max(V))
        if self.detected_max is None or vmax > self.detected_max:
            self.detected_max = vmax

    def for_evolution(self):
        """Endpoint the evolution plots should use.

        Priority: explicit --vmax > detected max from family A > evolution's own
        STYLE default (returned as None so the caller falls back to it).
        """
        if self.user_vmax is not None:
            return float(self.user_vmax)
        if self.detected_max is not None:
            return float(self.detected_max)
        return None


ENDPOINT = _Endpoint()


# =============================================================================
# STYLE_CASES  —  family A (non-evolution).  Identical to plot_all_cases.py.
# =============================================================================
STYLE_CASES = {
    # --- fonts ---
    "font_family":      "STIXGeneral",
    "font_size_base":   16,
    "font_size_label":  25,
    "font_size_tick":   15,
    "font_size_legend": 16,
    "font_size_annot":  11,

    # --- line / marker ---
    "line_width":   2.5,
    "marker_size":  10,
    "cap_size":     5,
    "elinewidth":   2.0,

    # --- colours ---
    "color_VT":     "#2B6CB0",
    "color_zeta":   "#C53030",
    "heatmap_cmap": "viridis",
    "heatmap_text": "white",
    "overlay_colors": ["#2B6CB0", "purple", "darkorange", "red", "green", "brown"],

    "iv_seed": 41,

    "trim_to_common_V": False,
    "common_V_max":     12.0,

    "VA_CLIP":      (0.0, 20.0),

    # --- output ---
    "dpi":          200,
    "fig_format":   "png",

    # --- axes ---
    "ymin_zero":    True,
    "tight_pad":    0.4,
}


# =============================================================================
# STYLE_EVO  —  family B (evolution).  Identical to plot_connectivity_activation.py.
# =============================================================================
STYLE_EVO = {
    # --- fonts ---
    "font_family": "STIXGeneral",
    "font_size_base": 16,
    "font_size_label": 25,
    "font_size_tick": 15,
    "font_size_legend": 16,

    # --- line / marker ---
    "line_width": 2.5,
    "marker_size": 8,

    # --- output ---
    "dpi": 300,
    "fig_format": "png",

    # --- axes ---
    "grid": False,
    "V_MAX_PLOT": 12.0,
    "ymin_zero": True,
    "tight_pad": 0.4,
    "figsize": (6.0, 5.5),

    "trim_activation_plateau": True,
    "trim_algebraic_plateau": True,

    # Trim BOTH evolution curves (activated nodes and algebraic connectivity) at
    # the same transition voltage the IV curve uses: the first voltage at which
    # the activated-node count reaches ACTIVATION_TRANSITION_FRAC of the in-file
    # node total. This keeps the evolution plots' x-axis endpoint consistent with
    # the IV plot for the same condition (the IV CSV is trimmed upstream by the
    # same 90%-activation rule). When True, this transition-voltage trim is used
    # INSTEAD OF the independent plateau-detection trims above, which otherwise
    # cut at different voltages (e.g. 97%-of-max lambda_2 or first reach of the
    # exact activated-node maximum) and break that consistency.
    "trim_at_activation_transition": True,
    "y_zero_pad_frac": 0.06,
    "y_tight_margin_frac": 0.08,

    "colors": [
        "#2B6CB0", "purple", "darkorange", "red", "green", "brown",
        "teal", "magenta", "black", "gray", "olive", "cyan",
        "tab:blue", "tab:orange", "tab:green", "tab:red",
    ],
}

# Candidate activated-node column names, in order of preference (family B).
ACTIVATION_COLUMNS = ("activated_nodes_sampled_Va", "activated_nodes")

# Fraction of the in-file node total that defines the nonlinear->linear
# transition voltage (same rule the sweep applies when trimming each IV curve).
ACTIVATION_TRANSITION_FRAC = 0.90


def activation_transition_voltage(df, frac=ACTIVATION_TRANSITION_FRAC):
    """First voltage at which activated_nodes reaches `frac` of the node total.

    The node total is taken as the maximum activated-node count seen in this
    file (i.e. the post-pruning N for that network). Returns None if the
    activated-node column is missing/empty or the threshold is never reached.

    This reproduces, from the evolution data alone, the transition voltage the
    sweep used to trim the matching IV curve, so the evolution plots can be cut
    at exactly the same endpoint.
    """
    if df.empty or "activated_nodes_plot" not in df.columns or "V" not in df.columns:
        return None
    d = df.sort_values("V")
    V = pd.to_numeric(d["V"], errors="coerce").to_numpy(dtype=float)
    an = pd.to_numeric(d["activated_nodes_plot"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(V) & np.isfinite(an)
    V, an = V[m], an[m]
    if len(an) == 0:
        return None
    n_total = float(np.nanmax(an))
    if not np.isfinite(n_total) or n_total <= 0:
        return None
    reached = np.where(an >= frac * n_total)[0]
    if len(reached) == 0:
        return None
    return float(V[reached[0]])


def trim_at_transition(df, v_trans):
    """Keep only rows with V <= v_trans (the IV-matched transition voltage)."""
    if v_trans is None or df.empty or "V" not in df.columns:
        return df
    return df[df["V"] <= float(v_trans) + 1e-9].copy()


# =============================================================================
# STYLE APPLICATION  (one per family; called right before that family draws)
# =============================================================================
def apply_style_cases():
    plt.rcParams.update({
        "font.family":          STYLE_CASES["font_family"],
        "font.size":            STYLE_CASES["font_size_base"],
        "mathtext.fontset":     "stix",
        "axes.labelsize":       STYLE_CASES["font_size_label"],
        "xtick.labelsize":      STYLE_CASES["font_size_tick"],
        "ytick.labelsize":      STYLE_CASES["font_size_tick"],
        "legend.fontsize":      STYLE_CASES["font_size_legend"],
        "axes.grid":            False,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     6,
        "ytick.major.size":     6,
        "xtick.major.width":    1.4,
        "ytick.major.width":    1.4,
        "axes.spines.top":      True,
        "axes.spines.right":    True,
        "axes.linewidth":       1.4,
    })


def apply_style_evo():
    plt.rcParams.update({
        "font.family": STYLE_EVO["font_family"],
        "font.size": STYLE_EVO["font_size_base"],
        "mathtext.fontset": "stix",
        "axes.labelsize": STYLE_EVO["font_size_label"],
        "xtick.labelsize": STYLE_EVO["font_size_tick"],
        "ytick.labelsize": STYLE_EVO["font_size_tick"],
        "legend.fontsize": STYLE_EVO["font_size_legend"],
        "axes.grid": STYLE_EVO["grid"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.top": False,
        "ytick.right": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.4,
    })


def _save_cases(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.{STYLE_CASES['fig_format']}")
    fig.savefig(path, dpi=STYLE_CASES["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def savefig_evo(fig, outdir, name):
    outdir = __import__("pathlib").Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{STYLE_EVO['fig_format']}"
    fig.savefig(path, dpi=STYLE_EVO["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# =============================================================================
#  FAMILY A — non-evolution plots (from plot_all_cases.py, unchanged behaviour
#  except that I-V plotters now report drawn voltages to ENDPOINT)
# =============================================================================
# =============================================================================

def _common_V_mask(V):
    """Boolean mask keeping I-V points at or below the common upper voltage."""
    V = np.asarray(V, dtype=float)
    if not STYLE_CASES.get("trim_to_common_V", False):
        return np.ones(len(V), dtype=bool)
    return V <= STYLE_CASES["common_V_max"] + 1e-9


def _finalise_ax(ax, multi):
    ax.tick_params(top=False, right=False, which="both")
    if STYLE_CASES.get("ymin_zero", True):
        ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    handles_labels = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if multi and handles_labels:
        handles, labels = zip(*handles_labels)
        ax.legend(handles, labels, frameon=False)


def _finalise_overlay(ax):
    ax.tick_params(top=False, right=False, which="both")


def _iv_legend(ax):
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.2,
    )


def _apply_iv_xlim(ax):
    if STYLE_CASES.get("trim_to_common_V", False):
        ax.set_xlim(0, float(STYLE_CASES["common_V_max"]))


_YLABEL = {
    "VT":   r"$V_\mathrm{T}$ [V]",
    "zeta": r"$\zeta$ [-]",
}

_XLABEL = {
    "sigma_Va_target_V":  r"$\sigma_a$ [V]",
    "mean_Va_target_V":   r"$\langle V_a\rangle$ [V]",
    "N":                  r"$N$ [-]",
    "void_fraction":      r"$f_v$ [-]",
}


def _make_xlabel(xcol, fallback=None):
    return _XLABEL.get(xcol, fallback or xcol)


def plot_metric_vs_x(d, xcol, xlabel, case, outdir,
                     metric, ycol, ylabel, fname):
    d = d[np.isfinite(d[ycol])]
    xs = sorted(d[xcol].unique())
    mean   = [d[d[xcol] == x][ycol].mean() for x in xs]
    counts = [len(d[d[xcol] == x]) for x in xs]
    std    = [d[d[xcol] == x][ycol].std(ddof=1) if c > 1 else 0.0
              for x, c in zip(xs, counts)]
    multi  = any(c > 1 for c in counts)

    ylabel_final = _YLABEL.get(metric, ylabel)
    xlabel_final = _make_xlabel(xcol, xlabel)
    color = STYLE_CASES["color_VT"] if metric == "VT" else STYLE_CASES["color_zeta"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.errorbar(xs, mean,
                yerr=(std if multi else None),
                fmt="-o",
                ms=STYLE_CASES["marker_size"],
                lw=STYLE_CASES["line_width"],
                color=color,
                ecolor=color,
                capsize=STYLE_CASES["cap_size"],
                elinewidth=STYLE_CASES["elinewidth"],
                label="_nolegend_")
    ax.set_xlabel(xlabel_final)
    ax.set_ylabel(ylabel_final)
    _finalise_ax(ax, multi)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, f"{case}_{fname}")


def do_1d(d, xcol, xlabel, case, outdir):
    plot_metric_vs_x(d, xcol, xlabel, case, outdir,
                     "VT",   "fit_V_T_V", r"$V_T$ [V]",  "VT")
    plot_metric_vs_x(d, xcol, xlabel, case, outdir,
                     "zeta", "fit_zeta",  r"$\zeta$", "zeta")


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
                   cmap=STYLE_CASES["heatmap_cmap"], interpolation="nearest")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{y:g}" for y in ys])
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            if np.isfinite(Z[iy, ix]):
                ax.text(ix, iy, f"{Z[iy, ix]:.2f}", ha="center", va="center",
                        color=STYLE_CASES["heatmap_text"],
                        fontsize=STYLE_CASES["font_size_annot"], fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(clabel, rotation=270, labelpad=18)
    ax.set_xlabel(r"$N$ [-]")
    ax.set_ylabel(r"$\langle V_a\rangle$ [V]")
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, f"{case}_{fname}")


def do_heatmaps(d, case, outdir):
    plot_heatmap(d, case, outdir, "fit_zeta",
                 r"$\zeta$ [-]",         "zeta_heatmap")
    plot_heatmap(d, case, outdir, "fit_V_T_V",
                 r"$V_\mathrm{T}$ [V]", "VT_heatmap")


def plot_case(case, d, outdir):
    print(f"[{case}]  {len(d)} rows")
    if case == "case_1":
        do_1d(d, "sigma_Va_target_V",
              r"$\sigma_a$ [V]",
              case, outdir)
    elif case in ("case_2a", "case_2b"):
        do_1d(d, "mean_Va_target_V",
              r"$\langle V_a\rangle$ [V]",
              case, outdir)
    elif case == "case_3":
        do_heatmaps(d, case, outdir)
    elif case == "case_4":
        do_1d(d, "N", r"$N$ [-]", case, outdir)
    else:
        for xcol, xlabel in [
            ("N",                 r"$N$ [-]"),
            ("mean_Va_target_V",  r"$\langle V_a\rangle$ [V]"),
            ("sigma_Va_target_V", r"$\sigma_a$ [V]"),
        ]:
            if d[xcol].nunique() > 1:
                do_1d(d, xcol, xlabel, case, outdir)
                break
        else:
            print(f"  [skip] nothing varies in {case}")


def plot_distribution(path, outdir):
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

    colors = STYLE_CASES["overlay_colors"]
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
            lbl = rf"$\langle V_a\rangle$ = {mv:g} V"
            color = color_map[mv]

        ax.plot(
            x,
            df[c],
            linestyle=(0, (6, 3)) if shifted else "-",
            lw=STYLE_CASES["line_width"] + (0.3 if shifted else 0.0),
            color=color,
            label=lbl,
        )

    ax.set_xlabel(r"$V_a$ [V]")
    ax.set_ylabel("Density [-]")
    clip = STYLE_CASES.get("VA_CLIP")
    if clip is not None:
        ax.set_xlim(float(clip[0]), float(clip[1]))
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.2)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, f"{stem}")


def plot_iv_overlay(path, outdir, seed=None):
    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    if seed is None:
        seed = STYLE_CASES.get("iv_seed", 41)

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

    colors = STYLE_CASES["overlay_colors"]
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
        ENDPOINT.note_voltages(V)   # <-- record drawn endpoint for family B

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
            lw=STYLE_CASES["line_width"] + (0.3 if shifted else 0.0),
            color=color_map[val],
            label=lbl,
        )

    ax.set_xlabel(r"$V$ [V]")
    ax.set_ylabel(r"$I$ [nA]")
    _apply_iv_xlim(ax)
    _iv_legend(ax)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, f"{stem}_iv_overlay")


def plot_sampled_va_hist(path, outdir, bins=40):
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
    colors  = STYLE_CASES["overlay_colors"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for k, c in enumerate(cols):
        vals = df[c].dropna().to_numpy(dtype=float)
        mv, sv = parsed[c]
        lbl = (rf"$\sigma_a$ = {sv:g} V" if by_sigma
               else rf"$\langle V_a\rangle$ = {mv:g} V")
        ax.hist(vals, bins=edges, density=True, histtype="step",
                lw=STYLE_CASES["line_width"], color=colors[k % len(colors)], label=lbl)

    ax.set_xlabel(r"Activation voltage, $V_a$ [V]")
    ax.set_ylabel("Density [-]")

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.35)

    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.2,
    )
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, f"{stem}_hist")


def plot_case4_iv_by_N(path, outdir, seed=41):
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
    colors = STYLE_CASES["overlay_colors"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for k, N in enumerate(Ns):
        s = sub[sub["_N"] == N].sort_values("voltage_V")
        V = s["voltage_V"].to_numpy()
        y = (s[ycol] if ycol else s["current_A"] * 1e9).to_numpy()
        m = _common_V_mask(V)
        V, y = V[m], y[m]
        ENDPOINT.note_voltages(V)   # <-- record drawn endpoint for family B
        ax.plot(V, y, "-", lw=STYLE_CASES["line_width"],
                color=colors[k % len(colors)], label=rf"$N$ = {N:g}")
    ax.set_xlabel(r"$V$ [V]")
    ax.set_ylabel(r"$I$ [nA]")
    _apply_iv_xlim(ax)
    _iv_legend(ax)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, f"{stem}_seed{seed}_by_N")


def plot_caseR_aggregated(path, outdir):
    df = pd.read_csv(path)
    x  = df["void_fraction"].to_numpy()

    specs = [
        ("VT",   "VT_mean",   "VT_std",
         r"$V_\mathrm{T}$ [V]", "VT"),
        ("zeta", "zeta_mean", "zeta_std",
         r"$\zeta$",          "zeta"),
    ]
    for metric, mcol, scol, ylabel, fname in specs:
        y     = df[mcol].to_numpy()
        yerr  = df[scol].to_numpy() if scol in df.columns else None
        multi = yerr is not None and np.any(np.isfinite(yerr) & (yerr > 0))
        color = STYLE_CASES["color_VT"] if metric == "VT" else STYLE_CASES["color_zeta"]
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        ax.errorbar(
                    x, y,
                    yerr=(yerr if multi else None),
                    fmt="-o",
                    ms=STYLE_CASES["marker_size"],
                    lw=STYLE_CASES["line_width"],
                    color=color,
                    ecolor=color,
                    capsize=STYLE_CASES["cap_size"],
                    elinewidth=STYLE_CASES["elinewidth"],
                    )
        ax.set_xlabel(r"$f_v$ [-]")
        if metric == "zeta":
            ax.set_ylabel(r"$\zeta$ [-]")
        else:
            ax.set_ylabel(ylabel)
        _finalise_ax(ax, False)
        fig.tight_layout(pad=STYLE_CASES["tight_pad"])
        _save_cases(fig, outdir, f"caseR_{fname}")


def plot_caseR_iv(path, outdir, void_fractions=(0.00, 0.10, 0.15, 0.20)):
    df = pd.read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    def parse_fv(label):
        m = re.search(r"fv(\d+(?:\.\d+)?)", str(label))
        return float(m.group(1)) if m else None

    df       = df.copy()
    df["_fv"] = df["label"].map(parse_fv)
    wanted   = sorted(void_fractions)
    ycol     = "current_nA" if "current_nA" in df.columns else None
    colors   = STYLE_CASES["overlay_colors"]

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
        ENDPOINT.note_voltages(V)   # <-- record drawn endpoint for family B
        ax.plot(V, y, "-", lw=STYLE_CASES["line_width"],
                color=colors[k % len(colors)],
                label = rf"$f_v={fv:.2f}$")
        plotted += 1
    if plotted == 0:
        print("  [caseR I-V] nothing to plot")
        plt.close(fig)
        return
    ax.set_xlabel(r"$V$ [V]")
    ax.set_ylabel(r"$I$ [nA]")
    _apply_iv_xlim(ax)
    _iv_legend(ax)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, f"{stem}_iv_overlay")


def plot_sweep_table(path, outdir):
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
        mask = np.isfinite(V) & np.isfinite(y)
        if not mask.any():
            print(f"  [sweep table] {stem}: '{fname}' has no finite data — skipped")
            return
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        ax.plot(V[mask], y[mask], "-o", lw=STYLE_CASES["line_width"],
                ms=STYLE_CASES.get("marker_size", 5), color=color)
        ax.set_xlabel(r"$V$ [V]")
        ax.set_ylabel(ylabel)
        if fname == "iv":
            ENDPOINT.note_voltages(V[mask])   # <-- I-V panel reports its endpoint
        # Force the SAME x-axis window on every sweep-table panel so they all
        # share both endpoints instead of autoscaling per-metric.
        if STYLE_CASES.get("trim_to_common_V", False):
            ax.set_xlim(STYLE_CASES.get("common_V_min", 0.0),
                        float(STYLE_CASES["common_V_max"]))
        ax.tick_params(top=False, right=False, direction="out")
        fig.tight_layout(pad=STYLE_CASES["tight_pad"])
        _save_cases(fig, outdir, f"{stem}_{fname}")

    colors = STYLE_CASES["overlay_colors"]

    cur = col("total_current_chargeconserving_A")
    if cur is None:
        cur = col("total_current_A")
    if cur is not None:
        single_plot(cur * 1e9, r"$I$ [nA]", "iv", colors[0])
    else:
        print(f"  [sweep table] {stem}: no current column — I-V skipped")

    metrics = [
        ("activated_nodes",         r"Activated node count, $N_\mathrm{act}$ [-]", "activated_nodes"),
        ("conducting_edges",        r"Conducting edges [-]",                       "conducting_edges"),
        ("algebraic_connectivity",  r"$\lambda_2$ [-]",    "algebraic_connectivity"),
        ("conductance_S",           r"Conductance [S]",                            "conductance"),
        ("effective_resistance_ohm", r"$R_{eff}$ [$\Omega$]",           "effective_resistance"),
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
    cur_curves = {}
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

    common_V = None
    for s in cur_curves.values():
        common_V = set(s.index) if common_V is None else (common_V & set(s.index))
    common_V = np.array(sorted(common_V), dtype=float)
    if len(common_V) == 0:
        print("  [iv multi] seeds share no common voltages — cannot align.")
        return

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
        "current_mean_A": csv_mean,
        "current_std_A": csv_std,
    }).to_csv(csv_path, index=False)
    print(f"  [iv multi] wrote {csv_path}")

    m = _common_V_mask(common_V)
    Vp, meanp, lop, hip = common_V[m], mean[m], lo[m], hi[m]
    ENDPOINT.note_voltages(Vp)   # <-- record drawn endpoint for family B

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.fill_between(Vp, lop, hip, color="red", alpha=0.20, linewidth=0,
                    label=band_label)
    ax.plot(Vp, meanp, "-", lw=STYLE_CASES["line_width"], color="red", label="Current")

    if phases:
        for vb in phases:
            ax.axvline(float(vb), color="blue", ls="--", lw=1.8)

    ax.set_xlabel(r"$V$ [V]")
    ax.set_ylabel(r"$I$ [nA]")
    _apply_iv_xlim(ax)
    _iv_legend(ax)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, out_name)
    print(f"  [iv multi] {len(cur_curves)} seeds, {len(Vp)} common voltages, "
          f"band = {band_label}")


def plot_iv_aggregated(path, outdir, phases=None, out_name=None):
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
    ENDPOINT.note_voltages(V)   # <-- record drawn endpoint for family B

    if out_name is None:
        out_name = os.path.splitext(os.path.basename(path))[0]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.fill_between(V, mean - sd, mean + sd, color="red", alpha=0.20,
                    linewidth=0, label="Spread (±1σ)")
    ax.plot(V, mean, "-", lw=STYLE_CASES["line_width"], color="red", label="Current")
    if phases:
        for vb in phases:
            ax.axvline(float(vb), color="blue", ls="--", lw=1.8)
    ax.set_xlabel(r"$V$ [V]")
    ax.set_ylabel(r"$I$ [nA]")
    _apply_iv_xlim(ax)
    _iv_legend(ax)
    _finalise_overlay(ax)
    fig.tight_layout(pad=STYLE_CASES["tight_pad"])
    _save_cases(fig, outdir, out_name)
    print(f"  [iv aggregated] {len(V)} points from {os.path.basename(path)}")


# =============================================================================
# =============================================================================
#  FAMILY B — evolution plots (from plot_connectivity_activation.py).
#  Behaviour unchanged EXCEPT vmax now defaults to the shared endpoint.
# =============================================================================
# =============================================================================
from pathlib import Path as _Path


def _extract_float(pattern, text):
    match = re.search(pattern, str(text))
    return float(match.group(1)) if match else None


def _extract_int(pattern, text):
    match = re.search(pattern, str(text))
    return int(match.group(1)) if match else None


def parse_label_from_filename(path):
    stem = _Path(path).stem
    N = _extract_int(r"N(\d+)", stem)
    mean = _extract_float(r"mean(\d+(?:\.\d+)?)", stem)
    sigma = _extract_float(r"sigma(\d+(?:\.\d+)?)", stem)
    seed = _extract_int(r"seed(\d+)", stem)
    fv = _extract_float(r"fv(\d+(?:\.\d+)?)", stem)

    if "case1" in stem:
        return rf"$\sigma_a$ = {sigma:g} V"
    if "case2a" in stem:
        return rf"$\langle V_a\rangle$ = {mean:g} V"
    if "case2b" in stem:
        return rf"$\langle V_a\rangle$ = {mean:g} V"
    if "case4" in stem:
        return rf"$N$ = {N:g}" if N is not None else "Case 4"
    if "caseR" in stem:
        return rf"$f_v$ = {fv:.2f}" if fv is not None else "Case R"

    parts = []
    if N is not None:
        parts.append(rf"$N$ = {N:g}")
    if mean is not None:
        parts.append(rf"$\langle V_a\rangle$ = {mean:g} V")
    if sigma is not None:
        parts.append(rf"$\sigma$ = {sigma:g} V")
    if fv is not None:
        parts.append(rf"$f_v$ = {fv:.2f}")
    if seed is not None:
        parts.append(rf"seed={seed}")
    return ", ".join(parts) if parts else stem


def sort_key_from_filename(path):
    stem = _Path(path).stem
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
    df = pd.read_csv(path)
    if "V" not in df.columns:
        raise ValueError(f"{_Path(path).name} is missing required column 'V'.")

    act_col = next((c for c in ACTIVATION_COLUMNS if c in df.columns), None)
    if act_col is None:
        raise ValueError(
            f"{_Path(path).name} must contain one of {list(ACTIVATION_COLUMNS)}."
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
        df = df[df["V"] <= float(vmax)].copy()
    return df


def _finalise_ax_evo(ax, plotted, vmax, ymin_zero=None):
    use_zero = STYLE_EVO.get("ymin_zero", True) if ymin_zero is None else ymin_zero

    if use_zero:
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymax) and ymax > 0:
            pad = STYLE_EVO.get("y_zero_pad_frac", 0.06) * ymax
            ax.set_ylim(bottom=-pad)
        else:
            ax.set_ylim(bottom=-0.05)
    else:
        ax.relim()
        ax.autoscale(enable=True, axis="y")
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            span = ymax - ymin
            margin = STYLE_EVO.get("y_tight_margin_frac", 0.08) * span
            ax.set_ylim(ymin - margin, ymax + margin)

    if plotted > 1:
        ax.legend(frameon=False)

    ax.tick_params(top=False, right=False, direction="out")
    ax.grid(False)
    ax.set_xlim(left=0, right=vmax)


def trim_algebraic_plateau(df, ycol="algebraic_connectivity"):
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
    target = 0.97 * ymax
    idx_all = np.arange(len(y))
    hit = idx_all[np.isfinite(y) & (y >= target)]
    if len(hit) == 0:
        return d
    cut_idx = int(hit[0])
    return d.iloc[: cut_idx + 1].copy()


def trim_activation_plateau(df, ycol="activated_nodes_plot"):
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


def metadata_from_filename(path):
    stem = _Path(path).stem
    case_match = re.search(r"(case\d+[a-z]?|caseR)", stem, flags=re.IGNORECASE)
    case = case_match.group(1) if case_match else None
    return {
        "file": _Path(path).name,
        "label": parse_label_from_filename(path),
        "case": case,
        "N": _extract_int(r"N(\d+)", stem),
        "mean_Va_V": _extract_float(r"mean(\d+(?:\.\d+)?)", stem),
        "sigma_a_V": _extract_float(r"sigma(\d+(?:\.\d+)?)", stem),
        "seed": _extract_int(r"seed(\d+)", stem),
        "void_fraction": _extract_float(r"fv(\d+(?:\.\d+)?)", stem),
    }


def linear_fit_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def max_local_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        return np.nan
    order = np.argsort(x)
    x = x[order]; y = y[order]
    dx = np.diff(x); dy = np.diff(y)
    good = dx > 0
    if not np.any(good):
        return np.nan
    return float(np.nanmax(dy[good] / dx[good]))


def collect_algebraic_slope_rows(files, vmax):
    rows = []
    for path in files:
        df = read_evolution_file(path, vmax=vmax).sort_values("V")
        df = df[df["V"] <= float(vmax)].copy()
        d = df[np.isfinite(df["algebraic_connectivity"])].copy()
        meta = metadata_from_filename(path)

        if d.empty:
            rows.append({**meta, "n_points_used": 0, "V_min_used": np.nan,
                         "V_max_used": np.nan, "lambda2_min_used": np.nan,
                         "lambda2_max_used": np.nan,
                         "linear_slope_lambda2_per_V": np.nan,
                         "linear_intercept_lambda2": np.nan, "linear_fit_R2": np.nan,
                         "max_local_slope_lambda2_per_V": np.nan,
                         "status": "skipped_no_finite_algebraic_connectivity"})
            continue

        if STYLE_EVO.get("trim_algebraic_plateau", True):
            d = trim_algebraic_plateau(d, ycol="algebraic_connectivity")

        if d.empty or len(d) < 2:
            rows.append({**meta, "n_points_used": len(d),
                         "V_min_used": np.nan if d.empty else float(d["V"].min()),
                         "V_max_used": np.nan if d.empty else float(d["V"].max()),
                         "lambda2_min_used": np.nan if d.empty else float(d["algebraic_connectivity"].min()),
                         "lambda2_max_used": np.nan if d.empty else float(d["algebraic_connectivity"].max()),
                         "linear_slope_lambda2_per_V": np.nan,
                         "linear_intercept_lambda2": np.nan, "linear_fit_R2": np.nan,
                         "max_local_slope_lambda2_per_V": np.nan,
                         "status": "skipped_not_enough_points_after_trim"})
            continue

        V = d["V"].to_numpy(dtype=float)
        lam = d["algebraic_connectivity"].to_numpy(dtype=float)
        slope, intercept, r2 = linear_fit_stats(V, lam)
        local_slope = max_local_slope(V, lam)
        rows.append({**meta, "n_points_used": int(len(d)),
                     "V_min_used": float(np.nanmin(V)), "V_max_used": float(np.nanmax(V)),
                     "lambda2_min_used": float(np.nanmin(lam)),
                     "lambda2_max_used": float(np.nanmax(lam)),
                     "linear_slope_lambda2_per_V": slope,
                     "linear_intercept_lambda2": intercept, "linear_fit_R2": r2,
                     "max_local_slope_lambda2_per_V": local_slope, "status": "ok"})
    return rows


def write_algebraic_slope_csv(files, outdir, output_name, vmax):
    rows = collect_algebraic_slope_rows(files, vmax=vmax)
    outdir = _Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{output_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def plot_algebraic_connectivity(files, outdir, output_name, vmax):
    fig, ax = plt.subplots(figsize=STYLE_EVO["figsize"])
    colors = STYLE_EVO["colors"]

    plotted = 0
    for k, path in enumerate(files):
        df = read_evolution_file(path, vmax=vmax)
        df = df.sort_values("V")
        df = df[df["V"] <= float(vmax)].copy()

        # Compute the IV-matched transition voltage from the FULL activated-node
        # data (before restricting to finite-connectivity rows), so both
        # evolution curves can be trimmed at the same endpoint as the IV curve.
        v_trans = (activation_transition_voltage(df)
                   if STYLE_EVO.get("trim_at_activation_transition", True) else None)

        d = df[np.isfinite(df["algebraic_connectivity"])]

        if d.empty:
            print(
                f"Skipped algebraic connectivity for {_Path(path).name}: "
                "no finite values (column absent, empty, or all NaN)."
            )
            continue

        if STYLE_EVO.get("trim_at_activation_transition", True):
            # Trim at the IV curve's transition voltage (consistency-preserving).
            if v_trans is not None:
                d = trim_at_transition(d, v_trans)
        elif STYLE_EVO.get("trim_algebraic_plateau", True):
            d = trim_algebraic_plateau(d, ycol="algebraic_connectivity")

        if d.empty:
            print(f"Skipped algebraic connectivity for {_Path(path).name}: empty after trim.")
            continue

        ax.plot(
            d["V"],
            d["algebraic_connectivity"],
            "-o",
            linewidth=STYLE_EVO["line_width"],
            markersize=STYLE_EVO["marker_size"],
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

    ax.set_xlabel(r"$V$ [V]")
    ax.set_ylabel(r"$\lambda_2$ [-]")
    _finalise_ax_evo(ax, plotted, vmax, ymin_zero=False)
    fig.tight_layout(pad=STYLE_EVO["tight_pad"])
    savefig_evo(fig, outdir, output_name)


def plot_activated_nodes(files, outdir, output_name, vmax):
    fig, ax = plt.subplots(figsize=STYLE_EVO["figsize"])
    colors = STYLE_EVO["colors"]

    plotted = 0
    for k, path in enumerate(files):
        df = read_evolution_file(path, vmax=vmax)
        df = df.sort_values("V")
        df = df[df["V"] <= float(vmax)].copy()

        # IV-matched transition voltage from the full activated-node data.
        v_trans = (activation_transition_voltage(df)
                   if STYLE_EVO.get("trim_at_activation_transition", True) else None)

        d = df[np.isfinite(df["activated_nodes_plot"])]
        if d.empty:
            print(f"Skipped activated nodes for {_Path(path).name}: no finite values.")
            continue

        if STYLE_EVO.get("trim_at_activation_transition", True):
            if v_trans is not None:
                d = trim_at_transition(d, v_trans)
        elif STYLE_EVO.get("trim_activation_plateau", True):
            d = trim_activation_plateau(d, ycol="activated_nodes_plot")

        ax.plot(
            d["V"],
            d["activated_nodes_plot"],
            "-o",
            linewidth=STYLE_EVO["line_width"],
            markersize=STYLE_EVO["marker_size"],
            color=colors[k % len(colors)],
            label=parse_label_from_filename(path),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("No activated-node curves were plotted.")
        return

    ax.set_xlabel(r"$V$ [V]")
    ax.set_ylabel(r"Activated node count, $N_\mathrm{act}$ [-]")
    _finalise_ax_evo(ax, plotted, vmax)
    fig.tight_layout(pad=STYLE_EVO["tight_pad"])
    savefig_evo(fig, outdir, output_name)


def filter_files(files, group):
    if group is None:
        return files
    group = group.lower()
    return [f for f in files if group in _Path(f).name.lower()]


def expand_file_patterns(patterns):
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


def run_evolution_family(evo_files, outdir, evo_group, evo_prefix,
                         vmax, write_slopes=True,
                         keep_plateau=False, keep_alg_plateau=False):
    """Drive the family-B (evolution) plots, using the shared endpoint as vmax."""
    if keep_plateau:
        STYLE_EVO["trim_activation_plateau"] = False
    if keep_alg_plateau:
        STYLE_EVO["trim_algebraic_plateau"] = False

    apply_style_evo()   # switch rcParams to the evolution family's style

    files = expand_file_patterns(evo_files)
    files = filter_files(files, evo_group)
    if not files:
        print(f"[evolution] no matching evolution CSV files for "
              f"group={evo_group} — skipped")
        return

    files = sorted(files, key=sort_key_from_filename)

    prefix = evo_prefix if evo_prefix else (evo_group if evo_group else "all")

    print(f"[evolution] endpoint vmax = {vmax:g} V applied to all evolution plots")
    print("[evolution] files used:")
    for f in files:
        print(f"  {_Path(f).name}")

    plot_algebraic_connectivity(
        files, outdir, f"{prefix}_algebraic_connectivity_overlay", vmax=vmax)
    if write_slopes:
        write_algebraic_slope_csv(
            files, outdir, f"{prefix}_algebraic_connectivity_slopes", vmax=vmax)
    plot_activated_nodes(
        files, outdir, f"{prefix}_activated_nodes_overlay", vmax=vmax)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Unified plotting: case summaries + I-V/distribution overlays "
                    "(family A) plus per-voltage evolution overlays (family B), "
                    "with a shared, auto-detected x-axis endpoint.")

    # ---- family A (non-evolution) ----
    ap.add_argument("csv", nargs="?",
                    help="path to all_cases_VT_zeta.csv (optional)")
    ap.add_argument("--outdir", default="combined_plots", help="output directory")
    ap.add_argument("--only", help="plot only this case (e.g. case_4)")
    ap.add_argument("--dist", nargs="*", default=[],
                    help="one or more *_distribution_data.csv files to plot")
    ap.add_argument("--case4-iv", dest="case4_iv",
                    help="case4_VT_zeta_iv_data.csv -> I-V overlay by N")
    ap.add_argument("--seed", type=int, default=41,
                    help="seed to use for the case-4 I-V overlay (default 41)")
    ap.add_argument("--iv", nargs="*", default=[],
                    help="one or more *_VT_zeta_iv_data.csv files -> I-V overlay")
    ap.add_argument("--iv-seed", dest="iv_seed", type=int,
                    default=STYLE_CASES["iv_seed"],
                    help="representative seed for Case 1 / Case 2 I-V overlays (default 41)")
    ap.add_argument("--sampled", nargs="*", default=[],
                    help="one or more *_sampled_va.csv files -> histogram")
    ap.add_argument("--caseR-agg", dest="caseR_agg",
                    help="caseR_VT_zeta_aggregated.csv -> V_T and zeta vs void fraction")
    ap.add_argument("--caseR-iv", dest="caseR_iv",
                    help="caseR_iv_random_voids_data.csv -> I-V overlay")
    ap.add_argument("--sweep-table", dest="sweep_table", nargs="*", default=[],
                    help="one or more sweep_table_*.csv -> per-network panels")
    ap.add_argument("--sweep-iv-multi", dest="sweep_iv_multi", nargs="*", default=[],
                    help="multiple sweep_table_*.csv -> multiseed I-V band")
    ap.add_argument("--iv-aggregated", dest="iv_aggregated", nargs="*", default=[],
                    help="pre-aggregated iv_curve_multiseed_data.csv -> I-V band")
    ap.add_argument("--phases", nargs="*", type=float, default=None,
                    help="voltage breakpoints for dashed phase dividers")
    ap.add_argument("--spread", choices=["std", "minmax"], default="std",
                    help="multiseed I-V band: 'std' or 'minmax' (default std)")

    # ---- family B (evolution) ----
    ap.add_argument("--evolution", nargs="*", default=[],
                    help="evolution_*.csv files -> algebraic-connectivity and "
                         "activated-node overlays (family B)")
    ap.add_argument("--evo-group", dest="evo_group", default=None,
                    help="optional filename filter for --evolution files "
                         "(e.g. case1, case2a, case4, caseR)")
    ap.add_argument("--evo-prefix", dest="evo_prefix", default=None,
                    help="optional output filename prefix for evolution figures")
    ap.add_argument("--keep-plateau", action="store_true",
                    help="do not trim the activated-node saturation plateau")
    ap.add_argument("--keep-algebraic-plateau", action="store_true",
                    help="do not trim the algebraic-connectivity plateau")
    ap.add_argument("--no-slope-csv", action="store_true",
                    help="do not write the algebraic-connectivity slope CSV")

    # ---- shared ----
    ap.add_argument("--vmax", type=float, default=None,
                    help="common upper voltage [V] applied to BOTH families. If "
                         "omitted, family-A I-V curves end at their own stored "
                         "transition voltage, and the evolution plots use the "
                         "MAXIMUM voltage drawn across the family-A I-V curves.")
    ap.add_argument("--evo-vmax", dest="evo_vmax", type=float, default=None,
                    help="override ONLY the evolution-plot endpoint (ignores the "
                         "shared/auto-detected value). Rarely needed.")

    args = ap.parse_args()

    # Record the user's common cap (drives family A trimming AND the shared
    # endpoint handed to family B).
    ENDPOINT.user_vmax = args.vmax
    if args.vmax is not None:
        STYLE_CASES["trim_to_common_V"] = True
        STYLE_CASES["common_V_max"] = float(args.vmax)

    # -------- FAMILY A FIRST (so the endpoint can be detected) --------
    apply_style_cases()

    any_A = any([args.csv, args.dist, args.case4_iv, args.iv, args.sampled,
                 args.caseR_agg, args.caseR_iv, args.sweep_table,
                 args.sweep_iv_multi, args.iv_aggregated])

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

    # -------- FAMILY B (evolution), using the shared endpoint --------
    if args.evolution:
        # endpoint priority: --evo-vmax > --vmax > detected max > STYLE default
        if args.evo_vmax is not None:
            evo_vmax = float(args.evo_vmax)
            print(f"[evolution] using --evo-vmax override = {evo_vmax:g} V")
        else:
            shared = ENDPOINT.for_evolution()
            if shared is not None:
                evo_vmax = shared
                src = ("--vmax" if ENDPOINT.user_vmax is not None
                       else "auto-detected from family-A I-V curves")
                print(f"[evolution] endpoint {evo_vmax:g} V ({src})")
            else:
                evo_vmax = float(STYLE_EVO["V_MAX_PLOT"])
                print(f"[evolution] no family-A endpoint available; "
                      f"using STYLE_EVO default {evo_vmax:g} V")

        run_evolution_family(
            args.evolution, args.outdir, args.evo_group, args.evo_prefix,
            vmax=evo_vmax, write_slopes=not args.no_slope_csv,
            keep_plateau=args.keep_plateau,
            keep_alg_plateau=args.keep_algebraic_plateau,
        )

    if not any_A and not args.evolution:
        ap.error("provide family-A inputs (cases CSV, --dist, --iv, ...) and/or "
                 "family-B inputs (--evolution)")


if __name__ == "__main__":
    main()