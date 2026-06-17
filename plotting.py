"""
plotting.py — figures for parameter_sweep_cases.py

Restyled and rebuilt:
  * Transport exponent is labelled zeta (ζ) everywhere, never gamma.
  * "Activation voltage" (V_a) replaces "threshold voltage" in all labels.
  * Case 3 size axis is labelled "junction count" (number of nodes).
  * No path-enumeration diagnostics. The conductance-matrix heatmap is produced
    directly from the Kirchhoff Laplacian (see plot_G_matrix).

All visual constants are at the top of this file — edit them to restyle every
figure without touching the sweep driver.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# STYLE CONSTANTS  (edit here to restyle every figure)
FONT_TITLE      = 13
FONT_AXIS_LABEL = 13
FONT_TICK       = 11
FONT_LEGEND     = 10
FONT_COLORBAR   = 11

LINE_WIDTH   = 2.0
MARKER_SIZE  = 6
FIG_DPI      = 200
DIAG_DPI     = 200

# Symbols used in labels (single source of truth)
SYM_ZETA  = r"$\zeta$"
SYM_VA    = r"$V_a$"
SYM_MEANVA = r"$\langle V_a\rangle$"
SYM_SIGMA = r"$\sigma$"

def _apply_axes_style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=FONT_TITLE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(alpha=0.3)

def _series_label(r, mode):
    """Legend label for one result dict, depending on what is being varied."""
    if mode == "sigma":
        return rf"{SYM_SIGMA} = {r['sigma_va_target']:.0f} V"
    if mode == "mean":
        return rf"{SYM_MEANVA} = {r['mean_va_target']:.0f} V"
    if mode == "N":
        return rf"N = {r['N']}"
    return str(r.get("seed", ""))

def plot_iv_overlay(results, mode, title, outfile):
    """Overlay I-V curves for a list of results, coloured by the varied param."""
    results = sorted(results, key=lambda r: (
        r["sigma_va_target"] if mode == "sigma" else
        r["mean_va_target"]  if mode == "mean" else
        r["N"]))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for c, r in zip(colors, results):
        v = np.asarray(r["voltages"], dtype=float)
        i = np.asarray(r["currents"], dtype=float)
        v_end = float(r.get("transition_voltage_V", np.nan))
        if np.isfinite(v_end):
            keep = v <= v_end + 1e-9
            v, i = v[keep], i[keep]
        ax.plot(v, i * 1e9, lw=LINE_WIDTH, color=c, label=_series_label(r, mode))
    _apply_axes_style(ax, title, "Voltage (V)", "Current (nA)")
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_iv_overlay_case(results, title, outfile):
    """I-V overlay for the void-fraction sweep (Case R), coloured by void fraction."""
    results = sorted(results, key=lambda r: r.get("void_fraction", 0.0))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(results)))
    for c, r in zip(colors, results):
        fv = r.get("void_fraction", 0.0)
        v = np.asarray(r["voltages"], dtype=float)
        i = np.asarray(r["currents"], dtype=float)
        v_end = float(r.get("transition_voltage_V", np.nan))
        if np.isfinite(v_end):
            keep = v <= v_end + 1e-9
            v, i = v[keep], i[keep]
        ax.plot(v, i * 1e9, lw=LINE_WIDTH, color=c, label=rf"$f_v$ = {fv:.2f}")
    _apply_axes_style(ax, title, "Voltage (V)", "Current (nA)")
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# DISTRIBUTION OVERLAY  (activation-voltage distributions)
def plot_theoretical_distribution_overlay(results, mode, title, outfile,
                                          add_ref_line=False,
                                          clip=(0.0, 20.0)):
    """Overlay the activation-voltage normal distributions, CLIPPED to `clip`.

    The drawn Gaussian is truncated to the clip window [lo, hi] and renormalized
    so its area over that window is 1, matching the way activation voltages are
    sampled (the sampler clips V_a to [VA_MIN, VA_MAX]). With clip=(0,20) the
    sub-2 V left tail remains visible; the portion outside [0,20] is removed.
    """
    results = sorted(results, key=lambda r: (
        r["sigma_va_target"] if mode == "sigma" else r["mean_va_target"]))
    lo, hi = float(clip[0]), float(clip[1])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    x = np.linspace(lo, hi, 600)
    for c, r in zip(colors, results):
        mu, sg = r["mean_va_target"], r["sigma_va_target"]
        pdf = norm.pdf(x, mu, sg)
        # renormalize so the clipped curve integrates to 1 over [lo, hi]
        mass = norm.cdf(hi, mu, sg) - norm.cdf(lo, mu, sg)
        if mass > 0:
            pdf = pdf / mass
        ax.plot(x, pdf, lw=LINE_WIDTH, color=c, label=_series_label(r, mode))
    if add_ref_line and results:
        ax.axvline(results[0]["mean_va_target"], color="0.4", ls="--", lw=1)
    ax.set_xlim(lo, hi)
    _apply_axes_style(ax, title, rf"Activation voltage {SYM_VA} (V)", "Density")
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# ZETA vs JUNCTION COUNT
def plot_zeta_vs_N(results, title, outfile):
    """Transport exponent zeta as a function of junction count N."""
    results = sorted(results, key=lambda r: r["N"])
    Ns   = [r["N"] for r in results]
    zeta = [r.get("fit_zeta", r.get("fit_gamma", np.nan)) for r in results]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(Ns, zeta, "-o", lw=LINE_WIDTH, ms=MARKER_SIZE, color="tab:blue")
    _apply_axes_style(ax, title, "Junction count, N", SYM_ZETA)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_metric_heatmap(results, metric_key, clabel, title, outfile,
                        x_key="N", y_key="mean_va_target",
                        x_label="Junction count, N",
                        y_label=r"Mean activation voltage $\langle V_a\rangle$ (V)"):
    """
    Heatmap of a fitted metric over a 2D parameter grid.

    Cells are grouped by (x_key, y_key); if multiple seeds occupy a cell the
    mean of the metric over successful fits is shown. Empty/failed cells are
    left blank (NaN). Used for V_T and zeta over the (N, <V_a>) Case 3 grid.
    """
    import numpy as _np
    from collections import defaultdict

    def metric_of(r):
        if metric_key in ("fit_zeta", "fit_gamma"):
            return r.get("fit_zeta", r.get("fit_gamma", _np.nan))
        return r.get(metric_key, _np.nan)

    xs = sorted(set(r[x_key] for r in results))
    ys = sorted(set(r[y_key] for r in results))
    cell = defaultdict(list)
    for r in results:
        if not r.get("fit_success", True):
            continue
        v = metric_of(r)
        if v is not None and _np.isfinite(v):
            cell[(r[x_key], r[y_key])].append(float(v))

    Z = _np.full((len(ys), len(xs)), _np.nan)
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            vals = cell.get((xv, yv), [])
            if vals:
                Z[iy, ix] = _np.mean(vals)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{y:g}" for y in ys])
    # annotate each cell with its value
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            if _np.isfinite(Z[iy, ix]):
                ax.text(ix, iy, f"{Z[iy, ix]:.2f}", ha="center", va="center",
                        color="white", fontsize=FONT_TICK,
                        fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(clabel, rotation=270, labelpad=18, fontsize=FONT_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    _apply_axes_style(ax, title, x_label, y_label)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# METRIC vs VOID FRACTION  (Case R)
def plot_metric_vs_void(results, metric_key, ylabel, title, outfile):
    """Generic metric vs void fraction for the void sweep."""
    results = sorted(results, key=lambda r: r.get("void_fraction", 0.0))
    fv  = [r.get("void_fraction", 0.0) for r in results]
    # tolerate either fit_zeta or legacy fit_gamma for the exponent
    if metric_key in ("fit_zeta", "fit_gamma"):
        y = [r.get("fit_zeta", r.get("fit_gamma", np.nan)) for r in results]
    else:
        y = [r.get(metric_key, np.nan) for r in results]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(fv, y, "-o", lw=LINE_WIDTH, ms=MARKER_SIZE, color="tab:purple")
    _apply_axes_style(ax, title, r"Void fraction $f_v$", ylabel)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def save_topology_plot(n_junctions, mean_va, sigma_va, seed, outfile,
                       build_network_fn):
    """Draw the network topology, nodes coloured by activation voltage."""
    net = build_network_fn(n_junctions, mean_va, sigma_va, seed)
    pos = net.positions
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for i, j in net.G.edges():
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                lw=0.4, alpha=0.2, color="0.45")
    ids  = list(net.G.nodes())
    vals = [net.G.nodes[n]["Vth"] for n in ids]
    sc = ax.scatter(pos[ids, 0], pos[ids, 1], c=vals, s=16,
                    cmap="viridis", edgecolors="none")
    if net.source_nodes:
        s = np.array(sorted(net.source_nodes))
        ax.scatter(pos[s, 0], pos[s, 1], marker="s", s=55, c="tab:green",
                   edgecolors="k", linewidths=0.4)
    if net.drain_nodes:
        d = np.array(sorted(net.drain_nodes))
        ax.scatter(pos[d, 0], pos[d, 1], marker="s", s=55, c="tab:red",
                   edgecolors="k", linewidths=0.4)
    ax.set_aspect("equal")
    _apply_axes_style(ax, f"Network topology (N={n_junctions})",
                      "x position", "y position")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label(rf"Activation voltage {SYM_VA} (V)", rotation=270,
                   labelpad=16, fontsize=FONT_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def save_topology_plot_case(result, outfile):
    """Topology plot for a Case R result that already carries positions/edges/voids."""
    pos = np.asarray(result["positions"])
    edges = result.get("edges", [])
    voids = result.get("voids", [])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for e in edges:
        i, j = int(e[0]), int(e[1])
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                lw=0.4, alpha=0.2, color="0.45")
    src = set(result.get("source_nodes", []))
    drn = set(result.get("drain_nodes", []))
    other = [n for n in range(len(pos)) if n not in src and n not in drn]
    if other:
        ax.scatter(pos[other, 0], pos[other, 1], s=10, c="steelblue", alpha=0.5)
    if src:
        s = np.array(sorted(src))
        ax.scatter(pos[s, 0], pos[s, 1], marker="s", s=55, c="tab:green",
                   edgecolors="k", linewidths=0.4)
    if drn:
        d = np.array(sorted(drn))
        ax.scatter(pos[d, 0], pos[d, 1], marker="s", s=55, c="tab:red",
                   edgecolors="k", linewidths=0.4)
    for v in voids:
        circ = plt.Circle(v["center"], v["radius"], facecolor="red",
                          edgecolor="darkred", alpha=0.45, lw=0.8, zorder=0)
        ax.add_patch(circ)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fv = result.get("void_fraction", 0.0)
    _apply_axes_style(ax, f"Topology (N={result['N']}, $f_v$={fv:.2f})",
                      "x position", "y position")
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def _trim_rows_to_vend(rows, v_end):
    """Return rows whose 'V' is <= v_end (with a tiny tolerance). If v_end is
    not finite, return rows unchanged. Used to end the evolution/I-V curves at
    the nonlinear->linear transition voltage."""
    try:
        ve = float(v_end)
    except (TypeError, ValueError):
        return rows
    if not np.isfinite(ve):
        return rows
    return [r for r in rows if float(r.get("V", np.nan)) <= ve + 1e-9]

def plot_evolution(rows, fields, title, outfile, percolation_V=None, v_end=None):
    """
    Four-panel per-voltage network-evolution figure, plotted from the same rows
    written to the evolution CSV.

    Panels: activation growth; activated vs conducting edges; I-V (solver +
    charge-conserving); current concentration (participation + backbone).

    When v_end is finite, points beyond it are dropped so the curves end at the
    nonlinear->linear transition voltage rather than running to V_MAX.
    """
    import numpy as _np
    rows = _trim_rows_to_vend(rows, v_end)
    def col(name):
        return _np.array([r.get(name, _np.nan) for r in rows], dtype=float)

    V   = col("V")
    an  = col("activated_nodes")
    ae  = col("activated_edges")
    ce  = col("conducting_edges")
    I   = col("total_current_A")
    Icc = col("total_current_chargeconserving_A")
    part = col("participation_ratio")
    bb   = col("backbone_edges")

    if percolation_V is None:
        sdc = col("source_drain_connected")
        idx = _np.where(sdc >= 1)[0]
        percolation_V = float(V[idx[0]]) if len(idx) else None

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold")

    a = ax[0, 0]
    a.plot(V, an, "-o", ms=4, color="tab:blue", label="activated nodes")
    _apply_axes_style(a, "Activation grows as V crosses node $V_a$",
                      "Voltage (V)", "count")
    a.legend(fontsize=FONT_LEGEND)

    a = ax[0, 1]
    a.plot(V, ae, "-", color="tab:gray", label="activated edges")
    a.plot(V, ce, "-", color="tab:green", label="conducting edges")
    _apply_axes_style(a, "Activated vs conducting edges",
                      "Voltage (V)", "edge count")
    a.legend(fontsize=FONT_LEGEND)

    a = ax[1, 0]
    a.plot(V, I * 1e9, "-o", ms=4, color="tab:red", label="solver current")
    a.plot(V, Icc * 1e9, "--", color="tab:orange", label="charge-conserving")
    _apply_axes_style(a, "I-V curve (Kirchhoff)", "Voltage (V)", "Current (nA)")
    a.legend(fontsize=FONT_LEGEND)

    a = ax[1, 1]
    a.plot(V, part, "-", color="tab:purple", label="participation ratio")
    a.plot(V, bb, "-", color="tab:orange", label="backbone edges (>=1% max)")
    _apply_axes_style(a, "Current concentration", "Voltage (V)", "effective # edges")
    a.legend(fontsize=FONT_LEGEND)

    if percolation_V is not None:
        for a in ax.ravel():
            a.axvline(percolation_V, color="k", ls="--", lw=1, alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_structure_evolution(rows, fields, title, outfile, percolation_V=None, v_end=None):
    """
    Connected-component structure of the activated subgraph vs voltage.

    Panels:
      (1) largest-component fraction (percolation order parameter) — rises 0->1
          through the transition.
      (2) second-largest component size and number of components — the
          second-largest classically peaks at the percolation threshold.
    """
    import numpy as _np
    rows = _trim_rows_to_vend(rows, v_end)
    def col(name):
        return _np.array([r.get(name, _np.nan) for r in rows], dtype=float)

    V       = col("V")
    frac    = col("largest_cc_fraction")
    second  = col("second_cc_nodes")
    ncomp   = col("num_components")
    if percolation_V is None:
        sdc = col("source_drain_connected")
        idx = _np.where(sdc >= 1)[0]
        percolation_V = float(V[idx[0]]) if len(idx) else None

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold")

    a = ax[0]
    a.plot(V, frac, "-o", ms=4, color="tab:blue")
    _apply_axes_style(a, "Largest-component fraction\n(percolation order parameter)",
                      "Voltage (V)", "fraction of activated nodes")
    a.set_ylim(-0.02, 1.02)

    a = ax[1]
    a.plot(V, second, "-o", ms=4, color="tab:red", label="2nd-largest comp. (nodes)")
    a.set_ylabel("2nd-largest component (nodes)", fontsize=FONT_AXIS_LABEL,
                 color="tab:red")
    a.tick_params(axis="y", labelcolor="tab:red", labelsize=FONT_TICK)
    a.tick_params(axis="x", labelsize=FONT_TICK)
    a.set_xlabel("Voltage (V)", fontsize=FONT_AXIS_LABEL)
    a.set_title("Second-largest component & component count\n"
                "(2nd-largest peaks near threshold)", fontsize=FONT_TITLE)
    a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(V, ncomp, "-s", ms=3, color="tab:green", alpha=0.7,
            label="# components")
    a2.set_ylabel("number of components", fontsize=FONT_AXIS_LABEL,
                  color="tab:green")
    a2.tick_params(axis="y", labelcolor="tab:green", labelsize=FONT_TICK)

    if percolation_V is not None:
        for a in ax:
            a.axvline(percolation_V, color="k", ls="--", lw=1, alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_current_distribution_evolution(rows, fields, title, outfile,
                                        percolation_V=None, v_end=None):
    """
    Current-distribution shape vs voltage: how concentrated the flow is.

    Panels:
      (1) Gini coefficient and top-10% current fraction (inequality of flow).
      (2) coefficient of variation and max/mean ratio (spread / dominance).
    All four drop as voltage rises if current delocalizes onto more edges.
    """
    import numpy as _np
    rows = _trim_rows_to_vend(rows, v_end)
    def col(name):
        return _np.array([r.get(name, _np.nan) for r in rows], dtype=float)

    V     = col("V")
    gini  = col("current_gini")
    top10 = col("current_top10_fraction")
    cv    = col("current_cv")
    m2m   = col("current_max_to_mean")
    if percolation_V is None:
        sdc = col("source_drain_connected")
        idx = _np.where(sdc >= 1)[0]
        percolation_V = float(V[idx[0]]) if len(idx) else None

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold")

    a = ax[0]
    a.plot(V, gini, "-o", ms=4, color="tab:purple", label="Gini coefficient")
    a.plot(V, top10, "-s", ms=3, color="tab:orange",
           label="top-10% current fraction")
    _apply_axes_style(a, "Current inequality\n(1 = all in one edge, 0 = even)",
                      "Voltage (V)", "fraction / coefficient")
    a.set_ylim(-0.02, 1.02)
    a.legend(fontsize=FONT_LEGEND)

    a = ax[1]
    a.plot(V, cv, "-o", ms=4, color="tab:blue", label="coeff. of variation")
    a.plot(V, m2m, "-s", ms=3, color="tab:red", label="max / mean")
    _apply_axes_style(a, "Current spread & dominance",
                      "Voltage (V)", "ratio")
    a.legend(fontsize=FONT_LEGEND)

    if percolation_V is not None:
        for a in ax:
            a.axvline(percolation_V, color="k", ls="--", lw=1, alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_spectral_evolution(rows, fields, title, outfile, percolation_V=None, v_end=None):
    """
    Full-system-matrix spectral metrics vs voltage.

    Panels:
      (1) effective source-to-drain resistance (log scale) — the network's
          overall resistance from the full node+edge matrix; matches V/I.
      (2) algebraic connectivity (Fiedler value) — how robustly the conducting
          network holds together. Rises as more nodes activate; a dip would flag
          the conducting network nearly pinching into two pieces (a bottleneck).
    """
    import numpy as _np
    rows = _trim_rows_to_vend(rows, v_end)
    def col(name):
        return _np.array([r.get(name, _np.nan) for r in rows], dtype=float)

    V    = col("V")
    Reff = col("effective_resistance_ohm")
    ac   = col("algebraic_connectivity")
    if percolation_V is None:
        sdc = col("source_drain_connected")
        idx = _np.where(sdc >= 1)[0]
        percolation_V = float(V[idx[0]]) if len(idx) else None

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold")

    a = ax[0]
    mask = _np.isfinite(Reff) & (Reff > 0)
    a.semilogy(V[mask], Reff[mask], "-o", ms=4, color="tab:red")
    _apply_axes_style(a, "Effective source-drain resistance",
                      "Voltage (V)", r"$R_{eff}$ ($\Omega$)")

    a = ax[1]
    mask2 = _np.isfinite(ac)
    a.plot(V[mask2], ac[mask2], "-o", ms=4, color="tab:blue")
    _apply_axes_style(a, "Algebraic connectivity (Fiedler value)\n"
                      "(network robustness; dips = near-bottleneck)",
                      "Voltage (V)", "algebraic connectivity")

    if percolation_V is not None:
        for a in ax:
            a.axvline(percolation_V, color="k", ls="--", lw=1, alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_metric_vs_parameter_errorbars(results, x_key, metric_key, xlabel, ylabel,
                                       title, outfile, label_prefix=None):
    """
    Generic parameter-sweep error-bar plot.

    Groups repeated-seed results by `x_key` and plots mean +/- sample std for
    `metric_key`. Failed fits and non-finite values are excluded. This is used
    for Case 1 and Case 2, where each sigma or mean activation-voltage value is
    now evaluated over all configured seeds.
    """
    import numpy as _np
    from collections import defaultdict

    def metric_of(r):
        if metric_key in ("fit_zeta", "fit_gamma"):
            return r.get("fit_zeta", r.get("fit_gamma", _np.nan))
        return r.get(metric_key, _np.nan)

    groups = defaultdict(list)
    for r in results:
        ok = r.get("fit_success", True)
        v = metric_of(r)
        x = r.get(x_key, _np.nan)
        if ok and x is not None and _np.isfinite(x) and v is not None and _np.isfinite(v):
            groups[float(x)].append(float(v))

    xs = sorted(groups.keys())
    means = _np.array([_np.mean(groups[x]) for x in xs])
    stds = _np.array([_np.std(groups[x], ddof=1) if len(groups[x]) > 1 else 0.0
                      for x in xs])
    nseed = [len(groups[x]) for x in xs]
    nmax = max(nseed, default=0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(xs, means, yerr=stds, fmt="-o", ms=MARKER_SIZE, lw=LINE_WIDTH,
                color="tab:blue", ecolor="tab:blue", elinewidth=1.2,
                capsize=4, capthick=1.2,
                label=f"mean $\\pm$ std (n={nmax} seeds)")
    _apply_axes_style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_zeta_vs_N_errorbars(results, title, outfile):
    """
    Transport exponent zeta vs junction count N, with error bars (mean +/- std
    over seeds). `results` may contain multiple entries per N (one per seed);
    they are grouped by N and reduced to mean +/- std. Failed fits (NaN / not
    success) are dropped from the statistics.
    """
    import numpy as _np
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        z = r.get("fit_zeta", r.get("fit_gamma", _np.nan))
        ok = r.get("fit_success", True)
        if ok and z is not None and _np.isfinite(z):
            groups[r["N"]].append(float(z))

    Ns = sorted(groups.keys())
    means = _np.array([_np.mean(groups[n]) for n in Ns])
    stds = _np.array([_np.std(groups[n], ddof=1) if len(groups[n]) > 1 else 0.0
                      for n in Ns])
    nmax = max((len(groups[n]) for n in Ns), default=0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(Ns, means, yerr=stds, fmt="-o", ms=MARKER_SIZE, lw=LINE_WIDTH,
                color="tab:blue", ecolor="tab:blue", elinewidth=1.2,
                capsize=4, capthick=1.2,
                label=f"mean $\\pm$ std (n={nmax} seeds)")
    _apply_axes_style(ax, title, "Junction count, N", SYM_ZETA)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_metric_vs_void_errorbars(results, metric_key, ylabel, title, outfile):
    """
    Metric vs void fraction with error bars (mean +/- std over seeds).

    `results` may contain MULTIPLE entries per void fraction (one per seed);
    they are grouped by void_fraction and reduced to mean +/- std. Points with a
    single seed get no visible bar (std = 0). Failed fits (NaN) are dropped from
    the statistics.
    """
    import numpy as _np
    from collections import defaultdict

    def metric_of(r):
        if metric_key in ("fit_zeta", "fit_gamma"):
            return r.get("fit_zeta", r.get("fit_gamma", _np.nan))
        return r.get(metric_key, _np.nan)

    groups = defaultdict(list)
    for r in results:
        v = metric_of(r)
        if v is not None and _np.isfinite(v):
            groups[r.get("void_fraction", 0.0)].append(float(v))

    fvs = sorted(groups.keys())
    means = _np.array([_np.mean(groups[f]) for f in fvs])
    stds = _np.array([_np.std(groups[f], ddof=1) if len(groups[f]) > 1 else 0.0
                      for f in fvs])
    nseed = [len(groups[f]) for f in fvs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(fvs, means, yerr=stds, fmt="-o", ms=MARKER_SIZE,
                lw=LINE_WIDTH, color="tab:blue", ecolor="tab:blue",
                elinewidth=1.2, capsize=4, capthick=1.2,
                label=f"mean $\\pm$ std (n={max(nseed)} seeds)")
    _apply_axes_style(ax, title, r"Void fraction $f_v$", ylabel)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# NETWORK SNAPSHOTS  (conduction region spreading; edge width/color = |current|)
def plot_snapshots(net, snap_voltages, outfile, title=None, R_MIN=1.0):
    """
    Multi-panel network snapshots showing the conduction region spread with
    voltage. Edges are drawn with width and color proportional to |current|.
    Edge currents are computed on the spot from the live network (via
    sweep_analysis._edge_currents), so nothing large needs to be stored.

    Parameters
    ----------
    net           : a built NanoparticleNetwork (sources/drains identified)
    snap_voltages : list of voltages to show as panels
    outfile       : output PNG path
    title         : optional suptitle (defaults to a standard caption)
    """
    import numpy as _np
    from matplotlib.collections import LineCollection
    import sweep_analysis as _sa

    pos = net.positions
    snap_voltages = sorted(set(float(v) for v in snap_voltages))

    # compute edge currents at each snapshot voltage
    ec_by_V = {}
    for V in snap_voltages:
        activated = {n for n in net.G.nodes() if net.G.nodes[n]["Vth"] <= V}
        _, _, ec = _sa._edge_currents(net, activated, V)
        ec_by_V[V] = ec

    # global max |current| for consistent width/color scaling across panels
    all_max = max((max((abs(c) for c in ec.values()), default=0.0)
                   for ec in ec_by_V.values()), default=1.0) or 1.0

    n_panels = len(snap_voltages)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    src_set = set(net.source_nodes)
    drn_set = set(net.drain_nodes)

    for ax, V in zip(axes, snap_voltages):
        ec = ec_by_V[V]
        # faint backbone of all edges
        segs_all = [[(pos[i][0], pos[i][1]), (pos[j][0], pos[j][1])]
                    for i, j in net.G.edges()]
        ax.add_collection(LineCollection(segs_all, colors="lightgray",
                                         linewidths=0.2, alpha=0.25, zorder=1))
        # current-carrying edges
        if ec:
            segs, widths, vals = [], [], []
            for (i, j), c in ec.items():
                segs.append([(pos[i][0], pos[i][1]), (pos[j][0], pos[j][1])])
                mag = abs(c)
                widths.append(0.3 + 4.0 * mag / all_max)
                vals.append(mag / all_max)
            lc = LineCollection(segs, array=_np.array(vals), cmap="plasma",
                                linewidths=widths, zorder=3)
            lc.set_clim(0, 1)
            ax.add_collection(lc)
        # nodes
        activated = {n for n in net.G.nodes() if net.G.nodes[n]["Vth"] <= V}
        off = [n for n in net.G.nodes() if n not in activated]
        on = list(activated)
        if off:
            ax.scatter(pos[off, 0], pos[off, 1], s=4, c="lightgray",
                       alpha=0.6, zorder=2)
        if on:
            ax.scatter(pos[on, 0], pos[on, 1], s=22, c="steelblue",
                       edgecolors="navy", linewidths=0.3, alpha=0.85, zorder=4)
        if src_set:
            s = _np.array(sorted(src_set))
            ax.scatter(pos[s, 0], pos[s, 1], s=28, marker="s", c="green",
                       edgecolors="k", lw=0.4, zorder=5)
        if drn_set:
            d = _np.array(sorted(drn_set))
            ax.scatter(pos[d, 0], pos[d, 1], s=28, marker="s", c="red",
                       edgecolors="k", lw=0.4, zorder=5)
        n_cond = len(ec)
        n_act = len(activated)
        ax.set_title(f"V = {V:g} V\n{n_act} activated nodes, "
                     f"{n_cond} conducting edges", fontsize=10)
        ax.set_xlim(-0.02, net.domain[0] + 0.02)
        ax.set_ylim(-0.02, net.domain[1] + 0.02)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # build a consistent legend (proxy handles) on the first panel
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="green",
               markeredgecolor="k", markersize=7, label="source"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="red",
               markeredgecolor="k", markersize=7, label="drain"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="steelblue",
               markeredgecolor="navy", markersize=7, label="activated node"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="lightgray",
               markeredgecolor="none", markersize=6, label="inactive node"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("|edge current| / max", fontsize=9)
    if title is None:
        title = "Conduction region spreading with voltage\n(edge width & color = current magnitude)"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outfile

# CONDUCTANCE MATRIX HEATMAP  (Kirchhoff Laplacian; no path enumeration)
def plot_G_matrix(net, voltage, outfile, R_MIN=1.0):
    """
    log10 conductance-matrix heatmap over the activated subgraph at one voltage.
    Built directly from edge conductances (the Laplacian), with no path search.
    Writes a companion long-form CSV of nonzero entries.
    """
    import csv
    Vr = float(voltage)
    activated = sorted(n for n in net.G.nodes() if net.G.nodes[n]["Vth"] <= Vr)
    if not activated:
        return None
    idx = {v: k for k, v in enumerate(activated)}
    N = len(activated)
    G = np.zeros((N, N))
    for (i, j) in net.G.edges():
        if i in idx and j in idx:
            g = 1.0 / max(net.G[i][j]["R_edge"], R_MIN)
            ki, kj = idx[i], idx[j]
            G[ki, ki] += g; G[kj, kj] += g
            G[ki, kj] -= g; G[kj, ki] -= g

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(np.abs(G) + 1e-20), cmap="plasma",
                   aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}|G_{ij}|$ (S)", rotation=270, labelpad=16,
                   fontsize=FONT_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    src = [k for k, n in enumerate(activated) if n in set(net.source_nodes)]
    drn = [k for k, n in enumerate(activated) if n in set(net.drain_nodes)]
    ax.set_xticks(src + drn)
    ax.set_xticklabels(["S"] * len(src) + ["D"] * len(drn), fontsize=6, color="white")
    ax.set_yticks(src + drn)
    ax.set_yticklabels(["S"] * len(src) + ["D"] * len(drn), fontsize=6, color="white")
    _apply_axes_style(ax, f"Conductance matrix at V={Vr:g} V ({N} active nodes)",
                      "matrix index", "matrix index")
    fig.tight_layout()
    fig.savefig(outfile, dpi=DIAG_DPI, bbox_inches="tight")
    plt.close(fig)

    csv_path = str(outfile).rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "col_index", "row_node", "col_node", "conductance_S"])
        for i in range(N):
            for j in range(N):
                if G[i, j] != 0.0:
                    w.writerow([i, j, activated[i], activated[j], G[i, j]])
    return outfile
