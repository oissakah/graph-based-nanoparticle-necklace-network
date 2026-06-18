"""
parameter_sweep_cases.py

Parameter sweep: Cases 1-4 plus Case R (random void fraction).

Cases 1-4  — activation-voltage width/mean and junction-count sweeps using
             analytical normal distributions and the Kirchhoff solver.
  Case 1   — vary activation-voltage width (standard deviation) at fixed mean.
  Case 2   — vary mean activation voltage at two widths.
  Case 3   — vary junction count (number of nodes) crossed with mean.
  Case 4   — vary junction count at fixed mean and width.

Case R     — fixed N=500, random void-fraction sweep (uniform-radius random
             void placement).

The transport exponent fitted to I = A (V - V_T)^zeta is reported as zeta
throughout (symbol ζ). "Activation voltage" (V_a) is used in place of
"threshold voltage" in labels.

Outputs
-------
Each run creates timestamped sub-directories under IV_results/:
  parameter_sweep_cases_<timestamp>/   — Cases 1-4 figures + CSVs
  case_R_random_voids_<timestamp>/     — Case R figures + CSVs

For every individual run a per-voltage network-evolution CSV is also written
(activated_nodes/edges, conducting_edges, source_drain_connected,
total_current_A, conductance_S, participation_ratio/backbone_edges).

Plot appearance
---------------
All font sizes, line widths, marker sizes and output DPI are controlled in
plotting.py — edit the constants at the top of that file to restyle every
figure without touching this module.
"""

import os
# worker process and does not fight with multiprocessing for CPU cores.
os.environ.setdefault("OMP_NUM_THREADS",    "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS",    "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

import sys
import pickle
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import pandas as pd

from nanoparticle_network import NanoparticleNetwork

EDGE_K             = 2.0e10
NODE_R_SCALE       = 5.0e8
CONNECTION_RADIUS  = 0.15
DOMAIN_SIZE        = (1.0, 1.0)
LEFT_THRESH        = 0.15
RIGHT_THRESH       = 0.85
CALCULATION_METHOD = "kirchhoff"   # "kirchhoff" or "bfs"
TUNNELING_EFFECTS  = True

V_START    = 0.0
V_MAX      = 16.0
V_STEP     = 0.5
FIT_WINDOW = 10.0

# Use the same snapshot voltages for all snapshot/diagnostic figures.
SNAPSHOT_VOLTAGES = [1.0, 2.0, 6.0, 14.0]

VA_MIN   = 0.0
VA_MAX   = 20.0

# Unified random seeds — single source of truth for all cases.
# Loaded from optimized_config.yaml ("seeds:") so this script and the
# sweep_analysis / run_sweep_analysis tools use the SAME seeds. Falls back to a
# default list if the config file is absent (this script can run without it).
# Each case draws seeds deterministically as SEEDS[i % len(SEEDS)].
def _load_seeds():
    default = [41, 51, 61, 71, 81]
    try:
        import yaml
        with open("optimized_config.yaml") as _f:
            cfg = yaml.safe_load(_f)
        seeds = cfg.get("seeds")
        if seeds:
            return [int(s) for s in seeds]
    except Exception:
        pass
    return default

SEEDS = _load_seeds()

def seed_for(i):
    """Deterministic seed for the i-th swept parameter value."""
    return SEEDS[i % len(SEEDS)]

RESULTS_ROOT = Path("IV_results")

# Limit multiprocessing on Windows to avoid SciPy DLL import failures caused by
# too many worker processes exhausting RAM/page-file space. Increase to 3 or 4
# only if your computer has enough memory.
MAX_WORKERS = 2

def pool_workers(n_tasks):
    """Return a safe worker count for multiprocessing pools."""
    return max(1, min(MAX_WORKERS, int(n_tasks)))

SIGMA_VALUES_CASE1 = [1.0, 3.0, 5.0, 7.0]   # activation-voltage widths
CONST_MEAN_CASE1   = 8.0
N_CASE12           = 500

MEAN_VALUES_CASE2  = [4.0, 6.0, 8.0, 10.0]
SIGMA_VALUES_CASE2 = [1.0, 3.0, 5.0, 7.0]   # Case 2 now includes all activation-voltage widths
SIGMA_CASE2A       = 1.0                    # legacy alias: low-width subset
SIGMA_CASE2B       = 5.0                    # legacy alias: high-width subset

N_VALUES_CASE3    = [200, 400, 600, 800]    # junction counts
MEAN_VALUES_CASE3 = [4.0, 6.0, 8.0, 10.0]
SIGMA_CASE3       = 3.0

N_VALUES_CASE4          = [200, 400, 600, 800]
CONST_MEAN_CASE4        = 6.0
SIGMA_CASE4             = 3.0
TOPOLOGY_REFERENCE_MEAN = 6.0

# CASE R CONSTANTS (random void-fraction sweep)

VOID_FRACTIONS_CASER = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
CONST_MEAN_CASER     = 6.0
SIGMA_CASER          = 3.0
CASER_REF_RADIUS     = 0.15
CASER_REF_N          = 500

# Fixed void radius (domain units). All voids are identical circles; the void
# count needed to reach a target area fraction follows directly from this.
VOID_RADIUS = 0.08

def ensure_dir(path):
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def _save_csv(df, outfile):
    """
    Write a DataFrame to CSV alongside a companion PNG or data file.

    Mirrors the same helper in plotting.py so CSV files can also be
    generated from pure-data functions (e.g. make_fit_table).

    Parameters
    ----------
    df      : pandas.DataFrame — data to export
    outfile : str or Path      — destination path (any extension is accepted;
                                 the extension is NOT changed here)
    """
    csv_path = Path(outfile)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")

def first_positive_span_idx(v, i, span=1.0):
    """
    Return the index of the first voltage at which current has been
    continuously positive for at least `span` volts.

    This noise-robust detector ignores isolated positive current spikes;
    the onset is only declared real once current stays positive over a
    contiguous voltage window of width >= span.

    Parameters
    ----------
    v    : array-like — voltages (monotonically increasing)
    i    : array-like — currents (same length as v)
    span : float      — minimum continuous positive window in V

    Returns
    -------
    int or None — onset index, or None if no percolation detected.
    """
    v = np.asarray(v)
    i = np.asarray(i)
    pos   = i > 0
    start = None
    for idx, is_pos in enumerate(pos):
        if is_pos:
            if start is None:
                start = idx
            if v[idx] - v[start] >= span:
                return start
        else:
            start = None
    return None

def connectivity_transition_voltage(V, ac, plateau_frac=0.95,
                                    band=0.15, smooth_win=5):
    """
    Detect the nonlinear->linear transition voltage from an algebraic-
    connectivity curve.

    Physically: as bias rises, more junctions activate and the conducting
    cluster grows; algebraic connectivity climbs and then *flattens* once
    almost all activated nodes are connected. That plateau onset marks the end
    of the nonlinear regime — beyond it the I-V grows roughly linearly. This
    returns the voltage at that plateau onset.

    Method (robust to plateau noise):
      1. light moving-average smoothing (window `smooth_win`);
      2. estimate the plateau level as the median of the upper-voltage tail
         (last 25% of points);
      3. the transition is the FIRST voltage whose smoothed connectivity is
         >= plateau_frac * plateau AND which never afterwards drops below
         (plateau_frac - band) * plateau (so single-point noise dips on the
         plateau do not disqualify an early, correct crossing).

    Parameters
    ----------
    V            : array-like — voltages (any order; sorted internally)
    ac           : array-like — algebraic connectivity at each V
    plateau_frac : float — fraction of the plateau that counts as "reached"
    band         : float — tolerance (fraction of plateau) for later dips
    smooth_win   : int   — moving-average window (points)

    Returns
    -------
    float or None — transition voltage, or None if it cannot be determined.
    """
    V = np.asarray(V, dtype=float)
    ac = np.asarray(ac, dtype=float)
    m = np.isfinite(V) & np.isfinite(ac)
    V, ac = V[m], ac[m]
    if len(ac) < 4:
        return None
    order = np.argsort(V)
    V, ac = V[order], ac[order]

    if smooth_win > 1 and len(ac) >= smooth_win:
        kern = np.ones(smooth_win) / smooth_win
        acs = np.convolve(ac, kern, mode="same")
    else:
        acs = ac.copy()

    plateau = np.median(acs[int(0.75 * len(acs)):])
    if not np.isfinite(plateau) or plateau <= 0:
        return None

    target = plateau_frac * plateau
    floor = (plateau_frac - band) * plateau
    for idx in range(len(acs)):
        if acs[idx] >= target and np.all(acs[idx:] >= floor):
            return float(V[idx])
    # never settles cleanly -> fall back to the peak-connectivity voltage
    return float(V[int(np.argmax(acs))])

# Fraction of total nodes that must be activated to declare the
# nonlinear->linear transition (the point where "almost all" nodes are on).
ACTIVATION_TRANSITION_FRAC = 0.90

def activation_transition_voltage(V, activated, n_total,
                                  frac=ACTIVATION_TRANSITION_FRAC):
    """
    Detect the nonlinear->linear transition as the onset voltage at which
    almost all nodes have activated.

    Physically: as bias rises, junctions switch on one by one; the I-V is
    strongly nonlinear while new nodes keep activating, and becomes
    quasi-linear once essentially every node is already on (no new conducting
    paths appear, only the existing network is driven harder). This returns the
    FIRST voltage at which the activated-node count reaches `frac` of the total
    node count.

    Parameters
    ----------
    V         : array-like — voltages (any order; sorted internally)
    activated : array-like — activated-node count at each V
    n_total   : int        — total number of nodes in the network
    frac      : float      — fraction of total nodes that counts as "almost all"

    Returns
    -------
    float or None — transition voltage, or None if it cannot be determined.
    """
    V = np.asarray(V, dtype=float)
    activated = np.asarray(activated, dtype=float)
    m = np.isfinite(V) & np.isfinite(activated)
    V, activated = V[m], activated[m]
    if len(V) == 0 or not n_total or n_total <= 0:
        return None
    order = np.argsort(V)
    V, activated = V[order], activated[order]

    target = frac * float(n_total)
    reached = np.where(activated >= target)[0]
    if len(reached) == 0:
        return None
    return float(V[reached[0]])

def _transition_voltage_from_evolution(evo):
    """
    Pull V and the activated-node count out of a sweep_analysis evolution
    result and return the activation-based transition voltage: the first
    voltage at which almost all nodes (>= ACTIVATION_TRANSITION_FRAC of the
    total) have activated. Returns NaN if unavailable.
    """
    rows = evo.get("rows") if isinstance(evo, dict) else None
    if not rows:
        return float("nan")
    n_total = evo.get("n_total_nodes")
    if not n_total:
        # fall back to the largest activated count seen in the sweep
        n_total = max((r.get("activated_nodes", 0) or 0) for r in rows)
    V = np.array([r.get("V") for r in rows], dtype=float)
    an = np.array([r.get("activated_nodes", np.nan) for r in rows], dtype=float)
    vt = activation_transition_voltage(V, an, n_total)
    return float(vt) if vt is not None else float("nan")

def fit_power_law(v_arr, i_arr, fit_window=FIT_WINDOW, v_step=V_STEP):
    """
    Fit I = A (V - V_T)^zeta on the conducting branch.

    Robust version:
      * Fits in LOG space (log I = log A + zeta * log(V - V_T)) so the result is
        independent of the absolute current scale (~1e-9 A here would otherwise
        make curve_fit's relative tolerances declare premature convergence and
        leave zeta parked at its initial guess).
      * Refines V_T by a small 1-D scan, then takes zeta as the slope of a linear
        fit in log-log space (scale-free, well-conditioned).
      * Returns success=False with a 'reason' when the data genuinely cannot
        support a 3-parameter power law, instead of fabricating zeta=1.5.

    Returned dict keys: success, V_T, zeta, A, R2, reason.
    """
    nan = float("nan")
    perc_idx = first_positive_span_idx(v_arr, i_arr, span=max(1.0, v_step))
    if perc_idx is None:
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason="no_conduction")

    V_T_guess = float(v_arr[perc_idx])
    fit_window_clamped = min(fit_window, max(0.0, float(v_arr[-1]) - V_T_guess))
    mask = (v_arr >= V_T_guess) & (v_arr <= V_T_guess + fit_window_clamped) & (i_arr > 0)
    V_fit = np.asarray(v_arr[mask], dtype=float)
    I_fit = np.asarray(i_arr[mask], dtype=float)

    if len(V_fit) < 4:
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason=f"too_few_points({len(V_fit)})")

    def loglog_slope(V_T):
        """Linear fit log(I) = log A + zeta log(V - V_T); return (zeta, logA, R2)."""
        dV = V_fit - V_T
        good = dV > 0
        if good.sum() < 4:
            return None
        x = np.log(dV[good])
        y = np.log(I_fit[good])
        # linear least squares
        A_mat = np.vstack([x, np.ones_like(x)]).T
        (zeta, logA), res, *_ = np.linalg.lstsq(A_mat, y, rcond=None)
        y_pred = zeta * x + logA
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
        return float(zeta), float(logA), float(R2)

    # Scan V_T just below the first-conduction voltage to maximize log-log R2.
    # V_T must be < smallest V_fit so that (V - V_T) > 0 for all fit points.
    v_min = V_fit.min()
    best = None
    # candidate V_T values: from a bit below percolation up to just under v_min
    lo = max(0.0, V_T_guess - 2.0 * max(v_step, 0.5))
    hi = v_min - 1e-6
    if hi <= lo:
        hi = lo + 1e-6
    for V_T_try in np.linspace(lo, hi, 40):
        out = loglog_slope(V_T_try)
        if out is None:
            continue
        zeta, logA, R2 = out
        if best is None or (np.isfinite(R2) and R2 > best[3]):
            best = (V_T_try, zeta, logA, R2)

    if best is None:
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason="loglog_failed")

    V_T_fit, zeta_fit, logA_fit, R2 = best
    A_fit = float(np.exp(logA_fit))

    # Sanity / honesty checks
    if not np.isfinite(zeta_fit) or not np.isfinite(R2):
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason="nonfinite")
    if R2 < 0.80:
        return dict(success=False, V_T=float(V_T_fit), zeta=float(zeta_fit),
                    A=A_fit, R2=float(R2), reason=f"poor_fit_R2={R2:.3f}")
    if zeta_fit <= 0.1 or zeta_fit >= 10.0:
        return dict(success=False, V_T=float(V_T_fit), zeta=float(zeta_fit),
                    A=A_fit, R2=float(R2), reason="zeta_out_of_range")

    return dict(success=True, V_T=float(V_T_fit), zeta=float(zeta_fit),
                A=A_fit, R2=float(R2), reason="ok")

# CASES 1–4: NETWORK BUILDER AND RUNNERS

def build_network(n_junctions, mean_va, sigma_va, seed):
    net = NanoparticleNetwork(
        n_junctions=n_junctions,
        connection_radius=CONNECTION_RADIUS,
        domain=DOMAIN_SIZE,
    )
    node_Va = {
        "type": "normal",
        "mean": float(mean_va),
        "std":  float(sigma_va),
        "min":  float(VA_MIN),
        "max":  float(VA_MAX),
    }
    net.generate_network(
        seed=int(seed),
        node_Vth=node_Va,       # API keyword must stay 'node_Vth'
        edge_k=float(EDGE_K),
        node_r_scale=float(NODE_R_SCALE),
    )
    net.identify_sources_drains(
        left_thresh=float(LEFT_THRESH),
        right_thresh=float(RIGHT_THRESH),
    )
    return net

def run_iv(n_junctions, mean_va, sigma_va, seed):
    net = build_network(n_junctions, mean_va, sigma_va, seed)

    if CALCULATION_METHOD.lower() == "kirchhoff":
        iv = net.calculate_iv_curve_kirchhoff(
            V_start=V_START, V_max=V_MAX, V_step=V_STEP
        )
    else:
        iv = net.calculate_iv_curve(
            V_start=V_START, V_max=V_MAX, V_step=V_STEP,
            tunneling_effects=TUNNELING_EFFECTS,
        )

    node_va = np.array([net.G.nodes[n]["Vth"] for n in net.G.nodes()], dtype=float)
    peak_idx = int(np.argmax(iv["currents"]))
    perc_idx = next((i for i, c in enumerate(iv["currents"]) if c > 0), None)
    fit      = fit_power_law(
        np.asarray(iv["voltages"], dtype=float),
        np.asarray(iv["currents"], dtype=float),
    )

    # per-voltage network-evolution metrics
    import sweep_analysis as _sa
    evo = _sa.sweep(net, V_START, V_MAX, V_STEP)

    # nonlinear->linear transition voltage from the connectivity plateau
    transition_V = _transition_voltage_from_evolution(evo)

    return {
        "N":                     int(n_junctions),
        "mean_va_target":       float(mean_va),
        "sigma_va_target":      float(sigma_va),
        "seed":                  int(seed),
        "voltages":              np.asarray(iv["voltages"], dtype=float),
        "currents":              np.asarray(iv["currents"], dtype=float),
        "conductances":          np.asarray(iv["conductances"], dtype=float),
        "node_va":              node_va,
        "n_nodes_actual":        int(net.G.number_of_nodes()),
        "n_edges":               int(net.G.number_of_edges()),
        "n_sources":             int(len(net.source_nodes)),
        "n_drains":              int(len(net.drain_nodes)),
        "sampled_mean_va":      float(np.mean(node_va)),
        "sampled_std_va":       float(np.std(node_va)),
        "peak_current_A":        float(iv["currents"][peak_idx]),
        "peak_voltage_V":        float(iv["voltages"][peak_idx]),
        "percolation_voltage_V": float(iv["voltages"][perc_idx]) if perc_idx is not None else np.nan,
        "transition_voltage_V":  float(transition_V),
        "max_conductance_S":     float(np.max(iv["conductances"])),
        "fit_V_T_V":             float(fit["V_T"])   if fit["success"] else np.nan,
        "fit_zeta":              float(fit["zeta"]) if fit["success"] else np.nan,
        "fit_R2":                float(fit["R2"])    if fit["success"] else np.nan,
        "fit_success":           bool(fit["success"]),
        "fit_reason":            fit.get("reason", ""),
        "evolution_rows":        evo["rows"],
        "evolution_fields":      evo["fieldnames"],
    }

def worker(args):
    return run_iv(*args)

def make_fit_table(results, case_name):
    """
    Build a summary DataFrame of fitted transport parameters for one sweep case.

    Returns
    -------
    pandas.DataFrame with activation-voltage columns and the transport
    exponent reported as 'fit_zeta'.
    """
    rows = []
    for r in results:
        rows.append({
            "case":                  case_name,
            "N":                     r["N"],
            "mean_Va_target_V":      r["mean_va_target"],
            "sigma_Va_target_V":     r["sigma_va_target"],
            "seed":                  r["seed"],
            "sampled_mean_Va_V":     r["sampled_mean_va"],
            "sampled_sigma_Va_V":    r["sampled_std_va"],
            "percolation_voltage_V": r["percolation_voltage_V"],
            "transition_voltage_V":  r.get("transition_voltage_V", np.nan),
            "fit_V_T_V":             r["fit_V_T_V"],
            "fit_zeta":              r["fit_zeta"],
            "fit_R2":                r["fit_R2"],
            "peak_current_A":        r["peak_current_A"],
            "max_conductance_S":     r["max_conductance_S"],
        })
    return pd.DataFrame(rows)

def representative_results(results, group_key):
    """Return one representative result per group, using the lowest seed."""
    reps = {}
    for r in results:
        key = r[group_key]
        if key not in reps or r["seed"] < reps[key]["seed"]:
            reps[key] = r
    return [reps[k] for k in sorted(reps)]

def aggregate_fit_table(results, group_key, group_col_name):
    """
    Aggregate V_T and zeta by a swept parameter using successful finite fits.

    Writes one row per parameter value with mean, sample standard deviation, and
    n_seeds. Failed fits are excluded from the statistics.
    """
    rows = []
    for val in sorted(set(r[group_key] for r in results)):
        grp = [r for r in results if r[group_key] == val]
        vt = np.array([r["fit_V_T_V"] for r in grp
                       if r.get("fit_success", True) and np.isfinite(r["fit_V_T_V"])],
                      dtype=float)
        zt = np.array([r["fit_zeta"] for r in grp
                       if r.get("fit_success", True) and np.isfinite(r["fit_zeta"])],
                      dtype=float)
        rows.append({
            group_col_name: val,
            "n_seeds": int(max(len(vt), len(zt))),
            "VT_mean": float(np.mean(vt)) if len(vt) else np.nan,
            "VT_std": float(np.std(vt, ddof=1)) if len(vt) > 1 else 0.0,
            "zeta_mean": float(np.mean(zt)) if len(zt) else np.nan,
            "zeta_std": float(np.std(zt, ddof=1)) if len(zt) > 1 else 0.0,
        })
    return pd.DataFrame(rows)

def save_summary_csv(rows, out_csv):
    """
    Write a flat list of scalar result dicts to CSV.

    Parameters
    ----------
    rows    : list of dicts — one dict per simulation run
    out_csv : Path          — destination CSV path

    Notes
    -----
    Array-valued keys (voltages, currents, node_va, etc.) should be
    excluded from `rows` before calling this function.
    """
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df

# Keys that must never go into the flat summary CSVs (arrays / nested lists)
_ARRAY_KEYS = {"voltages", "currents", "conductances", "node_va",
               "positions", "edges", "source_nodes", "drain_nodes", "voids",
               "evolution_rows", "evolution_fields"}

def _scalar_row(r):
    """Drop array/nested keys so a result dict can go into a flat CSV."""
    return {k: v for k, v in r.items() if k not in _ARRAY_KEYS}

def _iv_rows_for_result(r, trim_at_transition=True):
    """
    Build I-V CSV rows (label, voltage_V, current_A, current_nA) for one result.

    When trim_at_transition is True, points beyond that run's connectivity
    transition voltage are dropped, so the exported I-V curve ends at the
    nonlinear->linear transition rather than running to V_MAX. Falls back to
    the full curve if the transition voltage is unavailable.
    """
    lbl = (f"N={r['N']}_mean{r['mean_va_target']:.0f}"
           f"_sigma{r['sigma_va_target']:.0f}_seed{r['seed']}")
    v_end = r.get("transition_voltage_V", float("nan"))
    rows = []
    for v, i in zip(r["voltages"], r["currents"]):
        if (trim_at_transition and np.isfinite(v_end)
                and float(v) > float(v_end) + 1e-9):
            continue
        rows.append({"label": lbl,
                     "voltage_V": float(v),
                     "current_A": float(i),
                     "current_nA": float(i) * 1e9})
    return rows

def save_distribution_csv(results, outdir, fname, mode="sigma",
                          v_min=0.0, v_max=20.0, n_points=600):
    """
    Write the activation-voltage Gaussian distribution curves to CSV, so the
    distribution overlay figure can be reproduced/replotted from data.

    The Gaussian is CLIPPED to [v_min, v_max] (default [0, 20]) and renormalized
    so its area over that window is 1 — matching the clipped plot and the way
    V_a is sampled. One shared `V_a` column plus one density column per distinct
    curve, named `density_mean{m}_sigma{s}`.
    """
    import pandas as _pd
    from scipy.stats import norm as _norm
    # unique (mean, sigma) pairs, in stable order
    seen, pairs = set(), []
    for r in results:
        key = (round(r["mean_va_target"], 6), round(r["sigma_va_target"], 6))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    x = np.linspace(v_min, v_max, n_points)
    data = {"V_a": x}
    for (m, s) in pairs:
        if s <= 0:
            continue
        pdf = np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        mass = _norm.cdf(v_max, m, s) - _norm.cdf(v_min, m, s)
        if mass > 0:
            pdf = pdf / mass
        data[f"density_mean{m:g}_sigma{s:g}"] = pdf
    _pd.DataFrame(data).to_csv(outdir / fname, index=False)

def save_sampled_va_csv(results, outdir, fname):
    """
    Write the ACTUAL sampled (and clipped) activation voltages V_a for each
    curve, so the distribution can be plotted as a histogram of the real node
    values rather than the analytic Gaussian. Columns are named
    `va_mean{m}_sigma{s}`; one column per distinct (mean, sigma) curve, each
    holding that representative network's node_va values (length = N, so columns
    may differ in length and are NaN-padded).
    """
    import pandas as _pd

    # one representative result per (mean, sigma) — first seed encountered
    seen, cols = set(), {}
    for r in results:
        key = (round(r["mean_va_target"], 6), round(r["sigma_va_target"], 6))
        if key in seen or "node_va" not in r:
            continue
        seen.add(key)
        m, s = key
        cols[f"va_mean{m:g}_sigma{s:g}"] = np.asarray(r["node_va"], dtype=float)

    if not cols:
        return

    maxlen = max(len(v) for v in cols.values())
    data = {
        k: np.concatenate([v, np.full(maxlen - len(v), np.nan)])
        for k, v in cols.items()
    }
    _pd.DataFrame(data).to_csv(Path(outdir) / fname, index=False)

def save_snapshots(results, outdir, case_name, snap_voltages=None):
    """
    Spatial network snapshots (node/edge layout, conduction-region spreading)
    for each representative result in `results`.

    The live network objects are built inside worker processes and not returned,
    so each representative network is REBUILT here from its (N, mean, sigma,
    seed) — deterministic because build_network is seeded, so the rebuilt network
    is identical to the one that produced the result. plotting.plot_snapshots
    then draws multi-panel snapshots (edges colored/scaled by |current|).

    snap_voltages defaults to four voltages spread across [V_START, V_MAX].
    """
    import plotting as _P
    snap_dir = Path(outdir) / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    if snap_voltages is None:
        # four evenly spaced snapshot voltages within the swept range,
        # skipping V_START (no conduction there)
        lo = V_START + (V_MAX - V_START) * 0.125
        snap_voltages = list(np.linspace(lo, V_MAX, 4))

    for r in results:
        # rebuild the identical network deterministically from its parameters
        net = build_network(int(r["N"]), float(r["mean_va_target"]),
                            float(r["sigma_va_target"]), int(r["seed"]))
        tag = (f"{case_name}_N{int(r['N'])}_mean{r['mean_va_target']:.0f}"
               f"_sigma{r['sigma_va_target']:.0f}_seed{int(r['seed'])}")
        title = (rf"{case_name} snapshots — $N={int(r['N'])}$, "
                 rf"$\langle V_a\rangle={r['mean_va_target']:.0f}$ V, "
                 rf"$\sigma={r['sigma_va_target']:.0f}$ V")
        outfile = snap_dir / f"snapshots_{tag}.png"
        try:
            _P.plot_snapshots(net, snap_voltages, str(outfile), title=title)
            print(f"  wrote {outfile}")
        except Exception as e:
            print(f"  [snapshots] {tag}: failed ({e})")

def save_evolution_csvs(results, outdir, case_name):
    """
    Write one per-voltage network-evolution CSV per run, plus a matching
    four-panel evolution plot next to it. Columns:
      V, activated_nodes, activated_edges, conducting_nodes, conducting_edges,
      source_drain_connected, total_current_A, total_current_chargeconserving_A,
      conductance_S, backbone_edges, participation_ratio.
    """
    import plotting as _P
    evo_dir = Path(outdir) / "evolution_csv"
    evo_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        rows = r.get("evolution_rows")
        if not rows:
            continue
        fields = r.get("evolution_fields") or list(rows[0].keys())
        tag = (f"{case_name}_N{r['N']}_mean{r['mean_va_target']:.0f}"
               f"_sigma{r['sigma_va_target']:.0f}_seed{r['seed']}")
        fv = r.get("void_fraction")
        if fv is not None:
            tag += f"_fv{fv:.2f}"
        df = pd.DataFrame(rows, columns=fields)
        df.to_csv(evo_dir / f"evolution_{tag}.csv", index=False)
        # matching plot next to the CSV
        title = (rf"Network evolution — N={r['N']}, "
                 rf"$\langle V_a\rangle$={r['mean_va_target']:.0f} V, "
                 rf"$\sigma$={r['sigma_va_target']:.0f} V")
        if fv is not None:
            title += rf", $f_v$={fv:.2f}"
        _P.plot_evolution(rows, fields, title,
                          evo_dir / f"evolution_{tag}.png")
        _P.plot_structure_evolution(rows, fields,
                          "Component structure — " + title,
                          evo_dir / f"structure_{tag}.png")
        _P.plot_current_distribution_evolution(rows, fields,
                          "Current distribution — " + title,
                          evo_dir / f"current_dist_{tag}.png")
        _P.plot_spectral_evolution(rows, fields,
                          "Spectral metrics — " + title,
                          evo_dir / f"spectral_{tag}.png")

def scaled_radius_for_N(n):
    """Scale connection radius so mean node degree stays constant with N."""
    return CASER_REF_RADIUS * np.sqrt(CASER_REF_N / float(n))

def make_voids_random_sampling(target_fv, seed, radius=None, n_junctions=500):
    """
    Fixed-radius random void placement.

    All voids are identical circles of radius `radius`. The number of voids
    needed to reach the target area fraction f_v on the unit domain follows
    directly from the void area:

        n_v = round( f_v * L^2 / (pi * radius^2) ),   L = 1.

    Void centres are drawn uniformly over the interior of the domain (subject
    to a minimal boundary buffer so each void lies fully inside). Voids may
    overlap; no inter-void separation constraint is imposed. Because overlap
    is allowed, the realized depleted area can be slightly below n_v * pi r^2,
    but for the fractions studied here (f_v <= 0.25) the difference is small.

    Parameters
    ----------
    target_fv   : float — target void area fraction (0 to ~0.30)
    seed        : int   — random seed
    radius      : float or None — void radius in domain units (default VOID_RADIUS)
    n_junctions : int   — kept for interface compatibility (unused)

    Returns
    -------
    list of dict — each with keys 'center' (tuple) and 'radius' (float)
    """
    if target_fv <= 0.0:
        return []
    r = float(radius) if radius is not None else float(VOID_RADIUS)

    n_v = int(round(target_fv / (np.pi * r * r)))   # L = 1
    if n_v < 1:
        return []

    rng = np.random.default_rng(seed)
    buf = 0.02  # minimal boundary buffer so voids stay inside the domain
    lo, hi = buf + r, 1.0 - buf - r
    if hi <= lo:   # radius too large for the domain; clamp centre to mid-domain
        lo = hi = 0.5

    voids = []
    for _ in range(n_v):
        cx = rng.uniform(lo, hi)
        cy = rng.uniform(lo, hi)
        voids.append({"center": (float(cx), float(cy)), "radius": r})
    return voids

def compute_void_stats(voids, target_fv):
    if not voids:
        return dict(
            n_voids=0, achieved_void_area=0.0,
            achieved_fv_error=float(target_fv),
            mean_void_radius=np.nan, std_void_radius=np.nan,
            min_void_radius=np.nan,  max_void_radius=np.nan,
        )
    radii    = np.array([v["radius"] for v in voids])
    achieved = float(np.sum(np.pi * radii ** 2))
    return dict(
        n_voids=int(len(voids)),
        achieved_void_area=achieved,
        achieved_fv_error=float(abs(achieved - target_fv)),
        mean_void_radius=float(np.mean(radii)),
        std_void_radius=float(np.std(radii)),
        min_void_radius=float(np.min(radii)),
        max_void_radius=float(np.max(radii)),
    )

def point_in_void(point, voids):
    x, y = point
    return any(
        (x - v["center"][0]) ** 2 + (y - v["center"][1]) ** 2 < v["radius"] ** 2
        for v in voids
    )

def segment_intersects_circle(p1, p2, center, radius):
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    c  = np.array(center, float)
    d  = p2 - p1
    if np.allclose(d, 0.0):
        return np.linalg.norm(p1 - c) <= radius
    t = np.dot(c - p1, d) / np.dot(d, d)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p1 + t * d - c) < radius

def segment_intersects_any_void(p1, p2, voids):
    return any(
        segment_intersects_circle(p1, p2, v["center"], v["radius"])
        for v in voids
    )

def generate_positions_with_voids(n, domain_size, voids, seed):
    rng        = np.random.default_rng(seed)
    pts        = []
    max_trials = n * 20000
    trials     = 0
    while len(pts) < n and trials < max_trials:
        p = (rng.uniform(0.0, domain_size[0]), rng.uniform(0.0, domain_size[1]))
        if not point_in_void(p, voids):
            pts.append(p)
        trials += 1
    if len(pts) < n:
        raise RuntimeError(
            f"Could only place {len(pts)}/{n} nodes outside voids. "
            "Reduce void fraction or increase domain size."
        )
    return np.array(pts, dtype=float)

def solve_network_iv(net):
    if CALCULATION_METHOD.lower() == "kirchhoff":
        return net.calculate_iv_curve_kirchhoff(
            V_start=V_START, V_max=V_MAX, V_step=V_STEP
        )
    return net.calculate_iv_curve(
        V_start=V_START, V_max=V_MAX, V_step=V_STEP,
        tunneling_effects=TUNNELING_EFFECTS,
    )

# MAIN

def _worker_case_r(args):
    """Top-level worker: one Case R (N, mean, sigma, seed, fv) combination."""
    import networkx as _nx
    from scipy.spatial import cKDTree as _cKDTree

    N, mean, sigma, seed_r, fv = args
    voids     = make_voids_random_sampling(float(fv), seed_r, n_junctions=N)
    positions = generate_positions_with_voids(N, DOMAIN_SIZE, voids, seed_r)

    net = NanoparticleNetwork(
        n_junctions=N,
        connection_radius=scaled_radius_for_N(N),
        domain=DOMAIN_SIZE,
    )
    net.positions = positions
    net.G = _nx.Graph()

    rng     = np.random.default_rng(seed_r + 1000)
    raw_vth = np.clip(rng.normal(float(mean), float(sigma), size=N),
                      VA_MIN, VA_MAX)
    for i in range(N):
        Va = float(raw_vth[i])
        net.G.add_node(i, pos=positions[i], Vth=Va,
                       R_node=float(NODE_R_SCALE) * Va, activated=False)

    rc     = scaled_radius_for_N(N)
    _tree  = _cKDTree(positions)
    _pairs = _tree.query_pairs(r=rc, output_type="ndarray")
    for i, j in _pairs:
        if not segment_intersects_any_void(positions[i], positions[j], voids):
            dij    = float(np.linalg.norm(positions[i] - positions[j]))
            net.G.add_edge(i, j, R_edge=float(EDGE_K) * dij, distance=dij)

    changed = True
    while changed:
        low     = [n for n in net.G.nodes() if net.G.degree(n) < 2]
        changed = bool(low)
        net.G.remove_nodes_from(low)
    net.identify_sources_drains(
        left_thresh=float(LEFT_THRESH), right_thresh=float(RIGHT_THRESH))

    iv  = solve_network_iv(net)
    fit = fit_power_law(np.asarray(iv["voltages"], float),
                        np.asarray(iv["currents"],  float))

    # per-voltage network-evolution metrics (activated/conducting/current/etc.)
    import sweep_analysis as _sa
    evo = _sa.sweep(net, V_START, V_MAX, V_STEP)

    # nonlinear->linear transition voltage from the connectivity plateau
    transition_V = _transition_voltage_from_evolution(evo)

    _el = list(net.G.edges())
    return {
        "N":                int(N),
        "mean_va_target":   float(mean),
        "sigma_va_target":  float(sigma),
        "seed":             int(seed_r),
        "void_fraction":        float(fv),
        "void_fraction_target": float(fv),
        "void_method":      "random",
        "voltages":         np.asarray(iv["voltages"], float),
        "currents":         np.asarray(iv["currents"],  float),
        "fit_V_T_V":        float(fit["V_T"])   if fit["success"] else np.nan,
        "fit_zeta":         float(fit["zeta"]) if fit["success"] else np.nan,
        "fit_A":            float(fit["A"])     if fit["success"] else np.nan,
        "fit_R2":           float(fit["R2"])    if fit["success"] else np.nan,
        "fit_success":      bool(fit["success"]),
        "fit_reason":       fit.get("reason", ""),
        "transition_voltage_V": float(transition_V),
        "n_voids":          int(len(voids)),
        "mean_void_radius": float(np.mean([v["radius"] for v in voids]))
                            if voids else np.nan,
        "positions":        np.asarray(net.positions),
        "edges":            (np.array(_el, dtype=np.int32)
                             if _el else np.zeros((0, 2), dtype=np.int32)),
        "source_nodes":     list(net.source_nodes),
        "drain_nodes":      list(net.drain_nodes),
        "voids":            voids,
        # per-voltage evolution rows (list of dicts) for the evolution CSV
        "evolution_rows":   evo["rows"],
        "evolution_fields": evo["fieldnames"],
    }

def run_G_matrix_diagnostics(results, diag_base_dir, case_name,
                             diag_voltages, void_fraction=None):
    """
    Kirchhoff conductance-matrix heatmaps (no path enumeration).

    For each result, rebuild its network and write a log10 |G| heatmap + CSV
    at each requested voltage via plotting.plot_G_matrix.
    """
    import plotting as _P
    diag_base_dir = Path(diag_base_dir)
    diag_base_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        tag = (f"{case_name}_N{r['N']}"
               f"_mean{r['mean_va_target']:.0f}"
               f"_sigma{r['sigma_va_target']:.0f}"
               f"_seed{r['seed']}")
        fv = r.get("void_fraction", void_fraction)
        if fv is None or fv == 0.0:
            net = build_network(int(r["N"]), float(r["mean_va_target"]),
                                float(r["sigma_va_target"]), int(r["seed"]))
        else:
            net = _build_void_network(int(r["N"]), float(r["mean_va_target"]),
                                      float(r["sigma_va_target"]),
                                      int(r["seed"]), float(fv))[0]
        for V in diag_voltages:
            _P.plot_G_matrix(net, float(V),
                             diag_base_dir / f"{tag}_G_heatmap_V{float(V):.1f}.png")
    print(f"  [G-matrix] heatmaps -> {diag_base_dir}")

def run_snapshots(results, snap_base_dir, case_name,
                  snap_voltages=None, void_fraction=None):
    """
    Network snapshot panels (conduction region spreading; edge width/color =
    |current|) for each result. Rebuilds each network once and draws the panels
    at the chosen voltages. Defaults to low / mid / high of the sweep range.
    """
    import plotting as _P
    snap_base_dir = Path(snap_base_dir)
    snap_base_dir.mkdir(parents=True, exist_ok=True)
    if snap_voltages is None:
        snap_voltages = SNAPSHOT_VOLTAGES
    snap_voltages = [v for v in snap_voltages if V_START <= v <= V_MAX]

    for r in results:
        tag = (f"{case_name}_N{r['N']}"
               f"_mean{r['mean_va_target']:.0f}"
               f"_sigma{r['sigma_va_target']:.0f}"
               f"_seed{r['seed']}")
        fv = r.get("void_fraction", void_fraction)
        if fv is not None:
            tag += f"_fv{fv:.2f}"
        if fv is None or fv == 0.0:
            net = build_network(int(r["N"]), float(r["mean_va_target"]),
                                float(r["sigma_va_target"]), int(r["seed"]))
        else:
            net = _build_void_network(int(r["N"]), float(r["mean_va_target"]),
                                      float(r["sigma_va_target"]),
                                      int(r["seed"]), float(fv))[0]
        _P.plot_snapshots(
            net, snap_voltages,
            snap_base_dir / f"snapshots_{tag}.png",
            title=(f"Conduction region spreading with voltage — {tag}\n"
                   f"(edge width & color = current magnitude)"))

        #      positions) so the panels can be replotted independently ----
        import sweep_analysis as _sa
        pos = net.positions
        rows = []
        for V in snap_voltages:
            activated = {n for n in net.G.nodes()
                         if net.G.nodes[n]["Vth"] <= V}
            _, _, ec = _sa._edge_currents(net, activated, V)
            for (i, j), cur in ec.items():
                rows.append({
                    "V": float(V),
                    "node_i": int(i), "node_j": int(j),
                    "x_i": float(pos[i][0]), "y_i": float(pos[i][1]),
                    "x_j": float(pos[j][0]), "y_j": float(pos[j][1]),
                    "current_A": float(cur), "abs_current_A": abs(float(cur)),
                })
        pd.DataFrame(rows).to_csv(
            snap_base_dir / f"edge_currents_snapshots_{tag}.csv", index=False)
        # node activation states at each snapshot voltage 
        nrows = []
        for V in snap_voltages:
            for n in net.G.nodes():
                nrows.append({
                    "V": float(V), "node": int(n),
                    "x": float(pos[n][0]), "y": float(pos[n][1]),
                    "activated": bool(net.G.nodes[n]["Vth"] <= V),
                    "is_source": n in net.source_nodes,
                    "is_drain": n in net.drain_nodes,
                })
        pd.DataFrame(nrows).to_csv(
            snap_base_dir / f"snapshot_nodes_{tag}.csv", index=False)
    print(f"  [snapshots] -> {snap_base_dir}")

def _build_void_network(n, mean_va, sigma_va, seed, void_fraction):
    """Build a random-void network (shared by Case R diagnostics)."""
    import networkx as nx
    from scipy.spatial import cKDTree
    voids = make_voids_random_sampling(float(void_fraction), seed=seed + 500,
                                       n_junctions=n)
    net = NanoparticleNetwork(n_junctions=n,
                              connection_radius=scaled_radius_for_N(n),
                              domain=DOMAIN_SIZE)
    net.positions = generate_positions_with_voids(n, DOMAIN_SIZE, voids, seed)
    net.G = nx.Graph()
    rng = np.random.default_rng(seed + 1000)
    raw_vth = np.clip(rng.normal(float(mean_va), float(sigma_va), size=n),
                      VA_MIN, VA_MAX)
    for i in range(n):
        Va = float(raw_vth[i])
        net.G.add_node(i, pos=net.positions[i], Vth=Va,
                       R_node=float(NODE_R_SCALE) * Va, activated=False)
    tree = cKDTree(net.positions)
    for i, j in tree.query_pairs(r=net.connection_radius, output_type="ndarray"):
        if not segment_intersects_any_void(net.positions[i], net.positions[j], voids):
            dij = float(np.linalg.norm(net.positions[i] - net.positions[j]))
            net.G.add_edge(i, j, resistance=float(EDGE_K) * dij, distance=dij,
                           length=dij, R_edge=float(EDGE_K) * dij, activated=False)
    while True:
        low = [nd for nd, deg in net.G.degree() if deg < 2]
        if not low:
            break
        net.G.remove_nodes_from(low)
    net.identify_sources_drains(left_thresh=float(LEFT_THRESH),
                                right_thresh=float(RIGHT_THRESH))
    return net, voids

def main():
    import plotting as P

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Cases 1–4 directories ─────────────────────────────────────────────────
    outdir      = RESULTS_ROOT / f"parameter_sweep_cases_{timestamp}"
    plotdir     = outdir / "plots"          # ← all figures go here
    ensure_dir(outdir)
    ensure_dir(plotdir)

    print("=" * 70)
    print("RUNNING PARAMETER SWEEP CASES 1–5")
    print("=" * 70)
    print(f"Output root : {RESULTS_ROOT}")
    print(f"Cases 1–4  : {outdir}")
    print(f"Plots folder: {plotdir}")

    all_rows       = []
    all_fit_tables = []

    # ── CASE 1: vary σ at fixed ⟨V_a⟩ = 8 V, N = 500 ───────────────────────
    print("\n[Case 1] Vary σ …")
    case1_tasks = [
        (N_CASE12, CONST_MEAN_CASE1, sigma, seed)
        for sigma in SIGMA_VALUES_CASE1
        for seed in SEEDS
    ]
    with Pool(processes=pool_workers(len(case1_tasks)), maxtasksperchild=4) as pool:
        case1_results = list(pool.imap_unordered(worker, case1_tasks, chunksize=1))
    case1_results.sort(key=lambda r: (r["sigma_va_target"], r["seed"]))
    case1_rep = representative_results(case1_results, "sigma_va_target")

    for r in case1_results:
        r["case"] = "case_1"
        all_rows.append(_scalar_row(r))

    P.plot_theoretical_distribution_overlay(
        case1_rep, mode="sigma",
        title=rf"$V_a$ distributions: varying width $\sigma$ "
              rf"($\langle V_a\rangle={CONST_MEAN_CASE1:.0f}\ \mathrm{{V}}$ fixed)",
        outfile=plotdir / "case1_distribution_constant_mean_vary_sigma_N500.png",
        add_ref_line=True,
    )
    save_distribution_csv(case1_results, outdir,
                          "case1_distribution_data.csv", mode="sigma")
    save_sampled_va_csv(case1_results, outdir, "case1_sampled_va.csv")
    P.plot_iv_overlay(
        case1_rep, mode="sigma",
        title=rf"I-V curves at $N={N_CASE12}$: varying activation-voltage width $\sigma$",
        outfile=plotdir / "case1_iv_constant_mean_vary_sigma_N500.png",
    )
    fit_case1 = make_fit_table(case1_results, "case_1")
    fit_case1.to_csv(outdir / "case1_VT_zeta.csv", index=False)
    agg_case1 = aggregate_fit_table(case1_results, "sigma_va_target", "sigma_Va_target_V")
    agg_case1.to_csv(outdir / "case1_VT_zeta_aggregated.csv", index=False)
    P.plot_metric_vs_parameter_errorbars(
        case1_results, "sigma_va_target", "fit_V_T_V",
        rf"Activation-voltage width $\sigma$ (V)", r"$V_T$ (V)",
        rf"$V_T$ vs $\sigma$ at $\langle V_a\rangle={CONST_MEAN_CASE1:.0f}$ V",
        plotdir / "case1_VT_vs_sigma_errorbars.png",
    )
    P.plot_metric_vs_parameter_errorbars(
        case1_results, "sigma_va_target", "fit_zeta",
        rf"Activation-voltage width $\sigma$ (V)", r"$\zeta$",
        rf"$\zeta$ vs $\sigma$ at $\langle V_a\rangle={CONST_MEAN_CASE1:.0f}$ V",
        plotdir / "case1_zeta_vs_sigma_errorbars.png",
    )
    save_evolution_csvs(case1_rep, outdir, "case1")

    # Export individual I-V curve data for case1
    # Each row is one (label, voltage, current) data point.
    iv_rows_case1 = []
    for r in case1_results:
        iv_rows_case1.extend(_iv_rows_for_result(r))
    _save_csv(pd.DataFrame(iv_rows_case1),
              outdir / "case1_VT_zeta_iv_data.csv")
    all_fit_tables.append(fit_case1)

    # ── CASE 2: vary ⟨V_a⟩ at multiple σ values ───────────────────────────
    # This replaces the older Case 2A/2B-only structure.  Case 2 now runs the
    # mean-voltage sweep for σ = 1, 3, 5, and 7 V.  For backward compatibility,
    # σ=1 is still also saved as case2a and σ=5 as case2b.
    print("[Case 2] Vary ⟨V_a⟩ at σ = 1, 3, 5, and 7 V …")

    case2_results_by_sigma = {}
    case2_rep_by_sigma = {}
    case2a_results = []
    case2b_results = []

    for sigma_case2 in SIGMA_VALUES_CASE2:
        sigma_tag = f"sigma{sigma_case2:g}".replace(".", "p")
        case_name_sigma = f"case2_{sigma_tag}"
        print(f"[Case 2] Running mean sweep at σ = {sigma_case2:g} V …")

        case2_tasks = [
            (N_CASE12, mean, sigma_case2, seed)
            for mean in MEAN_VALUES_CASE2
            for seed in SEEDS
        ]
        with Pool(processes=pool_workers(len(case2_tasks)), maxtasksperchild=4) as pool:
            case2_results = list(pool.imap_unordered(worker, case2_tasks, chunksize=1))
        case2_results.sort(key=lambda r: (r["mean_va_target"], r["seed"]))
        case2_rep = representative_results(case2_results, "mean_va_target")

        case2_results_by_sigma[float(sigma_case2)] = case2_results
        case2_rep_by_sigma[float(sigma_case2)] = case2_rep

        # Legacy names so older analysis/replot scripts still find case2a/case2b.
        if float(sigma_case2) == float(SIGMA_CASE2A):
            case2a_results = case2_results
            legacy_case_name = "case_2a"
            legacy_prefix = "case2a"
        elif float(sigma_case2) == float(SIGMA_CASE2B):
            case2b_results = case2_results
            legacy_case_name = "case_2b"
            legacy_prefix = "case2b"
        else:
            legacy_case_name = case_name_sigma
            legacy_prefix = case_name_sigma

        for r in case2_results:
            r["case"] = legacy_case_name
            all_rows.append(_scalar_row(r))

        # Distribution overlay and data for this σ.
        P.plot_theoretical_distribution_overlay(
            case2_rep, mode="mean",
            title=rf"$V_a$ Distributions: Varying $\langle V_a\rangle$ "
                  rf"($\sigma={sigma_case2:.0f}\ \mathrm{{V}}$ fixed)",
            outfile=plotdir / f"{legacy_prefix}_distribution_constant_sigma{sigma_case2:g}_vary_mean_N500.png",
        )
        save_distribution_csv(case2_results, outdir,
                              f"{legacy_prefix}_distribution_data.csv", mode="mean")
        save_sampled_va_csv(case2_results, outdir, f"{legacy_prefix}_sampled_va.csv")

        # I-V overlay for this σ.
        P.plot_iv_overlay(
            case2_rep, mode="mean",
            title=rf"I-V Curves at $N={N_CASE12}$: $\sigma={sigma_case2:.0f}\ \mathrm{{V}}$ fixed",
            outfile=plotdir / f"{legacy_prefix}_iv_constant_sigma{sigma_case2:g}_vary_mean_N500.png",
        )

        # Fit tables and aggregated tables.
        fit_case2_sigma = make_fit_table(case2_results, legacy_case_name)
        fit_case2_sigma.to_csv(outdir / f"{legacy_prefix}_VT_zeta.csv", index=False)
        agg_case2_sigma = aggregate_fit_table(case2_results, "mean_va_target", "mean_Va_target_V")
        agg_case2_sigma.to_csv(outdir / f"{legacy_prefix}_VT_zeta_aggregated.csv", index=False)

        P.plot_metric_vs_parameter_errorbars(
            case2_results, "mean_va_target", "fit_V_T_V",
            rf"Mean activation voltage $\langle V_a\rangle$ (V)", r"$V_T$ (V)",
            rf"$V_T$ vs $\langle V_a\rangle$ at $\sigma={sigma_case2:.0f}$ V",
            plotdir / f"{legacy_prefix}_VT_vs_meanVa_errorbars.png",
        )
        P.plot_metric_vs_parameter_errorbars(
            case2_results, "mean_va_target", "fit_zeta",
            rf"Mean activation voltage $\langle V_a\rangle$ (V)", r"$\zeta$",
            rf"$\zeta$ vs $\langle V_a\rangle$ at $\sigma={sigma_case2:.0f}$ V",
            plotdir / f"{legacy_prefix}_zeta_vs_meanVa_errorbars.png",
        )

        save_evolution_csvs(case2_rep, outdir, legacy_prefix)

        # Snapshot voltages now match the other cases exactly.
        run_snapshots(
            case2_rep,
            snap_base_dir = plotdir / f"{legacy_prefix}_snapshots",
            case_name     = legacy_prefix,
            snap_voltages = SNAPSHOT_VOLTAGES,
        )

        # Export individual I-V curve data for this σ.
        iv_rows_case2 = []
        for r in case2_results:
            iv_rows_case2.extend(_iv_rows_for_result(r))
        _save_csv(pd.DataFrame(iv_rows_case2),
                  outdir / f"{legacy_prefix}_VT_zeta_iv_data.csv")

        all_fit_tables.append(fit_case2_sigma)

    # Combined Case 2 table across all σ values.
    case2_all_results = [r for group in case2_results_by_sigma.values() for r in group]
    if case2_all_results:
        make_fit_table(case2_all_results, "case_2_all_sigma").to_csv(
            outdir / "case2_all_sigma_VT_zeta.csv", index=False
        )

    # ── CASE 3: vary ⟨V_a⟩ and N at σ = 3 V ────────────────────────────────
    print("[Case 3] Vary ⟨V_a⟩ × N …")
    case3_tasks  = []
    _c3 = 0
    for N in N_VALUES_CASE3:
        for mean in MEAN_VALUES_CASE3:
            case3_tasks.append((N, mean, SIGMA_CASE3, seed_for(_c3)))
            _c3 += 1

    with Pool(processes=pool_workers(len(case3_tasks)), maxtasksperchild=4) as pool:
        case3_results = list(pool.imap_unordered(worker, case3_tasks, chunksize=1))

    for r in case3_results:
        r["case"] = "case_3"
        all_rows.append(_scalar_row(r))

    fit_case3 = make_fit_table(case3_results, "case_3")
    fit_case3.to_csv(outdir / "case3_VT_zeta.csv", index=False)
    save_evolution_csvs(case3_results, outdir, "case3")
    all_fit_tables.append(fit_case3)

    # ── Case 3 heatmaps — V_T and zeta over the (N, <V_a>) grid ───────────────
    print("[Case 3] Generating V_T and zeta heatmaps over (N, <V_a>) …")
    P.plot_metric_heatmap(
        case3_results, "fit_zeta", r"$\zeta$",
        rf"$\zeta$ over $(N,\ \langle V_a\rangle)$ at $\sigma={SIGMA_CASE3:.0f}$ V",
        plotdir / "case3_zeta_heatmap_N_meanVa.png",
    )
    P.plot_metric_heatmap(
        case3_results, "fit_V_T_V", r"$V_T$ (V)",
        rf"$V_T$ over $(N,\ \langle V_a\rangle)$ at $\sigma={SIGMA_CASE3:.0f}$ V",
        plotdir / "case3_VT_heatmap_N_meanVa.png",
    )

    # ── Case 3 diagnostic — conductance-matrix heatmaps at 8 V ────────────────
    print("[Case 3] Generating conductance-matrix heatmaps (8 V) …")
    run_G_matrix_diagnostics(
        case3_results,
        diag_base_dir = plotdir / "case3_Gmatrix",
        case_name     = "case3",
        diag_voltages = SNAPSHOT_VOLTAGES,
    )

    # ── Case 3 snapshots — conduction region spreading ────────────────────────
    print("[Case 3] Generating network snapshots …")
    run_snapshots(
        case3_results,
        snap_base_dir = plotdir / "case3_snapshots",
        case_name     = "case3",
        snap_voltages = SNAPSHOT_VOLTAGES,
    )

    for N in N_VALUES_CASE3:
        subset = [r for r in case3_results if r["N"] == N]
        P.plot_theoretical_distribution_overlay(
            subset, mode="mean",
            title=rf"$V_a$ Distributions: Varying $\langle V_a\rangle$ "
                  rf"($\sigma={SIGMA_CASE3:.0f}\ \mathrm{{V}}$ fixed, $N={N}$)",
            outfile=plotdir / f"case3_distribution_constant_sigma_vary_mean_N{N}.png",
        )
        save_distribution_csv(subset, outdir,
                              f"case3_distribution_data_N{N}.csv", mode="mean")
        save_sampled_va_csv(subset, outdir, f"case3_sampled_va_N{N}.csv")
        P.plot_iv_overlay(
            subset, mode="mean",
            title=rf"I-V Curves: $\sigma={SIGMA_CASE3:.0f}\ \mathrm{{V}}$ fixed at $N={N}$",
            outfile=plotdir / f"case3_iv_constant_sigma_vary_mean_N{N}.png",
        )
        P.save_topology_plot(
            n_junctions=N,
            mean_va=TOPOLOGY_REFERENCE_MEAN,
            sigma_va=SIGMA_CASE3,
            seed=seed_for(0),
            outfile=plotdir / f"case3_topology_N{N}.png",
            build_network_fn=build_network,
        )

    # ── CASE 4: vary N at fixed ⟨V_a⟩ = 6 V, σ = 3 V ───────────────────────
    print("[Case 4] Vary N (all seeds per N for error bars) …")
    case4_tasks = [
        (N, CONST_MEAN_CASE4, SIGMA_CASE4, seed)
        for N in N_VALUES_CASE4
        for seed in SEEDS
    ]
    with Pool(processes=pool_workers(len(case4_tasks)), maxtasksperchild=4) as pool:
        case4_results = list(pool.imap_unordered(worker, case4_tasks, chunksize=1))
    case4_results.sort(key=lambda r: (r["N"], r["seed"]))

    # representative (first seed) per N for single-network artifacts / I-V overlay
    _rep4 = {}
    for r in case4_results:
        if r["N"] not in _rep4 or r["seed"] < _rep4[r["N"]]["seed"]:
            _rep4[r["N"]] = r
    case4_rep = [_rep4[n] for n in sorted(_rep4)]

    for r in case4_results:
        r["case"] = "case_4"
        all_rows.append(_scalar_row(r))

    P.plot_iv_overlay(
        case4_rep, mode="N",
        title=rf"I-V Curves: varying $N$ at "
              rf"$\langle V_a\rangle={CONST_MEAN_CASE4:.0f}\ \mathrm{{V}}$ "
              rf"and $\sigma={SIGMA_CASE4:.0f}\ \mathrm{{V}}$",
        outfile=plotdir / "case4_iv_vary_N_same_graph.png",
    )
    fit_case4 = make_fit_table(case4_results, "case_4")
    fit_case4.to_csv(outdir / "case4_VT_zeta.csv", index=False)
    # aggregated zeta per N (mean/std/n)
    _agg4 = []
    for N in sorted(set(r["N"] for r in case4_results)):
        grp = [r["fit_zeta"] for r in case4_results
               if r["N"] == N and r.get("fit_success", True)
               and np.isfinite(r["fit_zeta"])]
        _agg4.append({"N": N, "n_seeds": len(grp),
                      "zeta_mean": float(np.mean(grp)) if grp else np.nan,
                      "zeta_std": float(np.std(grp, ddof=1)) if len(grp) > 1 else 0.0})
    pd.DataFrame(_agg4).to_csv(outdir / "case4_zeta_vs_N_aggregated.csv", index=False)
    save_evolution_csvs(case4_rep, outdir, "case4")

    # Export individual I-V curve data for case4
    # Each row is one (label, voltage, current) data point.
    iv_rows_case4 = []
    for r in case4_results:
        iv_rows_case4.extend(_iv_rows_for_result(r))
    _save_csv(pd.DataFrame(iv_rows_case4),
              outdir / "case4_VT_zeta_iv_data.csv")
    all_fit_tables.append(fit_case4)

    P.plot_zeta_vs_N_errorbars(
        case4_results,
        title=rf"$\zeta$ vs $N$ at "
              rf"$\langle V_a\rangle={CONST_MEAN_CASE4:.0f}\ \mathrm{{V}}$ "
              rf"and $\sigma={SIGMA_CASE4:.0f}\ \mathrm{{V}}$",
        outfile=plotdir / "case4_zeta_vs_N.png",
    )

    # ── Summary tables for Cases 1–4 ─────────────────────────────────────────
    save_summary_csv(all_rows, outdir / "parameter_sweep_case_summary.csv")
    pd.concat(all_fit_tables, ignore_index=True).to_csv(
        outdir / "all_cases_VT_zeta.csv", index=False
    )
    # ── Pickle: save all plot-relevant data so replot.py can regenerate ──────
    # Only the keys actually needed by plotting.py are saved, keeping the
    # file small.  Large node_va / conductances arrays are excluded.
    _keep14 = {"N", "mean_va_target", "sigma_va_target", "seed",
                "voltages", "currents", "fit_zeta", "fit_V_T_V"}
    sweep_data = {
        "case1":  [{k: v for k, v in r.items() if k in _keep14} for r in case1_results],
        "case2a": [{k: v for k, v in r.items() if k in _keep14} for r in case2a_results],
        "case2b": [{k: v for k, v in r.items() if k in _keep14} for r in case2b_results],
        "case2_by_sigma": {
            float(sigma): [{k: v for k, v in r.items() if k in _keep14} for r in results]
            for sigma, results in case2_results_by_sigma.items()
        },
        "case3":  [{k: v for k, v in r.items() if k in _keep14} for r in case3_results],
        "case4":  [{k: v for k, v in r.items() if k in _keep14} for r in case4_results],
        # topology plots need the network parameters, not result arrays
        "topology_params": {
            "N_values_case3": N_VALUES_CASE3,
            "sigma_case3":    SIGMA_CASE3,
            "reference_mean": TOPOLOGY_REFERENCE_MEAN,
            "seeds":          SEEDS,
            "N_values_case4": N_VALUES_CASE4,
            "sigma_case4":    SIGMA_CASE4,
            "mean_case4":     CONST_MEAN_CASE4,
        },
    }
    pickle_path = outdir / "sweep_results_cases1_4.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(sweep_data, f)
    print(f"Results pickled to : {pickle_path}")

    with open(outdir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Parameter sweep Cases 1–4 completed.\n\n"
            "Folder layout:\n"
            "  plots/                      — all figures\n"
            "  *.csv                       — numerical results\n"
            "  sweep_results_cases1_4.pkl  — saved results for replot.py\n\n"
            "To regenerate figures without re-running the sweep:\n"
            "  python replot.py --cases14 <path/to/sweep_results_cases1_4.pkl>\n\n"
            "Font sizes and plot style: edit plotting.py (top of file).\n"
        )
    print(f"Cases 1–4 saved to : {outdir}")
    print(f"Figures saved to   : {plotdir}\n")

    # CASE R: Random void-fraction sweep (N=500, uniform-radius random voids)
    print("[Case R] Random void-fraction sweep (N=500) ...")

    outdirR  = RESULTS_ROOT / f"case_R_random_voids_{timestamp}"
    plotdirR = outdirR / "plots"
    ensure_dir(outdirR)
    ensure_dir(plotdirR)
    print(f"Case R      : {outdirR}")
    print(f"Plots folder: {plotdirR}")

    # Run ALL seeds at each void fraction so V_T / zeta get error bars.
    cR_tasks = [
        (500, CONST_MEAN_CASER, SIGMA_CASER, seed, fv)
        for fv in VOID_FRACTIONS_CASER
        for seed in SEEDS
    ]
    with Pool(processes=pool_workers(len(cR_tasks)), maxtasksperchild=4) as pool:
        cR_results = list(pool.imap_unordered(_worker_case_r, cR_tasks, chunksize=1))
    cR_results.sort(key=lambda r: (r["void_fraction"], r["seed"]))

    # representative (first seed) per void fraction, for single-network artifacts
    rep_by_fv = {}
    for r in cR_results:
        fv = r["void_fraction"]
        if fv not in rep_by_fv or r["seed"] < rep_by_fv[fv]["seed"]:
            rep_by_fv[fv] = r
    cR_rep = [rep_by_fv[fv] for fv in sorted(rep_by_fv)]

    # I-V overlay coloured by void fraction (representative seed per fv)
    P.plot_iv_overlay_case(
        cR_rep,
        r"I-V at $N=500$: random void placement",
        plotdirR / "caseR_iv_random_voids_N500.png",
    )
    # I-V overlay data (one row per void fraction per voltage) so the overlay
    # figure can be replotted/restyled from CSV.
    iv_rows_caseR = []
    for r in cR_rep:
        lbl = f"fv{r['void_fraction']:.2f}"
        v_end = r.get("transition_voltage_V", float("nan"))
        for v, i in zip(r["voltages"], r["currents"]):
            if np.isfinite(v_end) and float(v) > float(v_end) + 1e-9:
                continue
            iv_rows_caseR.append({"label": lbl,
                                  "voltage_V": float(v),
                                  "current_A": float(i),
                                  "current_nA": float(i) * 1e9})
    pd.DataFrame(iv_rows_caseR).to_csv(
        outdirR / "caseR_iv_random_voids_data.csv", index=False)
    # V_T and zeta vs void fraction — WITH ERROR BARS over all seeds
    P.plot_metric_vs_void_errorbars(
        cR_results, "fit_V_T_V", r"$V_T$ (V)",
        r"$V_T$ vs void fraction (random placement, $N=500$)",
        plotdirR / "caseR_VT_vs_void_random_N500.png",
    )
    P.plot_metric_vs_void_errorbars(
        cR_results, "fit_zeta", r"$\zeta$",
        r"$\zeta$ vs void fraction (random placement, $N=500$)",
        plotdirR / "caseR_zeta_vs_void_random_N500.png",
    )
    # topology per void fraction (representative seed)
    for r in cR_rep:
        fv_pct = int(round(100 * r["void_fraction"]))
        P.save_topology_plot_case(
            r, plotdirR / f"caseR_topology_N500_void{fv_pct}.png")

    # summary CSV: ALL seeds (so you can recompute spreads yourself)
    pd.DataFrame([_scalar_row(r) for r in cR_results]).to_csv(
        outdirR / "caseR_VT_zeta_random_voids.csv", index=False)
    # aggregated CSV: mean / std / n per void fraction
    _agg_rows = []
    for fv in sorted(set(r["void_fraction"] for r in cR_results)):
        grp = [r for r in cR_results if r["void_fraction"] == fv]
        vt = np.array([r["fit_V_T_V"] for r in grp if np.isfinite(r["fit_V_T_V"])])
        zt = np.array([r["fit_zeta"] for r in grp if np.isfinite(r["fit_zeta"])])
        _agg_rows.append({
            "void_fraction": fv,
            "n_seeds": len(grp),
            "VT_mean": float(np.mean(vt)) if len(vt) else np.nan,
            "VT_std": float(np.std(vt, ddof=1)) if len(vt) > 1 else 0.0,
            "zeta_mean": float(np.mean(zt)) if len(zt) else np.nan,
            "zeta_std": float(np.std(zt, ddof=1)) if len(zt) > 1 else 0.0,
        })
    pd.DataFrame(_agg_rows).to_csv(
        outdirR / "caseR_VT_zeta_aggregated.csv", index=False)

    # per-voltage evolution CSVs (representative seed per fv)
    save_evolution_csvs(cR_rep, outdirR, "caseR")

    # conductance-matrix heatmaps at 8 V (representative seed)
    print("[Case R] Generating conductance-matrix heatmaps (8 V) ...")
    run_G_matrix_diagnostics(
        cR_rep,
        diag_base_dir = plotdirR / "caseR_Gmatrix",
        case_name     = "caseR",
        diag_voltages = SNAPSHOT_VOLTAGES,
        void_fraction = None,
    )

    # network snapshots — conduction region spreading (representative seed)
    print("[Case R] Generating network snapshots ...")
    run_snapshots(
        cR_rep,
        snap_base_dir = plotdirR / "caseR_snapshots",
        case_name     = "caseR",
        snap_voltages = SNAPSHOT_VOLTAGES,
        void_fraction = None,
    )
    print(f"  Random void sweep saved to: {outdirR}")

    print("All cases complete.")

if __name__ == "__main__":
    freeze_support()
    main()
