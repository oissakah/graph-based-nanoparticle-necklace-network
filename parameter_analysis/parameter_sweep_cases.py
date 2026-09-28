"""
parameter_sweep_cases.py

Parameter sweep: Cases 1-4 plus Case R (random void fraction).

Cases 1-4  — activation-voltage width/mean and junction-count sweeps using
             analytical normal distributions and the Kirchhoff solver.
  Case 1   — vary activation-voltage width (standard deviation) at fixed mean.
  Case 2   — vary mean activation voltage at several widths.
  Case 3   — vary junction count (number of nodes) crossed with mean.
  Case 4   — vary junction count at fixed mean and width.

Case R     — fixed N=500, random void-fraction sweep (uniform-radius random
             void placement).

The transport exponent fitted to I = A (V - V_T)^zeta is reported as zeta
throughout (symbol ζ). "Activation voltage" (V_a) is used in place of
"threshold voltage" in labels.
"""

import os
# Must be set before ANY numerical library import so BLAS uses 1 thread per
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


# =============================================================================
# SHARED CONSTANTS
# =============================================================================

EDGE_K             = 2.0e10
NODE_RESISTANCE_OHM = 3.5e9
CONNECTION_RADIUS  = 0.15
DOMAIN_SIZE        = (1.0, 1.0)
LEFT_THRESH        = 0.15
RIGHT_THRESH       = 0.85

V_START    = 0.0
V_MAX      = 16.0
V_STEP     = 0.5
FIT_WINDOW = 10.0

# Use the same snapshot voltages for all snapshot/diagnostic figures.
SNAPSHOT_VOLTAGES = [1.0, 2.0, 6.0, 8.0, 14.0]

VA_MIN   = 0.0
VA_MAX   = 20.0

# Va controls activation timing only. Junction resistance is fixed separately,
# preventing the mean-Va sweep from changing two physical quantities at once.

# -----------------------------------------------------------------------------
# Unified random seeds — single source of truth for all cases.
# -----------------------------------------------------------------------------
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

# Each worker runs a dense per-voltage eigendecomposition, so keep a conservative
# default. On Slurm, honor the CPUs assigned to this task instead of seeing and
# accidentally using every CPU on the node. NANONECKLACE_WORKERS is an optional
# explicit override for local or cluster runs.
def _configured_worker_cap(default=6):
    raw = (os.environ.get("NANONECKLACE_WORKERS")
           or os.environ.get("SLURM_CPUS_PER_TASK"))
    if raw is None:
        return int(default)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(default)


MAX_WORKERS = _configured_worker_cap()

def pool_workers(n_tasks):
    """Return a safe worker count for multiprocessing pools.

    Bounded by (a) the requested cap MAX_WORKERS, (b) the number of tasks, and
    (c) the number of CPU cores actually available, so it never oversubscribes.
    """
    return max(1, min(MAX_WORKERS, int(n_tasks), cpu_count()))


# =============================================================================
# CASES 1-4 CONSTANTS
# =============================================================================

SIGMA_VALUES_CASE1 = [1.0, 3.0, 5.0, 7.0]
CONST_MEAN_CASE1   = 8.0
N_CASE12           = 500

MEAN_VALUES_CASE2  = [4.0, 6.0, 8.0, 10.0]
SIGMA_VALUES_CASE2 = [1.0, 3.0, 5.0, 7.0]
SIGMA_CASE2A       = 1.0
SIGMA_CASE2B       = 5.0

N_VALUES_CASE3    = [200, 400, 600, 800]
MEAN_VALUES_CASE3 = [4.0, 6.0, 8.0, 10.0]
SIGMA_CASE3       = 3.0

N_VALUES_CASE4          = [200, 400, 600, 800]
CONST_MEAN_CASE4        = 6.0
SIGMA_CASE4             = 3.0
TOPOLOGY_REFERENCE_MEAN = 6.0


# =============================================================================
# CASE R CONSTANTS (random void-fraction sweep)
# =============================================================================

VOID_FRACTIONS_CASER = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
CONST_MEAN_CASER     = 6.0
SIGMA_CASER          = 3.0
CASER_REF_RADIUS     = 0.15
CASER_REF_N          = 500
VOID_RADIUS          = 0.08


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _save_csv(df, outfile):
    csv_path = Path(outfile)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def first_positive_span_idx(v, i, span=1.0):
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


ACTIVATION_TRANSITION_FRAC = 0.90


def activation_saturation_voltage(V, activated, n_total,
                                  frac=ACTIVATION_TRANSITION_FRAC):
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


def _activation_saturation_voltage_from_evolution(evo):
    rows = evo.get("rows") if isinstance(evo, dict) else None
    if not rows:
        return float("nan")
    n_total = evo.get("n_total_nodes")
    if not n_total:
        n_total = max((r.get("activated_nodes", 0) or 0) for r in rows)
    V = np.array([r.get("V") for r in rows], dtype=float)
    an = np.array([r.get("activated_nodes", np.nan) for r in rows], dtype=float)
    vt = activation_saturation_voltage(V, an, n_total)
    return float(vt) if vt is not None else float("nan")


def fit_power_law(v_arr, i_arr, V_T=None, V_stop=None,
                  fit_window=FIT_WINDOW):
    """Fit ``I = A (V - V_T)^zeta`` with the threshold held fixed.

    In the current framework, ``V_T`` is not an extrapolated fit parameter.
    It is the first sampled voltage with nonzero source--drain current, which
    is also the first source--drain percolation voltage.  Only ``A`` and
    ``zeta`` are fitted.  The threshold point itself is excluded because
    ``log(V - V_T)`` is undefined there.

    ``V_stop`` should be the first voltage at which 90% of the nodes are
    active.  If that saturation point is not reached, the fit uses the
    remaining available nonlinear window, capped by ``fit_window``.
    """
    nan = float("nan")
    v_arr = np.asarray(v_arr, dtype=float)
    i_arr = np.asarray(i_arr, dtype=float)
    finite = np.isfinite(v_arr) & np.isfinite(i_arr)
    v_arr, i_arr = v_arr[finite], i_arr[finite]
    if len(v_arr) == 0:
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason="no_data")

    if V_T is None or not np.isfinite(V_T):
        positive = np.flatnonzero(i_arr > 0)
        if len(positive) == 0:
            return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                        reason="no_conduction")
        V_T = float(v_arr[positive[0]])
    else:
        V_T = float(V_T)

    fallback_stop = min(float(np.max(v_arr)), V_T + float(fit_window))
    if V_stop is None or not np.isfinite(V_stop) or float(V_stop) <= V_T:
        V_stop = fallback_stop
    else:
        V_stop = min(float(V_stop), fallback_stop)

    mask = (v_arr > V_T) & (v_arr <= V_stop) & (i_arr > 0)
    V_fit = v_arr[mask]
    I_fit = i_arr[mask]
    if len(V_fit) < 4:
        return dict(success=False, V_T=V_T, zeta=nan, A=nan, R2=nan,
                    reason=f"too_few_points({len(V_fit)})")

    x = np.log(V_fit - V_T)
    y = np.log(I_fit)
    design = np.vstack([x, np.ones_like(x)]).T
    (zeta_fit, logA_fit), *_ = np.linalg.lstsq(design, y, rcond=None)
    y_pred = zeta_fit * x + logA_fit
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    A_fit = float(np.exp(logA_fit))

    if not np.isfinite(zeta_fit) or not np.isfinite(R2):
        return dict(success=False, V_T=V_T, zeta=nan, A=nan, R2=nan,
                    reason="nonfinite")
    if R2 < 0.80:
        return dict(success=False, V_T=V_T, zeta=float(zeta_fit),
                    A=A_fit, R2=float(R2), reason=f"poor_fit_R2={R2:.3f}")
    if zeta_fit <= 0.1 or zeta_fit >= 10.0:
        return dict(success=False, V_T=V_T, zeta=float(zeta_fit),
                    A=A_fit, R2=float(R2), reason="zeta_out_of_range")

    return dict(success=True, V_T=V_T, zeta=float(zeta_fit),
                A=A_fit, R2=float(R2), reason="ok")


# =============================================================================
# CASES 1–4: NETWORK BUILDER AND RUNNERS
# =============================================================================

def build_network(n_junctions, mean_va, sigma_va, seed):
    net = NanoparticleNetwork(
        n_junctions=n_junctions,
        # IMPORTANT: N is the network-density control parameter in Cases 3/4.
        # Keep the geometric connection rule fixed so increasing N genuinely
        # increases the number of nearby neighbours / conducting connections.
        connection_radius=connection_radius_for_N(n_junctions),
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
        node_Va=node_Va,
        edge_k=float(EDGE_K),
        node_resistance_ohm=float(NODE_RESISTANCE_OHM),
    )
    net.identify_sources_drains(
        left_thresh=float(LEFT_THRESH),
        right_thresh=float(RIGHT_THRESH),
    )
    return net


def run_iv(n_junctions, mean_va, sigma_va, seed):
    net = build_network(n_junctions, mean_va, sigma_va, seed)

    import sweep_analysis as _sa
    evo = _sa.sweep(net, V_START, V_MAX, V_STEP)
    voltages = np.asarray([row["V"] for row in evo["rows"]], dtype=float)
    currents = np.asarray([row["total_current_A"] for row in evo["rows"]], dtype=float)
    conductances = np.asarray([row["conductance_S"] for row in evo["rows"]], dtype=float)
    node_va = np.array([net.G.nodes[n]["Va"] for n in net.G.nodes()], dtype=float)
    peak_idx = int(np.argmax(currents))
    saturation_V = _activation_saturation_voltage_from_evolution(evo)
    percolation_V = (float(evo["percolation_V"])
                     if evo["percolation_V"] is not None else np.nan)
    fit = fit_power_law(
        voltages, currents, V_T=percolation_V, V_stop=saturation_V)

    return {
        "N":                     int(n_junctions),
        "mean_va_target":       float(mean_va),
        "sigma_va_target":      float(sigma_va),
        "seed":                  int(seed),
        "voltages":              voltages,
        "currents":              currents,
        "conductances":          conductances,
        "node_va":              node_va,
        "n_nodes_actual":        int(net.G.number_of_nodes()),
        "n_edges":               int(net.G.number_of_edges()),
        "n_sources":             int(len(net.source_nodes)),
        "n_drains":              int(len(net.drain_nodes)),
        "sampled_mean_va":      float(np.mean(node_va)),
        "sampled_std_va":       float(np.std(node_va)),
        "connection_radius_used": float(net.connection_radius),
        "peak_current_A":        float(currents[peak_idx]),
        "peak_voltage_V":        float(voltages[peak_idx]),
        "percolation_voltage_V": percolation_V,
        "edge_disjoint_pathways_at_Vperc": int(evo["percolation_pathways"]),
        "active_nodes_at_Vperc": int(evo["percolation_active_nodes"]),
        "activation_saturation_voltage_V": float(saturation_V),
        "max_conductance_S":     float(np.max(conductances)),
        # Kept for CSV compatibility. By definition this equals V_perc even
        # when too few post-threshold points are available to fit zeta.
        "fit_V_T_V":             float(fit["V_T"]),
        "fit_zeta":              float(fit["zeta"]) if fit["success"] else np.nan,
        "fit_R2":                float(fit["R2"])    if fit["success"] else np.nan,
        "fit_success":           bool(fit["success"]),
        "fit_reason":            fit.get("reason", ""),
        "evolution_rows":        evo["rows"],
        "evolution_fields":      evo["fieldnames"],
    }


def worker(args):
    return run_iv(*args)


# =============================================================================
# SHARED CSV HELPERS
# =============================================================================

def make_fit_table(results, case_name):
    rows = []
    for r in results:
        rows.append({
            "case":                  case_name,
            "N":                     r["N"],
            "mean_Va_target_V":      r["mean_va_target"],
            "sigma_Va_target_V":     r["sigma_va_target"],
            "seed":                  r["seed"],
            "connection_radius_used": r.get("connection_radius_used", np.nan),
            "sampled_mean_Va_V":     r["sampled_mean_va"],
            "sampled_sigma_Va_V":    r["sampled_std_va"],
            "percolation_voltage_V": r["percolation_voltage_V"],
            "edge_disjoint_pathways_at_Vperc": r.get(
                "edge_disjoint_pathways_at_Vperc", 0),
            "active_nodes_at_Vperc": r.get("active_nodes_at_Vperc", 0),
            "activation_saturation_voltage_V": r.get(
                "activation_saturation_voltage_V", np.nan),
            "fit_V_T_V":             r["fit_V_T_V"],
            "fit_zeta":              r["fit_zeta"],
            "fit_R2":                r["fit_R2"],
            "peak_current_A":        r["peak_current_A"],
            "max_conductance_S":     r["max_conductance_S"],
        })
    return pd.DataFrame(rows)


def representative_results(results, group_key):
    reps = {}
    for r in results:
        key = r[group_key]
        if key not in reps or r["seed"] < reps[key]["seed"]:
            reps[key] = r
    return [reps[k] for k in sorted(reps)]


def aggregate_fit_table(results, group_key, group_col_name):
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
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


_ARRAY_KEYS = {"voltages", "currents", "conductances", "node_va",
               "positions", "edges", "source_nodes", "drain_nodes", "voids",
               "evolution_rows", "evolution_fields"}


def _scalar_row(r):
    return {k: v for k, v in r.items() if k not in _ARRAY_KEYS}


def _iv_rows_for_result(r, trim_at_transition=True):
    lbl = (f"N={r['N']}_mean{r['mean_va_target']:.0f}"
           f"_sigma{r['sigma_va_target']:.0f}_seed{r['seed']}")
    v_end = r.get("activation_saturation_voltage_V", float("nan"))
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
    import pandas as _pd
    from scipy.stats import norm as _norm
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
    import pandas as _pd
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


def save_evolution_csvs(results, outdir, case_name):
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
        # Store the study-defining metadata beside every per-voltage row.
        # This makes publication plots auditable and prevents accidental use of
        # stale N-sweep results generated with a different connection radius.
        df["study_N"] = int(r["N"])
        df["study_mean_Va_target_V"] = float(r["mean_va_target"])
        df["study_sigma_Va_target_V"] = float(r["sigma_va_target"])
        df["study_seed"] = int(r["seed"])
        df["connection_radius_used"] = float(
            r.get("connection_radius_used", CONNECTION_RADIUS)
        )
        if fv is not None:
            df["study_void_fraction"] = float(fv)
        df.to_csv(evo_dir / f"evolution_{tag}.csv", index=False)
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


# =============================================================================
# CASE R: RANDOM VOID GENERATION
# =============================================================================

def connection_radius_for_N(n):
    """Return the fixed geometric connection radius used in N-density studies.

    ``N`` is intentionally the network-density parameter in Cases 3 and 4.
    Therefore the local connection criterion must remain unchanged when N is
    varied.  Scaling ``r_c`` as N**(-1/2) would hold the expected degree nearly
    constant and would turn the study into a system-size study rather than a
    density study.

    The argument is retained for a uniform call signature and future metadata
    checks; the returned radius is the same for every N.
    """
    _ = int(n)
    return float(CONNECTION_RADIUS)


def scaled_radius_for_N(n):
    """Backward-compatible alias for older scripts.

    Historical V14 code used this function name for an N-dependent radius.
    It now deliberately returns the fixed connection radius so old imports do
    not silently reintroduce the size-compensated topology.
    """
    return connection_radius_for_N(n)


def make_voids_random_sampling(target_fv, seed, radius=None, n_junctions=500):
    if target_fv <= 0.0:
        return []
    r = float(radius) if radius is not None else float(VOID_RADIUS)

    n_v = int(round(target_fv / (np.pi * r * r)))
    if n_v < 1:
        return []

    rng = np.random.default_rng(seed)
    buf = 0.02
    lo, hi = buf + r, 1.0 - buf - r
    if hi <= lo:
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
            n_voids=0, achieved_void_area=0.0, achieved_void_fraction=0.0,
            achieved_fv_error=float(target_fv),
            mean_void_radius=np.nan, std_void_radius=np.nan,
            min_void_radius=np.nan,  max_void_radius=np.nan,
        )
    radii    = np.array([v["radius"] for v in voids])
    # Deterministic midpoint-grid union area: overlapping circles are counted
    # once, unlike the former sum-of-circle-areas estimate.
    resolution = 700
    axis = (np.arange(resolution, dtype=float) + 0.5) / resolution
    x, y = np.meshgrid(axis, axis, indexing="xy")
    covered = np.zeros_like(x, dtype=bool)
    for void in voids:
        cx, cy = void["center"]
        covered |= (x - cx) ** 2 + (y - cy) ** 2 <= void["radius"] ** 2
    achieved = float(np.mean(covered))
    return dict(
        n_voids=int(len(voids)),
        achieved_void_area=achieved,
        achieved_void_fraction=achieved,
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


def sample_truncated_normal_array(rng, mean, std, size, vmin, vmax):
    """Vector rejection sampler with no artificial mass at either bound."""
    if std <= 0:
        return np.full(size, np.clip(mean, vmin, vmax), dtype=float)
    values = np.empty(size, dtype=float)
    pending = np.arange(size)
    while pending.size:
        draws = rng.normal(float(mean), float(std), size=pending.size)
        accepted = (draws >= vmin) & (draws <= vmax)
        values[pending[accepted]] = draws[accepted]
        pending = pending[~accepted]
    return values


# =============================================================================
# MAIN
# =============================================================================


def _worker_case_r(args):
    """Top-level worker: one Case R (N, mean, sigma, seed, fv) combination."""
    N, mean, sigma, seed_r, fv = args
    net, voids = _build_void_network(N, mean, sigma, seed_r, fv)

    import sweep_analysis as _sa
    evo = _sa.sweep(net, V_START, V_MAX, V_STEP)
    voltages = np.asarray([row["V"] for row in evo["rows"]], dtype=float)
    currents = np.asarray([row["total_current_A"] for row in evo["rows"]], dtype=float)
    saturation_V = _activation_saturation_voltage_from_evolution(evo)
    percolation_V = (float(evo["percolation_V"])
                     if evo["percolation_V"] is not None else np.nan)
    fit = fit_power_law(
        voltages, currents, V_T=percolation_V, V_stop=saturation_V)
    void_stats = compute_void_stats(voids, float(fv))

    _el = list(net.G.edges())
    return {
        "N":                int(N),
        "mean_va_target":   float(mean),
        "sigma_va_target":  float(sigma),
        "seed":             int(seed_r),
        "void_fraction":        float(fv),
        "void_fraction_target": float(fv),
        "void_fraction_achieved": void_stats["achieved_void_fraction"],
        "void_fraction_error": void_stats["achieved_fv_error"],
        "void_method":      "random",
        "voltages":         voltages,
        "currents":         currents,
        "percolation_voltage_V": percolation_V,
        "edge_disjoint_pathways_at_Vperc": int(evo["percolation_pathways"]),
        "active_nodes_at_Vperc": int(evo["percolation_active_nodes"]),
        "fit_V_T_V":        float(fit["V_T"]),
        "fit_zeta":         float(fit["zeta"]) if fit["success"] else np.nan,
        "fit_A":            float(fit["A"])     if fit["success"] else np.nan,
        "fit_R2":           float(fit["R2"])    if fit["success"] else np.nan,
        "fit_success":      bool(fit["success"]),
        "fit_reason":       fit.get("reason", ""),
        "activation_saturation_voltage_V": float(saturation_V),
        "n_voids":          int(len(voids)),
        "mean_void_radius": float(np.mean([v["radius"] for v in voids]))
                            if voids else np.nan,
        "positions":        np.asarray(net.positions),
        "edges":            (np.array(_el, dtype=np.int32)
                             if _el else np.zeros((0, 2), dtype=np.int32)),
        "source_nodes":     list(net.source_nodes),
        "drain_nodes":      list(net.drain_nodes),
        "voids":            voids,
        "evolution_rows":   evo["rows"],
        "evolution_fields": evo["fieldnames"],
    }


def run_G_matrix_diagnostics(results, diag_base_dir, case_name,
                             diag_voltages, void_fraction=None):
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

        import sweep_analysis as _sa
        pos = net.positions
        rows = []
        for V in snap_voltages:
            activated = net.activated_nodes(V)
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
        nrows = []
        for V in snap_voltages:
            for n in net.G.nodes():
                nrows.append({
                    "V": float(V), "node": int(n),
                    "x": float(pos[n][0]), "y": float(pos[n][1]),
                    "activated": bool(net.G.nodes[n]["Va"] <= V),
                    "is_source": n in net.source_nodes,
                    "is_drain": n in net.drain_nodes,
                })
        pd.DataFrame(nrows).to_csv(
            snap_base_dir / f"snapshot_nodes_{tag}.csv", index=False)
    print(f"  [snapshots] -> {snap_base_dir}")


def run_percolation_current_outputs(results, output_dir, case_name,
                                    void_fraction=None):
    """Write |Iij| distributions, maps, and edge data at every run's Vperc."""
    import plotting as _P
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in results:
        voltage = r.get("percolation_voltage_V", np.nan)
        if not np.isfinite(voltage):
            continue
        fv = r.get("void_fraction", void_fraction)
        if fv is None or fv == 0.0:
            net = build_network(int(r["N"]), float(r["mean_va_target"]),
                                float(r["sigma_va_target"]), int(r["seed"]))
        else:
            net = _build_void_network(int(r["N"]), float(r["mean_va_target"]),
                                      float(r["sigma_va_target"]),
                                      int(r["seed"]), float(fv))[0]
        tag = (f"{case_name}_N{r['N']}_mean{r['mean_va_target']:.0f}"
               f"_sigma{r['sigma_va_target']:.0f}_seed{r['seed']}")
        if fv is not None:
            tag += f"_fv{fv:.2f}"
        tag += f"_Vperc{float(voltage):g}"
        title_tail = (rf"$N={r['N']}$, "
                      rf"$\langle V_a\rangle={r['mean_va_target']:.0f}$ V, "
                      rf"$\sigma={r['sigma_va_target']:.0f}$ V")
        if fv is not None:
            title_tail += rf", $f_v={fv:.2f}$"
        import sweep_analysis as _sa
        active_at_perc = set(net.activated_nodes(voltage))
        pathway_count = _sa.count_edge_disjoint_pathways(net, active_at_perc)
        _P.plot_percolation_current_distribution(
            net, voltage, output_dir / f"current_distribution_{tag}.png",
            title=rf"Edge-current distribution at $V_{{perc}}={float(voltage):g}$ V — {title_tail}")
        _P.plot_percolation_current_snapshot(
            net, voltage, output_dir / f"edge_current_snapshot_{tag}.png",
            title=rf"$|I_{{ij}}|$ at $V_{{perc}}={float(voltage):g}$ V — {title_tail}",
            independent_pathways=pathway_count)

        solution = net.solve_active_network(active_at_perc, voltage)
        edge_rows = []
        for (i, j), current in solution["edge_currents"].items():
            edge_rows.append({
                "case": case_name, "N": int(r["N"]),
                "mean_Va_target_V": float(r["mean_va_target"]),
                "sigma_Va_target_V": float(r["sigma_va_target"]),
                "seed": int(r["seed"]), "void_fraction": fv,
                "percolation_voltage_V": float(voltage),
                "edge_disjoint_pathways_at_Vperc": pathway_count,
                "active_nodes_at_Vperc": len(active_at_perc),
                "node_i": int(i), "node_j": int(j),
                "x_i": float(net.positions[i][0]), "y_i": float(net.positions[i][1]),
                "x_j": float(net.positions[j][0]), "y_j": float(net.positions[j][1]),
                "Va_i_V": float(net.G.nodes[i]["Va"]),
                "Va_j_V": float(net.G.nodes[j]["Va"]),
                "edge_total_resistance_ohm": float(net.total_edge_resistance(i, j)),
                "current_A": float(current), "abs_current_A": abs(float(current)),
            })
        pd.DataFrame(edge_rows).to_csv(
            output_dir / f"edge_currents_{tag}.csv", index=False)

        # Topology sidecars make the percolation snapshot exactly reproducible
        # from CSV while allowing the publication plotter to apply one common
        # typographic/axis style to every default and parameter condition.
        node_rows = []
        for node in net.G.nodes():
            node_rows.append({
                "node": int(node),
                "x": float(net.positions[node][0]),
                "y": float(net.positions[node][1]),
                "Va_V": float(net.G.nodes[node]["Va"]),
                "active_at_Vperc": int(node in active_at_perc),
                "is_source": int(node in net.source_nodes),
                "is_drain": int(node in net.drain_nodes),
            })
        pd.DataFrame(node_rows).to_csv(
            output_dir / f"network_nodes_{tag}.csv", index=False)
        topology_rows = []
        for i, j in net.G.edges():
            topology_rows.append({
                "node_i": int(i), "node_j": int(j),
                "x_i": float(net.positions[i][0]),
                "y_i": float(net.positions[i][1]),
                "x_j": float(net.positions[j][0]),
                "y_j": float(net.positions[j][1]),
            })
        pd.DataFrame(topology_rows).to_csv(
            output_dir / f"network_edges_{tag}.csv", index=False)

        # Sparse voltage-resolved edge history used by the publication
        # voltage-edge heatmap and backbone-turnover diagnostic. Only the
        # conducting region from Vperc through the 90%-activation transition is
        # retained, which limits file size without discarding the transition.
        transition_v = r.get("activation_saturation_voltage_V", np.nan)
        history_voltages = []
        for row in r.get("evolution_rows", []):
            value_v = float(row.get("V", np.nan))
            if not np.isfinite(value_v) or value_v < float(voltage) - 1e-9:
                continue
            if np.isfinite(transition_v) and value_v > float(transition_v) + 1e-9:
                continue
            history_voltages.append(value_v)
        history_rows = []
        for history_v in history_voltages:
            history_solution = net.solve_active_network(
                net.activated_nodes(history_v), history_v)
            for (i, j), current in history_solution["edge_currents"].items():
                magnitude = abs(float(current))
                if magnitude <= 0:
                    continue
                history_rows.append({
                    "case": case_name, "N": int(r["N"]),
                    "mean_Va_target_V": float(r["mean_va_target"]),
                    "sigma_Va_target_V": float(r["sigma_va_target"]),
                    "seed": int(r["seed"]), "void_fraction": fv,
                    "percolation_voltage_V": float(voltage),
                    "edge_disjoint_pathways_at_Vperc": pathway_count,
                    "active_nodes_at_Vperc": len(active_at_perc),
                    "V": float(history_v),
                    "node_i": int(i), "node_j": int(j),
                    "x_i": float(net.positions[i][0]),
                    "y_i": float(net.positions[i][1]),
                    "x_j": float(net.positions[j][0]),
                    "y_j": float(net.positions[j][1]),
                    "current_A": float(current),
                    "abs_current_A": magnitude,
                })
        pd.DataFrame(history_rows).to_csv(
            output_dir / f"edge_current_history_{tag}.csv", index=False)
        summary.append({
            "case": case_name, "N": int(r["N"]),
            "mean_Va_target_V": float(r["mean_va_target"]),
            "sigma_Va_target_V": float(r["sigma_va_target"]),
            "seed": int(r["seed"]), "void_fraction": fv,
            "percolation_voltage_V": float(voltage),
            "edge_disjoint_pathways_at_Vperc": pathway_count,
            "active_nodes_at_Vperc": len(active_at_perc),
            "total_current_A": solution["total_current_A"],
            "n_current_carrying_edges": len(solution["edge_currents"]),
            "current_balance_error_A": solution["current_balance_error_A"],
        })
    if summary:
        summary_path = output_dir / f"percolation_current_summary_{case_name}.csv"
        pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"  [percolation-current distributions and snapshots] -> {output_dir}")


def _build_void_network(n, mean_va, sigma_va, seed, void_fraction):
    """Build a random-void network (shared by Case R diagnostics)."""
    import networkx as nx
    from scipy.spatial import cKDTree
    voids = make_voids_random_sampling(float(void_fraction), seed=seed,
                                       n_junctions=n)
    net = NanoparticleNetwork(n_junctions=n,
                              connection_radius=connection_radius_for_N(n),
                              domain=DOMAIN_SIZE)
    net.positions = generate_positions_with_voids(n, DOMAIN_SIZE, voids, seed)
    net.G = nx.Graph()
    rng = np.random.default_rng(seed + 1000)
    raw_va = sample_truncated_normal_array(
        rng, float(mean_va), float(sigma_va), n, VA_MIN, VA_MAX)
    for i in range(n):
        va = float(raw_va[i])
        net.G.add_node(i, pos=net.positions[i], Va=va,
                       R_node=float(NODE_RESISTANCE_OHM), activated=False)
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

    outdir      = RESULTS_ROOT / f"parameter_sweep_cases_{timestamp}"
    plotdir     = outdir / "plots"
    ensure_dir(outdir)
    ensure_dir(plotdir)

    print("=" * 70)
    print("RUNNING PARAMETER SWEEP CASES")
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

    # Case 1 snapshots for ALL activation-voltage widths (sigma = 1, 3, 5, 7 V),
    # one snapshot set per sigma, at the common SNAPSHOT_VOLTAGES.
    print("[Case 1] Generating snapshots for all sigma values …")
    run_snapshots(
        case1_rep,
        snap_base_dir = plotdir / "case1_snapshots",
        case_name     = "case1",
        snap_voltages = SNAPSHOT_VOLTAGES,
    )
    run_percolation_current_outputs(
        case1_rep, outdir / "current_distribution_plots", "case1")

    iv_rows_case1 = []
    for r in case1_results:
        iv_rows_case1.extend(_iv_rows_for_result(r))
    _save_csv(pd.DataFrame(iv_rows_case1),
              outdir / "case1_VT_zeta_iv_data.csv")
    all_fit_tables.append(fit_case1)

    # ── CASE 2: vary ⟨V_a⟩ at multiple σ values ───────────────────────────
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

        P.plot_theoretical_distribution_overlay(
            case2_rep, mode="mean",
            title=rf"$V_a$ Distributions: Varying $\langle V_a\rangle$ "
                  rf"($\sigma={sigma_case2:.0f}\ \mathrm{{V}}$ fixed)",
            outfile=plotdir / f"{legacy_prefix}_distribution_constant_sigma{sigma_case2:g}_vary_mean_N500.png",
        )
        save_distribution_csv(case2_results, outdir,
                              f"{legacy_prefix}_distribution_data.csv", mode="mean")
        save_sampled_va_csv(case2_results, outdir, f"{legacy_prefix}_sampled_va.csv")

        P.plot_iv_overlay(
            case2_rep, mode="mean",
            title=rf"I-V Curves at $N={N_CASE12}$: $\sigma={sigma_case2:.0f}\ \mathrm{{V}}$ fixed",
            outfile=plotdir / f"{legacy_prefix}_iv_constant_sigma{sigma_case2:g}_vary_mean_N500.png",
        )

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

        run_snapshots(
            case2_rep,
            snap_base_dir = plotdir / f"{legacy_prefix}_snapshots",
            case_name     = legacy_prefix,
            snap_voltages = SNAPSHOT_VOLTAGES,
        )
        run_percolation_current_outputs(
            case2_rep, outdir / "current_distribution_plots", legacy_prefix)

        iv_rows_case2 = []
        for r in case2_results:
            iv_rows_case2.extend(_iv_rows_for_result(r))
        _save_csv(pd.DataFrame(iv_rows_case2),
                  outdir / f"{legacy_prefix}_VT_zeta_iv_data.csv")

        all_fit_tables.append(fit_case2_sigma)

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

    print("[Case 3] Generating conductance-matrix heatmaps …")
    run_G_matrix_diagnostics(
        case3_results,
        diag_base_dir = plotdir / "case3_Gmatrix",
        case_name     = "case3",
        diag_voltages = SNAPSHOT_VOLTAGES,
    )

    print("[Case 3] Generating network snapshots …")
    run_snapshots(
        case3_results,
        snap_base_dir = plotdir / "case3_snapshots",
        case_name     = "case3",
        snap_voltages = SNAPSHOT_VOLTAGES,
    )
    run_percolation_current_outputs(
        case3_results, outdir / "current_distribution_plots", "case3")

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

    iv_rows_case4 = []
    for r in case4_results:
        iv_rows_case4.extend(_iv_rows_for_result(r, trim_at_transition=False))
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
    run_percolation_current_outputs(
        case4_rep, outdir / "current_distribution_plots", "case4")

    save_summary_csv(all_rows, outdir / "parameter_sweep_case_summary.csv")
    pd.concat(all_fit_tables, ignore_index=True).to_csv(
        outdir / "all_cases_VT_zeta.csv", index=False
    )
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
            "  sweep_results_cases1_4.pkl  — saved results for replot.py\n"
        )
    print(f"Cases 1–4 saved to : {outdir}")
    print(f"Figures saved to   : {plotdir}\n")

    # =========================================================================
    # CASE R: Random void-fraction sweep
    # =========================================================================
    print("[Case R] Random void-fraction sweep (N=500) ...")

    outdirR  = RESULTS_ROOT / f"case_R_random_voids_{timestamp}"
    plotdirR = outdirR / "plots"
    ensure_dir(outdirR)
    ensure_dir(plotdirR)
    print(f"Case R      : {outdirR}")
    print(f"Plots folder: {plotdirR}")

    cR_tasks = [
        (500, CONST_MEAN_CASER, SIGMA_CASER, seed, fv)
        for fv in VOID_FRACTIONS_CASER
        for seed in SEEDS
    ]
    with Pool(processes=pool_workers(len(cR_tasks)), maxtasksperchild=4) as pool:
        cR_results = list(pool.imap_unordered(_worker_case_r, cR_tasks, chunksize=1))
    cR_results.sort(key=lambda r: (r["void_fraction"], r["seed"]))

    rep_by_fv = {}
    for r in cR_results:
        fv = r["void_fraction"]
        if fv not in rep_by_fv or r["seed"] < rep_by_fv[fv]["seed"]:
            rep_by_fv[fv] = r
    cR_rep = [rep_by_fv[fv] for fv in sorted(rep_by_fv)]

    P.plot_iv_overlay_case(
        cR_rep,
        r"I-V at $N=500$: random void placement",
        plotdirR / "caseR_iv_random_voids_N500.png",
    )
    iv_rows_caseR = []
    for r in cR_rep:
        lbl = f"fv{r['void_fraction']:.2f}"
        v_end = r.get("activation_saturation_voltage_V", float("nan"))
        for v, i in zip(r["voltages"], r["currents"]):
            if np.isfinite(v_end) and float(v) > float(v_end) + 1e-9:
                continue
            iv_rows_caseR.append({"label": lbl,
                                  "voltage_V": float(v),
                                  "current_A": float(i),
                                  "current_nA": float(i) * 1e9})
    pd.DataFrame(iv_rows_caseR).to_csv(
        outdirR / "caseR_iv_random_voids_data.csv", index=False)
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
    for r in cR_rep:
        fv_pct = int(round(100 * r["void_fraction"]))
        P.save_topology_plot_case(
            r, plotdirR / f"caseR_topology_N500_void{fv_pct}.png")

    pd.DataFrame([_scalar_row(r) for r in cR_results]).to_csv(
        outdirR / "caseR_VT_zeta_random_voids.csv", index=False)
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

    save_evolution_csvs(cR_rep, outdirR, "caseR")

    print("[Case R] Generating conductance-matrix heatmaps ...")
    run_G_matrix_diagnostics(
        cR_rep,
        diag_base_dir = plotdirR / "caseR_Gmatrix",
        case_name     = "caseR",
        diag_voltages = SNAPSHOT_VOLTAGES,
        void_fraction = None,
    )

    print("[Case R] Generating network snapshots ...")
    run_snapshots(
        cR_rep,
        snap_base_dir = plotdirR / "caseR_snapshots",
        case_name     = "caseR",
        snap_voltages = SNAPSHOT_VOLTAGES,
        void_fraction = None,
    )
    run_percolation_current_outputs(
        cR_rep, outdirR / "current_distribution_plots", "caseR")
    print(f"  Random void sweep saved to: {outdirR}")

    print("All cases complete.")


if __name__ == "__main__":
    freeze_support()
    main()
