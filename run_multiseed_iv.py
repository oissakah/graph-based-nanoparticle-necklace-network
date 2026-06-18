"""
run_multiseed_iv.py — full per-seed sweeps + multiseed I-V band.

For EVERY seed in optimized_config.yaml this runs the FULL Kirchhoff sweep
(algebraic_connectivity = True), so each seed's per-voltage table includes the
activated-node count AND the algebraic connectivity. It then builds the
mean +/- std I-V band across seeds.

Seeds are run in parallel across N_WORKERS processes (default 3).

Outputs (in outdir, default Results_multiseed/):
  per seed:
    sweep_table_seed<seed>_<mode>.csv     full table
                                          (incl. activated_nodes,
                                           algebraic_connectivity)
  aggregate:
    iv_curve_multiseed_<mode>.png             mean line + (mean +/- std) band
    iv_curve_multiseed_<mode>_byseed.csv      each seed's current per voltage
    iv_curve_multiseed_<mode>_aggregated.csv  mean / std / n per voltage

Usage:
    python run_multiseed_iv.py [outdir]

NOTE ON RESOURCES (read if you hit memory errors):
  algebraic_connectivity does a DENSE eigendecomposition at every voltage and
  is memory-heavy. Running several seeds at once multiplies that memory use,
  and on Windows each worker re-imports scipy. If you see "paging file is too
  small" or MemoryError, set N_WORKERS = 1 below (sequential) and/or enlarge
  the Windows paging file. N_WORKERS = 3 is requested here; drop it if needed.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path
from multiprocessing import Pool, freeze_support
import yaml
import numpy as np
import pandas as pd

from nanoparticle_network import NanoparticleNetwork
import sweep_analysis as sa
import visualize_sweep as vz

# Number of seeds to run in parallel. Lower to 1 if memory is tight.
N_WORKERS = 3


def _run_one_seed(args):
    """Worker: full-fidelity sweep for one seed; returns (seed, result dict)."""
    seed, config, V_start, V_max, V_step = args
    net = sa.make_network_from_config(NanoparticleNetwork, config, seed)
    res = sa.sweep(net, V_start, V_max, V_step, algebraic_connectivity=True)
    return seed, res


def main():
    config = yaml.safe_load(open("optimized_config.yaml"))
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Results_multiseed")
    outdir.mkdir(parents=True, exist_ok=True)

    mode = config["electrical"].get("resistance_model", "edge")
    seeds = [int(s) for s in config["seeds"]]

    sw = config["voltage_sweep"]
    V_start = float(sw["V_start"]); V_max = float(sw["V_max"]); V_step = float(sw["V_step"])

    n_workers = min(N_WORKERS, len(seeds))
    print(f"Full sweeps over seeds {seeds} (mode={mode}) on {n_workers} workers")
    print("  (algebraic_connectivity = True -> activated_nodes + connectivity "
          "in each table)")

    tasks = [(s, config, V_start, V_max, V_step) for s in seeds]
    with Pool(processes=n_workers, maxtasksperchild=1) as pool:
        results = list(pool.imap_unordered(_run_one_seed, tasks))

    results.sort(key=lambda sr: seeds.index(sr[0]))
    results_by_seed = [res for _, res in results]

    for s, res in results:
        sa.write_csv(res, str(outdir / f"sweep_table_seed{s}_{mode}.csv"))
        print(f"  seed {s}: percolation_V = {res.get('percolation_V')}  "
              f"-> sweep_table_seed{s}_{mode}.csv")

    tag = f"allseeds_{mode}"
    vz.plot_iv_curve_multiseed(
        results_by_seed, tag, str(outdir / f"iv_curve_multiseed_{mode}.png"))

    base_V = np.array([r["V"] for r in results_by_seed[0]["rows"]])
    byseed = {"voltage_V": base_V}
    stack = np.full((len(seeds), len(base_V)), np.nan)
    for k, (s, res) in enumerate(zip(seeds, results_by_seed)):
        cur = {round(r["V"], 10): r["total_current_A"] for r in res["rows"]}
        col = np.array([cur.get(round(float(v), 10), np.nan) for v in base_V])
        stack[k] = col
        byseed[f"current_nA_seed{s}"] = col * 1e9
    pd.DataFrame(byseed).to_csv(
        outdir / f"iv_curve_multiseed_{mode}_byseed.csv", index=False)

    I_mean = np.nanmean(stack, axis=0) * 1e9
    I_std = (np.nanstd(stack, axis=0, ddof=1) * 1e9
             if len(seeds) > 1 else np.zeros_like(I_mean))
    n_per_V = np.sum(np.isfinite(stack), axis=0)
    pd.DataFrame({
        "voltage_V": base_V,
        "current_mean_nA": I_mean,
        "current_std_nA": I_std,
        "n_seeds": n_per_V,
    }).to_csv(outdir / f"iv_curve_multiseed_{mode}_aggregated.csv", index=False)

    print(f"\nDone. Outputs in: {outdir.resolve()}")
    for p in sorted(outdir.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    freeze_support()
    main()
