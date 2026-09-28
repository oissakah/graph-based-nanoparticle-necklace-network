"""
run_sweep_analysis.py — produce the per-voltage table, evolution + snapshot
figures, and conductance-matrix heatmaps/CSVs for the production config.

Usage:
    python run_sweep_analysis.py [seed] [outdir]

Defaults to the first seed in optimized_config.yaml. Honors resistance_model
(edge or node) straight from the config, so both modes work unchanged.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path
import yaml
import numpy as np

from nanoparticle_network import NanoparticleNetwork
import sweep_analysis as sa
import visualize_sweep as vz


def main():
    config = yaml.safe_load(open("optimized_config.yaml"))
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else int(config["seeds"][0])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Results")
    outdir.mkdir(parents=True, exist_ok=True)

    mode = config["electrical"].get("resistance_model", "edge")
    tag = f"seed{seed}_{mode}"

    sw = config["voltage_sweep"]
    V_start = float(sw["V_start"]); V_max = float(sw["V_max"]); V_step = float(sw["V_step"])

    # voltages at which to dump the conductance matrix (CSV + npy + heatmap)
    G_voltages = [v for v in (1.0, 2.0, 6.0, 14.0) if V_start <= v <= V_max]
    # voltages for the network snapshot panels (conduction-region spreading)
    snap_voltages = [v for v in (1.0, 2.0, 6.0, 14.0) if V_start <= v <= V_max]

    net = sa.make_network_from_config(NanoparticleNetwork, config, seed)

    res = sa.sweep(net, V_start, V_max, V_step,
                   G_voltages=G_voltages,
                   G_out_prefix=str(outdir / f"Gmatrix_{tag}"))

    sa.write_csv(res, str(outdir / f"sweep_table_{tag}.csv"))

    # standalone Kirchhoff I-V curve: focused CSV + single-panel plot
    sa.write_iv_csv(res, str(outdir / f"iv_curve_{tag}.csv"))
    vz.plot_iv_curve(res, tag, str(outdir / f"iv_curve_{tag}.png"))

    # per-edge currents (with positions) so snapshots can be replotted from CSV
    sa.write_edge_currents_csv(
        net, res, str(outdir / f"edge_currents_{tag}.csv"),
        conducting_only=True)
    # smaller companion: only the snapshot voltages
    if snap_voltages:
        sa.write_edge_currents_csv(
            net, res, str(outdir / f"edge_currents_snapshots_{tag}.csv"),
            conducting_only=True, voltages=snap_voltages)

    vz.plot_evolution(res, tag, str(outdir / f"evolution_{tag}.png"))
    vz.plot_snapshots(net, res, tag, str(outdir / f"snapshots_{tag}.png"),
                      snap_voltages=snap_voltages if snap_voltages else None)
    vz.plot_percolation_current_outputs(
        net, res, tag, outdir / "current_distribution_plots")
    for V in G_voltages:
        vz.plot_Gmatrix(net, V, tag, str(outdir / f"Gmatrix_{tag}_V{V:g}.png"))

    print(f"\nDone. percolation_V = {res['percolation_V']}")
    print(f"Outputs in: {outdir.resolve()}")
    for p in sorted(outdir.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
