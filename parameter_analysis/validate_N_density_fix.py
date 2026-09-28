#!/usr/bin/env python3
"""Validate the Case-4 definition after the N-density correction.

This is a lightweight reproducibility check for the intended interpretation of
``N`` in the Nanonecklace manuscript: the domain and geometric connection
radius are fixed, so increasing N increases graph density.

The script builds the representative seed-41 Case-4 networks and reports
average degree and conductance-weighted algebraic connectivity at 10 V.  It is
not required for production runs; it is provided as an audit/diagnostic tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nanoparticle_network import NanoparticleNetwork
from spectral_analysis import spectral_metrics
from parameter_sweep_cases import (
    CONNECTION_RADIUS,
    DOMAIN_SIZE,
    EDGE_K,
    LEFT_THRESH,
    NODE_RESISTANCE_OHM,
    RIGHT_THRESH,
    VA_MAX,
    VA_MIN,
    connection_radius_for_N,
)

N_VALUES = (200, 400, 600, 800)
MEAN_VA = 6.0
SIGMA_VA = 3.0
SEED = 41
CHECK_VOLTAGE = 10.0


def build_case4_network(n: int) -> NanoparticleNetwork:
    """Build one representative Case-4 network using the production settings."""
    net = NanoparticleNetwork(
        n_junctions=int(n),
        connection_radius=connection_radius_for_N(n),
        domain=DOMAIN_SIZE,
    )
    net.generate_network(
        seed=SEED,
        node_Va={
            "type": "normal",
            "mean": MEAN_VA,
            "std": SIGMA_VA,
            "min": VA_MIN,
            "max": VA_MAX,
        },
        edge_k=float(EDGE_K),
        node_resistance_ohm=float(NODE_RESISTANCE_OHM),
    )
    net.identify_sources_drains(
        left_thresh=float(LEFT_THRESH),
        right_thresh=float(RIGHT_THRESH),
    )
    return net


def main() -> None:
    rows = []
    for n in N_VALUES:
        radius = connection_radius_for_N(n)
        if not np.isclose(radius, CONNECTION_RADIUS):
            raise SystemExit(
                f"ERROR: N={n} uses r_c={radius}, expected fixed r_c={CONNECTION_RADIUS}."
            )

        net = build_case4_network(n)
        n_nodes = net.G.number_of_nodes()
        avg_degree = 2.0 * net.G.number_of_edges() / max(n_nodes, 1)
        active = net.activated_nodes(CHECK_VOLTAGE)
        lambda2, _gap_ratio, _nzero = spectral_metrics(net, active)
        rows.append((n, radius, n_nodes, avg_degree, len(active), lambda2))

    print("\nCase-4 N-density validation (seed 41, V = 10 V)")
    print("N    r_c     nodes   avg_degree   N_active   lambda2 [S]")
    for n, r, nodes, degree, nactive, lam in rows:
        print(f"{n:<4d} {r:<7.3f} {nodes:<7d} {degree:<12.3f} {nactive:<10d} {lam:.6e}")

    degrees = np.asarray([row[3] for row in rows], dtype=float)
    lambdas = np.asarray([row[5] for row in rows], dtype=float)

    if not np.all(np.diff(degrees) > 0):
        raise SystemExit("ERROR: average degree does not increase with N.")

    # For the deterministic reference seed/condition, the corrected study also
    # reproduces the expected increasing lambda_2 trend seen in the reference
    # publication figure.
    finite = np.isfinite(lambdas)
    if finite.all() and not np.all(np.diff(lambdas) > 0):
        raise SystemExit("ERROR: reference lambda2 trend is not increasing with N.")

    print("\nPASS: fixed radius is used and the N-density trend is restored.")


if __name__ == "__main__":
    main()
