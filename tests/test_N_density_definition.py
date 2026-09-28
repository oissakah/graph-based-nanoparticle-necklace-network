"""Regression tests for the N-density study definition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "parameter_analysis"
if str(PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(PARAM_DIR))

import parameter_sweep_cases as psc
from nanoparticle_network import NanoparticleNetwork


def _build(n: int) -> NanoparticleNetwork:
    net = NanoparticleNetwork(
        n_junctions=n,
        connection_radius=psc.connection_radius_for_N(n),
        domain=psc.DOMAIN_SIZE,
    )
    net.generate_network(
        seed=41,
        node_Va={
            "type": "normal",
            "mean": 6.0,
            "std": 3.0,
            "min": psc.VA_MIN,
            "max": psc.VA_MAX,
        },
        edge_k=float(psc.EDGE_K),
        node_resistance_ohm=float(psc.NODE_RESISTANCE_OHM),
    )
    return net


def test_N_sweep_uses_fixed_connection_radius():
    radii = [psc.connection_radius_for_N(n) for n in (200, 400, 600, 800)]
    assert np.allclose(radii, psc.CONNECTION_RADIUS)

    # Backward-compatible helper must not reintroduce the old N^-1/2 scaling.
    legacy_radii = [psc.scaled_radius_for_N(n) for n in (200, 400, 600, 800)]
    assert np.allclose(legacy_radii, psc.CONNECTION_RADIUS)


def test_fixed_radius_makes_N_a_density_parameter():
    small = _build(200)
    large = _build(800)

    deg_small = 2.0 * small.G.number_of_edges() / small.G.number_of_nodes()
    deg_large = 2.0 * large.G.number_of_edges() / large.G.number_of_nodes()

    # With the same domain and r_c, a larger N must produce a denser graph.
    assert deg_large > deg_small
