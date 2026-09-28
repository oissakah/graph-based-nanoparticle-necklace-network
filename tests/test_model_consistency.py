import sys
import unittest
from pathlib import Path

import numpy as np
import networkx as nx
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoparticle_network import NanoparticleNetwork
import spectral_analysis
import sweep_analysis


class ModelConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.net = NanoparticleNetwork(120, 0.20, (1.0, 1.0))
        self.net.generate_network(
            seed=41,
            node_Va={"type": "normal", "mean": 6.0, "std": 2.0,
                     "min": 0.0, "max": 20.0},
            edge_k=2.0e10,
            node_resistance_ohm=3.5e9,
        )
        self.net.identify_sources_drains(0.15, 0.85)

    def first_conducting_solution(self):
        for voltage in np.arange(0.0, 12.5, 0.5):
            solution = self.net.solve_active_network(
                self.net.activated_nodes(voltage), voltage)
            if solution["total_current_A"] > 0:
                return voltage, solution
        self.fail("Test network did not percolate.")

    def test_source_and_drain_currents_match(self):
        _, solution = self.first_conducting_solution()
        self.assertTrue(np.isclose(
            solution["source_current_A"], solution["drain_current_A"],
            rtol=1e-10, atol=1e-22))

    def test_effective_resistance_matches_voltage_over_current(self):
        voltage, solution = self.first_conducting_solution()
        resistance = spectral_analysis.effective_resistance(
            self.net, self.net.activated_nodes(voltage), V_test=1.0)
        self.assertTrue(np.isclose(
            voltage / solution["total_current_A"], resistance,
            rtol=1e-10, atol=1e-6))

    def test_circuit_laplacian_is_symmetric_and_psd(self):
        voltage, _ = self.first_conducting_solution()
        system = self.net.build_active_laplacian(self.net.activated_nodes(voltage))
        matrix = system["laplacian"].toarray()
        self.assertTrue(np.allclose(matrix, matrix.T, rtol=0, atol=1e-18))
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-18)

    def test_activation_and_resistance_are_decoupled(self):
        resistances = {self.net.G.nodes[n]["R_node"] for n in self.net.G.nodes()}
        activation_voltages = {self.net.G.nodes[n]["Va"] for n in self.net.G.nodes()}
        self.assertEqual(len(resistances), 1)
        self.assertGreater(len(activation_voltages), 1)

    def test_edge_disjoint_pathway_count(self):
        graph = nx.Graph()
        graph.add_edges_from([
            (0, 1), (1, 3),
            (0, 2), (2, 3),
        ])
        network = SimpleNamespace(
            G=graph, source_nodes={0}, drain_nodes={3})
        self.assertEqual(
            sweep_analysis.count_edge_disjoint_pathways(
                network, {0, 1, 2, 3}),
            2)

        graph.add_edge(3, 4)
        network.drain_nodes = {4}
        self.assertEqual(
            sweep_analysis.count_edge_disjoint_pathways(
                network, {0, 1, 2, 3, 4}),
            1)


if __name__ == "__main__":
    unittest.main()
