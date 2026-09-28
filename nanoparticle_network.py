"""Core voltage-gated Kirchhoff model for a nanonecklace network.

The model is deliberately narrow and internally consistent:

* junction ``i`` has a phenomenological activation voltage ``Va_i``;
* it is electrically available when the global applied voltage satisfies
  ``Va_i <= V``;
* an edge is available only when both endpoint junctions are active;
* the resulting undirected resistor network is solved exactly with Kirchhoff
  nodal analysis;
* no path-current approximation or additional transport law is present.

If junction resistance is enabled, a fixed resistance independent of ``Va`` is
assigned to each junction. Half of each endpoint junction resistance is added
to the connecting edge. Along an unbranched path this counts every internal
junction resistance exactly once, while retaining a symmetric passive circuit
Laplacian and an unambiguous current on every undirected edge.
"""

from __future__ import annotations

import pickle
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class NanoparticleNetwork:
    """Spatial graph representation of a voltage-activated resistor network."""

    R_MIN_OHM = 1.0

    def __init__(self, n_junctions=50, connection_radius=0.2,
                 domain=(1.0, 1.0)):
        self.G = nx.Graph()
        self.n_junctions = int(n_junctions)
        self.connection_radius = float(connection_radius)
        self.domain = tuple(domain)
        self.positions = None
        self.source_nodes = []
        self.drain_nodes = []
        self.source_node = None
        self.drain_node = None

    def generate_network(self, seed=None, node_Va=(0.1, 0.3, 0.5),
                         edge_k=1000.0, node_resistance_ohm=0.0):
        """Generate a random geometric network.

        ``node_Va`` may be discrete values or a distribution dictionary.
        ``edge_k`` sets ``R_edge = edge_k * distance``. A fixed
        ``node_resistance_ohm`` is independent of activation voltage; zero
        selects the edge-only model.
        """
        rng = np.random.default_rng(seed)
        width, height = self.domain
        self.positions = rng.random((self.n_junctions, 2)) * [width, height]
        use_distribution = isinstance(node_Va, dict)

        for i in range(self.n_junctions):
            va = (self._sample_Va_distribution(node_Va, rng)
                  if use_distribution else float(rng.choice(node_Va)))
            self.G.add_node(
                i, pos=self.positions[i], Va=va,
                R_node=float(node_resistance_ohm), activated=False,
            )

        for i in range(self.n_junctions):
            for j in range(i + 1, self.n_junctions):
                distance = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if distance < self.connection_radius:
                    resistance = float(edge_k) * distance
                    self.G.add_edge(
                        i, j, resistance=resistance, distance=distance,
                        length=distance, R_edge=resistance, activated=False,
                    )

        self._remove_nonjunction_nodes()
        print(f"Generated network: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    @staticmethod
    def _sample_Va_distribution(dist_config, rng):
        """Sample one activation voltage, using true truncation when bounded."""
        kind = str(dist_config.get("type", "uniform")).lower()
        if kind == "uniform":
            return float(rng.uniform(dist_config.get("min", 0.0),
                                     dist_config.get("max", 10.0)))
        if kind in {"normal", "gaussian"}:
            mean = float(dist_config.get("mean", 8.0))
            std = float(dist_config.get("std", 2.0))
            vmin = float(dist_config.get("min", 0.0))
            vmax = float(dist_config.get("max", np.inf))
            if std <= 0:
                return float(np.clip(mean, vmin, vmax))
            for _ in range(10000):
                value = float(rng.normal(mean, std))
                if vmin <= value <= vmax:
                    return value
            raise RuntimeError("Could not sample the requested truncated normal.")
        if kind == "lognormal":
            return float(rng.lognormal(dist_config.get("mean", 2.0),
                                       dist_config.get("sigma", 0.5)))
        if kind == "exponential":
            return float(dist_config.get("offset", 0.0)
                         + rng.exponential(dist_config.get("scale", 5.0)))
        if kind == "gamma":
            return float(dist_config.get("offset", 0.0)
                         + rng.gamma(dist_config.get("shape", 2.0),
                                     dist_config.get("scale", 3.0)))
        if kind == "weibull":
            return float(dist_config.get("scale", 8.0)
                         * rng.weibull(dist_config.get("shape", 2.0)))
        if kind == "bimodal":
            mode1, mode2 = dist_config.get("mode1"), dist_config.get("mode2")
            if mode1 is None or mode2 is None:
                raise ValueError("Bimodal distribution requires mode1 and mode2.")
            chosen = mode1 if rng.random() < dist_config.get("weight", 0.5) else mode2
            return NanoparticleNetwork._sample_Va_distribution(chosen, rng)
        raise ValueError(f"Unknown activation-voltage distribution: {kind}")

    def _remove_nonjunction_nodes(self):
        """Iteratively remove degree-0/1 sites and relabel remaining nodes."""
        total_removed = 0
        while True:
            low_degree = [node for node, degree in self.G.degree() if degree < 2]
            if not low_degree:
                break
            self.G.remove_nodes_from(low_degree)
            total_removed += len(low_degree)
        if not total_removed:
            return
        remaining = sorted(self.G.nodes())
        self.positions = self.positions[remaining]
        mapping = {old: new for new, old in enumerate(remaining)}
        self.G = nx.relabel_nodes(self.G, mapping, copy=True)
        for node in self.G.nodes():
            self.G.nodes[node]["pos"] = self.positions[node]
        self.n_junctions = self.G.number_of_nodes()
        print(f"Removed {total_removed} nodes with degree < 2")

    def set_electrodes(self, source=0, drain=None):
        """Set single source/drain electrodes (backward-compatible helper)."""
        self.source_node = int(source)
        self.drain_node = int(drain) if drain is not None else self.n_junctions - 1
        self.source_nodes = [self.source_node]
        self.drain_nodes = [self.drain_node]

    def identify_sources_drains(self, left_thresh=0.15, right_thresh=0.85):
        """Identify electrode-contact nodes from normalized x-position."""
        width = self.domain[0]
        self.source_nodes = [
            node for node in self.G.nodes()
            if self.positions[node][0] <= float(left_thresh) * width
        ]
        self.drain_nodes = [
            node for node in self.G.nodes()
            if self.positions[node][0] >= float(right_thresh) * width
        ]
        print(f"Identified {len(self.source_nodes)} source nodes and "
              f"{len(self.drain_nodes)} drain nodes")
        return self.source_nodes, self.drain_nodes

    def save_network(self, filename):
        with open(filename, "wb") as stream:
            pickle.dump(self.G, stream)
        np.save(str(filename).replace(".pkl", "_positions.npy"), self.positions)

    def load_network(self, filename):
        with open(filename, "rb") as stream:
            self.G = pickle.load(stream)
        self.positions = np.load(str(filename).replace(".pkl", "_positions.npy"))
        self.n_junctions = self.G.number_of_nodes()

    def activated_nodes(self, applied_voltage):
        """Return nodes satisfying the global phenomenological gating rule."""
        return {
            node for node in self.G.nodes()
            if self.G.nodes[node]["Va"] <= float(applied_voltage)
        }

    def _bridging_nodes(self, activated_nodes: Iterable[int]):
        """Keep activated components that touch both electrode sets."""
        active = set(activated_nodes)
        if not active:
            return set()
        active_sources = set(self.source_nodes) & active
        active_drains = set(self.drain_nodes) & active
        if not active_sources or not active_drains:
            return set()
        bridging = set()
        for component in nx.connected_components(self.G.subgraph(active)):
            if component & active_sources and component & active_drains:
                bridging.update(component)
        return bridging

    def total_edge_resistance(self, i, j):
        """Resistance of one connection, including adjacent junction halves."""
        electrodes = set(self.source_nodes) | set(self.drain_nodes)
        ri = 0.0 if i in electrodes else self.G.nodes[i].get("R_node", 0.0)
        rj = 0.0 if j in electrodes else self.G.nodes[j].get("R_node", 0.0)
        return max(float(self.G[i][j]["R_edge"]) + 0.5 * (ri + rj),
                   self.R_MIN_OHM)

    def build_active_laplacian(self, activated_nodes):
        """Build the exact symmetric circuit Laplacian used by the solver."""
        working = self._bridging_nodes(activated_nodes)
        if not working:
            return None
        nodes = sorted(working)
        index = {node: k for k, node in enumerate(nodes)}
        matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
        edge_conductance = {}
        for i, j in self.G.subgraph(working).edges():
            conductance = 1.0 / self.total_edge_resistance(i, j)
            edge_conductance[(i, j)] = conductance
            ii, jj = index[i], index[j]
            matrix[ii, ii] += conductance
            matrix[jj, jj] += conductance
            matrix[ii, jj] -= conductance
            matrix[jj, ii] -= conductance
        return {
            "laplacian": csr_matrix(matrix), "nodes": nodes, "index": index,
            "working_nodes": working, "edge_conductance": edge_conductance,
            "active_sources": sorted(set(self.source_nodes) & working),
            "active_drains": sorted(set(self.drain_nodes) & working),
        }

    def solve_active_network(self, activated_nodes, applied_voltage):
        """Solve one active graph and derive all currents from that solution."""
        system = self.build_active_laplacian(activated_nodes)
        empty = {
            "total_current_A": 0.0, "source_current_A": 0.0,
            "drain_current_A": 0.0, "current_balance_error_A": 0.0,
            "node_potentials": {}, "edge_currents": {}, "working_nodes": set(),
        }
        if system is None:
            return empty

        nodes, index = system["nodes"], system["index"]
        sources, drains = system["active_sources"], system["active_drains"]
        boundary = sources + drains
        boundary_set = set(boundary)
        unknown = [node for node in nodes if node not in boundary_set]
        potentials = np.zeros(len(nodes), dtype=float)
        for source in sources:
            potentials[index[source]] = float(applied_voltage)

        if unknown:
            unknown_idx = np.array([index[node] for node in unknown], dtype=int)
            boundary_idx = np.array([index[node] for node in boundary], dtype=int)
            lap = system["laplacian"].tocsr()
            lhs = lap[unknown_idx][:, unknown_idx]
            rhs = -lap[unknown_idx][:, boundary_idx] @ potentials[boundary_idx]
            try:
                solved = spsolve(lhs, rhs)
            except Exception:
                solved = np.full(len(unknown), np.nan)
            if not np.all(np.isfinite(solved)):
                failed = dict(empty)
                failed["current_balance_error_A"] = np.nan
                return failed
            potentials[unknown_idx] = solved

        raw_edge_currents = {}
        source_current = drain_current = 0.0
        source_set, drain_set = set(sources), set(drains)
        for (i, j), conductance in system["edge_conductance"].items():
            current = conductance * (potentials[index[i]] - potentials[index[j]])
            raw_edge_currents[(i, j)] = float(current)
            if i in source_set and j not in source_set:
                source_current += current
            elif j in source_set and i not in source_set:
                source_current -= current
            if i in drain_set and j not in drain_set:
                drain_current -= current
            elif j in drain_set and i not in drain_set:
                drain_current += current

        source_current = abs(float(source_current))
        drain_current = abs(float(drain_current))
        total_current = 0.5 * (source_current + drain_current)
        # Exclude roundoff-level loop currents from plots/statistics while using
        # every edge above for the electrode-current balance.
        max_edge_current = max((abs(value) for value in raw_edge_currents.values()),
                               default=0.0)
        current_tolerance = max(1e-30, 1e-12 * max_edge_current)
        edge_currents = {
            edge: value for edge, value in raw_edge_currents.items()
            if abs(value) > current_tolerance
        }
        return {
            "total_current_A": total_current,
            "source_current_A": source_current,
            "drain_current_A": drain_current,
            "current_balance_error_A": abs(source_current - drain_current),
            "node_potentials": {
                node: float(potentials[index[node]]) for node in nodes
            },
            "edge_currents": edge_currents,
            "working_nodes": set(nodes),
        }

    def _solve_kirchhoff(self, activated_nodes, V_applied):
        """Backward-compatible tuple interface to the canonical solver."""
        result = self.solve_active_network(activated_nodes, V_applied)
        return result["total_current_A"], result["node_potentials"]

    def calculate_iv_curve_kirchhoff(self, V_start=0.0, V_max=2.0,
                                     V_step=0.01):
        """Sweep global voltage and solve the activated resistor graph."""
        if not self.source_nodes or not self.drain_nodes:
            raise ValueError("Identify sources and drains before solving.")
        voltages, currents, conductances = [], [], []
        voltage = float(V_start)
        while voltage <= float(V_max) + 1e-10:
            voltage = round(voltage, 10)
            result = self.solve_active_network(self.activated_nodes(voltage), voltage)
            current = result["total_current_A"]
            voltages.append(voltage)
            currents.append(current)
            conductances.append(current / voltage if voltage > 1e-12 else 0.0)
            voltage = round(voltage + float(V_step), 10)
        return {
            "voltages": np.asarray(voltages),
            "currents": np.asarray(currents),
            "conductances": np.asarray(conductances),
            "num_paths": np.zeros(len(voltages)),
            "path_details": [[] for _ in voltages],
        }
