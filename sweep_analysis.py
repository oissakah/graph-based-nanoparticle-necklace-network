"""
sweep_analysis.py — Per-voltage network-evolution analysis for NanoparticleNetwork,
wired to the optimized_config.yaml production setup.

Outputs you can plot yourself:
  1. Per-voltage table (CSV): V, activated_nodes, activated_edges, conducting_nodes,
     conducting_edges, source_drain_connected, total_current_A, conductance_S,
     backbone_edges, participation_ratio.
  2. Conductance matrix G at each requested voltage: a long-form CSV of nonzero
     G_ij entries plus a dense .npy and a node-index sidecar, so you can render
     the heatmap however you like. visualize_sweep.py also writes a PNG.

Works for BOTH edge-only (resistance_model: edge) and fixed junction-resistance
(resistance_model: node) modes. Total current and every edge current come from
one canonical Kirchhoff solution; source/drain imbalance is reported explicitly.

No simple-path enumeration is used. Connectivity identifies the first spanning
state, and one unit-capacity max-flow calculation at that voltage measures the
number of edge-disjoint source-to-drain pathways.
"""

import csv
import numpy as np
import networkx as nx


def count_edge_disjoint_pathways(net, activated_nodes):
    """Count independent activated source-to-drain channels.

    The returned integer is the maximum number of source-to-drain paths that
    do not share an internal graph edge. By max-flow/min-cut duality it is also
    the unit-capacity source-drain minimum-cut size. Electrode-to-boundary-node
    links are assigned effectively infinite capacity, so the count measures
    bottlenecks inside the activated nanonecklace rather than at the artificial
    super-source/super-drain connections.

    This is deliberately *not* a count of every simple path: looped networks
    can contain an enormous number of overlapping simple paths that do not
    represent independent transport channels.
    """
    active = set(activated_nodes)
    sources = [node for node in net.source_nodes if node in active]
    drains = [node for node in net.drain_nodes if node in active]
    if not sources or not drains:
        return 0

    graph = net.G.subgraph(active)
    if graph.number_of_edges() == 0:
        return 0

    super_source = ("__electrode__", "source")
    super_drain = ("__electrode__", "drain")
    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(graph.nodes())

    # Reciprocal unit-capacity arcs are the standard flow representation of
    # an undirected unit-capacity edge. Opposing flow can always be cancelled,
    # leaving the same integral edge-disjoint-path count.
    for node_i, node_j in graph.edges():
        flow_graph.add_edge(node_i, node_j, capacity=1)
        flow_graph.add_edge(node_j, node_i, capacity=1)

    electrode_capacity = max(1, graph.number_of_edges() + 1)
    for node in sources:
        flow_graph.add_edge(super_source, node,
                            capacity=electrode_capacity)
    for node in drains:
        flow_graph.add_edge(node, super_drain,
                            capacity=electrode_capacity)

    value = nx.maximum_flow_value(
        flow_graph, super_source, super_drain, capacity="capacity",
        flow_func=nx.algorithms.flow.shortest_augmenting_path)
    return int(round(float(value)))


def make_network_from_config(NanoparticleNetwork, config, seed):
    """Mirror make_network() in optimized_final_system.py so behaviour matches."""
    netcfg = config["network"]
    thcfg = config["threshold_distribution"]
    elec = config["electrical"]
    net = NanoparticleNetwork(
        n_junctions=int(netcfg["n_junctions"]),
        connection_radius=float(netcfg["connection_radius"]),
        domain=tuple(netcfg["domain_size"]),
    )
    net.generate_network(
        seed=int(seed),
        node_Va={
            "type": thcfg["type"], "mean": float(thcfg["mean"]),
            "std": float(thcfg["std"]), "min": float(thcfg["min"]),
            "max": float(thcfg["max"]),
        },
        edge_k=float(elec["edge_k"]),
        node_resistance_ohm=float(elec.get("node_resistance_ohm", 0.0))
        if elec.get("resistance_model", "edge") == "node" else 0.0,
    )
    net.identify_sources_drains(
        left_thresh=float(netcfg["left_thresh"]),
        right_thresh=float(netcfg["right_thresh"]),
    )
    return net


def conductance_matrix(net, activated_nodes):
    """
    Full active-circuit Laplacian used by the solver. In junction-resistance mode
    each edge conductance includes half the fixed resistance of each internal
    endpoint, so this matrix and the current calculation are exactly consistent.
    """
    system = net.build_active_laplacian(activated_nodes)
    if system is None:
        return np.zeros((0, 0)), []
    return system["laplacian"].toarray(), system["nodes"]


def _edge_currents(net, activated_nodes, V_applied):
    """Return canonical total current, node potentials, and signed edge currents."""
    solution = net.solve_active_network(activated_nodes, V_applied)
    return (solution["total_current_A"], solution["node_potentials"],
            solution["edge_currents"])


def sweep(net, V_start, V_max, V_step, current_frac=0.01,
          G_voltages=None, G_out_prefix=None, effective_resistance=True,
          algebraic_connectivity=True):
    """Run the sweep, collect per-voltage metrics, optionally export G matrices.

    effective_resistance=True (cheap, one extra linear solve per voltage): adds
        source-to-drain effective resistance from the full node+edge matrix
        (matches V/I exactly).
    algebraic_connectivity=True (EXPENSIVE: a dense eigendecomposition per
        voltage, roughly doubles sweep time on the production network): adds the
        Fiedler value and spectral gap ratio. Set False to skip for speed; it is
        most informative on fragmented/voided networks and nearly flat on the
        dense baseline.

    Returns dict: rows, fieldnames, edge_currents_by_V, percolation_V,
                  n_total_nodes, n_total_edges."""
    if not net.source_nodes or not net.drain_nodes:
        raise ValueError("Identify sources and drains first.")

    G_voltages_set = set(round(float(v), 10) for v in (G_voltages or []))
    rows, edge_currents_by_V, percolation_V = [], {}, None
    percolation_pathways = 0
    percolation_active_nodes = 0
    n_total_nodes = net.G.number_of_nodes()
    n_total_edges = net.G.number_of_edges()

    V = V_start
    while V <= V_max + 1e-10:
        Vr = round(V, 10)
        activated_nodes = net.activated_nodes(Vr)
        activated_edges = [(i, j) for i, j in net.G.edges()
                           if i in activated_nodes and j in activated_nodes]

        G_act = net.G.subgraph(activated_nodes)
        src_comp = set()
        for s in net.source_nodes:
            if s in G_act:
                src_comp |= nx.node_connected_component(G_act, s)
        drn_comp = set()
        for d in net.drain_nodes:
            if d in G_act:
                drn_comp |= nx.node_connected_component(G_act, d)
        connected = len(src_comp & drn_comp) > 0
        if connected and percolation_V is None:
            percolation_V = Vr
            percolation_pathways = count_edge_disjoint_pathways(
                net, activated_nodes)
            percolation_active_nodes = len(activated_nodes)

        # --- connected-component structure of the activated subgraph ---
        # Component sizes (in nodes) of the activated subgraph. The largest and
        # second-largest are the classic percolation order parameters: the
        # second-largest tends to peak near the percolation threshold.
        comp_sizes = sorted((len(c) for c in nx.connected_components(G_act)),
                            reverse=True)
        num_components   = len(comp_sizes)
        largest_cc_nodes = comp_sizes[0] if comp_sizes else 0
        second_cc_nodes  = comp_sizes[1] if len(comp_sizes) > 1 else 0
        # fraction of activated nodes in the largest component (percolation
        # strength / order parameter): rises from ~0 to ~1 through the transition
        largest_cc_fraction = (largest_cc_nodes / len(activated_nodes)
                               if activated_nodes else 0.0)
        # mean component size excluding the largest (susceptibility-like; also
        # peaks near threshold)
        if num_components > 1:
            mean_finite_cc = float(np.mean(comp_sizes[1:]))
        else:
            mean_finite_cc = 0.0

        solution = net.solve_active_network(activated_nodes, Vr)
        total_current = solution["total_current_A"]
        phi = solution["node_potentials"]
        edge_currents = solution["edge_currents"]
        edge_currents_by_V[Vr] = edge_currents

        if edge_currents:
            mags = np.array([abs(c) for c in edge_currents.values()])
            max_mag = mags.max()
            backbone_edges = int(np.sum(mags >= current_frac * max_mag))
            s = mags.sum()
            participation = float((s * s) / np.sum(mags * mags)) if s > 0 else 0.0
            # --- current-distribution shape statistics ---
            # How concentrated is the flow? These describe the *shape* of the
            # edge-current magnitude distribution at this voltage.
            mean_mag = float(mags.mean())
            max_to_mean = float(max_mag / mean_mag) if mean_mag > 0 else 0.0
            # coefficient of variation: spread relative to mean (0 = uniform flow,
            # large = a few edges dominate)
            cv_current = float(mags.std() / mean_mag) if mean_mag > 0 else 0.0
            # Gini coefficient: 0 = perfectly even current across edges,
            # 1 = all current in one edge. A clean inequality measure of how
            # unequally current is shared.
            sorted_mags = np.sort(mags)
            n_e = len(sorted_mags)
            cum = np.cumsum(sorted_mags)
            gini_current = float((2.0 * np.sum((np.arange(1, n_e + 1)) * sorted_mags)
                                  - (n_e + 1) * cum[-1]) / (n_e * cum[-1])) \
                if cum[-1] > 0 else 0.0
            # fraction of total current carried by the top 10% of edges
            k = max(1, int(np.ceil(0.10 * n_e)))
            top10_current_fraction = float(sorted_mags[-k:].sum() / cum[-1]) \
                if cum[-1] > 0 else 0.0
        else:
            backbone_edges, participation = 0, 0.0
            max_to_mean = cv_current = gini_current = top10_current_fraction = 0.0

        if Vr in G_voltages_set and G_out_prefix and activated_nodes:
            Gmat, active_list = conductance_matrix(net, activated_nodes)
            np.save(f"{G_out_prefix}_V{Vr:g}.npy", Gmat)
            with open(f"{G_out_prefix}_V{Vr:g}_nodeindex.csv", 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['matrix_index', 'node_id', 'is_source', 'is_drain'])
                for k, nid in enumerate(active_list):
                    w.writerow([k, nid, int(nid in net.source_nodes),
                                int(nid in net.drain_nodes)])
            with open(f"{G_out_prefix}_V{Vr:g}.csv", 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['row_index', 'col_index', 'row_node', 'col_node',
                            'conductance_S'])
                N = Gmat.shape[0]
                for i in range(N):
                    for j in range(N):
                        if Gmat[i, j] != 0.0:
                            w.writerow([i, j, active_list[i], active_list[j],
                                        Gmat[i, j]])

        conductance = total_current / Vr if Vr > 1e-12 else 0.0

        # full-system-matrix spectral metrics
        R_eff = alg_conn = gap_ratio = np.nan
        if activated_nodes and (effective_resistance or algebraic_connectivity):
            import spectral_analysis as _sp
            if effective_resistance:
                R_eff = _sp.effective_resistance(net, activated_nodes, V_test=1.0)
            if algebraic_connectivity:
                alg_conn, gap_ratio, _ = _sp.spectral_metrics(net, activated_nodes)

        rows.append({
            'V': Vr,
            'activated_nodes': len(activated_nodes),
            'activated_edges': len(activated_edges),
            'conducting_nodes': len(phi),
            'conducting_edges': len(edge_currents),
            'source_drain_connected': int(connected),
            'total_current_A': total_current,
            'source_current_A': solution["source_current_A"],
            'drain_current_A': solution["drain_current_A"],
            'current_balance_error_A': solution["current_balance_error_A"],
            'conductance_S': conductance,
            'backbone_edges': backbone_edges,
            'participation_ratio': participation,
            # connected-component structure
            'num_components': num_components,
            'largest_cc_nodes': largest_cc_nodes,
            'second_cc_nodes': second_cc_nodes,
            'largest_cc_fraction': largest_cc_fraction,
            'mean_finite_cc': mean_finite_cc,
            # current-distribution shape
            'current_cv': cv_current,
            'current_gini': gini_current,
            'current_max_to_mean': max_to_mean,
            'current_top10_fraction': top10_current_fraction,
            # full-system-matrix spectral metrics
            'effective_resistance_ohm': R_eff,
            'algebraic_connectivity': alg_conn,
            'spectral_gap_ratio': gap_ratio,
        })
        V = round(V + V_step, 10)

    # This is a run-level property evaluated at the first connected voltage.
    # Repeating it in every row makes the value available in any sliced or
    # aggregated evolution CSV without implying that it was recomputed later.
    for row in rows:
        row['edge_disjoint_pathways_at_Vperc'] = percolation_pathways
        row['active_nodes_at_Vperc'] = percolation_active_nodes

    fieldnames = ['V', 'activated_nodes', 'activated_edges', 'conducting_nodes',
                  'conducting_edges', 'source_drain_connected', 'total_current_A',
                  'source_current_A', 'drain_current_A',
                  'current_balance_error_A',
                  'conductance_S', 'backbone_edges', 'participation_ratio',
                  'num_components', 'largest_cc_nodes', 'second_cc_nodes',
                  'largest_cc_fraction', 'mean_finite_cc',
                  'current_cv', 'current_gini', 'current_max_to_mean',
                  'current_top10_fraction',
                  'effective_resistance_ohm', 'algebraic_connectivity',
                  'spectral_gap_ratio', 'edge_disjoint_pathways_at_Vperc',
                  'active_nodes_at_Vperc']
    return {
        'rows': rows, 'fieldnames': fieldnames,
        'edge_currents_by_V': edge_currents_by_V,
        'percolation_V': percolation_V,
        'percolation_pathways': percolation_pathways,
        'percolation_active_nodes': percolation_active_nodes,
        'n_total_nodes': n_total_nodes, 'n_total_edges': n_total_edges,
    }


def write_csv(result, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=result['fieldnames'])
        w.writeheader()
        for row in result['rows']:
            w.writerow(row)
    return path


def write_iv_csv(result, path):
    """Focused I-V table with current-conservation diagnostics.
    A clean companion to the full per-voltage table for plotting the I-V curve."""
    rows = result['rows']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['V', 'current_A', 'source_current_A', 'drain_current_A',
                    'current_balance_error_A', 'conductance_S'])
        for r in rows:
            w.writerow([r['V'], r['total_current_A'], r['source_current_A'],
                        r['drain_current_A'], r['current_balance_error_A'],
                        r['conductance_S']])
    return path


def write_edge_currents_csv(net, result, path, conducting_only=True,
                            voltages=None):
    """
    Export per-edge currents across the sweep, with node positions, so the
    network snapshots can be reproduced from CSV alone.

    One row per edge per voltage. The run-level
    edge_disjoint_pathways_at_Vperc value is repeated so a standalone edge CSV
    retains the independent-pathway count used in its snapshot title.
    current_A is signed (positive = flow node_i -> node_j).

    Parameters
    ----------
    conducting_only : if True (default), only edges carrying nonzero current are
        written. Set False to write every activated edge (0 current included) —
        much larger.
    voltages : optional list of voltages to restrict the export to (e.g. just the
        snapshot voltages). If None, all swept voltages are written.
    """
    pos = net.positions
    ecbv = result['edge_currents_by_V']
    Vset = set(round(float(v), 10) for v in voltages) if voltages is not None else None

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['V', 'edge_disjoint_pathways_at_Vperc',
                    'active_nodes_at_Vperc',
                    'node_i', 'node_j', 'x_i', 'y_i', 'x_j', 'y_j',
                    'current_A', 'abs_current_A'])
        for Vr in sorted(ecbv.keys()):
            if Vset is not None and Vr not in Vset:
                continue
            ec = ecbv[Vr]
            for (i, j), c in ec.items():
                w.writerow([Vr, result.get('percolation_pathways', 0),
                            result.get('percolation_active_nodes', 0), i, j,
                            f"{pos[i][0]:.6f}", f"{pos[i][1]:.6f}",
                            f"{pos[j][0]:.6f}", f"{pos[j][1]:.6f}",
                            f"{c:.8e}", f"{abs(c):.8e}"])
    return path
