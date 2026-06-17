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

Works for BOTH edge-only (resistance_model: edge) and node-resistance
(resistance_model: node) modes. Edge currents are reconstructed from the node
potentials returned by NanoparticleNetwork._solve_kirchhoff, accounting for the
node-splitting used in node-resistance mode. Verified against Kirchhoff's current
law (net current into every internal node ~1e-16 A).

No simple-path enumeration is used. The only graph traversal is connectivity
(networkx.node_connected_component), which is what "is source linked to drain"
genuinely requires.
"""

import csv
import numpy as np
import networkx as nx

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
        node_Vth={
            "type": thcfg["type"], "mean": float(thcfg["mean"]),
            "std": float(thcfg["std"]), "min": float(thcfg["min"]),
            "max": float(thcfg["max"]),
        },
        edge_k=float(elec["edge_k"]),
        node_r_scale=float(elec["node_r_scale"])
        if elec.get("resistance_model", "edge") == "node" else 0.0,
    )
    net.identify_sources_drains(
        left_thresh=float(netcfg["left_thresh"]),
        right_thresh=float(netcfg["right_thresh"]),
    )
    return net

def conductance_matrix(net, activated_nodes, R_MIN=1.0):
    """
    Conductance (Laplacian) matrix over the activated node set, in edge-resistance
    terms — exactly what plot_G_matrix_heatmap in the production script builds.
    Node resistors are a separate element and are NOT in this off-diagonal
    structure, so the heatmap is identical in both modes; currents differ and are
    handled separately.
    """
    active_list = sorted(activated_nodes)
    local_idx = {v: k for k, v in enumerate(active_list)}
    N = len(active_list)
    G = np.zeros((N, N))
    for (i, j) in net.G.edges():
        if i in local_idx and j in local_idx:
            g = 1.0 / max(net.G[i][j]['R_edge'], R_MIN)
            ki, kj = local_idx[i], local_idx[j]
            G[ki, ki] += g; G[kj, kj] += g
            G[ki, kj] -= g; G[kj, ki] -= g
    return G, active_list

def _edge_currents(net, activated_nodes, V_applied):
    """(total_current, node_potentials, edge_currents) correct for both modes.
    edge_currents: {(i, j): signed current}, +ve means flow i->j."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    total_current, phi = net._solve_kirchhoff(activated_nodes, V_applied)
    edge_currents = {}
    if not phi:
        return total_current, phi, edge_currents

    R_MIN = 1.0
    working = set(phi.keys())
    G_sub = net.G.subgraph(working)
    has_node_resistance = any(net.G.nodes[n].get('R_node', 0.0) > 0.0 for n in working)

    if not has_node_resistance:
        for (i, j) in G_sub.edges():
            if i in phi and j in phi:
                R_e = max(net.G[i][j]['R_edge'], R_MIN)
                I_ij = (phi[i] - phi[j]) / R_e
                if abs(I_ij) > 1e-30:
                    edge_currents[(i, j)] = I_ij
        return total_current, phi, edge_currents

    # Node-resistance: rebuild the split system to recover in/out terminal
    # potentials, then take current across the EDGE element only.
    active_sources = [n for n in net.source_nodes if n in working]
    active_drains = [n for n in net.drain_nodes if n in working]
    all_electrodes = set(active_sources) | set(active_drains)
    internal_nodes = [n for n in sorted(working) if n not in all_electrodes]
    electrode_nodes = [n for n in sorted(working) if n in all_electrodes]
    N_int = len(internal_nodes)
    int_idx = {n: i for i, n in enumerate(internal_nodes)}
    el_idx = {n: i for i, n in enumerate(electrode_nodes)}
    MAT_SIZE = 2 * N_int + len(electrode_nodes)

    def row_in(n):
        return 2 * int_idx[n] if n in int_idx else 2 * N_int + el_idx[n]

    def row_out(n):
        return 2 * int_idx[n] + 1 if n in int_idx else 2 * N_int + el_idx[n]

    bfs_depth = {n: 999 for n in working}
    q = list(active_sources)
    for s in active_sources:
        bfs_depth[s] = 0
    h = 0
    while h < len(q):
        cur = q[h]; h += 1
        for nbr in G_sub.neighbors(cur):
            if nbr in bfs_depth and bfs_depth[nbr] == 999:
                bfs_depth[nbr] = bfs_depth[cur] + 1
                q.append(nbr)

    G_mat = lil_matrix((MAT_SIZE, MAT_SIZE), dtype=float)
    for n in internal_nodes:
        g_n = 1.0 / max(net.G.nodes[n].get('R_node', 0.0), R_MIN)
        ri, ro = row_in(n), row_out(n)
        G_mat[ri, ri] += g_n; G_mat[ro, ro] += g_n
        G_mat[ri, ro] -= g_n; G_mat[ro, ri] -= g_n
    edge_orientation = {}
    for (i, j) in G_sub.edges():
        g_e = 1.0 / max(net.G[i][j]['R_edge'], R_MIN)
        if bfs_depth[i] <= bfs_depth[j]:
            src_n, drn_n = i, j
        else:
            src_n, drn_n = j, i
        edge_orientation[(i, j)] = (src_n, drn_n)
        rso, rdi = row_out(src_n), row_in(drn_n)
        G_mat[rso, rso] += g_e; G_mat[rdi, rdi] += g_e
        G_mat[rso, rdi] -= g_e; G_mat[rdi, rso] -= g_e
    b = np.zeros(MAT_SIZE)
    for src in active_sources:
        for row in (row_in(src), row_out(src)):
            G_mat[row, :] = 0; G_mat[row, row] = 1.0; b[row] = V_applied
    for drn in active_drains:
        for row in (row_in(drn), row_out(drn)):
            G_mat[row, :] = 0; G_mat[row, row] = 1.0; b[row] = 0.0
    try:
        psi = spsolve(G_mat.tocsr(), b)
        if not np.all(np.isfinite(psi)):
            return total_current, phi, {}
    except Exception:
        return total_current, phi, {}

    for (i, j) in G_sub.edges():
        src_n, drn_n = edge_orientation[(i, j)]
        R_e = max(net.G[i][j]['R_edge'], R_MIN)
        I = (psi[row_out(src_n)] - psi[row_in(drn_n)]) / R_e
        I_ij = I if src_n == i else -I
        if abs(I_ij) > 1e-30:
            edge_currents[(i, j)] = I_ij
    return total_current, phi, edge_currents

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
    n_total_nodes = net.G.number_of_nodes()
    n_total_edges = net.G.number_of_edges()

    V = V_start
    while V <= V_max + 1e-10:
        Vr = round(V, 10)
        activated_nodes = {n for n in net.G.nodes() if net.G.nodes[n]['Vth'] <= Vr}
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

        total_current, phi, edge_currents = _edge_currents(net, activated_nodes, Vr)
        edge_currents_by_V[Vr] = edge_currents

        if edge_currents:
            mags = np.array([abs(c) for c in edge_currents.values()])
            max_mag = mags.max()
            backbone_edges = int(np.sum(mags >= current_frac * max_mag))
            s = mags.sum()
            participation = float((s * s) / np.sum(mags * mags)) if s > 0 else 0.0
            # charge-conserving current = net current leaving the source set,
            # to machine precision). May differ from the solver's own
            # total_current in node-resistance mode (see README).
            src_set = set(net.source_nodes)
            I_cc = 0.0
            for (i, j), c in edge_currents.items():
                if i in src_set and j not in src_set:
                    I_cc += c
                elif j in src_set and i not in src_set:
                    I_cc -= c
            I_cc = abs(I_cc)

            # How concentrated is the flow? These describe the *shape* of the
            # edge-current magnitude distribution at this voltage.
            mean_mag = float(mags.mean())
            max_to_mean = float(max_mag / mean_mag) if mean_mag > 0 else 0.0
            # coefficient of variation: spread relative to mean (0 = uniform flow,
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
            backbone_edges, participation, I_cc = 0, 0.0, 0.0
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
            'total_current_chargeconserving_A': I_cc,
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

    fieldnames = ['V', 'activated_nodes', 'activated_edges', 'conducting_nodes',
                  'conducting_edges', 'source_drain_connected', 'total_current_A',
                  'total_current_chargeconserving_A',
                  'conductance_S', 'backbone_edges', 'participation_ratio',
                  'num_components', 'largest_cc_nodes', 'second_cc_nodes',
                  'largest_cc_fraction', 'mean_finite_cc',
                  'current_cv', 'current_gini', 'current_max_to_mean',
                  'current_top10_fraction',
                  'effective_resistance_ohm', 'algebraic_connectivity',
                  'spectral_gap_ratio']
    return {
        'rows': rows, 'fieldnames': fieldnames,
        'edge_currents_by_V': edge_currents_by_V,
        'percolation_V': percolation_V,
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
    """Focused I-V table: just voltage, current (both conventions), conductance.
    A clean companion to the full per-voltage table for plotting the I-V curve."""
    rows = result['rows']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['V', 'current_A', 'current_chargeconserving_A',
                    'conductance_S'])
        for r in rows:
            w.writerow([r['V'], r['total_current_A'],
                        r['total_current_chargeconserving_A'],
                        r['conductance_S']])
    return path

def write_edge_currents_csv(net, result, path, conducting_only=True,
                            voltages=None):
    """
    Export per-edge currents across the sweep, with node positions, so the
    network snapshots can be reproduced from CSV alone.

    One row per edge per voltage:
      V, node_i, node_j, x_i, y_i, x_j, y_j, current_A, abs_current_A
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
        w.writerow(['V', 'node_i', 'node_j', 'x_i', 'y_i', 'x_j', 'y_j',
                    'current_A', 'abs_current_A'])
        for Vr in sorted(ecbv.keys()):
            if Vset is not None and Vr not in Vset:
                continue
            ec = ecbv[Vr]
            for (i, j), c in ec.items():
                w.writerow([Vr, i, j,
                            f"{pos[i][0]:.6f}", f"{pos[i][1]:.6f}",
                            f"{pos[j][0]:.6f}", f"{pos[j][1]:.6f}",
                            f"{c:.8e}", f"{abs(c):.8e}"])
    return path
