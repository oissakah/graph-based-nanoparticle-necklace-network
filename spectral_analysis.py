"""
spectral_analysis.py — spectral / effective-resistance metrics from the FULL
node+edge system matrix (the same node-split operator _solve_kirchhoff builds).

Verified properties of that matrix (see check): symmetric, real eigenvalues,
positive semi-definite weighted Laplacian. Standard spectral graph theory
therefore applies — BUT the entire spectrum sits at ~1e-8 because of the large
conductances (edge_k ~ 2e10, node_r_scale ~ 5e8), so EVERY eigenvalue threshold
here is RELATIVE to the largest eigenvalue, never absolute.

Metrics provided per voltage:
  * effective_resistance_ohm : source-to-drain effective resistance from the
        full matrix (matches V / I_solver to working precision).
  * algebraic_connectivity   : smallest NONZERO eigenvalue (Fiedler value) of the
        full system Laplacian — how robustly the conducting network holds
        together; dips flag near-bottlenecks.
  * spectral_gap_ratio       : algebraic_connectivity / largest eigenvalue, a
        scale-free version of the same.
"""

import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Relative tolerance for declaring an eigenvalue "zero" (scaled to spectrum max)
EIG_REL_TOL = 1e-9
R_MIN = 1.0


def build_full_system(net, activated_nodes):
    """
    Build the full node+edge system matrix on the source-to-drain bridging set,
    identically to _solve_kirchhoff's node-resistance branch, but WITHOUT
    boundary conditions (the raw Laplacian operator) and return the index maps
    needed to apply boundary conditions for effective resistance.

    Returns dict or None (None if no source-drain bridge exists).
    """
    working = set(activated_nodes)
    G_sub0 = net.G.subgraph(working)
    asrc = [n for n in net.source_nodes if n in working]
    adrn = [n for n in net.drain_nodes if n in working]
    if not asrc or not adrn:
        return None
    reach_s = set()
    for s in asrc:
        if s in G_sub0:
            reach_s |= nx.node_connected_component(G_sub0, s)
    reach_d = set()
    for d in adrn:
        if d in G_sub0:
            reach_d |= nx.node_connected_component(G_sub0, d)
    work = reach_s & reach_d
    if not work:
        return None

    G_sub = net.G.subgraph(work)
    asrc = [n for n in asrc if n in work]
    adrn = [n for n in adrn if n in work]
    all_el = set(asrc) | set(adrn)
    internal = [n for n in sorted(work) if n not in all_el]
    electrode = [n for n in sorted(work) if n in all_el]
    N_int, N_el = len(internal), len(electrode)
    SZ = 2 * N_int + N_el
    int_idx = {n: i for i, n in enumerate(internal)}
    el_idx = {n: i for i, n in enumerate(electrode)}

    def row_in(n):
        return 2 * int_idx[n] if n in int_idx else 2 * N_int + el_idx[n]

    def row_out(n):
        return 2 * int_idx[n] + 1 if n in int_idx else 2 * N_int + el_idx[n]

    depth = {n: 999 for n in work}
    q = list(asrc)
    for s in asrc:
        depth[s] = 0
    h = 0
    while h < len(q):
        c = q[h]; h += 1
        for nb in G_sub.neighbors(c):
            if nb in depth and depth[nb] == 999:
                depth[nb] = depth[c] + 1
                q.append(nb)

    M = lil_matrix((SZ, SZ))
    for n in internal:
        g = 1.0 / max(net.G.nodes[n].get("R_node", 0.0), R_MIN)
        ri, ro = row_in(n), row_out(n)
        M[ri, ri] += g; M[ro, ro] += g
        M[ri, ro] -= g; M[ro, ri] -= g
    for (i, j) in G_sub.edges():
        ge = 1.0 / max(net.G[i][j]["R_edge"], R_MIN)
        sn, dn = (i, j) if depth[i] <= depth[j] else (j, i)
        rso, rdi = row_out(sn), row_in(dn)
        M[rso, rso] += ge; M[rdi, rdi] += ge
        M[rso, rdi] -= ge; M[rdi, rso] -= ge

    return {
        "M": M, "SZ": SZ, "N_int": N_int, "N_el": N_el,
        "row_in": row_in, "row_out": row_out,
        "asrc": asrc, "adrn": adrn,
        "internal": internal, "electrode": electrode,
    }


def effective_resistance(net, activated_nodes, V_test=1.0):
    """
    Source-to-drain effective resistance from the full system matrix.
    Pins sources at V_test, drains at 0, solves, and returns V_test / I_total.
    Matches the solver's V/I to working precision. Returns np.nan if no bridge.
    """
    sysd = build_full_system(net, activated_nodes)
    if sysd is None:
        return np.nan
    M = sysd["M"].tolil()
    SZ = sysd["SZ"]
    row_in, row_out = sysd["row_in"], sysd["row_out"]
    asrc, adrn = sysd["asrc"], sysd["adrn"]

    b = np.zeros(SZ)
    for s in asrc:
        for r in (row_in(s), row_out(s)):
            M[r, :] = 0; M[r, r] = 1.0; b[r] = V_test
    for d in adrn:
        for r in (row_in(d), row_out(d)):
            M[r, :] = 0; M[r, r] = 1.0; b[r] = 0.0
    try:
        phi = spsolve(M.tocsr(), b)
        if not np.all(np.isfinite(phi)):
            return np.nan
    except Exception:
        return np.nan

    # total current into drains = sum over drain-incident edges
    G_sub = net.G.subgraph(set(sysd["internal"]) | set(sysd["electrode"]))
    depth_src = set(asrc)
    I_tot = 0.0
    for d in adrn:
        for nb in G_sub.neighbors(d):
            ge = 1.0 / max(net.G[d][nb]["R_edge"], R_MIN)
            # current from neighbor-out into drain-in
            I_tot += ge * (phi[row_out(nb)] - phi[row_in(d)])
    I_tot = abs(I_tot)
    return (V_test / I_tot) if I_tot > 0 else np.nan


def spectral_metrics(net, activated_nodes):
    """
    Algebraic connectivity (Fiedler value) and spectral gap ratio of the full
    system Laplacian, using a RELATIVE eigenvalue threshold.

    Returns (algebraic_connectivity, spectral_gap_ratio, n_zero_eigs).
    All np.nan / 0 if no bridge.
    """
    sysd = build_full_system(net, activated_nodes)
    if sysd is None:
        return np.nan, np.nan, 0
    M = sysd["M"].toarray()
    # symmetric PSD Laplacian -> use eigvalsh (real, sorted ascending)
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))  # symmetrize defensively
    ev = np.sort(ev)
    emax = ev[-1]
    if emax <= 0:
        return np.nan, np.nan, 0
    tol = EIG_REL_TOL * emax
    n_zero = int(np.sum(ev < tol))
    # algebraic connectivity = smallest eigenvalue above the zero threshold
    nonzero = ev[ev >= tol]
    alg_conn = float(nonzero[0]) if len(nonzero) else np.nan
    gap_ratio = float(alg_conn / emax) if np.isfinite(alg_conn) else np.nan
    return alg_conn, gap_ratio, n_zero
