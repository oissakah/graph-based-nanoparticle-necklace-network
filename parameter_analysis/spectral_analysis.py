"""Spectral and effective-resistance metrics for the canonical circuit.

The raw matrix is exactly the symmetric active-circuit Laplacian returned by
``NanoparticleNetwork.build_active_laplacian``. No second circuit construction
is maintained here, preventing solver/diagnostic drift.
"""

import numpy as np

EIG_REL_TOL = 1e-9


def build_full_system(net, activated_nodes):
    """Return the same active circuit system used for Kirchhoff solving."""
    return net.build_active_laplacian(activated_nodes)


def effective_resistance(net, activated_nodes, V_test=1.0):
    """Source-to-drain effective resistance, ``V_test / I``."""
    solution = net.solve_active_network(activated_nodes, float(V_test))
    current = solution["total_current_A"]
    return float(V_test) / current if current > 0 else np.nan


def spectral_metrics(net, activated_nodes):
    """Return Fiedler value, scale-free gap ratio, and zero-eigenvalue count."""
    system = build_full_system(net, activated_nodes)
    if system is None:
        return np.nan, np.nan, 0
    matrix = system["laplacian"].toarray()
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    eigenvalues = np.sort(eigenvalues)
    maximum = eigenvalues[-1]
    if maximum <= 0:
        return np.nan, np.nan, 0
    tolerance = EIG_REL_TOL * maximum
    n_zero = int(np.sum(eigenvalues < tolerance))
    nonzero = eigenvalues[eigenvalues >= tolerance]
    fiedler = float(nonzero[0]) if len(nonzero) else np.nan
    ratio = float(fiedler / maximum) if np.isfinite(fiedler) else np.nan
    return fiedler, ratio, n_zero
