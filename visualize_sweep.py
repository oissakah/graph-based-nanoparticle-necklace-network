"""
visualize_sweep.py — figures for the voltage-sweep analysis (production config).

Produces:
  1. <tag>_evolution.png : multi-panel curves vs V (activation growth, conduction
     onset, current with both conventions, current concentration).
  2. <tag>_snapshots.png : the network at several voltages, edges drawn with
     thickness/color proportional to |current|, showing the conduction region
     spreading as V rises.
  3. <tag>_Gmatrix_V{V}.png : log10 conductance-matrix heatmap at selected
     voltages (CSV + .npy written by sweep_analysis.sweep()).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sweep_analysis as sa

def plot_evolution(result, tag, outpath):
    rows = result['rows']
    V = np.array([r['V'] for r in rows])
    an = np.array([r['activated_nodes'] for r in rows])
    ae = np.array([r['activated_edges'] for r in rows])
    ce = np.array([r['conducting_edges'] for r in rows])
    I = np.array([r['total_current_A'] for r in rows])
    Icc = np.array([r['total_current_chargeconserving_A'] for r in rows])
    part = np.array([r['participation_ratio'] for r in rows])
    bb = np.array([r['backbone_edges'] for r in rows])
    pV = result['percolation_V']

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Network evolution vs applied voltage — {tag}', fontsize=14, fontweight='bold')

    a = ax[0, 0]
    a.plot(V, an, '-o', ms=3, color='tab:blue', label='activated nodes')
    a.set_xlabel('V (V)'); a.set_ylabel('count')
    a.set_title('Activation grows as V crosses node $V_a$')
    a.legend(); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(V, ae, '-', color='tab:gray', label='activated edges')
    a.plot(V, ce, '-', color='tab:green', label='conducting edges')
    a.set_xlabel('V (V)'); a.set_ylabel('edge count')
    a.set_title('Activated vs conducting edges\n(gap = on but not on a source-drain path)')
    a.legend(); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(V, I * 1e9, '-o', ms=3, color='tab:red', label='solver current')
    a.plot(V, Icc * 1e9, '--', color='tab:orange', label='charge-conserving')
    a.set_xlabel('V (V)'); a.set_ylabel('total current (nA)')
    a.set_title('I-V curve (Kirchhoff)\nsolver vs charge-conserving (see README)')
    a.legend(); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(V, part, '-', color='tab:purple', label='participation ratio')
    a.plot(V, bb, '-', color='tab:orange', label='backbone edges (>=1% of max)')
    a.set_xlabel('V (V)'); a.set_ylabel('effective # edges')
    a.set_title('Current concentration\n(higher = flow spread over more edges)')
    a.legend(); a.grid(alpha=0.3)

    for a in ax.ravel():
        if pV is not None:
            a.axvline(pV, color='k', ls='--', lw=1, alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_snapshots(net, result, tag, outpath, snap_voltages=None, n_snaps=4):
    rows = result['rows']
    ecbv = result['edge_currents_by_V']
    pos = net.positions
    pV = result['percolation_V']

    Vs_with_current = [r['V'] for r in rows if r['conducting_edges'] > 0]
    if not Vs_with_current:
        print(f'[{tag}] no conduction in range; skipping snapshots')
        return None

    if snap_voltages is None:
        lo, hi = Vs_with_current[0], rows[-1]['V']
        targets = np.linspace(lo, hi, n_snaps)
        avail = np.array([r['V'] for r in rows])
        snap_Vs = sorted(set(round(float(avail[np.argmin(np.abs(avail - t))]), 10)
                             for t in targets))
    else:
        snap_Vs = sorted(set(round(float(v), 10) for v in snap_voltages))

    n_panels = len(snap_Vs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    all_max = max((max((abs(c) for c in ecbv.get(round(v, 10), {}).values()), default=0.0)
                   for v in snap_Vs), default=1.0) or 1.0

    for ax, V in zip(axes, snap_Vs):
        Vr = round(V, 10)
        ec = ecbv.get(Vr, {})
        segs_all = [[(pos[i][0], pos[i][1]), (pos[j][0], pos[j][1])]
                    for i, j in net.G.edges()]
        ax.add_collection(LineCollection(segs_all, colors='lightgray',
                                         linewidths=0.2, alpha=0.25, zorder=1))
        if ec:
            segs, widths, vals = [], [], []
            for (i, j), c in ec.items():
                segs.append([(pos[i][0], pos[i][1]), (pos[j][0], pos[j][1])])
                mag = abs(c)
                widths.append(0.3 + 4.0 * mag / all_max)
                vals.append(mag / all_max)
            lc = LineCollection(segs, array=np.array(vals), cmap='plasma',
                                linewidths=widths, zorder=3)
            lc.set_clim(0, 1)
            ax.add_collection(lc)
        activated = {n for n in net.G.nodes() if net.G.nodes[n]['Vth'] <= Vr}
        off = [n for n in net.G.nodes() if n not in activated]
        on = list(activated)
        if off:
            ax.scatter(pos[off, 0], pos[off, 1], s=4, c='lightgray',
                       alpha=0.6, zorder=2)
        if on:
            ax.scatter(pos[on, 0], pos[on, 1], s=22, c='steelblue',
                       edgecolors='navy', linewidths=0.3, alpha=0.85, zorder=4)
        ax.scatter(pos[net.source_nodes, 0], pos[net.source_nodes, 1],
                   s=28, marker='s', c='green', edgecolors='k', lw=0.4, zorder=5)
        ax.scatter(pos[net.drain_nodes, 0], pos[net.drain_nodes, 1],
                   s=28, marker='s', c='red', edgecolors='k', lw=0.4, zorder=5)
        row = next(r for r in rows if r['V'] == Vr)
        ax.set_title(f'V = {Vr:g} V\n{len(activated)} activated nodes, '
                     f'{row["conducting_edges"]} conducting edges, '
                     f'I = {row["total_current_A"]*1e9:.3g} nA', fontsize=10)
        ax.set_xlim(-0.02, net.domain[0] + 0.02)
        ax.set_ylim(-0.02, net.domain[1] + 0.02)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor='green',
               markeredgecolor='k', markersize=7, label='source'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='red',
               markeredgecolor='k', markersize=7, label='drain'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='steelblue',
               markeredgecolor='navy', markersize=7, label='activated node'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='lightgray',
               markeredgecolor='none', markersize=6, label='inactive node'),
    ]
    axes[0].legend(handles=legend_handles, loc='upper left', fontsize=8)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label('|edge current| / max', fontsize=9)
    fig.suptitle(f'Conduction region spreading with voltage — {tag}\n'
                 f'(edge width & color = current magnitude)',
                 fontsize=13, fontweight='bold')
    plt.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_iv_curve(result, tag, outpath, show_chargeconserving=True):
    """Standalone Kirchhoff I-V curve (single panel). Primary curve is the
    solver current (matches optimized_final_system.py); the charge-conserving
    current is shown as a dashed reference if requested."""
    rows = result['rows']
    V = np.array([r['V'] for r in rows])
    I = np.array([r['total_current_A'] for r in rows])
    Icc = np.array([r['total_current_chargeconserving_A'] for r in rows])
    pV = result['percolation_V']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(V, I * 1e9, '-o', ms=4, color='tab:blue', label='Kirchhoff (solver)')
    if show_chargeconserving:
        ax.plot(V, Icc * 1e9, '--', color='tab:orange',
                label='charge-conserving')
    if pV is not None:
        ax.axvline(pV, color='k', ls=':', lw=1, alpha=0.6)
        ax.text(pV, ax.get_ylim()[1] * 0.05, f' percolation V={pV:g}',
                fontsize=9, rotation=90, va='bottom')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (nA)')
    ax.set_title(f'Kirchhoff I-V curve — {tag}')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_iv_curve_multiseed(results_by_seed, tag, outpath):
    """
    I-V curve with seed spread: mean +/- std shaded band over multiple seeds.

    Parameters
    ----------
    results_by_seed : list of sweep() result dicts, one per seed. They must share
        the same voltage grid (same V_start/V_max/V_step), which they do when run
        from the same config.
    Uses the solver current (total_current_A).
    """
    # align on the common voltage grid (use the first result's voltages)
    base_rows = results_by_seed[0]['rows']
    V = np.array([r['V'] for r in base_rows])
    n_seed = len(results_by_seed)

    # stack currents: shape (n_seed, n_voltage)
    I_stack = np.full((n_seed, len(V)), np.nan)
    for k, res in enumerate(results_by_seed):
        rows = res['rows']
        # map voltage -> current for robust alignment
        cur = {round(r['V'], 10): r['total_current_A'] for r in rows}
        for vi, v in enumerate(V):
            I_stack[k, vi] = cur.get(round(float(v), 10), np.nan)

    I_mean = np.nanmean(I_stack, axis=0) * 1e9   # nA
    I_std = np.nanstd(I_stack, axis=0, ddof=1) * 1e9 if n_seed > 1 else np.zeros_like(I_mean)

    pV = results_by_seed[0].get('percolation_V')

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(V, I_mean, '-o', ms=4, color='tab:blue',
            label=f'mean (n={n_seed} seeds)')
    ax.fill_between(V, I_mean - I_std, I_mean + I_std, color='tab:blue',
                    alpha=0.25, label='$\\pm$ std')
    if pV is not None:
        ax.axvline(pV, color='k', ls=':', lw=1, alpha=0.6)
        ax.text(pV, ax.get_ylim()[1] * 0.05, f' percolation V={pV:g}',
                fontsize=9, rotation=90, va='bottom')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (nA)')
    ax.set_title(f'Kirchhoff I-V curve with seed spread — {tag}')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_structure(result, tag, outpath):
    """Connected-component structure of the activated subgraph vs voltage."""
    rows = result['rows']
    V = np.array([r['V'] for r in rows])
    frac = np.array([r.get('largest_cc_fraction', np.nan) for r in rows])
    second = np.array([r.get('second_cc_nodes', np.nan) for r in rows])
    ncomp = np.array([r.get('num_components', np.nan) for r in rows])
    pV = result['percolation_V']

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Component structure — {tag}', fontsize=13, fontweight='bold')

    a = ax[0]
    a.plot(V, frac, '-o', ms=4, color='tab:blue')
    a.set_xlabel('V (V)'); a.set_ylabel('fraction of activated nodes')
    a.set_title('Largest-component fraction\n(percolation order parameter)')
    a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3)

    a = ax[1]
    a.plot(V, second, '-o', ms=4, color='tab:red', label='2nd-largest comp. (nodes)')
    a.set_xlabel('V (V)'); a.set_ylabel('2nd-largest component (nodes)', color='tab:red')
    a.tick_params(axis='y', labelcolor='tab:red')
    a.set_title('Second-largest component & component count\n(2nd-largest peaks near threshold)')
    a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(V, ncomp, '-s', ms=3, color='tab:green', alpha=0.7)
    a2.set_ylabel('number of components', color='tab:green')
    a2.tick_params(axis='y', labelcolor='tab:green')

    for a in ax:
        if pV is not None:
            a.axvline(pV, color='k', ls='--', lw=1, alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_current_distribution(result, tag, outpath):
    """Current-distribution shape vs voltage (how concentrated the flow is)."""
    rows = result['rows']
    V = np.array([r['V'] for r in rows])
    gini = np.array([r.get('current_gini', np.nan) for r in rows])
    top10 = np.array([r.get('current_top10_fraction', np.nan) for r in rows])
    cv = np.array([r.get('current_cv', np.nan) for r in rows])
    m2m = np.array([r.get('current_max_to_mean', np.nan) for r in rows])
    pV = result['percolation_V']

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Current distribution — {tag}', fontsize=13, fontweight='bold')

    a = ax[0]
    a.plot(V, gini, '-o', ms=4, color='tab:purple', label='Gini coefficient')
    a.plot(V, top10, '-s', ms=3, color='tab:orange', label='top-10% current fraction')
    a.set_xlabel('V (V)'); a.set_ylabel('fraction / coefficient')
    a.set_title('Current inequality\n(1 = all in one edge, 0 = even)')
    a.set_ylim(-0.02, 1.02); a.legend(); a.grid(alpha=0.3)

    a = ax[1]
    a.plot(V, cv, '-o', ms=4, color='tab:blue', label='coeff. of variation')
    a.plot(V, m2m, '-s', ms=3, color='tab:red', label='max / mean')
    a.set_xlabel('V (V)'); a.set_ylabel('ratio')
    a.set_title('Current spread & dominance')
    a.legend(); a.grid(alpha=0.3)

    for a in ax:
        if pV is not None:
            a.axvline(pV, color='k', ls='--', lw=1, alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_spectral(result, tag, outpath):
    """Full-system-matrix spectral metrics vs voltage."""
    rows = result['rows']
    V = np.array([r['V'] for r in rows])
    Reff = np.array([r.get('effective_resistance_ohm', np.nan) for r in rows])
    ac = np.array([r.get('algebraic_connectivity', np.nan) for r in rows])
    pV = result['percolation_V']

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Spectral metrics — {tag}', fontsize=13, fontweight='bold')

    a = ax[0]
    mask = np.isfinite(Reff) & (Reff > 0)
    a.semilogy(V[mask], Reff[mask], '-o', ms=4, color='tab:red')
    a.set_xlabel('V (V)'); a.set_ylabel(r'$R_{eff}$ ($\Omega$)')
    a.set_title('Effective source-drain resistance')
    a.grid(alpha=0.3)

    a = ax[1]
    mask2 = np.isfinite(ac)
    a.plot(V[mask2], ac[mask2], '-o', ms=4, color='tab:blue')
    a.set_xlabel('V (V)'); a.set_ylabel('algebraic connectivity')
    a.set_title('Algebraic connectivity (Fiedler value)\n(robustness; dips = near-bottleneck)')
    a.grid(alpha=0.3)

    for a in ax:
        if pV is not None:
            a.axvline(pV, color='k', ls='--', lw=1, alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_Gmatrix(net, voltage, tag, outpath):
    """log10 conductance-matrix heatmap at one voltage (built fresh here)."""
    Vr = round(float(voltage), 10)
    activated = {n for n in net.G.nodes() if net.G.nodes[n]['Vth'] <= Vr}
    if not activated:
        print(f'[{tag}] no active nodes at V={Vr}; skipping G heatmap')
        return None
    Gmat, active_list = sa.conductance_matrix(net, activated)
    N = Gmat.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(np.abs(Gmat) + 1e-20), cmap='plasma',
                   aspect='equal', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_{10}|G_{ij}|$ (S)', rotation=270, labelpad=16)
    src_idx = [k for k, n in enumerate(active_list) if n in set(net.source_nodes)]
    drn_idx = [k for k, n in enumerate(active_list) if n in set(net.drain_nodes)]
    ticks = src_idx + drn_idx
    labels = ['S'] * len(src_idx) + ['D'] * len(drn_idx)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6, color='white')
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6, color='white')
    ax.set_xlabel('matrix index'); ax.set_ylabel('matrix index')
    ax.set_title(f'Conductance matrix at V={Vr:g} V  ({N} active nodes) — {tag}')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return outpath
