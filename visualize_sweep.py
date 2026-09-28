"""
visualize_sweep.py — figures for the voltage-sweep analysis (production config).

Produces:
  1. <tag>_evolution.png : multi-panel curves vs V (activation growth, conduction
     onset, canonical Kirchhoff current, current concentration).
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
from pathlib import Path
import sweep_analysis as sa


def plot_evolution(result, tag, outpath):
    rows = result['rows']
    V = np.array([r['V'] for r in rows])
    an = np.array([r['activated_nodes'] for r in rows])
    ae = np.array([r['activated_edges'] for r in rows])
    ce = np.array([r['conducting_edges'] for r in rows])
    I = np.array([r['total_current_A'] for r in rows])
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
    a.plot(V, I * 1e9, '-o', ms=3, color='tab:red', label='Kirchhoff current')
    a.set_xlabel('V (V)'); a.set_ylabel('total current (nA)')
    a.set_title('Macroscopic I-V response')
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
        activated = net.activated_nodes(Vr)
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


def plot_iv_curve(result, tag, outpath):
    """Standalone canonical Kirchhoff I-V curve."""
    rows = result['rows']
    V = np.array([r['V'] for r in rows])
    I = np.array([r['total_current_A'] for r in rows])
    pV = result['percolation_V']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(V, I * 1e9, '-o', ms=4, color='tab:blue', label='Kirchhoff current')
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


def plot_percolation_current_outputs(net, result, tag, outdir):
    """Save the |I_ij| distribution, edge table, and current map at V_perc."""
    percolation_voltage = result.get('percolation_V')
    if percolation_voltage is None:
        print(f'[{tag}] no percolation in sweep; skipping percolation-current plots')
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    voltage = round(float(percolation_voltage), 10)
    edge_currents = result['edge_currents_by_V'].get(voltage, {})
    pathway_count = int(result.get('percolation_pathways', 0))
    pathway_word = 'pathway' if pathway_count == 1 else 'pathways'
    edges = [edge for edge, value in edge_currents.items() if abs(value) > 0]
    magnitudes = np.asarray(
        [abs(edge_currents[edge]) for edge in edges], dtype=float)
    if magnitudes.size == 0:
        return []

    stem = f'{tag}_Vperc{voltage:g}'
    csv_path = outdir / f'edge_currents_{stem}.csv'
    sa.write_edge_currents_csv(net, result, str(csv_path), voltages=[voltage])

    # CSV sidecars preserve the full topology and node roles so the publication
    # script can rebuild this snapshot with its common figure style.
    active = set(net.activated_nodes(voltage))
    node_rows = []
    for node in net.G.nodes():
        node_rows.append({
            'node': int(node), 'x': float(net.positions[node][0]),
            'y': float(net.positions[node][1]),
            'Va_V': float(net.G.nodes[node]['Va']),
            'active_at_Vperc': int(node in active),
            'is_source': int(node in net.source_nodes),
            'is_drain': int(node in net.drain_nodes),
        })
    import pandas as pd
    pd.DataFrame(node_rows).to_csv(outdir / f'network_nodes_{stem}.csv', index=False)
    edge_rows = []
    for i, j in net.G.edges():
        edge_rows.append({
            'node_i': int(i), 'node_j': int(j),
            'x_i': float(net.positions[i][0]), 'y_i': float(net.positions[i][1]),
            'x_j': float(net.positions[j][0]), 'y_j': float(net.positions[j][1]),
        })
    pd.DataFrame(edge_rows).to_csv(outdir / f'network_edges_{stem}.csv', index=False)

    from matplotlib.colors import LogNorm
    pos = net.positions
    edge_values = np.asarray([abs(edge_currents[edge]) * 1e9 for edge in edges])
    segments = [[tuple(pos[i]), tuple(pos[j])] for i, j in edges]
    vmin, vmax = edge_values.min(), edge_values.max()
    if vmax > vmin:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        scaled = ((np.log10(edge_values) - np.log10(vmin)) /
                  (np.log10(vmax) - np.log10(vmin)))
    else:
        norm = plt.Normalize(vmin=max(vmin * 0.9, 1e-30), vmax=vmax * 1.1)
        scaled = np.ones_like(edge_values)
    all_segments = [[tuple(pos[i]), tuple(pos[j])] for i, j in net.G.edges()]
    inactive = [node for node in net.G.nodes() if node not in active]
    active_list = sorted(active)

    values_na = magnitudes * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5),
                             gridspec_kw={'width_ratios': [1.0, 1.12]})
    positive_min, positive_max = values_na.min(), values_na.max()
    if positive_max > positive_min:
        bins = np.geomspace(positive_min, positive_max, 28)
        counts, _, _ = axes[0].hist(
            values_na, bins=bins, color='#3567A8', edgecolor='white')
        axes[0].set_xscale('log')
    else:
        counts, _, _ = axes[0].hist(
            values_na, bins=1, color='#3567A8', edgecolor='white')
    if int(round(float(counts.sum()))) != len(edges):
        raise RuntimeError('Histogram and snapshot current-edge counts differ.')
    axes[0].set_xlabel(r'Edge-current magnitude, $|I_{ij}|$ [nA]')
    axes[0].set_ylabel('Current-carrying edge count [-]')
    axes[0].grid(False)
    axes[0].text(
        0.04, 0.95,
        rf'$V_{{perc}}$ = {voltage:g} V'
        '\n' + rf'$N_{{path}}$ = {pathway_count}'
        '\n' + rf'$N_{{active}}$ = {len(active)}',
        transform=axes[0].transAxes, va='top', ha='left')

    axes[1].add_collection(LineCollection(
        all_segments, colors='0.85', linewidths=0.25, alpha=0.35, zorder=1))
    combined_collection = LineCollection(
        segments, array=edge_values, cmap='plasma', norm=norm,
        linewidths=0.5 + 4.0 * scaled, zorder=3)
    axes[1].add_collection(combined_collection)
    if inactive:
        axes[1].scatter(pos[inactive, 0], pos[inactive, 1], s=4,
                        c='0.8', zorder=2)
    axes[1].scatter(pos[active_list, 0], pos[active_list, 1], s=16,
                    c='steelblue', edgecolors='navy', linewidths=0.2, zorder=4)
    if net.source_nodes:
        src = np.asarray(sorted(net.source_nodes))
        axes[1].scatter(pos[src, 0], pos[src, 1], s=30, marker='s', c='green',
                        edgecolors='k', linewidths=0.3, zorder=5)
    if net.drain_nodes:
        drn = np.asarray(sorted(net.drain_nodes))
        axes[1].scatter(pos[drn, 0], pos[drn, 1], s=30, marker='s', c='red',
                        edgecolors='k', linewidths=0.3, zorder=5)
    combined_cbar = fig.colorbar(
        combined_collection, ax=axes[1], fraction=0.045, pad=0.03)
    combined_cbar.set_label(r'$|I_{ij}|$ [nA]')
    axes[1].set_xlim(-0.02, net.domain[0] + 0.02)
    axes[1].set_ylim(-0.02, net.domain[1] + 0.02)
    axes[1].set_aspect('equal'); axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(
        rf'$|I_{{ij}}|$ at $V_{{perc}}={voltage:g}$ V — {tag}'
        f'\n{len(active)} active nodes; {len(edges)} current-carrying edges',
        fontweight='bold')
    fig.tight_layout()
    distribution_path = outdir / f'current_distribution_{stem}.png'
    fig.savefig(distribution_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.add_collection(LineCollection(all_segments, colors='0.85', linewidths=0.25,
                                     alpha=0.35, zorder=1))
    collection = LineCollection(segments, array=edge_values, cmap='plasma', norm=norm,
                                linewidths=0.5 + 4.0 * scaled, zorder=3)
    ax.add_collection(collection)
    if inactive:
        ax.scatter(pos[inactive, 0], pos[inactive, 1], s=4, c='0.8', zorder=2)
    ax.scatter(pos[active_list, 0], pos[active_list, 1], s=16, c='steelblue',
               edgecolors='navy', linewidths=0.2, zorder=4)
    if net.source_nodes:
        src = np.asarray(sorted(net.source_nodes))
        ax.scatter(pos[src, 0], pos[src, 1], s=30, marker='s', c='green',
                   edgecolors='k', linewidths=0.3, zorder=5)
    if net.drain_nodes:
        drn = np.asarray(sorted(net.drain_nodes))
        ax.scatter(pos[drn, 0], pos[drn, 1], s=30, marker='s', c='red',
                   edgecolors='k', linewidths=0.3, zorder=5)
    colorbar = fig.colorbar(collection, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label(r'Edge-current magnitude, $|I_{ij}|$ (nA)')
    ax.set_xlim(-0.02, net.domain[0] + 0.02)
    ax.set_ylim(-0.02, net.domain[1] + 0.02)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf'$|I_{{ij}}|$ at $V_{{perc}}={voltage:g}$ V — {tag}'
                 f'\n{len(active)} active nodes; {len(edge_currents)} current-carrying edges; '
                 f'{pathway_count} independent S-D {pathway_word}',
                 fontweight='bold')
    fig.tight_layout()
    snapshot_path = outdir / f'edge_current_snapshot_{stem}.png'
    fig.savefig(snapshot_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return [distribution_path, snapshot_path, csv_path]


def plot_Gmatrix(net, voltage, tag, outpath):
    """log10 conductance-matrix heatmap at one voltage (built fresh here)."""
    Vr = round(float(voltage), 10)
    activated = net.activated_nodes(Vr)
    if not activated:
        print(f'[{tag}] no active nodes at V={Vr}; skipping G heatmap')
        return None
    Gmat, active_list = sa.conductance_matrix(net, activated)
    if Gmat.size == 0:
        print(f'[{tag}] no source-drain bridge at V={Vr}; skipping circuit heatmap')
        return None
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
