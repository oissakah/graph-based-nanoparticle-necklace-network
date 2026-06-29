# Nanoparticle Network Kirchhoff Solver

Graph-based simulation tools for studying voltage-activated transport, percolation, current redistribution, and spectral connectivity in nanoparticle necklace networks.

The model represents a nanoparticle necklace film as a spatial graph. Nodes represent junctions with activation voltages `V_a`, and edges represent inter-junction necklace segments with distance-dependent resistance. At each applied voltage, nodes with `V_a <= V` become active and the active source-drain subnetwork is solved using Kirchhoff nodal analysis.

## Repository contents

| File | Purpose |
| --- | --- |
| `nanoparticle_network.py` | Core graph model, activation-voltage sampling, electrode detection, and Kirchhoff solver. |
| `optimized_config.yaml` | Production configuration for the default network, electrical model, voltage sweep, and seed list. |
| `sweep_analysis.py` | Per-voltage sweep metrics, conductance-matrix export, edge-current reconstruction, and CSV writers. |
| `spectral_analysis.py` | Full node+edge system-matrix effective resistance and algebraic-connectivity metrics. |
| `visualize_sweep.py` | I-V, evolution, snapshot, conductance-matrix, structure, current-distribution, and spectral plots. |
| `run_sweep_analysis.py` | Main single-seed workflow for tables, I-V curves, snapshots, edge-current CSVs, and G-matrix outputs. |
| `run_multiseed_iv.py` | Full multiseed workflow that runs all seeds and produces a mean ± standard deviation I-V band. |
| `parameter_sweep_cases.py` | Publication parameter-sweep driver for activation-voltage width, mean activation voltage, junction count, and void-fraction cases. |
| `plotting.py` | Plotting helpers used by the parameter-sweep workflow. |
| `publication_figures.py` | For producing publication figures. |

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
scipy
networkx
matplotlib
PyYAML
```

## Quick start: single-seed voltage sweep

Run the default seed listed first in `optimized_config.yaml`:

```bash
python run_sweep_analysis.py
```

Run a specific seed:

```bash
python run_sweep_analysis.py 51
```

Run a specific seed and write to a custom output folder:

```bash
python run_sweep_analysis.py 51 Results_seed51
```

The single-seed workflow produces:

- `sweep_table_seed<seed>_<mode>.csv`
- `iv_curve_seed<seed>_<mode>.csv`
- `iv_curve_seed<seed>_<mode>.png`
- `evolution_seed<seed>_<mode>.png`
- `snapshots_seed<seed>_<mode>.png`
- `edge_currents_seed<seed>_<mode>.csv`
- `edge_currents_snapshots_seed<seed>_<mode>.csv`
- `Gmatrix_seed<seed>_<mode>_V*.csv`
- `Gmatrix_seed<seed>_<mode>_V*.npy`
- `Gmatrix_seed<seed>_<mode>_V*.png`

By default, `run_sweep_analysis.py` exports conductance matrices and network snapshots at 1, 2, 6, and 14 V when those voltages fall within the configured voltage range.

## Multiseed I-V spread

To run all seeds in `optimized_config.yaml` and generate a mean ± standard deviation I-V band:

```bash
python run_multiseed_iv.py
```

To choose an output directory:

```bash
python run_multiseed_iv.py Results_multiseed
```

The multiseed workflow produces one full sweep table per seed and three aggregate files:

- `iv_curve_multiseed_<mode>.png`
- `iv_curve_multiseed_<mode>_byseed.csv`
- `iv_curve_multiseed_<mode>_aggregated.csv`

`run_multiseed_iv.py` computes algebraic connectivity for every seed. This uses dense eigendecomposition and can be memory intensive. If memory errors occur, reduce `N_WORKERS` inside `run_multiseed_iv.py`.

## Configuration

The main settings are stored in `optimized_config.yaml`.

```yaml
network:
  n_junctions: 500
  connection_radius: 0.15
  domain_size: [1.0, 1.0]
  left_thresh: 0.15
  right_thresh: 0.85

threshold_distribution:
  type: normal
  mean: 4.0
  std: 1.0
  min: 0.0
  max: 20.0

electrical:
  edge_k: 2.0e10
  resistance_model: node
  node_r_scale: 5.0e8
  r_floor: 0.5

voltage_sweep:
  V_start: 0.0
  V_max: 16.0
  V_step: 0.5

seeds: [41, 51, 61, 71, 81]
```

### Activation-voltage sampling

For the normal activation-voltage distribution, the code uses rejection sampling to produce a true truncated normal distribution between `min` and `max`. This avoids the artificial pile-up at the lower bound that occurs when out-of-range values are simply clipped. The activation rule remains:

```text
node is active when V_a <= V
```

### Node-resistance model

When `resistance_model: node`, each internal node is split into an input and output terminal connected by a node resistance:

```text
R_node = node_r_scale * max(V_a, r_floor)
```

The `r_floor` parameter prevents very low activation-voltage nodes from producing nearly zero node resistance. It affects node resistance only; it does not change the activation threshold.

## Main quantities in the sweep table

`sweep_analysis.py` writes one row per voltage step. Important columns include:

| Column | Meaning |
| --- | --- |
| `V` | Applied voltage. |
| `activated_nodes` | Nodes satisfying `V_a <= V`. |
| `activated_edges` | Edges whose endpoints are both activated. |
| `conducting_nodes` | Nodes in the active source-drain conducting set returned by the Kirchhoff solve. |
| `conducting_edges` | Edges carrying nonzero current. |
| `source_drain_connected` | Boolean indicator for an active source-drain bridge. |
| `total_current_A` | Solver current from the Kirchhoff nodal analysis. |
| `total_current_chargeconserving_A` | Current reconstructed from edge currents. Useful for checking conservation. |
| `conductance_S` | `total_current_A / V`. |
| `backbone_edges` | Number of current-carrying edges above 1% of the maximum edge current at that voltage. |
| `largest_cc_fraction` | Fraction of activated nodes in the largest activated component. |
| `current_gini` | Inequality of current distribution across conducting edges. |
| `current_top10_fraction` | Fraction of current carried by the top 10% of conducting edges. |
| `effective_resistance_ohm` | Full-system source-drain effective resistance. |
| `algebraic_connectivity` | Fiedler value of the full active node+edge system matrix. |
| `spectral_gap_ratio` | Algebraic connectivity normalized by the largest eigenvalue. |
| `publication_figures` | Generates published figures. |

## Conductance matrix outputs

At selected voltages, `sweep_analysis.py` exports the active edge-resistance Laplacian:

- dense matrix: `Gmatrix_<tag>_V*.npy`
- long-form CSV: `Gmatrix_<tag>_V*.csv`
- node-index sidecar: `Gmatrix_<tag>_V*_nodeindex.csv`
- heatmap: `Gmatrix_<tag>_V*.png`

The G-matrix heatmap is built from edge resistances only. Node resistors are handled in the Kirchhoff solve and spectral metrics but are not included in the off-diagonal edge-only heatmap.

## Publication figure workflow

The parameter-sweep workflow is handled by `parameter_sweep_cases.py` and the publication plotting scripts.

Typical plotting commands include:

```bash
python publication_figures.py all_cases_VT_zeta.csv --dist case1_distribution_data.csv case2a_distribution_data.csv case2b_distribution_data.csv --iv case1_VT_zeta_iv_data.csv case2a_VT_zeta_iv_data.csv --iv-seed 41 --case4-iv case4_VT_zeta_iv_data.csv --caseR-agg caseR_VT_zeta_aggregated.csv --caseR-iv caseR_iv_random_voids_data.csv --sweep-table sweep_table_seed41_node.csv --outdir plots

python publication_figures.py all_cases_VT_zeta.csv --dist case1_distribution_data_with_dummy.csv case2a_distribution_data.csv case2b_distribution_data.csv --iv case1_VT_zeta_iv_data_with_dummy.csv case2a_VT_zeta_iv_data.csv --iv-seed 41 --case4-iv case4_VT_zeta_iv_data.csv --caseR-agg caseR_VT_zeta_aggregated.csv --caseR-iv caseR_iv_random_voids_data.csv --sweep-table sweep_table_seed41_node.csv --outdir plots 

python publication_figures.py --evolution evolution_caseR_N500_mean6_sigma3_seed41_fv0.00.csv evolution_caseR_N500_mean6_sigma3_seed41_fv0.10.csv evolution_caseR_N500_mean6_sigma3_seed41_fv0.15.csv evolution_caseR_N500_mean6_sigma3_seed41_fv0.20.csv --evo-group caseR --outdir plots_connectactivate 

python publication_figures.py --evolution evolution_case1_N500_mean8_sigma1_seed41.csv evolution_case1_N500_mean8_sigma3_seed41.csv evolution_case1_N500_mean8_sigma5_seed41.csv evolution_case1_N500_mean8_sigma7_seed41.csv --evo-group case1 --outdir plots_connectactivate

python publication_figures.py --evolution evolution_case2a_N500_mean4_sigma1_seed41.csv evolution_case2a_N500_mean6_sigma1_seed41.csv evolution_case2a_N500_mean8_sigma1_seed41.csv evolution_case2a_N500_mean10_sigma1_seed41.csv --evo-group case2a --outdir plots_connectactivate 

python publication_figures.py --evolution evolution_case4_N200_mean6_sigma3_seed41.csv evolution_case4_N400_mean6_sigma3_seed41.csv evolution_case4_N600_mean6_sigma3_seed41.csv evolution_case4_N800_mean6_sigma3_seed41.csv --evo-group case4 --outdir plots_connectactivate

```


## Notes and limitations

- The publication workflow uses the Kirchhoff solver. The legacy path-enumeration/BFS utilities in `nanoparticle_network.py` are retained for comparison, but the per-voltage analysis does not use simple-path enumeration.
- Algebraic connectivity is informative for fragmented or voided networks but can be nearly flat in dense baseline networks.
- The multiseed shaded band uses five seeds by default. Treat ± standard deviation as an indication of seed-to-seed variability, not as a formal confidence interval.
- Generated outputs are excluded from version control through `.gitignore`.

## Recommended citation language

If this repository is used in a manuscript, describe it as a graph-based Kirchhoff nodal-analysis framework for activation-voltage-gated nanoparticle necklace networks. The transport exponent is reported as `zeta` (`\zeta`) throughout.
