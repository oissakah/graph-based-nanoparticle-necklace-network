# Nanoparticle Necklace Network Simulator

Python code for simulating transport through nanoparticle-necklace networks using graph-based Kirchhoff analysis. The model represents junctions as graph nodes, necklace connections as graph edges, and voltage-dependent transport as progressive activation of nodes with activation voltage $V_a$.

The repository supports two workflows:

1. **Full parameter sweeps** for publication figures: activation-voltage width, mean activation voltage, junction count, and random void fraction.
2. **Single-network voltage-sweep diagnostics**: per-voltage connectivity metrics, edge currents, conductance-matrix heatmaps, and network snapshots.

## Main files

| File | Purpose |
|---|---|
| `nanoparticle_network.py` | Core `NanoparticleNetwork` graph model and Kirchhoff I--V solver. |
| `parameter_sweep_cases.py` | Runs Cases 1--4 and Case R parameter sweeps and exports CSVs/figures. |
| `plotting.py` | Plotting helpers used by `parameter_sweep_cases.py`. |
| `sweep_analysis.py` | Per-voltage network evolution metrics, edge-current reconstruction, and conductance-matrix export. |
| `spectral_analysis.py` | Effective resistance, algebraic connectivity, and spectral-gap metrics from the full node/edge system matrix. |
| `visualize_sweep.py` | Diagnostic plots for single-network sweeps. |
| `run_sweep_analysis.py` | Runs the single-network diagnostic workflow from `optimized_config.yaml`. |
| `plot_all_cases.py` | Regenerates publication-style summary plots from exported sweep CSVs. |
| `plot_connectivity_activation.py` | Overlays activated-node and algebraic-connectivity curves from evolution CSVs. |
| `optimized_config.yaml` | Default production configuration and seed list. |
| `What I'm doing.txt` | Working notes with commands used to generate manuscript figures. |

## Requirements

Tested with Python 3.10+.

Install dependencies with:

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

## Quick checks before running large sweeps

Compile all scripts:

```bash
python -m py_compile nanoparticle_network.py sweep_analysis.py spectral_analysis.py visualize_sweep.py run_sweep_analysis.py plot_connectivity_activation.py plot_all_cases.py parameter_sweep_cases.py plotting.py
```

Run a single-network diagnostic using the first seed in `optimized_config.yaml`:

```bash
python run_sweep_analysis.py
```

Run a specific seed and output folder:

```bash
python run_sweep_analysis.py 51 Results_seed51
```

This writes files such as:

- `sweep_table_seed41_node.csv`
- `iv_curve_seed41_node.csv`
- `iv_curve_seed41_node.png`
- `edge_currents_seed41_node.csv`
- `edge_currents_snapshots_seed41_node.csv`
- `evolution_seed41_node.png`
- `snapshots_seed41_node.png`
- `Gmatrix_seed41_node_V*.csv/.npy/.png`

## Full parameter sweep

Run all sweep cases:

```bash
python parameter_sweep_cases.py
```

The script writes timestamped folders under `IV_results/`:

```text
IV_results/
  parameter_sweep_cases_<timestamp>/
  case_R_random_voids_<timestamp>/
```

### Sweep definitions

| Case | Parameters |
|---|---|
| Case 1 | Vary activation-voltage width, $\sigma = 1, 3, 5, 7$ V, at fixed $N=500$ and $\langle V_a\rangle=8$ V. |
| Case 2A | Vary mean activation voltage, $\langle V_a\rangle = 4, 6, 8, 10$ V, at fixed $N=500$ and $\sigma=1$ V. |
| Case 2B | Vary mean activation voltage, $\langle V_a\rangle = 4, 6, 8, 10$ V, at fixed $N=500$ and $\sigma=5$ V. |
| Case 3 | 2D sweep over junction count $N=200,400,600,800$ and mean activation voltage $\langle V_a\rangle = 4,6,8,10$ V, at fixed $\sigma=3$ V. |
| Case 4 | Vary junction count $N=200,400,600,800$ at fixed $\langle V_a\rangle=6$ V and $\sigma=3$ V. |
| Case R | Random void-fraction sweep, $f_v = 0.00,0.05,0.10,0.15,0.20,0.25$, at fixed $N=500$, $\langle V_a\rangle=6$ V, and $\sigma=3$ V. |

All cases use the seeds listed in `optimized_config.yaml` unless that file is absent, in which case the code falls back to `[41, 51, 61, 71, 81]`.

## Publication figure commands

After running `parameter_sweep_cases.py`, copy or run commands from the generated output folders as needed. The following command matches the currently uploaded `plot_all_cases.py` interface:

```bash
python plot_all_cases.py all_cases_VT_zeta.csv \
  --dist case1_distribution_data.csv case2a_distribution_data.csv case2b_distribution_data.csv \
  --iv case1_VT_zeta_iv_data.csv case2a_VT_zeta_iv_data.csv \
  --iv-seed 41 \
  --case4-iv case4_VT_zeta_iv_data.csv \
  --caseR-agg caseR_VT_zeta_aggregated.csv \
  --caseR-iv caseR_iv_random_voids_data.csv \
  --sweep-table sweep_table_seed41_node.csv \
  --outdir plots
```

Optional flags supported by `plot_all_cases.py` include:

```bash
--only case_4
--sampled case1_sampled_va.csv
--sweep-iv-multi iv_curve_spread_allseeds_node_byseed.csv
--iv-aggregated iv_curve_spread_allseeds_node_aggregated.csv
--vmax 12
```

The older commands in `What I'm doing.txt` that call `plot_all_cases_pub.py` or `plot_all_cases_dummy_dashed.py` require those separate scripts. They are not part of the current uploaded code bundle unless you add them to the repository.

## Connectivity and activation overlays

Example commands using exported evolution CSVs:

```bash
python plot_connectivity_activation.py \
  evolution_case1_N500_mean8_sigma1_seed41.csv \
  evolution_case1_N500_mean8_sigma3_seed41.csv \
  evolution_case1_N500_mean8_sigma5_seed41.csv \
  evolution_case1_N500_mean8_sigma7_seed41.csv \
  --group case1 --outdir plots_connectactivate --vmax 12
```

```bash
python plot_connectivity_activation.py \
  evolution_case2a_N500_mean4_sigma1_seed41.csv \
  evolution_case2a_N500_mean6_sigma1_seed41.csv \
  evolution_case2a_N500_mean8_sigma1_seed41.csv \
  evolution_case2a_N500_mean10_sigma1_seed41.csv \
  --group case2a --outdir plots_connectactivate --vmax 12
```

```bash
python plot_connectivity_activation.py \
  evolution_case4_N200_mean6_sigma3_seed41.csv \
  evolution_case4_N400_mean6_sigma3_seed41.csv \
  evolution_case4_N600_mean6_sigma3_seed41.csv \
  evolution_case4_N800_mean6_sigma3_seed41.csv \
  --group case4 --outdir plots_connectactivate --vmax 12
```

```bash
python plot_connectivity_activation.py \
  evolution_caseR_N500_mean6_sigma3_seed41_fv0.00.csv \
  evolution_caseR_N500_mean6_sigma3_seed41_fv0.10.csv \
  evolution_caseR_N500_mean6_sigma3_seed41_fv0.15.csv \
  evolution_caseR_N500_mean6_sigma3_seed41_fv0.20.csv \
  --group caseR --outdir plots_connectactivate --vmax 12
```

## Model notes

- Nodes are junctions in the nanoparticle-necklace network.
- Edges are necklace connections between junctions.
- Activation voltages are sampled from distributions and stored internally under the legacy node key `Vth`; manuscript text and plot labels use activation voltage $V_a$.
- Edge resistance is proportional to inter-node distance: `R_edge = edge_k * distance`.
- Node resistance can be enabled with `resistance_model: node`, using `R_node = node_r_scale * max(V_a, r_floor)`.
- The Kirchhoff solver uses the active subgraph at each applied voltage and solves for source/drain currents.
- No simple-path enumeration is used in the current analysis workflow.

## Important caveats

1. **Runtime:** Full parameter sweeps can be slow, especially for node-resistance mode and algebraic-connectivity calculations.
2. **Current convention:** In node-resistance mode, the evolution table reports both `total_current_A` and `total_current_chargeconserving_A`. The fitted $V_T$ and $\zeta$ are based on the exported solver current.
3. **Case R spectral metrics:** For speed, some Case R workers may skip algebraic connectivity. When this happens, `plot_connectivity_activation.py` skips missing/NaN algebraic-connectivity curves instead of failing.
4. **Generated outputs:** Do not commit large `IV_results/`, `Results/`, or plot-output folders unless intentionally archiving a result set.

## Suggested repository layout

```text
nanoparticle-necklace-network/
  README.md
  requirements.txt
  .gitignore
  optimized_config.yaml
  nanoparticle_network.py
  parameter_sweep_cases.py
  plotting.py
  sweep_analysis.py
  spectral_analysis.py
  visualize_sweep.py
  run_sweep_analysis.py
  plot_all_cases.py
  plot_connectivity_activation.py
  What I'm doing.txt
```
