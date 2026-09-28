# Parameter analysis

`parameter_sweep_cases.py` runs the activation-disorder, mean-activation,
network-density, and random-void studies using the canonical Kirchhoff solver.

```bash
python parameter_analysis/parameter_sweep_cases.py
```

Cases 3 and 4 keep the domain and connection radius fixed at `r_c = 0.15`, so
changing `N` changes network density. The threshold convention is
`V_T = V_perc`, the first sampled voltage with a spanning active component and
nonzero device current. The fit holds this threshold fixed and estimates only
`A` and `zeta` before 90% activation saturation.

Run `python parameter_analysis/validate_N_density_fix.py` for a quick check of
the density-study definition.
