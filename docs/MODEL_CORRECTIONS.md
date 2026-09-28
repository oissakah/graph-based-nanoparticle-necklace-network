# Model consistency corrections

1. **One transport engine.** Voltage gating is followed by Kirchhoff nodal analysis. The device current and every edge current come from the same potential solution.
2. **Symmetric junction resistance.** Half of each endpoint junction resistance is assigned to an active connection. This replaces the orientation-dependent split-node construction with a symmetric passive circuit Laplacian.
3. **Activation/resistance decoupling.** `Va_i` controls activation only. Junction resistance is fixed independently at `node_resistance_ohm`.
4. **No unused tunneling law.** The unsupported tunneling flag and legacy path-current calculation are absent from the active framework.
5. **Bounded activation sampling.** Bounded normal activation distributions use rejection sampling rather than clipping, avoiding artificial probability mass at the limits.
6. **Canonical current reconstruction.** Source current, drain current, device current, and edge currents are calculated from the same Kirchhoff solution. Their imbalance is retained as a numerical diagnostic.
7. **Fixed threshold convention.** `V_T` is defined as the first sampled source–drain percolation voltage, so `V_T = V_perc`. It is held fixed when fitting `A` and `zeta`.
8. **Nonlinear fit window.** The power-law fit uses positive-current points strictly above `V_T` and ends at 90% node activation, or at the available fit-window limit if saturation is not reached.
9. **N-density definition.** Cases 3 and 4 keep the domain and connection radius fixed (`r_c = 0.15`) as `N` varies. The former `N^(-1/2)` radius scaling is not used.
10. **Void fraction.** Case R stores requested and achieved void fractions. Achieved void area is based on the geometric union, so overlaps are counted once.
11. **Spectral consistency.** Spectral diagnostics use the same conductance-weighted active-circuit Laplacian as the electrical solver. Raw `lambda_2` is reported in siemens `[S]`.
