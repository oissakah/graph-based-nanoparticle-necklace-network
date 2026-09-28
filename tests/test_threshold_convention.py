"""Tests for the fixed ``V_T = V_perc`` power-law convention."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "parameter_analysis"
if str(PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(PARAM_DIR))

import parameter_sweep_cases as psc


def test_threshold_is_first_positive_voltage_and_is_not_refitted():
    voltage = np.arange(0.0, 9.0, 1.0)
    current = np.zeros_like(voltage)
    current[2:] = 3.0e-9 * (voltage[2:] - 2.0) ** 2
    current[2] = 1.0e-15

    fit = psc.fit_power_law(voltage, current, V_T=2.0, V_stop=8.0)

    assert fit["success"]
    assert fit["V_T"] == 2.0
    assert np.isclose(fit["zeta"], 2.0, rtol=1e-12)


def test_failed_exponent_fit_still_retains_percolation_threshold():
    voltage = np.array([0.0, 1.0, 2.0, 3.0])
    current = np.array([0.0, 1.0e-12, 2.0e-12, 3.0e-12])

    fit = psc.fit_power_law(voltage, current, V_T=1.0, V_stop=3.0)

    assert not fit["success"]
    assert fit["V_T"] == 1.0
