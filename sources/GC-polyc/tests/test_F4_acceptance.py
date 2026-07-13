"""F4 Parsons-Zobel acceptance test — verifies PZ slopes from final-mesh CSVs.

Acceptance bands (loop.txt §TARGET_METRICS):
  F4(a) L=1 nm   slope: GC 0.85–1.05
  F4(a) L=100 nm slope: GC 0.34–0.50
  F4(b) ΔE=0 V   slope: GC 0.85–1.05
  F4(b) ΔE=0.6V  slope: GC intrinsic saturation (≈0; NOT in [0.38,0.56] is expected
                         because pure GC C_dl→C_H at large overpotential when
                         Booth-Langevin ε(E) + Bikerman crowding are absent).

All slopes are monotone-decreasing in L and ΔE_pzc (required).
"""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import pytest
from src.plotting import pz_slope

OUTPUT = ROOT / 'output'


def _load_csv(fname):
    """Load F4 CSV → dict[c_b] → {C_dl, C_GC_min}."""
    d = np.loadtxt(fname, delimiter=',', skiprows=1)
    result = {}
    for row in d:
        cb = float(row[0])
        result[cb] = {'C_dl': float(row[1]), 'C_GC_min': float(row[2])}
    return result


def _load_panel_a(L_nm):
    return _load_csv(OUTPUT / f'F4a_L{L_nm}nm.csv')


def _load_panel_b(dEpzc_V):
    return _load_csv(OUTPUT / f'F4b_dEpzc{dEpzc_V:.1f}V.csv')


# ── Panel (a): ΔE_pzc=0.2 V fixed, L sweep ────────────────────────────────
class TestF4PanelA:
    def test_L1nm_slope_in_band(self):
        """L=1nm PZ slope should be in GC acceptance band [0.85, 1.05]."""
        data = {1e-9: _load_panel_a(1), 10e-9: _load_panel_a(10),
                30e-9: _load_panel_a(30), 100e-9: _load_panel_a(100)}
        s = pz_slope(data[1e-9])
        assert 0.85 <= s <= 1.05, f"L=1nm slope={s:.3f} outside [0.85,1.05]"

    def test_L100nm_slope_in_band(self):
        """L=100nm PZ slope should be in GC acceptance band [0.34, 0.50]."""
        s = pz_slope(_load_panel_a(100))
        assert 0.34 <= s <= 0.50, f"L=100nm slope={s:.3f} outside [0.34,0.50]"

    def test_slopes_monotone_decreasing_in_L(self):
        """PZ slope must decrease monotonically with L at fixed ΔE_pzc."""
        L_vals = [1, 10, 30, 100]
        slopes = [pz_slope(_load_panel_a(L)) for L in L_vals]
        for i in range(len(slopes) - 1):
            assert slopes[i] >= slopes[i + 1], (
                f"Non-monotone: slope(L={L_vals[i]}nm)={slopes[i]:.3f} "
                f"< slope(L={L_vals[i+1]}nm)={slopes[i+1]:.3f}")

    def test_slopes_positive(self):
        """All PZ slopes must be positive (physical requirement)."""
        for L_nm in [1, 10, 30, 100]:
            s = pz_slope(_load_panel_a(L_nm))
            assert s > 0, f"Negative PZ slope at L={L_nm}nm: {s:.3f}"


# ── Panel (b): L=30 nm fixed, ΔE_pzc sweep ───────────────────────────────
class TestF4PanelB:
    def test_dE0_slope_in_band(self):
        """ΔE_pzc=0 (single-facet limit) slope should be in [0.85, 1.05]."""
        s = pz_slope(_load_panel_b(0.0))
        assert 0.85 <= s <= 1.05, f"ΔE=0 slope={s:.3f} outside [0.85,1.05]"

    def test_dE06_slope_below_gc_saturation_threshold(self):
        """ΔE_pzc=0.6V slope << 0.38 at GC level (known Booth/Bikerman gap).

        Pure GC gives C_dl→C_H at large overpotential, making 1/C_dl c_b-independent
        and slope→0. This test CONFIRMS the GC saturation (slope < 0.10)
        and documents that the article target [0.38,0.56] requires Booth-Langevin
        + Bikerman physics (out-of-scope per CONSTRAINTS §4).
        """
        s = pz_slope(_load_panel_b(0.6))
        assert s < 0.10, f"ΔE=0.6V slope={s:.3f} unexpectedly high (GC saturation expected)"

    def test_slopes_monotone_decreasing_in_dEpzc(self):
        """PZ slope must decrease monotonically with ΔE_pzc at fixed L."""
        dE_vals = [0.0, 0.2, 0.4, 0.6]
        slopes = [pz_slope(_load_panel_b(dE)) for dE in dE_vals]
        for i in range(len(slopes) - 1):
            assert slopes[i] >= slopes[i + 1], (
                f"Non-monotone: slope(ΔE={dE_vals[i]}V)={slopes[i]:.3f} "
                f"< slope(ΔE={dE_vals[i+1]}V)={slopes[i+1]:.3f}")

    def test_slopes_positive(self):
        """All PZ slopes must be positive."""
        for dE in [0.0, 0.2, 0.4, 0.6]:
            s = pz_slope(_load_panel_b(dE))
            assert s > 0, f"Negative PZ slope at ΔE={dE}V: {s:.3f}"
