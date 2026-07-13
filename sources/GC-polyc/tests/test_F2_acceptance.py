"""F2 capacitance-curve acceptance test — topology + position from final-mesh CSVs.

Panel parameters (CONSTRAINTS §3 F2 / run_F2.py):
  a: ΔE_pzc=0.6V, c_b=0.1mM   b: ΔE_pzc=0.6V, c_b=10mM
  c: ΔE_pzc=0.2V, c_b=0.1mM   d: ΔE_pzc=0.2V, c_b=10mM

Acceptance (CONSTRAINTS §3 F2):
  - n_minima topology consistent with L/λ_D regime.
  - Minimum positions within ±50 mV of expected E_pzc in deep-bistable regime.
  - C_dl > 0 everywhere (physical requirement).
"""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import pytest

OUTPUT = ROOT / 'output'


def _load(panel, L_nm):
    f = OUTPUT / f'F2{panel}_L{L_nm}nm.csv'
    d = np.loadtxt(f, delimiter=',', skiprows=1)
    return d[:, 0], d[:, 2]   # E_M_V, C_dl_Fm2


def _count_minima(cdl):
    n = len(cdl)
    return sum(1 for i in range(1, n - 1)
               if cdl[i] < cdl[i - 1] and cdl[i] < cdl[i + 1])


def _min_positions(em, cdl):
    """Return E_M at all local minima."""
    n = len(cdl)
    return [em[i] for i in range(1, n - 1)
            if cdl[i] < cdl[i - 1] and cdl[i] < cdl[i + 1]]


def test_all_csvs_exist():
    for panel in ['a', 'b', 'c', 'd']:
        for L in [1, 10, 100]:
            f = OUTPUT / f'F2{panel}_L{L}nm.csv'
            assert f.exists(), f"Missing: {f}"


def test_4panels_png_exists():
    assert (OUTPUT / 'F2_4panels.png').exists()


def test_cdl_positive_everywhere():
    """C_dl > 0 for all panels and L values."""
    for panel in ['a', 'b', 'c', 'd']:
        for L in [1, 10, 100]:
            em, cdl = _load(panel, L)
            assert float(np.min(cdl)) > 0, (
                f"F2{panel} L={L}nm: C_dl has non-positive value {np.min(cdl):.3e}")


# ── Panel (a): ΔE=0.6V, c_b=0.1mM, λ_D≈30nm ────────────────────────────
class TestF2PanelA:
    def test_L1nm_single_min(self):
        """L/λD≈0.033 → averaging → 1-min."""
        _, cdl = _load('a', 1)
        assert _count_minima(cdl) == 1, f"F2a L=1nm n_min={_count_minima(cdl)}"

    def test_L10nm_single_min(self):
        """L/λD≈0.33 → still averaging → 1-min."""
        _, cdl = _load('a', 10)
        assert _count_minima(cdl) == 1, f"F2a L=10nm n_min={_count_minima(cdl)}"

    def test_L100nm_two_min(self):
        """L/λD≈3.3, eDE/kT≈23 → bistable → 2-min."""
        _, cdl = _load('a', 100)
        assert _count_minima(cdl) == 2, f"F2a L=100nm n_min={_count_minima(cdl)}"

    def test_L100nm_minima_symmetric_about_zero(self):
        """F2a L=100nm: 2 minima are symmetric about E_M=0 (x=0.5 symmetry).

        L/λD≈3.3 (partial bistability) — minima appear at ~±0.225V rather
        than ±0.3V due to facet cross-talk. No tight position tolerance here;
        topology (n_min=2) and symmetry (|pos1|≈|pos2|) are the checks.
        """
        em, cdl = _load('a', 100)
        positions = _min_positions(em, cdl)
        assert len(positions) == 2, f"Expected 2 minima, got {len(positions)}"
        assert abs(positions[0] + positions[1]) <= 0.05, (
            f"F2a L=100nm minima not symmetric: {positions[0]:.3f}V, {positions[1]:.3f}V")


# ── Panel (b): ΔE=0.6V, c_b=10mM, λ_D≈3nm ──────────────────────────────
class TestF2PanelB:
    def test_L1nm_single_min(self):
        """L/λD≈0.33 → averaging → 1-min."""
        _, cdl = _load('b', 1)
        assert _count_minima(cdl) == 1, f"F2b L=1nm n_min={_count_minima(cdl)}"

    def test_L10nm_two_min(self):
        """L/λD≈3.3, eDE/kT≈23 → bistable → 2-min."""
        _, cdl = _load('b', 10)
        assert _count_minima(cdl) == 2, f"F2b L=10nm n_min={_count_minima(cdl)}"

    def test_L100nm_two_min(self):
        """L/λD≈33 → deep bistable → 2-min."""
        _, cdl = _load('b', 100)
        assert _count_minima(cdl) == 2, f"F2b L=100nm n_min={_count_minima(cdl)}"

    def test_L100nm_minima_at_epzc(self):
        """F2b L=100nm (deep bistable): minima within ±50mV of ±0.3V."""
        em, cdl = _load('b', 100)
        positions = _min_positions(em, cdl)
        assert len(positions) == 2, f"Expected 2 minima, got {len(positions)}"
        target = 0.3
        for pos in positions:
            assert abs(abs(pos) - target) <= 0.05, (
                f"F2b L=100nm min at {pos:.3f}V, >50mV from ±{target}V")


# ── Panel (c): ΔE=0.2V, c_b=0.1mM ──────────────────────────────────────
class TestF2PanelC:
    def test_all_L_single_min(self):
        """ΔE=0.2V, c_b=0.1mM: GC bistability absent → 1-min for all L."""
        for L in [1, 10, 100]:
            _, cdl = _load('c', L)
            n = _count_minima(cdl)
            assert n == 1, f"F2c L={L}nm n_min={n} (expected 1)"


# ── Panel (d): ΔE=0.2V, c_b=10mM ───────────────────────────────────────
class TestF2PanelD:
    def test_all_L_single_min(self):
        """ΔE=0.2V, c_b=10mM: small ΔE → below bistability threshold → 1-min."""
        for L in [1, 10, 100]:
            _, cdl = _load('d', L)
            n = _count_minima(cdl)
            assert n == 1, f"F2d L={L}nm n_min={n} (expected 1)"
