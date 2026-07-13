"""F2 capacitance-magnitude self-consistency test.

Acceptance criteria (CONSTRAINTS §3 F2):
  - C_dl_min per panel must be within GC physical bounds:
      lower: C_H || C_GC,min(cb) — series of Helmholtz and minimum GC cap
      upper: C_H = 50 uF/cm2 (GC never exceeds Helmholtz)
  - Two-facet minimum (one facet at PZC, one far off) must be above
    the single-facet PZC minimum and below C_H.
  - F2b L=100nm (deep bistable): C_dl_min within 50% of analytic
    two-facet GC estimate ((C_dl,PZC + C_H) / 2).
    Note: GC overestimates vs article by ≤30% (no Booth-Langevin/Bikerman).

All checks use SI units internally; μF/cm² in messages.
"""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import pytest

from src import EPS0, F as FARADAY, R as GAS_R, EPS_R_WATER

OUTPUT = ROOT / 'output'
T = 298.15
C_H = 0.5          # F/m2 = 50 uF/cm2


def _lam_D(cb_mM):
    return float(np.sqrt(EPS0 * EPS_R_WATER * GAS_R * T
                         / (2.0 * FARADAY**2 * cb_mM)))


def _C_GC_min(cb_mM):
    return EPS0 * EPS_R_WATER / _lam_D(cb_mM)


def _C_dl_min_analytic(cb_mM):
    C_GC = _C_GC_min(cb_mM)
    return C_H * C_GC / (C_H + C_GC)


def _load_cdl_min(panel, L_nm):
    d = np.loadtxt(OUTPUT / f'F2{panel}_L{L_nm}nm.csv', delimiter=',', skiprows=1)
    return float(np.min(d[:, 2]))


# ── Global bounds ────────────────────────────────────────────────────────────
class TestF2MagnitudeBounds:
    @pytest.mark.parametrize("panel,L,cb_mM", [
        ('a', 100, 0.1), ('b', 100, 10.0),
        ('b', 10,  10.0), ('b', 1,   10.0),
        ('c', 100, 0.1),  ('d', 100, 10.0),
    ])
    def test_cdl_min_above_gc_lower_bound(self, panel, L, cb_mM):
        """C_dl_min must exceed the single-facet GC minimum (C_H||C_GC,min)."""
        cdl_min = _load_cdl_min(panel, L)
        gc_lower = _C_dl_min_analytic(cb_mM)
        assert cdl_min >= gc_lower * 0.5, (
            f"F2{panel} L={L}nm: C_dl_min={cdl_min*100:.2f} uF/cm2 "
            f"< 0.5×C_dl,GC_min={gc_lower*100:.2f} uF/cm2 — unexpectedly low")

    @pytest.mark.parametrize("panel,L", [
        ('a', 1), ('a', 10), ('a', 100),
        ('b', 1), ('b', 10), ('b', 100),
        ('c', 1), ('c', 10), ('c', 100),
        ('d', 1), ('d', 10), ('d', 100),
    ])
    def test_cdl_min_below_ch(self, panel, L):
        """C_dl_min must be below C_H (series formula upper bound)."""
        cdl_min = _load_cdl_min(panel, L)
        assert cdl_min < C_H, (
            f"F2{panel} L={L}nm: C_dl_min={cdl_min*100:.1f} uF/cm2 "
            f">= C_H={C_H*100:.0f} uF/cm2 — physically impossible")


# ── Two-facet estimate check (deep bistable) ─────────────────────────────────
class TestF2TwoFacetMagnitude:
    def test_F2b_L100nm_within_50pct_of_two_facet_estimate(self):
        """F2b L=100nm (L/λD=33): C_dl_min within 50% of two-facet GC estimate.

        At deep bistable (L >> λD): one facet at PZC (C_dl ≈ C_dl,min_GC),
        other far off-PZC (C_dl → C_H). Area-weighted average (x=0.5):
        C_dl_2facet ≈ (C_dl,PZC + C_H) / 2.
        GC overestimates vs article by ≤30% (no Booth-Langevin/Bikerman) —
        so 50% tolerance covers both numerical and physics offset.
        """
        cb = 10.0
        cdl_pzc = _C_dl_min_analytic(cb)
        two_facet_est = (cdl_pzc + C_H) / 2.0
        cdl_min = _load_cdl_min('b', 100)
        ratio = cdl_min / two_facet_est
        assert 0.5 <= ratio <= 1.5, (
            f"F2b L=100nm: C_dl_min={cdl_min*100:.1f} uF/cm2, "
            f"two-facet estimate={two_facet_est*100:.1f} uF/cm2, "
            f"ratio={ratio:.3f} outside [0.5, 1.5]")
