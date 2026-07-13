from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import pytest
from src.gc_analytic import C_GC, C_DH, sigma_GC, solve_psi0


C_B_TEST = 1.0      # mol/m³ (= 1 mM)
C_H_TEST = 0.5      # F/m²   (= 50 µF/cm²)
SMALL_PSI = [1e-4, 2e-4, 5e-4]  # V — well within DH regime (F|ψ|/2RT < 0.01)


def test_dh_limit():
    """G2 acceptance: |C_GC(ψ_0) − C_DH|/C_DH < 1e-4 for small ψ_0."""
    C_dh = C_DH(C_B_TEST)
    for psi in SMALL_PSI:
        for sign in (+1, -1):
            psi_0 = sign * psi
            Cgc = C_GC(psi_0, C_B_TEST)
            rel_err = abs(Cgc - C_dh) / C_dh
            assert rel_err < 1e-4, (
                f"ψ_0={psi_0*1e3:.2f} mV: |C_GC-C_DH|/C_DH={rel_err:.2e} ≥ 1e-4"
            )


def test_sigma_gc_odd():
    """σ_GC is an odd function of ψ_0."""
    psi_vals = [0.05, 0.1, 0.2, 0.3]
    for psi in psi_vals:
        s_pos = sigma_GC(+psi, C_B_TEST)
        s_neg = sigma_GC(-psi, C_B_TEST)
        assert abs(s_pos + s_neg) / abs(s_pos) < 1e-12, (
            f"σ_GC not odd at ψ_0={psi} V"
        )


def test_sigma_gc_sign():
    """σ_GC has the same sign as ψ_0 (positive electrode → positive charge)."""
    assert sigma_GC(+0.1, C_B_TEST) > 0
    assert sigma_GC(-0.1, C_B_TEST) < 0
    assert abs(sigma_GC(0.0, C_B_TEST)) < 1e-20


def test_solve_psi0_at_pzc():
    """At E_M = E_pzc, ψ_0 must be zero (no charge at PZC)."""
    E_pzc = 0.1
    psi_0 = solve_psi0(E_M=E_pzc, E_pzc=E_pzc, C_H=C_H_TEST, c_b=C_B_TEST)
    assert abs(psi_0) < 1e-10, f"ψ_0={psi_0:.2e} V at PZC (expected 0)"


def test_solve_psi0_positive_overpotential():
    """At E_M > E_pzc, ψ_0 > 0 (electrode positive, solution near surface positive)."""
    psi_0 = solve_psi0(E_M=0.2, E_pzc=0.0, C_H=C_H_TEST, c_b=C_B_TEST)
    assert psi_0 > 0, f"Expected ψ_0 > 0 for E_M > E_pzc, got {psi_0}"


def test_cdl_single_series():
    """C_dl_single < min(C_H, C_GC) — series combination is always smaller."""
    from src.gc_analytic import C_dl_single
    E_pzc = 0.0
    for E_M in np.linspace(-0.3, 0.3, 7):
        psi_0 = solve_psi0(E_M, E_pzc, C_H_TEST, C_B_TEST)
        Cgc = C_GC(psi_0, C_B_TEST)
        Cdl = C_dl_single(E_M, E_pzc, C_H_TEST, C_B_TEST)
        assert Cdl < C_H_TEST + 1e-10
        assert Cdl < Cgc + 1e-10
