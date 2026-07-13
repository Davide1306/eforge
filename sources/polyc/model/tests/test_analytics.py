"""
G1 SMOKE — analytic layer (eqs 10-13). Pins the eq-10 GC capacitance hand value
(the CG cross-check) and the eq-13 C_ads peak. Falsifier (worker EDIT RATIONALE
iter2): C_GC(E_pzc) != eps_S/lam_D, or numeric C_ads max != closed-form peak / not
at theta=1/2.
"""
import numpy as np

from src import analytics as A
from src import constants as C


def _rel(a, b):
    return abs(a - b) / abs(b)


def test_C_GC_minimum_at_pzc_hand_value():
    lam = C.lambda_D(C.mM_to_M(0.1))
    cmin = A.C_GC(0.3, 0.3, lam)               # E = E_pzc
    assert _rel(cmin, C.eps_S / lam) < 1e-12   # identity at the minimum
    assert _rel(C.SI_to_uFcm2(cmin), 2.284) < 1e-2   # hand value ~2.284 uF/cm^2


def test_C_GC_symmetric_and_convex():
    lam = C.lambda_D(C.mM_to_M(1.0))
    E = np.linspace(-0.5, 0.5, 101)
    c = A.C_GC(E, 0.0, lam)
    assert np.argmin(c) == 50                  # minimum at E = E_pzc = 0
    assert _rel(A.C_GC(0.2, 0.0, lam), A.C_GC(-0.2, 0.0, lam)) < 1e-12  # even


def test_C_ads_peak_hand_value():
    assert _rel(C.SI_to_uFcm2(A.C_ads_peak(0.01)), 23.4) < 1e-2


def test_C_ads_peaks_at_half_coverage():
    E = np.linspace(-0.5, 1.5, 4001)
    c = A.C_ads(E, 0.45, C.c_b_default, 0.01)  # c_b^(A-) = c_b (eq-12 assumption)
    imax = int(np.argmax(c))
    th = A.theta_ads(E[imax], 0.45, C.c_b_default)
    assert _rel(th, 0.5) < 1e-2
    assert _rel(c[imax], A.C_ads_peak(0.01)) < 1e-3


def test_GCS_series_below_both():
    lam = C.lambda_D(C.mM_to_M(0.1))
    cgc = A.C_GC(0.3, 0.3, lam)
    cdl = A.C_dl_GCS(C.C_H_default, cgc)
    assert cdl < cgc and cdl < C.C_H_default
    assert _rel(1.0 / cdl, 1.0 / C.C_H_default + 1.0 / cgc) < 1e-12
