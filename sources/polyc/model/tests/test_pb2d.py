"""
G3 — 2-D two-facet PB solver. Falsifiers (worker EDIT RATIONALE iter7):
(1) identical facets must give a y-uniform solution matching the 1-D solver;
(2) distinct facets must give a y-varying phi_HP higher over the facet whose PZC
    is nearer E_M (correct charge sign).
"""
import numpy as np

from src import constants as C
from src import pb1d, pb2d


def _rel(a, b):
    return abs(a - b) / abs(b)


def test_2d_reduces_to_1d_for_identical_facets():
    cb = C.mM_to_M(0.1)
    L = 10e-9
    C_H = C.uFcm2_to_SI(50.0)
    E_pzc, E_M = 0.0, 0.2
    s2 = pb2d.solve_2d(E_M, cb, L, 0.5, E_pzc, E_pzc, C_H, C_H, Ny=12, Nz=48)
    assert s2["success"]
    # identical facets -> y-uniform solution (1e-6 V is the sparse-Newton
    # convergence floor; ~1e-5 relative, far below any acceptance band)
    assert np.max(np.ptp(s2["phi"], axis=0)) < 1e-6
    # q_free matches the 1-D Robin solver
    s1 = pb1d.solve_robin(E_M, E_pzc, C_H, cb, n=48)
    assert _rel(s2["q_free"], s1["q_free"]) < 1e-2


def test_2d_distinct_facets_phiHP_varies_with_correct_sign():
    cb = C.mM_to_M(10.0)
    L = 10e-9
    C_H = C.uFcm2_to_SI(50.0)
    # Table-S1 base pair: facet1 E_pzc=+0.3, facet2 E_pzc=-0.1; probe at E_M=0.1
    s = pb2d.solve_2d(0.1, cb, L, 0.5, 0.3, -0.1, C_H, C_H, Ny=16, Nz=48)
    assert s["success"]
    phiHP, Epc = s["phi_HP"], s["E_pzc_col"]
    assert np.ptp(phiHP) > 1e-3                       # genuinely 2-D (not uniform)
    f1 = Epc > 0                                      # facet 1 (E_pzc = +0.3)
    # E_M - E_pzc: facet1 = -0.2 (phi_HP<0), facet2 = +0.2 (phi_HP>0)
    assert phiHP[~f1].mean() > phiHP[f1].mean()


def test_2d_converges_clean():
    cb = C.mM_to_M(1.0)
    L = 10e-9
    C_H = C.uFcm2_to_SI(50.0)
    s = pb2d.solve_2d(0.0, cb, L, 0.5, 0.3, -0.1, C_H, C_H, Ny=12, Nz=40)
    assert s["success"] and s["res_inf"] < 1e-12


def test_newton_matches_hybr():
    # the sparse-Newton solver (default) must agree with dense hybr.
    cb = C.mM_to_M(0.1)
    C_H = C.uFcm2_to_SI(50.0)
    kw = dict(Ny=16, Nz=40)
    sN = pb2d.solve_2d(0.2, cb, 100e-9, 0.5, 0.3, -0.3, C_H, C_H, method="newton", **kw)
    sH = pb2d.solve_2d(0.2, cb, 100e-9, 0.5, 0.3, -0.3, C_H, C_H, method="hybr", **kw)
    assert sN["success"] and sH["success"]
    assert _rel(sN["q_free"], sH["q_free"]) < 1e-4


def test_minima_from_qfree_symmetric():
    # spline-derivative extraction must recover symmetric minima from a clean
    # antisymmetric q_free (no solver; deterministic). Target C_dl has two wells
    # at +-0.3 V; the integral is q_free.
    from scipy.integrate import cumulative_trapezoid
    E = np.linspace(-0.6, 0.6, 81)
    Ct = 8.0 - 4.0 * np.exp(-((E - 0.3) / 0.1) ** 2) - 4.0 * np.exp(-((E + 0.3) / 0.1) ** 2)
    q = cumulative_trapezoid(Ct, E, initial=0.0)
    Ef, Cf, mins = pb2d.minima_from_qfree(E, q)
    assert len(mins) == 2
    pos = sorted(m[0] for m in mins)
    assert abs(pos[0] + 0.3) < 0.02 and abs(pos[1] - 0.3) < 0.02       # B-pos 20 mV
    assert abs(mins[0][1] - mins[1][1]) < 0.02 * abs(mins[0][1])        # symmetric


def test_one_two_minimum_transition():
    # Fig 2 / F1: dE_pzc = 0.6 V (facets +-0.3), 0.1 mM (lam_D = 30.4 nm), x=0.5.
    # E7 sense: single minimum for L < lam_D, two minima for L > lam_D.
    cb = C.mM_to_M(0.1)
    C_H = C.uFcm2_to_SI(50.0)
    E = np.linspace(-0.6, 0.6, 25)

    q1, Cdl1 = pb2d.C_dl_2facet(E, cb, 1e-9, 0.5, 0.3, -0.3, C_H, C_H, Ny=12, Nz=32)
    m1 = pb2d.count_minima(C.SI_to_uFcm2(Cdl1), 0.05)
    assert len(m1) == 1 and abs(E[m1[0]]) <= 0.05          # single, near mean PZC (0)

    q2, Cdl2 = pb2d.C_dl_2facet(E, cb, 100e-9, 0.5, 0.3, -0.3, C_H, C_H, Ny=12, Nz=32)
    m2 = pb2d.count_minima(C.SI_to_uFcm2(Cdl2), 0.05)
    assert len(m2) == 2                                    # two minima
    pos = sorted(E[i] for i in m2)
    assert pos[0] < 0 < pos[1]                             # straddle the mean PZC
    assert abs(pos[0] + 0.3) <= 0.1 and abs(pos[1] - 0.3) <= 0.1   # near facet PZCs
