"""
G2 core — 1-D modified-PB solver. The CG cross-check (worker EDIT RATIONALE
iter4): in the dilute small-field limit the diffuse capacitance C_diff(psi_0->0)
must equal the analytic GC minimum eps_S/lambda_D (eq 10) to <1e-3. Falsifiers:
non-convergent solve, wrong sigma_M sign, or C_diff(0) off >1e-3.
"""
import numpy as np

from src import constants as C
from src import pb1d


def _rel(a, b):
    return abs(a - b) / abs(b)


def test_solver_converges_dilute():
    s = pb1d.solve_dirichlet(0.05, C.mM_to_M(0.1))
    assert s["success"]
    assert s["res_rel"] < 1e-8
    # boundary + decay sanity
    assert _rel(s["phi"][0], 0.05) < 1e-12
    assert abs(s["phi"][-1]) < 1e-9
    assert s["phi"][1] < s["phi"][0]          # decays from the wall


def test_profile_monotonic_decay():
    s = pb1d.solve_dirichlet(0.1, C.mM_to_M(1.0))
    assert s["success"]
    # monotone non-increasing for psi_0>0 (tol 1e-9 V absorbs ~1e-11 solver noise
    # in the far bulk where phi ~ 1e-10 is numerically zero on the 0.1 V scale)
    assert np.all(np.diff(s["phi"]) <= 1e-9)
    assert s["phi"].min() >= -1e-9                  # no unphysical overshoot


def test_sigma_M_sign():
    sp = pb1d.solve_dirichlet(+0.05, C.mM_to_M(0.1))
    sm = pb1d.solve_dirichlet(-0.05, C.mM_to_M(0.1))
    assert sp["sigma_M"] > 0 and sm["sigma_M"] < 0          # +wall -> +metal charge
    assert _rel(sp["sigma_M"], -sm["sigma_M"]) < 1e-2       # near-symmetric at small psi


def test_CG_cross_check_minimum():
    # C_diff(psi_0 -> 0) == eps_S / lambda_D  (analytic GC, eq 10) to <1e-3
    for mM in (0.1, 1.0):
        cdiff0 = pb1d.diffuse_capacitance(0.0, C.mM_to_M(mM))
        cgc = pb1d.C_GC_min(C.mM_to_M(mM))
        assert _rel(cdiff0, cgc) < 1e-3, (mM, cdiff0, cgc)


def test_capacitance_grows_with_potential():
    # GC differential capacitance is convex (cosh): C_diff(0.1 V) > C_diff(0)
    cb = C.mM_to_M(0.1)
    c0 = pb1d.diffuse_capacitance(0.0, cb)
    c1 = pb1d.diffuse_capacitance(0.1, cb)
    assert c1 > c0


# ---- Robin (Helmholtz) BC + C_dl(E_M), G2 single-facet (Fig S3 / V0) ----
def test_robin_converges_across_sweep():
    C_H = C.uFcm2_to_SI(25.0)
    cb = C.mM_to_M(0.1)
    for EM in (-0.5, -0.2, 0.0, 0.3, 0.6):
        s = pb1d.solve_robin(EM, 0.0, C_H, cb)
        assert s["success"] and s["res_rel"] < 1e-7, (EM, s["res_rel"])


def test_robin_GCS_consistency_at_pzc():
    # at E_pzc the interface is pure GCS (zero field, eps_eff = eps_S):
    # C_dl(E_pzc) == (1/C_H + lambda_D/eps_S)^-1  (eq 11) to <1%
    C_H = C.uFcm2_to_SI(25.0)
    cb = C.mM_to_M(0.1)
    q, Cdl = pb1d.C_dl_curve(np.array([-0.01, 0.0, 0.01]), 0.0, C_H, cb)
    assert _rel(Cdl[1], pb1d.C_dl_GCS_series(C_H, cb)) < 1e-2


def test_robin_minimum_at_pzc_dilute():
    # dilute single facet: the C_dl(E) minimum sits at E_pzc (GC well), B-pos 20 mV
    C_H = C.uFcm2_to_SI(25.0)
    cb = C.mM_to_M(0.1)
    E = np.linspace(-0.6, 0.6, 61)
    q, Cdl = pb1d.C_dl_curve(E, 0.0, C_H, cb)
    imin = int(np.argmin(Cdl))
    assert abs(E[imin]) <= 0.02
    assert Cdl.min() > 0


def test_robin_q_free_sign_and_monotonic():
    # q_free increases with E_M (capacitive), and crosses zero near E_pzc
    C_H = C.uFcm2_to_SI(25.0)
    cb = C.mM_to_M(0.1)
    E = np.linspace(-0.4, 0.4, 41)
    q, Cdl = pb1d.C_dl_curve(E, 0.0, C_H, cb)
    assert np.all(np.diff(q) > 0)              # monotincreasing
    assert q[0] < 0 < q[-1]                    # negative below pzc, positive above
