"""
test_validator_independence.py — LESSON G7 / BP-04 mutation-probe suite.

Normative, self-enforcing check that the CG cross-check VALIDATOR computes its
NUMERICAL side (the diffuse-layer surface charge sigma_M / diffuse capacitance)
from the converged Poisson-Boltzmann solution phi, and does NOT re-derive it from
the analytic Gouy-Chapman closed form (eq 10) that serves as the reference.

A tier/figure "pass" on the CG cross-check (convergence_proof grammar:
"CG cross-check rel_err < 1e-3") is only meaningful if CORRUPTING phi BREAKS the
agreement. If the numerical side were secretly the analytic form, the agreement
would survive any mutation of phi — a circular (self-validating) test.

LESSON G7: tier-acceptance tests that
compare a numerical output X to an analytic reference X_ref MUST compute X from
the converged numerical solution, NOT via the same closed form used for X_ref.
This file is that lesson's live mutation probe (BP-04: the single normative
check; pointers only elsewhere).

Degenerate limit used: dilute (c_b = 0.1 mM) + small wall potential
(psi_0 <= 0.1 V) -> Gouy-Chapman point-ion limit, where steric/polarization
corrections are O(1e-4) and C_diff -> C_GC = (eps_S/lambda_D) cosh(beta e_0 psi/2).
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import constants as C  # noqa: E402
from src import pb1d  # noqa: E402

CB_M = C.mM_to_M(0.1)   # dilute -> GC limit
PSI = 0.02             # V, inside the |E - E_pzc| <= 0.1 V GC window


def _cg_analytic(psi):
    """Analytic Gouy-Chapman differential capacitance (eq 10) at wall potential
    psi, single-facet dilute limit [F/m^2]. The REFERENCE side."""
    return pb1d.C_GC_min(CB_M) * np.cosh(C.beta * C.e_0 * psi / 2.0)


def _sigma_from_phi(phi, z, n_b, v):
    """Recompute sigma_M from an ARBITRARY phi profile via the SAME Gauss /
    neutrality integral the solver uses (pb1d internals) — the numerical side."""
    h, w = pb1d._control_volumes(z)
    _, rho = pb1d._assemble(phi, z, h, n_b, v)
    return float(-np.sum(rho * w))


def test_cg_crosscheck_passes_for_truth():
    """Sanity: the cross-check IS satisfiable — the converged numerical diffuse
    capacitance matches analytic CG within the 1e-3-class band at the GC limit.
    (If this failed, the probe below would be vacuous.)"""
    C_num = pb1d.diffuse_capacitance(PSI, CB_M)
    rel = abs(C_num - _cg_analytic(PSI)) / _cg_analytic(PSI)
    assert rel < 1e-2, f"CG cross-check must pass for the true solution; rel={rel:.2e}"


def test_sigma_is_a_function_of_phi():
    """The numerical side (sigma_M) returned by the solver equals sigma recomputed
    from its phi — i.e. sigma_M IS derived from phi, not stored independently."""
    s = pb1d.solve_dirichlet(PSI, CB_M)
    sigma_recomp = _sigma_from_phi(s["phi"], s["z"], s["n_b"], s["v"])
    assert abs(sigma_recomp - s["sigma_M"]) / abs(s["sigma_M"]) < 1e-9


def test_mutation_breaks_the_numerical_side():
    """CORE G7 PROBE: zeroing the interior of the converged phi MUST materially
    change sigma_M. If sigma_M (hence the cross-check's numerical side) were the
    analytic CG form, it would be invariant to phi and this would FAIL —
    exposing a circular validator."""
    s = pb1d.solve_dirichlet(PSI, CB_M)
    sigma_true = _sigma_from_phi(s["phi"], s["z"], s["n_b"], s["v"])
    phi_mut = s["phi"].copy()
    phi_mut[1:-1] = 0.0                      # corrupt the interior solution
    sigma_mut = _sigma_from_phi(phi_mut, s["z"], s["n_b"], s["v"])
    rel_change = abs(sigma_mut - sigma_true) / abs(sigma_true)
    assert rel_change > 0.5, (
        f"mutating phi must break the numerical side (it must NOT be the analytic "
        f"reference); rel_change={rel_change:.2e}")


def test_coarse_mesh_has_genuine_galerkin_error():
    """The numerical side carries a real, mesh-dependent discretization error:
    coarse and fine meshes must DIFFER. A vanishing (~1e-12) coarse-vs-fine
    difference would prove the numerical side is re-derived analytically
    (G7 violation), since Galerkin error makes exact agreement impossible."""
    C_coarse = pb1d.diffuse_capacitance(PSI, CB_M, n=65)
    C_fine = pb1d.diffuse_capacitance(PSI, CB_M, n=513)
    rel = abs(C_coarse - C_fine) / C_fine
    assert rel > 1e-6, (
        f"coarse vs fine mesh must differ (genuine Galerkin error, not analytic "
        f"re-derivation); rel={rel:.2e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
