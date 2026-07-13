"""
pb1d.py — 1-D modified Poisson-Boltzmann solver (single facet, normal direction
z from the Helmholtz plane into the bulk). Finite-volume discretization of
CONSTRAINTS §2 eq 1

    d/dz( eps_eff(phi, |phi'|) dphi/dz ) = - e_0 (n_+ - n_-),

with the field-dependent constitutive kernel from physics.py (eqs 2-4). The
conservative FV form guarantees discrete charge balance.

Boundary conditions:
  z = 0 :  'dirichlet'  phi(0) = psi_0          (pure diffuse layer; CG cross-check)
           'robin'      eq 5 Helmholtz BC       (added at G2 next iter -> C_dl vs E_M)
  z -> inf (truncated at z_max):  phi(z_max) = 0 (eq 6)

The diffuse-layer differential capacitance C_diff = d sigma_M / d psi_0 reduces,
in the dilute small-field limit, to the analytic Gouy-Chapman eq 10 minimum
eps_S / lambda_D -- the CG cross-check required by every convergence proof.
"""
import numpy as np
from scipy.optimize import root

from . import constants as C
from . import physics as PH


def make_grid(lam_D, n=257, z_max_over_lamD=25.0, stretch=7.0):
    """Nonuniform z grid [m], length n, clustered at the wall (z=0). The
    exponential stretch resolves the compact near-wall layer while reaching
    ~25 lambda_D in the bulk."""
    s = np.linspace(0.0, 1.0, n)
    return z_max_over_lamD * lam_D * (np.exp(stretch * s) - 1.0) / (np.exp(stretch) - 1.0)


def _control_volumes(z):
    h = np.diff(z)
    w = np.empty(z.size)
    w[0] = h[0] / 2.0
    w[-1] = h[-1] / 2.0
    w[1:-1] = 0.5 * (h[:-1] + h[1:])
    return h, w


def _charge_density(phi, E_node, n_b, v):
    """rho = e_0 (n_+ - n_-)  [C/m^3], via the overflow-stable number_densities
    (= -2 e_0 n_b sinh(beta e_0 phi)/D, but computed without exp/sinh overflow)."""
    n_plus, n_minus = PH.number_densities(phi, E_node, n_b, v)
    return C.e_0 * (n_plus - n_minus)


def _assemble(phi, z, h, n_b, v):
    """Face fluxes J [length n-1], node field E_node, charge density rho."""
    dphi = np.diff(phi)
    phi_f = 0.5 * (phi[:-1] + phi[1:])
    E_f = np.abs(dphi / h)
    eps_f = PH.eps_eff(phi_f, E_f, v)
    J = eps_f * dphi / h
    E_node = np.empty(z.size)
    E_node[1:-1] = 0.5 * (E_f[:-1] + E_f[1:])
    E_node[0] = E_f[0]
    E_node[-1] = E_f[-1]
    rho = _charge_density(phi, E_node, n_b, v)
    return J, rho


def _residual_dirichlet(phi_int, psi_0, z, h, w, n_b, v):
    phi = np.empty(z.size)
    phi[0] = psi_0
    phi[-1] = 0.0
    phi[1:-1] = phi_int
    J, rho = _assemble(phi, z, h, n_b, v)
    # interior nodes 1..N-1:  J_{i+1/2} - J_{i-1/2} + rho_i w_i = 0
    return (J[1:] - J[:-1]) + rho[1:-1] * w[1:-1]


def solve_dirichlet(psi_0, c_b_M, n=257, z_max_over_lamD=25.0, stretch=7.0,
                    phi_guess=None):
    """Solve the diffuse layer for a fixed wall potential psi_0 [V]. Returns a
    dict with z, phi, success, res_inf (raw inf-norm), res_rel (||R||/max|J|),
    sigma_M, lam_D, v, n_b."""
    lam = C.lambda_D(c_b_M)
    z = make_grid(lam, n, z_max_over_lamD, stretch)
    h, w = _control_volumes(z)
    n_b, v = C.n_b(c_b_M), C.v_frac(c_b_M)
    if phi_guess is None:
        phi_guess = psi_0 * np.exp(-z / lam)
    sol = root(_residual_dirichlet, np.asarray(phi_guess)[1:-1],
               args=(psi_0, z, h, w, n_b, v), method="hybr")
    phi = np.empty(z.size)
    phi[0] = psi_0
    phi[-1] = 0.0
    phi[1:-1] = sol.x
    R = _residual_dirichlet(sol.x, psi_0, z, h, w, n_b, v)
    J, rho = _assemble(phi, z, h, n_b, v)
    res_inf = float(np.linalg.norm(R, np.inf))
    res_rel = res_inf / (float(np.max(np.abs(J))) + 1e-300)
    sigma_M = float(-np.sum(rho * w))   # sigma_M = -Q_diff (Gauss + neutrality)
    return dict(z=z, phi=phi, success=bool(sol.success), res_inf=res_inf,
                res_rel=res_rel, sigma_M=sigma_M, lam_D=lam, v=v, n_b=n_b)


def diffuse_capacitance(psi_0, c_b_M, dpsi=5e-4, **kw):
    """C_diff = d sigma_M / d psi_0  [F/m^2] by central difference (warm-started)."""
    sp = solve_dirichlet(psi_0 + dpsi, c_b_M, **kw)
    sm = solve_dirichlet(psi_0 - dpsi, c_b_M, **kw)
    return (sp["sigma_M"] - sm["sigma_M"]) / (2.0 * dpsi)


def C_GC_min(c_b_M):
    """Analytic Gouy-Chapman differential-capacitance minimum eps_S/lambda_D
    (eq 10 at E = E_pzc) -- the CG cross-check target [F/m^2]."""
    return C.eps_S / C.lambda_D(c_b_M)


# --------------------------------------------------------------------------
# Robin (Helmholtz) boundary condition -- eq 5 -- and C_dl(E_M) -- eqs 8-9.
# E2 ambiguity: the printed eq 5 divides by eps_S ('eps_S', the default); the
# Gauss-continuity reading divides by eps_eff(0) ('eps_eff', deferred to an A9
# resolution if Fig S3 / §3 bands fail at large |E_M - E_pzc|).
# --------------------------------------------------------------------------
def _residual_robin(phi_unk, E_M, E_pzc, C_H, z, h, w, n_b, v, bc):
    phi = np.empty(z.size)
    phi[:-1] = phi_unk
    phi[-1] = 0.0
    Delta = E_M - E_pzc - phi[0]
    if bc == "eps_eff":
        # E2-alternative (Gauss continuity): eps_eff(0) phi'(0) = -C_H*Delta, so
        # the wall displacement (flux) IS -C_H*Delta -- the eps_eff(0) cancels.
        J_wall = -C_H * Delta
    else:                                                # "eps_S", as printed (eq 5 / eps_S)
        dphi0 = -(C_H / C.eps_S) * Delta
        J_wall = PH.eps_eff(phi[0], abs(dphi0), v) * dphi0
    J, rho = _assemble(phi, z, h, n_b, v)
    R = np.empty(z.size - 1)
    R[0] = J[0] - J_wall + rho[0] * w[0]                 # half-cell at z=0
    R[1:] = (J[1:] - J[:-1]) + rho[1:-1] * w[1:-1]       # interior nodes
    return R


def _robin_root(phi_guess_full, E_M, E_pzc, C_H, z, h, w, n_b, v, bc):
    sol = root(_residual_robin, np.asarray(phi_guess_full)[:-1],
               args=(E_M, E_pzc, C_H, z, h, w, n_b, v, bc), method="hybr")
    phi = np.empty(z.size)
    phi[:-1] = sol.x
    phi[-1] = 0.0
    R = _residual_robin(sol.x, E_M, E_pzc, C_H, z, h, w, n_b, v, bc)
    return sol, phi, R


def solve_robin(E_M, E_pzc, C_H, c_b_M, bc="eps_S", n=257, z_max_over_lamD=25.0,
                stretch=7.0, phi_guess=None):
    """Solve the single-facet interface (Helmholtz + diffuse) at metal potential
    E_M [V] with facet PZC E_pzc and Helmholtz capacitance C_H [F/m^2]. Returns a
    dict with z, phi, success, res_inf, res_rel, q_free, phi_0, lam_D, v, n_b.
    q_free is the eq-8 Gauss charge (= -integral rho), E3 sign so C_dl>0.

    Globalization: the PB equation is exponentially stiff at large |E_M - E_pzc|,
    so if the direct solve (from phi_guess or the GCS-linear estimate) fails we
    ramp E_M from the trivial E_pzc solution (phi == 0) in <=0.05 V steps,
    warm-starting each. Numerical only -- the converged solution is unique.

    bc selects the eq-5 reading (E2): "eps_S" (as printed, default) or "eps_eff"
    (Gauss continuity, eps_eff(0))."""
    if bc not in ("eps_S", "eps_eff"):
        raise ValueError(f"bc must be 'eps_S' or 'eps_eff', got {bc!r}")
    lam = C.lambda_D(c_b_M)
    z = make_grid(lam, n, z_max_over_lamD, stretch)
    h, w = _control_volumes(z)
    n_b, v = C.n_b(c_b_M), C.v_frac(c_b_M)
    if phi_guess is None:
        cgc = C.eps_S / lam                              # linear diffuse capacitance
        psi0 = (E_M - E_pzc) / (1.0 + cgc / C_H)         # GCS split of the total drop
        phi_guess = psi0 * np.exp(-z / lam)
    sol, phi, R = _robin_root(phi_guess, E_M, E_pzc, C_H, z, h, w, n_b, v, bc)
    if not sol.success:
        nstep = max(4, int(abs(E_M - E_pzc) / 0.05) + 1)
        guess = np.zeros(z.size)                         # phi == 0 solves eq 5 at E_pzc
        for EMi in np.linspace(E_pzc, E_M, nstep + 1)[1:]:
            sol, phi, R = _robin_root(guess, EMi, E_pzc, C_H, z, h, w, n_b, v, bc)
            guess = phi
    J, rho = _assemble(phi, z, h, n_b, v)
    res_inf = float(np.linalg.norm(R, np.inf))
    res_rel = res_inf / (float(np.max(np.abs(J))) + 1e-300)
    return dict(z=z, phi=phi, success=bool(sol.success), res_inf=res_inf,
                res_rel=res_rel, q_free=float(-np.sum(rho * w)), phi_0=float(phi[0]),
                lam_D=lam, v=v, n_b=n_b)


def C_dl_curve(E_M_arr, E_pzc, C_H, c_b_M, bc="eps_S", **kw):
    """Sweep E_M with continuation (warm start). Returns (q_free[E], C_dl[E]),
    C_dl = dq_free/dE_M (eq 9) by np.gradient."""
    E_M_arr = np.asarray(E_M_arr, dtype=float)
    q = np.empty(E_M_arr.size)
    phi_prev = None
    for k, EM in enumerate(E_M_arr):
        s = solve_robin(EM, E_pzc, C_H, c_b_M, bc=bc, phi_guess=phi_prev, **kw)
        q[k] = s["q_free"]
        phi_prev = s["phi"]
    return q, np.gradient(q, E_M_arr)


def C_dl_GCS_series(C_H, c_b_M):
    """GCS series capacitance at E_pzc (eq 11): (1/C_H + lambda_D/eps_S)^-1
    [F/m^2]. The low-field limit C_dl(E_pzc) must match this."""
    return 1.0 / (1.0 / C_H + C.lambda_D(c_b_M) / C.eps_S)
