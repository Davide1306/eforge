"""
pb2d.py — 2-D two-facet modified Poisson-Boltzmann solver (CONSTRAINTS §2).

In-plane y in [0, L] with TWO facet stripes (facet 1 occupies fraction x, facet 2
the rest), periodic in y (eq 7); normal z in [0, z_max] (truncated, eq 6). Each
facet carries its own E_pzc^i and C_H^i in the Robin BC (eq 5).

Conservative finite-volume discretization of div(eps_eff grad phi) = -rho with the
field-dependent kernel (physics.py). The 2-D field magnitude E = sqrt(Ey^2 + Ez^2)
enters eps_eff and the steric denominator. Reuses pb1d's nonuniform z-grid and
control volumes, and reduces EXACTLY to pb1d when the two facets are identical
(the solution becomes y-uniform).
"""
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve

from . import constants as C
from . import physics as PH
from . import pb1d

_SPARSITY_CACHE = {}


def facet_columns(y, L, x, E_pzc1, E_pzc2, C_H1, C_H2):
    """Per-column facet PZC and Helmholtz capacitance (facet 1 is y < x*L)."""
    f1 = y < x * L
    return np.where(f1, E_pzc1, E_pzc2), np.where(f1, C_H1, C_H2)


def make_grid_2d(c_b_M, L, Ny, Nz, z_max_over_lamD=25.0, stretch=7.0):
    lam = C.lambda_D(c_b_M)
    y = np.linspace(0.0, L, Ny, endpoint=False)        # periodic: y[Ny] == y[0] + L
    z = pb1d.make_grid(lam, Nz, z_max_over_lamD, stretch)
    return y, z, lam


def _node_fields(phi, hy, z):
    """Central node-field components Ey, Ez [V/m] on the full (Ny, Nz) grid."""
    Ey = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * hy)
    Ez = np.empty_like(phi)
    Ez[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (z[2:] - z[:-2])
    Ez[:, 0] = (phi[:, 1] - phi[:, 0]) / (z[1] - z[0])
    Ez[:, -1] = (phi[:, -1] - phi[:, -2]) / (z[-1] - z[-2])
    return Ey, Ez


def _residual_2d(phi_unk, y, z, hy, hz, wz, n_b, v, E_M, E_pzc_col, C_H_col, Ny, Nz,
                 bc="eps_S"):
    phi = np.empty((Ny, Nz))
    phi[:, :-1] = phi_unk.reshape(Ny, Nz - 1)
    phi[:, -1] = 0.0
    Ey_n, Ez_n = _node_fields(phi, hy, z)

    # z-faces (between k, k+1), k = 0..Nz-2
    dphi_z = phi[:, 1:] - phi[:, :-1]
    Ez_zf = dphi_z / hz
    Ey_zf = 0.5 * (Ey_n[:, :-1] + Ey_n[:, 1:])
    epsZ = PH.eps_eff(0.5 * (phi[:, :-1] + phi[:, 1:]), np.hypot(Ez_zf, Ey_zf), v)
    Fz = epsZ * dphi_z / hz

    # y-faces (between j, j+1, periodic), all k
    dphi_y = np.roll(phi, -1, axis=0) - phi
    Ez_yf = 0.5 * (Ez_n + np.roll(Ez_n, -1, axis=0))
    epsY = PH.eps_eff(0.5 * (phi + np.roll(phi, -1, axis=0)),
                      np.hypot(dphi_y / hy, Ez_yf), v)
    Fy = epsY * dphi_y / hy

    # wall (Robin, eq 5) flux per column at k=0; E2 reading selected by bc
    Delta = E_M - E_pzc_col - phi[:, 0]
    if bc == "eps_eff":
        J_wall = -C_H_col * Delta              # Gauss continuity (eps_eff(0) cancels)
    else:                                      # "eps_S", as printed
        dphi0 = -(C_H_col / C.eps_S) * Delta
        J_wall = PH.eps_eff(phi[:, 0], np.hypot(dphi0, Ey_n[:, 0]), v) * dphi0

    # charge density at nodes
    npl, nmi = PH.number_densities(phi, np.hypot(Ey_n, Ez_n), n_b, v)
    rho = C.e_0 * (npl - nmi)

    # FV divergence over each CV (Ny, Nz-1): k = 0..Nz-2
    Fz_lower = np.empty((Ny, Nz - 1))
    Fz_lower[:, 0] = J_wall
    Fz_lower[:, 1:] = Fz[:, :-1]
    z_div = (Fz - Fz_lower) * hy
    Fy_k = Fy[:, :-1]
    y_div = (Fy_k - np.roll(Fy_k, 1, axis=0)) * wz[:-1]
    src = rho[:, :-1] * hy * wz[:-1]
    return (z_div + y_div + src).ravel()


def _sparsity_2d(Ny, Nz):
    """Jacobian sparsity for the (Ny, Nz-1) unknown grid: each residual couples
    to the 3x3 (y,z) stencil (periodic in y; the bulk z column is fixed)."""
    key = (Ny, Nz)
    if key not in _SPARSITY_CACHE:
        nk = Nz - 1
        rows, cols = [], []
        for j in range(Ny):
            for k in range(nk):
                n = j * nk + k
                for dj in (-1, 0, 1):
                    jj = (j + dj) % Ny
                    for dk in (-1, 0, 1):
                        kk = k + dk
                        if 0 <= kk < nk:
                            rows.append(n)
                            cols.append(jj * nk + kk)
        N = Ny * nk
        _SPARSITY_CACHE[key] = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(N, N))
    return _SPARSITY_CACHE[key]


def _newton_2d(fun, x0, pattern, maxit=60):
    """Damped sparse Newton iterated to the residual FLOOR (stall). Colored
    finite-difference Jacobian (approx_derivative on the stencil sparsity) +
    sparse solve + backtracking; O(N) per Jacobian vs O(N^2) for dense hybr.
    Returns (x, res_inf).

    No absolute tol: the FV residual scale is config-dependent and the stiff PB
    is ill-conditioned (a small residual != a small solution error), so a fixed
    tol stops far too early on some configs. Instead iterate until Newton can no
    longer reduce ||R|| (the machine floor), which gives an accurate solution."""
    x = np.array(x0, dtype=float)
    R = fun(x)
    nrm = float(np.linalg.norm(R, np.inf))
    for _ in range(maxit):
        if nrm == 0.0:
            break
        J = approx_derivative(fun, x, method="2-point", sparsity=pattern)
        dx = spsolve(sp.csc_matrix(J), -R)
        alpha = 1.0
        while True:
            xn = x + alpha * dx
            Rn = fun(xn)
            nn = float(np.linalg.norm(Rn, np.inf))
            if nn < nrm or alpha < 1e-12:
                break
            alpha *= 0.5
        x, R = xn, Rn
        if nn >= 0.9 * nrm:              # cannot reduce further -> at the floor
            nrm = nn
            break
        nrm = nn
    return x, nrm


def solve_2d(E_M, c_b_M, L, x, E_pzc1, E_pzc2, C_H1, C_H2, Ny=24, Nz=48,
             z_max_over_lamD=25.0, stretch=7.0, method="newton", phi_guess=None,
             bc="eps_S"):
    """Solve the 2-D two-facet interface at metal potential E_M. Returns a dict
    with y, z, phi (Ny,Nz), success, res_inf, q_free (per electrode area, eq 8),
    phi_HP (= phi[:,0], the HP profile), lam_D, v, E_pzc_col.

    Globalization: if the direct solve (GCS-linear per-column guess) fails, ramp
    the facet-PZC contrast from the uniform mean (an easy y-uniform 1-D problem)
    out to the target gap, warm-starting each step."""
    y, z, lam = make_grid_2d(c_b_M, L, Ny, Nz, z_max_over_lamD, stretch)
    hy = L / Ny
    hz = np.diff(z)
    _, wz = pb1d._control_volumes(z)
    n_b, v = C.n_b(c_b_M), C.v_frac(c_b_M)
    E_pzc_col, C_H_col = facet_columns(y, L, x, E_pzc1, E_pzc2, C_H1, C_H2)

    def _flux_scale(phi):
        # order-of-magnitude scale of the FV residual terms (constant-eps flux)
        return float(np.max(np.abs(C.eps_S * (phi[:, 1:] - phi[:, :-1]) / hz))) * hy + 1e-300

    def _res_rel(phi, epcol, EM):
        R = _residual_2d(phi[:, :-1].ravel(), y, z, hy, hz, wz, n_b, v, EM,
                         epcol, C_H_col, Ny, Nz, bc)
        return float(np.linalg.norm(R, np.inf)) / _flux_scale(phi)

    def _once(EM, epcol, guess):
        a = (y, z, hy, hz, wz, n_b, v, EM, epcol, C_H_col, Ny, Nz, bc)
        x0 = np.asarray(guess)[:, :-1].ravel()   # exclude the Dirichlet bulk column
        if method == "newton":
            xx, _ = _newton_2d(lambda u: _residual_2d(u, *a), x0, _sparsity_2d(Ny, Nz))
        else:
            xx = root(_residual_2d, x0, args=a, method=method).x
        phi = np.empty((Ny, Nz))
        phi[:, :-1] = xx.reshape(Ny, Nz - 1)
        phi[:, -1] = 0.0
        return phi

    if phi_guess is None:
        psi0 = (E_M - E_pzc_col) / (1.0 + (C.eps_S / lam) / C_H_col)
        phi_guess = psi0[:, None] * np.exp(-z[None, :] / lam)
    phi = _once(E_M, E_pzc_col, phi_guess)
    if _res_rel(phi, E_pzc_col, E_M) > 1e-7:
        # ramp E_M AND the facet-PZC contrast together from the exact trivial
        # state (E_M = PZC = Emean -> phi == 0) out to the target, warm-started.
        Emean = 0.5 * (E_pzc1 + E_pzc2)
        nstep = max(6, int((abs(E_M - Emean) + abs(E_pzc1 - E_pzc2)) / 0.05) + 1)
        guess = np.zeros((Ny, Nz))
        for t in np.linspace(0.0, 1.0, nstep + 1):
            epcol, _ = facet_columns(y, L, x, Emean + t * (E_pzc1 - Emean),
                                     Emean + t * (E_pzc2 - Emean), C_H1, C_H2)
            phi = _once(Emean + t * (E_M - Emean), epcol, guess)
            guess = phi

    # success from the flux-normalized residual at the target (config-independent).
    res_rel = _res_rel(phi, E_pzc_col, E_M)
    res_inf = res_rel * _flux_scale(phi)
    npl, nmi = PH.number_densities(phi, np.hypot(*_node_fields(phi, hy, z)), n_b, v)
    rho = C.e_0 * (npl - nmi)
    q_free = float(-(hy / L) * np.sum(rho * wz[None, :]))
    return dict(y=y, z=z, phi=phi, success=bool(res_rel < 1e-7),
                res_inf=res_inf, res_rel=res_rel, q_free=q_free,
                phi_HP=phi[:, 0].copy(), lam_D=lam, v=v, E_pzc_col=E_pzc_col)


def C_dl_2facet(E_M_arr, c_b_M, L, x, E_pzc1, E_pzc2, C_H1, C_H2,
                Ny=24, Nz=48, **kw):
    """Sweep E_M (warm-started) for the two-facet electrode. Returns
    (q_free[E], C_dl[E] = dq_free/dE_M) per unit electrode area (eqs 8-9)."""
    E_M_arr = np.asarray(E_M_arr, dtype=float)
    nE = E_M_arr.size
    q = np.empty(nE)

    def _solve(EM, guess):
        return solve_2d(EM, c_b_M, L, x, E_pzc1, E_pzc2, C_H1, C_H2,
                        Ny=Ny, Nz=Nz, phi_guess=guess, **kw)

    # March OUTWARD from the point nearest the mean PZC (smallest drop, easiest),
    # so every point is warm-started from a converged neighbour -- no extreme
    # E_M is ever cold-started.
    c0 = int(np.argmin(np.abs(E_M_arr - 0.5 * (E_pzc1 + E_pzc2))))
    s = _solve(E_M_arr[c0], None)
    q[c0] = s["q_free"]
    phi_c = s["phi"]
    phi_prev = phi_c
    for k in range(c0 + 1, nE):
        s = _solve(E_M_arr[k], phi_prev)
        q[k] = s["q_free"]
        phi_prev = s["phi"]
    phi_prev = phi_c
    for k in range(c0 - 1, -1, -1):
        s = _solve(E_M_arr[k], phi_prev)
        q[k] = s["q_free"]
        phi_prev = s["phi"]
    return q, np.gradient(q, E_M_arr)


def count_minima(C_dl, prominence_frac=0.0):
    """Indices of interior local minima of C_dl(E). With prominence_frac > 0 keep
    only minima whose smaller adjacent rise exceeds prominence_frac*(max-min) --
    filters shallow numerical wiggles (B-count shallow-minimum convention)."""
    C = np.asarray(C_dl, dtype=float)
    rng = float(C.max() - C.min()) or 1.0
    out = []
    for i in range(1, len(C) - 1):
        if C[i] < C[i - 1] and C[i] < C[i + 1]:
            prom = min(C[:i].max() - C[i], C[i + 1:].max() - C[i])
            if prom >= prominence_frac * rng:
                out.append(i)
    return out


def minima_from_qfree(E_M_arr, q_free, n_fine=4001, prom_frac=0.02):
    """Clean C_dl(E) = dq_free/dE_M via a cubic spline of q_free(E_M), then locate
    minima on a fine grid (sub-grid B-pos resolution). q_free is smooth and
    accurate (antisymmetric to ~1e-15 for symmetric cases); np.gradient on the
    coarse sweep grid is noisy near the sharp minima, so the analytic spline
    derivative gives a clean, symmetric C_dl. Returns
    (E_fine, C_dl_fine, minima) with minima = [(E_min, C_min), ...]."""
    E = np.asarray(E_M_arr, dtype=float)
    cs = CubicSpline(E, np.asarray(q_free, dtype=float))
    Ef = np.linspace(float(E[0]), float(E[-1]), n_fine)
    Cf = cs(Ef, 1)                              # first derivative = C_dl
    rng = float(Cf.max() - Cf.min()) or 1.0
    peaks, _ = find_peaks(-Cf, prominence=prom_frac * rng)
    return Ef, Cf, [(float(Ef[i]), float(Cf[i])) for i in peaks]
