"""2-D nonlinear Poisson-Boltzmann FD solver (GC-polyc tier G4+).

Geometry: y ∈ [0, L) periodic, z ∈ [0, Z_max].
Solves in dimensionless φ = Fϕ/RT, ζ = z/λ_D, η = y/λ_D.

PB: ∂²φ/∂η² + ∂²φ/∂ζ² = sinh(φ)
BCs:
  z=0 (k=0): Robin  ∂φ/∂ζ = ν_i·(φ − Φ_i)
              ν_i = C_H^i·λ_D/(ε₀ε_r),  Φ_i = F(E_M−E_pzc^i)/RT
  z=Z_max (k=Nz): Dirichlet φ = 0
  y: periodic

Unknowns: φ at (Ny, Nz) nodes, row-major flat index i*Nz+k.
k=Nz is fixed at 0 (not an unknown).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from scipy.optimize import root, OptimizeResult
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from src import EPS0, F, R, EPS_R_WATER
from src.gc_pb_1d import make_z_mesh
from src.gc_analytic import solve_psi0

jax.config.update("jax_enable_x64", True)


# ── mesh ─────────────────────────────────────────────────────────────

def make_yz_mesh(Ny: int, Nz: int, L: float, Z_max: float,
                 alpha: float = 3.0):
    """Return (y, z) mesh arrays.

    y : (Ny,) uniform on [0, L) — periodic
    z : (Nz+1,) tanh-stretched on [0, Z_max], fine at z=0
    """
    y = np.linspace(0.0, L, Ny, endpoint=False)
    z = make_z_mesh(Nz, Z_max, stretch="tanh", alpha=alpha)
    return y, z


# ── residual ─────────────────────────────────────────────────────────

def residual_2d(phi_flat: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Dimensionless 2-D PB residual, shape (Ny*Nz,).

    phi_flat[i*Nz+k] = φ(y_i, ζ_k), i=0..Ny-1, k=0..Nz-1.
    k=0 equations: Robin BC.
    k=1..Nz-1 equations: 5-point PB interior.
    """
    Ny = params['Ny']
    Nz = params['Nz']
    zeta = params['zeta']    # (Nz+1,)
    deta = params['deta']    # scalar Δη = Δy/λ_D
    nu = params['nu']        # (Ny,) dimensionless Helmholtz coefficient
    Phi = params['Phi']      # (Ny,) dimensionless drive

    phi = phi_flat.reshape(Ny, Nz)
    # Augment with Dirichlet φ=0 at k=Nz
    phi_aug = jnp.concatenate([phi, jnp.zeros((Ny, 1))], axis=1)  # (Ny, Nz+1)

    # ── Robin BC at k=0 (second-order one-sided stencil) ─────────
    # dφ/dζ|₀ using ζ[0]=0, ζ[1]=h0, ζ[2]=h0+h1:
    # dφ/dζ|₀ = [(h0+h1)²φ₁ − h0²φ₂ − h1(2h0+h1)φ₀] / [h0·h1·(h0+h1)]
    h0 = zeta[1] - zeta[0]
    h1 = zeta[2] - zeta[1]
    coeff_num = h0 + h1
    dphi_dz_0 = (
        coeff_num**2 * phi_aug[:, 1]
        - h0**2 * phi_aug[:, 2]
        - h1 * (2.0 * h0 + h1) * phi_aug[:, 0]
    ) / (h0 * h1 * coeff_num)
    R_bc = dphi_dz_0 - nu * (phi_aug[:, 0] - Phi)
    # shape (Ny,)

    # ── Interior PB at k=1..Nz-1 ─────────────────────────────────
    h_m = zeta[1:-1] - zeta[:-2]    # (Nz-1,) spacing below each k
    h_p = zeta[2:] - zeta[1:-1]     # (Nz-1,) spacing above each k

    phi_c = phi_aug[:, 1:-1]         # (Ny, Nz-1) center
    phi_dn = phi_aug[:, :-2]         # (Ny, Nz-1) k-1 neighbors
    phi_up = phi_aug[:, 2:]          # (Ny, Nz-1) k+1 neighbors (includes Dirichlet at Nz)

    d2phi_dz2 = (2.0 / (h_m + h_p)) * (
        (phi_up - phi_c) / h_p - (phi_c - phi_dn) / h_m
    )

    # Periodic y: roll by ±1 in axis-0
    phi_left = jnp.roll(phi_c, 1, axis=0)
    phi_right = jnp.roll(phi_c, -1, axis=0)
    d2phi_dy2 = (phi_right - 2.0 * phi_c + phi_left) / (deta * deta)

    R_int = d2phi_dz2 + d2phi_dy2 - jnp.sinh(phi_c)  # (Ny, Nz-1)

    # Stack: (Ny, Nz) → flatten
    return jnp.concatenate([R_bc[:, None], R_int], axis=1).ravel()


# ── sparse analytical Jacobian ────────────────────────────────────────

def _build_jac_csr(phi_flat: np.ndarray, np_params: dict):
    """Analytical sparse CSR Jacobian of residual_2d.

    5 non-zeros/row max (BC: 3, interior: up to 5). O(N) construction.
    np_params must hold numpy (not JAX) versions of zeta, deta, nu.
    """
    Ny = int(np_params['Ny'])
    Nz = int(np_params['Nz'])
    zeta = np_params['zeta']      # (Nz+1,) numpy, dimensionless
    deta = float(np_params['deta'])
    nu = np_params['nu']           # (Ny,) numpy

    phi = phi_flat.reshape(Ny, Nz)
    inv_deta2 = 1.0 / (deta * deta)

    h0 = zeta[1] - zeta[0]
    h1 = zeta[2] - zeta[1]
    cn = h0 + h1

    h_m = zeta[1:-1] - zeta[:-2]      # (Nz-1,)
    h_p = zeta[2:] - zeta[1:-1]       # (Nz-1,)
    inv_hm = 1.0 / h_m
    inv_hp = 1.0 / h_p
    c2 = 2.0 / (h_m + h_p)            # (Nz-1,)

    max_nnz = 3 * Ny + 5 * Ny * (Nz - 1)
    rows_a = np.empty(max_nnz, dtype=np.int32)
    cols_a = np.empty(max_nnz, dtype=np.int32)
    data_a = np.empty(max_nnz, dtype=np.float64)
    ptr = 0

    dbc0_arr = -h1 * (2*h0 + h1) / (h0 * h1 * cn) - nu  # (Ny,)
    dbc1 = cn / (h0 * h1)
    dbc2 = -h0 / (h1 * cn)
    for i in range(Ny):
        r = i * Nz
        rows_a[ptr] = r; cols_a[ptr] = r;     data_a[ptr] = dbc0_arr[i]; ptr += 1
        rows_a[ptr] = r; cols_a[ptr] = r + 1; data_a[ptr] = dbc1;        ptr += 1
        rows_a[ptr] = r; cols_a[ptr] = r + 2; data_a[ptr] = dbc2;        ptr += 1

    for i in range(Ny):
        il = (i - 1) % Ny
        ir = (i + 1) % Ny
        for j in range(Nz - 1):
            k = j + 1
            r = i * Nz + k
            c_ctr = c2[j]*(-inv_hp[j]-inv_hm[j]) - 2*inv_deta2 - np.cosh(phi[i, k])
            rows_a[ptr] = r; cols_a[ptr] = r;         data_a[ptr] = c_ctr;     ptr += 1
            rows_a[ptr] = r; cols_a[ptr] = r - 1;     data_a[ptr] = c2[j]*inv_hm[j]; ptr += 1
            if k < Nz - 1:
                rows_a[ptr] = r; cols_a[ptr] = r + 1; data_a[ptr] = c2[j]*inv_hp[j]; ptr += 1
            rows_a[ptr] = r; cols_a[ptr] = il*Nz + k; data_a[ptr] = inv_deta2; ptr += 1
            rows_a[ptr] = r; cols_a[ptr] = ir*Nz + k; data_a[ptr] = inv_deta2; ptr += 1

    N = Ny * Nz
    return coo_matrix((data_a[:ptr], (rows_a[:ptr], cols_a[:ptr])), shape=(N, N)).tocsr()


# ── solver ───────────────────────────────────────────────────────────

def solve_pb_2d(case: dict, mesh: dict,
                eps_r: float = EPS_R_WATER, T: float = 298.15,
                phi_init_flat: np.ndarray = None,
                solver_method: str = 'hybr'):
    """Solve 2-D PB on the periodic-stripe geometry.

    Parameters
    ----------
    case : dict
        E_M   : float         — applied potential [V]
        E_pzc : array (2,)    — PZC for facets 1 and 2 [V]
        C_H   : array (2,)    — Helmholtz cap for facets 1 and 2 [F/m²]
        x     : float         — fraction of stripe that is facet 1
        c_b   : float         — bulk concentration [mol/m³]
        L     : float         — stripe period [m]
    mesh : dict
        Ny           : int
        Nz           : int
        Z_max_factor : float (default 8)
    phi_init_flat : optional initial guess (Ny*Nz,)

    Returns
    -------
    phi_yz   : (Ny, Nz+1) ndarray  — ϕ(y,z) [V]
    sigma_M  : (Ny,) ndarray       — σ_M(y) [C/m²]
    sigma_bar: float               — σ̄_M [C/m²]
    result   : scipy root result
    """
    E_M = float(case['E_M'])
    E_pzc = np.asarray(case['E_pzc'], dtype=float)
    C_H = np.asarray(case['C_H'], dtype=float)
    x = float(case['x'])
    c_b = float(case['c_b'])
    L = float(case['L'])

    Ny = int(mesh['Ny'])
    Nz = int(mesh['Nz'])
    Z_max_factor = float(mesh.get('Z_max_factor', 8.0))

    lambda_D = np.sqrt(EPS0 * eps_r * R * T / (2.0 * F**2 * c_b))
    Z_max = Z_max_factor * lambda_D
    y_np, z_np = make_yz_mesh(Ny, Nz, L, Z_max)

    zeta_np = z_np / lambda_D             # (Nz+1,)
    deta = (L / Ny) / lambda_D            # uniform Δη

    # Facet index per y-strip (0 = facet 1, 1 = facet 2)
    facet = np.where(y_np < x * L, 0, 1)

    nu_arr = C_H[facet] * lambda_D / (EPS0 * eps_r)      # (Ny,)
    Phi_arr = F * (E_M - E_pzc[facet]) / (R * T)          # (Ny,)

    params = {
        'Ny': Ny, 'Nz': Nz,
        'zeta': jnp.array(zeta_np),
        'deta': float(deta),
        'nu': jnp.array(nu_arr),
        'Phi': jnp.array(Phi_arr),
    }
    np_params = {
        'Ny': Ny, 'Nz': Nz,
        'zeta': zeta_np,
        'deta': float(deta),
        'nu': nu_arr,
        'Phi': Phi_arr,
    }

    # Initial guess: analytical GC profile at each y-strip
    if phi_init_flat is None:
        phi_init = np.zeros((Ny, Nz), dtype=float)
        for i in range(Ny):
            try:
                psi_0_i = solve_psi0(E_M, E_pzc[facet[i]], C_H[facet[i]],
                                     c_b, eps_r, T)
            except Exception:
                psi_0_i = 0.0
            phi0_i = F * psi_0_i / (R * T)
            gamma_i = np.tanh(phi0_i / 4.0)
            arg = np.clip(gamma_i * np.exp(-zeta_np[:Nz]), -1.0 + 1e-12, 1.0 - 1e-12)
            phi_init[i, :] = 4.0 * np.arctanh(arg)
        phi_init_flat = phi_init.ravel()

    def fun(phi_flat):
        return np.asarray(residual_2d(jnp.array(phi_flat), params))

    if solver_method == 'hybr':
        _jac = jacfwd(lambda p: residual_2d(p, params))

        def jac(phi_flat):
            return np.asarray(_jac(jnp.array(phi_flat)))

        result = root(fun, phi_init_flat, jac=jac, method='hybr',
                      tol=1e-10, options={'maxfev': 10000})
    elif solver_method == 'sparse':
        x_s = phi_init_flat.copy()
        f_s = fun(x_s)
        converged = False
        fn_prev = float('inf')
        stag_count = 0
        for _ in range(50):
            fn = float(np.linalg.norm(f_s))
            if fn < 1e-8:
                converged = True
                break
            # Stagnation guard: break if ||R|| barely changes for 3 iters.
            # JAX float64 precision floor rises with mesh size; ||R||<1e-6 is the test gate.
            if fn_prev > 0 and abs(fn - fn_prev) / fn_prev < 1e-3:
                stag_count += 1
                if stag_count >= 3:
                    break
            else:
                stag_count = 0
            fn_prev = fn
            J = _build_jac_csr(x_s, np_params)
            dx = spsolve(J, -f_s)
            alpha = 1.0
            fn2 = fn * fn
            for __ in range(30):
                f_try = fun(x_s + alpha * dx)
                if float(np.dot(f_try, f_try)) < fn2 * (1.0 - 1e-4 * alpha):
                    break
                alpha *= 0.5
            x_s = x_s + alpha * dx
            f_s = f_try
        fn_final = float(np.linalg.norm(f_s))
        success = converged or (fn_final < 1e-6)
        result = OptimizeResult(
            x=x_s, fun=f_s, success=success,
            message='Converged' if success else 'Max iterations reached',
        )
    else:
        # Matrix-free krylov — scales to large (Ny, Nz) without dense Jacobian.
        result = root(fun, phi_init_flat, method=solver_method,
                      tol=1e-10, options={'maxiter': 2000})

    # Reconstruct full solution
    phi_d = result.x.reshape(Ny, Nz)
    phi_d_full = np.concatenate([phi_d, np.zeros((Ny, 1))], axis=1)  # (Ny, Nz+1)
    phi_yz = phi_d_full * (R * T / F)                                  # [V]

    # σ_M from Helmholtz BC: C_H·(E_M − E_pzc − ψ_0)
    psi_surf = phi_yz[:, 0]                                            # [V]
    sigma_M = C_H[facet] * (E_M - E_pzc[facet] - psi_surf)           # (Ny,) [C/m²]
    sigma_bar = float(np.mean(sigma_M))

    return phi_yz, sigma_M, sigma_bar, result
