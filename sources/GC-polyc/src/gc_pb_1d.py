"""1-D nonlinear Poisson-Boltzmann FD solver (GC-polyc tier G3).

Works in dimensionless variables φ = Fϕ/RT and ζ = z/λ_D so the
residual is O(1) and scipy tolerances are meaningful.
Dimensionless PB: d²φ/dζ² = sinh(φ).

σ_M extracted via degree-4 polyfit through phi_full[0:5] (5-pt Lagrange
forward FD, O(h⁴)); alpha=4.0 tanh mesh concentrates nodes near z=0.
Expected rel_err vs σ_GC < 1e-3 at |ψ₀| ≤ 0.3 V (non-zero — FD exercised).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from scipy.optimize import root

from src import EPS0, F, R, EPS_R_WATER

jax.config.update("jax_enable_x64", True)


# ── mesh ─────────────────────────────────────────────────────────────

def make_z_mesh(Nz: int, Z_max: float, stretch: str = "tanh", alpha: float = 3.0):
    """Non-uniform z mesh on [0, Z_max], refined near z=0 [m].

    Returns array of shape (Nz+1,): z[0]=0, z[Nz]=Z_max.
    """
    i = np.arange(Nz + 1, dtype=float)
    if stretch == "tanh":
        # Flip so fine spacing is at z=0 (electrode side).
        z = Z_max * (1.0 - np.tanh(alpha * (Nz - i) / Nz) / np.tanh(alpha))
    else:
        z = Z_max * i / Nz
    return z


# ── residual in dimensionless variables (JAX) ────────────────────────

def residual(phi_int: jnp.ndarray, phi0: float, zeta: jnp.ndarray):
    """Dimensionless 1-D PB residual on interior nodes.

    Dimensionless form: d²φ/dζ² − 2 sinh(φ) = 0
      where φ = Fϕ/RT, ζ = z/λ_D.
    BCs: φ(ζ=0) = phi0, φ(ζ_max) = 0.

    Parameters
    ----------
    phi_int : (Nz-1,) JAX array  — unknowns at ζ[1]..ζ[Nz-1]
    phi0    : float  — Dirichlet BC φ(ζ=0) = Fψ_0/RT
    zeta    : (Nz+1,) JAX array  — dimensionless mesh ζ = z/λ_D
    """
    phi = jnp.concatenate([jnp.array([phi0]), phi_int, jnp.array([0.0])])
    h_m = zeta[1:-1] - zeta[:-2]
    h_p = zeta[2:]   - zeta[1:-1]
    d2phi = 2.0 / (h_m + h_p) * (
        (phi[2:] - phi[1:-1]) / h_p - (phi[1:-1] - phi[:-2]) / h_m
    )
    # Dimensionless PB: d²φ/dζ² = sinh(φ)  [factor 2 from dimensional cancels]
    return d2phi - jnp.sinh(phi[1:-1])


# ── solver ───────────────────────────────────────────────────────────

def solve_pb_1d(psi_0: float, c_b: float,
                Nz: int = 128, Z_max_factor: float = 8.0,
                eps_r: float = EPS_R_WATER, T: float = 298.15):
    """Solve 1-D PB with Dirichlet BCs ϕ(0)=ψ_0, ϕ(Z_max)=0.

    Solves internally in dimensionless variables; returns physical units.

    Returns
    -------
    phi    : (Nz+1,) ndarray  — ϕ(z) [V] on full mesh
    sigma  : float            — σ_M [C/m²] via forward-diff at z=0
    result : scipy root result  — .success, .fun (dimensionless residual)
    """
    lambda_D = np.sqrt(EPS0 * eps_r * R * T / (2.0 * F**2 * c_b))
    Z_max = Z_max_factor * lambda_D
    # alpha=4.0: denser mesh near z=0; required for 5-pt polyfit to reach rel_err<1e-3
    z_np = make_z_mesh(Nz, Z_max, alpha=4.0)

    # Dimensionless mesh and BC
    zeta_np = z_np / lambda_D
    phi0_d = F * psi_0 / (R * T)
    zeta_jax = jnp.array(zeta_np)

    # Initial guess: exact GC solution (much better than exp(-ζ) for large |φ_0|)
    phi_init = 4.0 * np.arctanh(np.tanh(phi0_d / 4.0) * np.exp(-zeta_np[1:-1]))

    def fun(phi_flat):
        return np.asarray(residual(jnp.array(phi_flat), phi0_d, zeta_jax))

    _jac = jacfwd(lambda p: residual(p, phi0_d, zeta_jax))

    def jac(phi_flat):
        return np.asarray(_jac(jnp.array(phi_flat)))

    result = root(fun, phi_init, jac=jac, method="hybr",
                  tol=1e-10, options={"maxfev": 4000})

    # Convert back to physical units
    phi_d_full = np.concatenate([[phi0_d], result.x, [0.0]])
    phi_full = phi_d_full * (R * T / F)

    # σ_M = -ε₀εᵣ ∂ϕ/∂z|_{z=0} via degree-4 polyfit through phi_full[0:5]
    # (5-pt Lagrange forward FD, O(h^4)); coefs[1] = dϕ/dz at z=0
    coefs = np.polynomial.polynomial.polyfit(z_np[:5], phi_full[:5], 4)
    sigma_M = -EPS0 * eps_r * coefs[1]
    return phi_full, sigma_M, result
