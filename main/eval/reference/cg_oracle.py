"""Chapman-Grahame closed-form oracle (frozen copy of GC-polyc src/gc_analytic.py).

Standalone: no imports from GC-polyc. Used by fast gates as the analytic
ground truth for single-facet configurations. All SI; c_b in mol/m^3
(1 mM = 1.0 numerically).
"""

import numpy as np
from scipy.optimize import brentq

EPS0 = 8.8541878128e-12
F = 96485.33212
R = 8.31446261815324
EPS_R_WATER = 78.5


def sigma_GC(psi_0, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Diffuse-layer surface charge density [C/m^2]."""
    return np.sqrt(8.0 * EPS0 * eps_r * R * T * c_b) * np.sinh(
        F * psi_0 / (2.0 * R * T))


def C_GC(psi_0, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Diffuse-layer capacitance [F/m^2]."""
    return C_DH(c_b, eps_r, T) * np.cosh(F * psi_0 / (2.0 * R * T))


def C_DH(c_b, eps_r=EPS_R_WATER, T=298.15):
    """Debye-Hueckel capacitance [F/m^2] = eps0*eps_r/lambda_D."""
    return np.sqrt(2.0 * F**2 * EPS0 * eps_r * c_b / (R * T))


def solve_psi0(E_M, E_pzc, C_H, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Solve sigma_GC(psi_0) = C_H (E_M - E_pzc - psi_0) for psi_0 [V]."""
    def f(psi):
        return sigma_GC(psi, c_b, eps_r, T) - C_H * (E_M - E_pzc - psi)
    return brentq(f, -3.0, 3.0, xtol=1e-12, rtol=1e-12)


def C_dl_single(E_M, E_pzc, C_H, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Total single-facet differential capacitance [F/m^2] (series C_H, C_GC)."""
    psi_0 = solve_psi0(E_M, E_pzc, C_H, c_b, eps_r, T)
    return 1.0 / (1.0 / C_H + 1.0 / C_GC(psi_0, c_b, eps_r, T))


def lambda_D(c_b, eps_r=EPS_R_WATER, T=298.15):
    """Debye length [m]."""
    return np.sqrt(EPS0 * eps_r * R * T / (2.0 * F**2 * c_b))
