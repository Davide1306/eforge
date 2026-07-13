"""Chapman-Grahame closed-form single-facet EDL (GC-polyc tier G2).

Equations from reference/CONSTRAINTS.md §2 (cross-check kernel for G3+).
All quantities in SI: c_b [mol/m³] == c_b [mM] numerically (1 mM = 1 mol/m³).
"""

import numpy as np
from scipy.optimize import brentq
from src import EPS0, F, R, EPS_R_WATER


def sigma_GC(psi_0, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Diffuse-layer surface charge density [C/m²].

    CONSTRAINTS §2: σ_GC(ψ_0) = √(8 ε₀ε_r RT c_b) · sinh(F ψ_0 / 2RT)
    Sign follows from sinh being odd: σ_GC > 0 when ψ_0 > 0.

    Parameters
    ----------
    psi_0 : float  Diffuse-layer potential at z=0 [V]
    c_b   : float  Bulk concentration [mol/m³]; 1 mM = 1 mol/m³
    """
    prefactor = np.sqrt(8.0 * EPS0 * eps_r * R * T * c_b)
    return prefactor * np.sinh(F * psi_0 / (2.0 * R * T))


def C_GC(psi_0, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Diffuse-layer capacitance [F/m²].

    CONSTRAINTS §2: C_GC = √(2 F² ε₀ε_r c_b / RT) · cosh(F ψ_0 / 2RT)
    Equals the DH limit C_DH at ψ_0 = 0.

    Parameters
    ----------
    psi_0 : float  Diffuse-layer potential [V]
    c_b   : float  Bulk concentration [mol/m³]
    """
    C_DH = np.sqrt(2.0 * F**2 * EPS0 * eps_r * c_b / (R * T))
    return C_DH * np.cosh(F * psi_0 / (2.0 * R * T))


def C_DH(c_b, eps_r=EPS_R_WATER, T=298.15):
    """Debye-Hückel (linearised GC) capacitance [F/m²] = ε₀ε_r / λ_D."""
    return np.sqrt(2.0 * F**2 * EPS0 * eps_r * c_b / (R * T))


def solve_psi0(E_M, E_pzc, C_H, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Solve σ_GC(ψ_0) = C_H · (E_M − E_pzc − ψ_0) for ψ_0 [V].

    CONSTRAINTS §2 BC: -ε₀ε_r (∂ϕ/∂z)|_{z=0} = C_H(E_M − E_pzc − ψ_0).
    """
    def f(psi):
        return sigma_GC(psi, c_b, eps_r, T) - C_H * (E_M - E_pzc - psi)

    return brentq(f, -3.0, 3.0, xtol=1e-12, rtol=1e-12)


def C_dl_single(E_M, E_pzc, C_H, c_b, eps_r=EPS_R_WATER, T=298.15):
    """Total single-facet differential capacitance [F/m²].

    Series combination: 1/C_dl = 1/C_H + 1/C_GC(ψ_0).
    """
    psi_0 = solve_psi0(E_M, E_pzc, C_H, c_b, eps_r, T)
    Cgc = C_GC(psi_0, c_b, eps_r, T)
    return 1.0 / (1.0 / C_H + 1.0 / Cgc)
