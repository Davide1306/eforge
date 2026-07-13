"""Closed-form analytic oracle for polyc_mpb2d (frozen copy of the reference
analytics + the constants the forms need).

Standalone: no external project imports. The polyc analogue of cg_oracle.py. Used by:
  - the deterministic floor (eval/validate_skeleton.py) as an analytic sanity
    cross-check of the inlined-solver minima magnitudes, and
  - the `cg_cross_check` gate (rel < 1e-3) for polyc_mpb2d.

All SI base units internally (F/m^2, V, m, 1/m^2, 1/m^3); concentrations are in
mol/L (M) where noted. Constants are transcribed VERBATIM from polyc
src/constants.py (Table S1); NOTE polyc T = 298.0 (NOT 298.15).

eq 10  C_GC(E, E_pzc, lam_D)   single-facet Gouy-Chapman differential capacitance
eq 11  C_dl_GCS(C_H, C_gc)     series 1/C_dl = 1/C_H + 1/C_GC
eq 13  C_ads_peak(theta_max)   closed-form Langmuir adsorption-capacitance peak
"""

import math

import numpy as np

# ----- General physical constants (polyc src/constants.py, Table S1) -----
k_B = 1.381e-23          # J/K
T = 298.0                # K   (polyc value; NOT 298.15)
e_0 = 1.6021e-19         # C
N_A = 6.02e23            # 1/mol
eps_0 = 8.85e-12         # F/m

# ----- Electrolyte / solvent permittivity (Table S1) -----
eps_S = 78.5 * eps_0     # F/m  bulk-solution permittivity (= 6.94725e-10)

# ----- Electrode (Table S1) -----
N_tot = 1.5029e19        # 1/m^2  surface site density = 4/(sqrt3 a_Pt^2)

# ----- derived (CONSTRAINTS section 1 "Derived") -----
beta = 1.0 / (k_B * T)   # 1/J ; ~2.4297e20


def lambda_D(c_b_M):
    """polyc Debye length [m] for bulk conc c_b in mol/L (M):
    sqrt(eps_S k_B T / (2 e_0^2 n_b)) with n_b = c_b_M * 1e3 * N_A [1/m^3]."""
    n_b = c_b_M * 1e3 * N_A
    return math.sqrt(eps_S * k_B * T / (2.0 * e_0**2 * n_b))


def C_GC(E, E_pzc, lam_D):
    """eq 10: C_GC(E) = (eps_S/lam_D) cosh(e_0 (E - E_pzc)/(2 k_B T)) [F/m^2].
    Minimum eps_S/lam_D at E = E_pzc; symmetric and convex in (E - E_pzc)."""
    E = np.asarray(E, dtype=float)
    return (eps_S / lam_D) * np.cosh(e_0 * (E - E_pzc) / (2.0 * k_B * T))


def C_dl_GCS(C_H, C_gc):
    """eq 11: series capacitance 1/C_dl = 1/C_H + 1/C_GC [F/m^2]."""
    return 1.0 / (1.0 / C_H + 1.0 / C_gc)


def theta_ads(E, E_ads0, c_b_Aminus):
    """eq 12 closed form: theta = sigma/(1+sigma),
    sigma = c_b_Aminus * exp(beta e_0 (E - E_ads0))."""
    E = np.asarray(E, dtype=float)
    sig = c_b_Aminus * np.exp(beta * e_0 * (E - E_ads0))
    return sig / (1.0 + sig)


def dtheta_dE(E, E_ads0, c_b_Aminus):
    """d theta/dE = beta e_0 theta (1 - theta)."""
    th = theta_ads(E, E_ads0, c_b_Aminus)
    return beta * e_0 * th * (1.0 - th)


def C_ads(E, E_ads0, c_b_Aminus, theta_max, N_tot_=None):
    """eq 13: C_ads = theta_max e_0 N_tot d theta/dE [F/m^2]."""
    Nt = N_tot if N_tot_ is None else N_tot_
    return theta_max * e_0 * Nt * dtheta_dE(E, E_ads0, c_b_Aminus)


def C_ads_peak(theta_max, N_tot_=None):
    """eq-13 closed-form peak (at theta = 1/2): theta_max e_0^2 N_tot beta / 4
    [F/m^2]. ~23.4 uF/cm^2 at theta_max = 1%."""
    Nt = N_tot if N_tot_ is None else N_tot_
    return theta_max * e_0**2 * Nt * beta / 4.0


# ----- unit helpers (article-facing cross-checks) -----
def mM_to_M(c):    # mM -> mol/L (M)
    return c * 1e-3


def SI_to_uFcm2(c):  # F/m^2 -> uF/cm^2  (1 uF/cm^2 = 1e-2 F/m^2)
    return c * 1e2
