"""
analytics.py — closed-form relations from CONSTRAINTS §2 used as (a) solver
cross-checks and (b) the adsorption deliverable building block.

  eq 10  C_GC(E)        single-facet Gouy-Chapman differential capacitance
  eq 11  GCS series     1/C_dl = 1/C_H + 1/C_GC
  eq 12  theta_ads(E)   Langmuir coverage (closed form)
  eq 13  C_ads(E)       Langmuir adsorption capacitance; C_ads_peak its maximum

All SI units (F/m^2, V, m, 1/m^2). The eq-10 GC capacitance is the CG cross-check
required by every G3+ convergence proof (it is the analytic dilute single-facet
limit of the full model).

Tracked ambiguity E4 (eq 12 c_b^(A-) standard state) is deferred: callers pass
c_b_Aminus explicitly. The PEAK value C_ads_peak is independent of E4 (it depends
only on theta_max, N_tot, beta), so it is pinned unambiguously at G1.
"""
import numpy as np

from . import constants as C


def C_GC(E, E_pzc, lam_D):
    """eq 10: C_GC(E) = (eps_S/lam_D) cosh(e_0 (E - E_pzc)/(2 k_B T)).
    Minimum eps_S/lam_D at E = E_pzc; symmetric and convex in (E - E_pzc)."""
    E = np.asarray(E, dtype=float)
    return (C.eps_S / lam_D) * np.cosh(C.e_0 * (E - E_pzc) / (2.0 * C.k_B * C.T))


def C_dl_GCS(C_H, C_gc):
    """eq 11: series capacitance 1/C_dl = 1/C_H + 1/C_GC."""
    return 1.0 / (1.0 / C_H + 1.0 / C_gc)


def theta_ads(E, E_ads0, c_b_Aminus):
    """eq 12 closed form: theta = sigma/(1+sigma),
    sigma = c_b_Aminus * exp(beta e_0 (E - E_ads0))."""
    E = np.asarray(E, dtype=float)
    sigma = c_b_Aminus * np.exp(C.beta * C.e_0 * (E - E_ads0))
    return sigma / (1.0 + sigma)


def dtheta_dE(E, E_ads0, c_b_Aminus):
    """d theta/dE = beta e_0 theta (1 - theta)."""
    th = theta_ads(E, E_ads0, c_b_Aminus)
    return C.beta * C.e_0 * th * (1.0 - th)


def C_ads(E, E_ads0, c_b_Aminus, theta_max, N_tot=None):
    """eq 13: C_ads = theta_max e_0 N_tot d theta/dE (positive for oxidative ads)."""
    Nt = C.N_tot if N_tot is None else N_tot
    return theta_max * C.e_0 * Nt * dtheta_dE(E, E_ads0, c_b_Aminus)


def C_ads_peak(theta_max, N_tot=None):
    """Closed-form peak (at theta = 1/2): C_ads_peak = theta_max e_0^2 N_tot beta/4.
    ~23.4 uF/cm^2 at theta_max = 1%."""
    Nt = C.N_tot if N_tot is None else N_tot
    return theta_max * C.e_0**2 * Nt * C.beta / 4.0
