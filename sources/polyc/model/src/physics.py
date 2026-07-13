"""
physics.py — pointwise constitutive kernel (CONSTRAINTS §2 eqs 2-4), called by
the modified-Poisson-Boltzmann residual at every grid point.

Given the local potential phi_S [V] and field magnitude E = |grad phi_S| [V/m]
(and the bulk inputs n_b, v for the electrolyte), it returns:
  denominator_D    eq 2 shared denominator D
  number_densities eq 2 ion densities n_+, n_-
  n_water          eq 4 local water density  (E1 fix: D_4 == D, see below)
  eps_eff          eq 3 field-dependent effective permittivity (Langevin)

E1 fix (CONSTRAINTS erratum E1): eq 4 as printed carries a spurious n_w^b inside
its denominator D_4, breaking dimensional consistency and the bulk limit. We
implement D_4 == D (eq 2's denominator); then n_w(0,0) = n_w^b exactly and the
eq-3 low-field closure eps_eff(0) = eps_S holds (the Table-S1 p-pin).

Dipole moment: P_DIP is the Table-S1 *defining formula*
p = sqrt(3 d_t^3 (eps_S - eps_opt) k_B T) at full precision (1.58007e-29 C*m), of
which the table's 1.58e-29 is the 3-sig-fig display. Using the formula value (not
the rounded display) makes the eq-3 closure eps_eff(E->0) = eps_S exact. No fitting.

All functions accept scalars or numpy arrays for (phi, E) and are vectorized.
"""
import numpy as np

from . import constants as C

# canonical solvent dipole (defining formula, full precision)
P_DIP = C.p_from_table()

# Overflow guard: clip the dimensionless exponent beta*e0*phi (and beta*p*E) so
# exp/sinh never overflow on a Newton solver's intermediate excursions. The
# physical solution has |phi| < 1 V (|arg| < 40); clipping at ~5 V (arg 200) is
# far outside the physical regime and never alters a converged result -- it only
# keeps a bad trial step finite instead of inf/nan.
_EXP_CLAMP = 200.0


def _bphi(phi):
    return np.clip(C.beta * C.e_0 * np.asarray(phi, dtype=float), -_EXP_CLAMP, _EXP_CLAMP)


def sinhc(u):
    """sinh(u)/u, with a small-u series (-> 1 at u = 0)."""
    u = np.clip(np.asarray(u, dtype=float), -_EXP_CLAMP, _EXP_CLAMP)
    scalar = (u.ndim == 0)
    u = np.atleast_1d(u)
    out = np.empty_like(u)
    small = np.abs(u) < 1e-4
    us = u[small]
    out[small] = 1.0 + us**2 / 6.0 + us**4 / 120.0
    ub = u[~small]
    out[~small] = np.sinh(ub) / ub
    return float(out[0]) if scalar else out


def langevin_L(u):
    """Langevin function L(u) = coth(u) - 1/u, with small-u series (-> u/3)."""
    u = np.asarray(u, dtype=float)
    scalar = (u.ndim == 0)
    u = np.atleast_1d(u)
    out = np.empty_like(u)
    small = np.abs(u) < 1e-4
    us = u[small]
    out[small] = us / 3.0 - us**3 / 45.0
    ub = u[~small]
    out[~small] = 1.0 / np.tanh(ub) - 1.0 / ub
    return float(out[0]) if scalar else out


def langevin_Lc(u):
    """L(u)/u, smooth at u = 0 (-> 1/3). Used by eps_eff so the E -> 0 limit needs
    no division by the field."""
    u = np.asarray(u, dtype=float)
    scalar = (u.ndim == 0)
    u = np.atleast_1d(u)
    out = np.empty_like(u)
    small = np.abs(u) < 1e-3
    us = u[small]
    out[small] = 1.0 / 3.0 - us**2 / 45.0 + 2.0 * us**4 / 945.0
    ub = u[~small]
    out[~small] = (1.0 / np.tanh(ub) - 1.0 / ub) / ub
    return float(out[0]) if scalar else out


def denominator_D(phi, E, v):
    """eq 2 shared denominator D(phi, E):
        D = (v/2)(g+ e^{-b e0 phi} + g- e^{+b e0 phi})
            + [1 - (v/2)(g+ + g-)] * sinh(bpE)/(bpE).
    Note D(0,0) = 1 for any v."""
    bphi = _bphi(phi)
    S = sinhc(C.beta * P_DIP * np.asarray(E, dtype=float))
    return (v / 2.0) * (C.gamma_p * np.exp(-bphi) + C.gamma_m * np.exp(bphi)) \
        + (1.0 - (v / 2.0) * (C.gamma_p + C.gamma_m)) * S


def number_densities(phi, E, n_b, v):
    """eq 2: n_+- = n_b e^{-+ b e0 phi} / D. Returns (n_plus, n_minus)."""
    bphi = _bphi(phi)
    D = denominator_D(phi, E, v)
    return n_b * np.exp(-bphi) / D, n_b * np.exp(bphi) / D


def n_water(phi, E, v):
    """eq 4 (E1 fix D_4 == D): n_w = n_w^b * [sinh(bpE)/(bpE)] / D."""
    S = sinhc(C.beta * P_DIP * np.asarray(E, dtype=float))
    return C.n_w_b * S / denominator_D(phi, E, v)


def eps_eff(phi, E, v):
    """eq 3: eps_eff = eps_opt + (n_w p / E) L(bpE)
            = eps_opt + n_w beta p^2 Lc(bpE)   (Lc = L/u, smooth at E = 0).
    Low-field limit eps_eff(0,0) = eps_S exactly."""
    u = C.beta * P_DIP * np.asarray(E, dtype=float)
    return C.eps_opt + n_water(phi, E, v) * C.beta * P_DIP**2 * langevin_Lc(u)
