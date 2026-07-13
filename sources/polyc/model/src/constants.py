"""
constants.py — frozen ground-truth parameters for the Liu2026 polycrystalline-EDL
reproduction (SI Table S1; CONSTRAINTS.md §1).

ALL quantities are SI base units internally:
  length m, charge C, energy J, temperature K, number density m^-3,
  potential V_SHE, capacitance-per-area F/m^2.

No fitting (CONSTRAINTS §0): every value is fixed by Table S1 or a §1
derived-consistency relation. Nothing here is tuned to a target curve.

Unit note: the article quotes capacitance in uF/cm^2 and concentration in mM/M.
Helpers (uFcm2_to_SI / SI_to_uFcm2 / mM_to_M) convert; *_uFcm2 / *_mM names hold
the article-facing values for readability.
"""
import math

# ----- General physical constants (Table S1) -----
k_B   = 1.381e-23      # J/K
T     = 298.0          # K
e_0   = 1.6021e-19     # C
N_A   = 6.02e23        # 1/mol
eps_0 = 8.85e-12       # F/m
c_w   = 55.6           # M  (pure-water molarity)

# ----- Electrolyte (Table S1) -----
eps_opt = 1.8  * eps_0   # F/m  optical permittivity (SI ref 4)
eps_HP  = 30.0 * eps_0   # F/m  metal<->HP gap; enters only via C_H = eps_HP/l_HP (SI ref 5)
eps_S   = 78.5 * eps_0   # F/m  bulk-solution permittivity (bulk water)
d_t     = 3.1e-10        # m    lattice size = (c_w N_A)^(-1/3); table-rounded 3.1 Angstrom
gamma_p = 3.0            # ion-size factor gamma_+  = (d_+/d_t)^3
gamma_m = 3.0            # ion-size factor gamma_-
p_dip   = 1.58e-29       # C*m  solvent dipole moment (pinned; see p_from_table)

# ----- Electrode (Table S1) -----
L_default = 10e-9        # m     unit-cell length (swept 1-100 nm)
x_default = 0.5          # facet-1 fraction
E_pzc_111 = +0.3         # V_SHE  111 / "T" facet (SI refs 3,6)
E_pzc_110 = -0.1         # V_SHE  110 / "S" facet (SI ref 7)
N_tot     = 1.5029e19    # 1/m^2  = 4/(sqrt3 a_Pt^2), a_Pt = 3.92 Angstrom

# ----- Adsorption (Table S1) -----
E_ads0_default    = 0.45  # V
theta_max_default = 0.01  # fraction (1%)

# ----- article-facing defaults (per-figure overrides live in CONSTRAINTS §3) -----
C_H_default_uFcm2 = 50.0  # uF/cm^2 (Fig S3 uses 25)
c_b_default_mM    = 0.1   # mM

# ----- unit conversions -----
def uFcm2_to_SI(c):   # uF/cm^2 -> F/m^2  (1 uF/cm^2 = 1e-2 F/m^2)
    return c * 1e-2

def SI_to_uFcm2(c):   # F/m^2 -> uF/cm^2
    return c * 1e2

def mM_to_M(c):       # mM -> mol/L (M)
    return c * 1e-3

C_H_default = uFcm2_to_SI(C_H_default_uFcm2)   # 0.5 F/m^2
c_b_default = mM_to_M(c_b_default_mM)          # 1e-4 M

# ----- derived quantities (CONSTRAINTS §1 "Derived" block) -----
beta = 1.0 / (k_B * T)            # 1/J ; ~2.430e20

# Bulk solvent number density. The two equivalent forms differ ~0.3% from d_t
# rounding; we adopt 1/d_t^3 as CANONICAL because it makes the eq-3 Langevin
# low-field limit eps_eff(E->0) = eps_S EXACT (the §1 p-pin presumes
# n_w -> n_w^b with eps_S = eps_opt + n_w^b p^2 beta/3). Derived-consistency,
# not a fit. n_w_b_conc is exposed for the cross-check.
n_w_b      = 1.0 / d_t**3          # 1/m^3 ; ~3.357e28  (canonical)
n_w_b_conc = c_w * 1e3 * N_A       # 1/m^3 ; ~3.347e28  (concentration form)


def n_b(c_b_M):
    """Bulk ion number density [1/m^3] for bulk conc c_b in mol/L (M)."""
    return c_b_M * 1e3 * N_A


def v_frac(c_b_M):
    """Bulk volume fraction of solvated ions, v = 2 d_t^3 n_b (eq-2 D term)."""
    return 2.0 * d_t**3 * n_b(c_b_M)


def lambda_D(c_b_M):
    """Debye length [m], monovalent: sqrt(eps_S k_B T / (2 e_0^2 n_b))."""
    return math.sqrt(eps_S * k_B * T / (2.0 * e_0**2 * n_b(c_b_M)))


def p_from_table():
    """Re-derive the solvent dipole p = sqrt(3 d_t^3 (eps_S - eps_opt) k_B T)
    (Table S1 definition; should equal p_dip to table rounding)."""
    return math.sqrt(3.0 * d_t**3 * (eps_S - eps_opt) * k_B * T)
