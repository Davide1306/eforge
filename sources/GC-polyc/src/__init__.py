"""GC-polyc — Gouy-Chapman model of the polycrystalline-electrode EDL.

Reproduces F1-F5 of:
  Liu, J.; Doblhoff-Dier, K.; Koper, M. T. M.
  "Modelling the Double Layer of Polycrystalline Electrodes:
   Capacitance, Potential of Zero Charge, and Parsons-Zobel Plot."
  ACS Electrochem. 2026, 2, 995-1004.
  DOI: 10.1021/acselectrochem.5c00544

Classical Gouy-Chapman ONLY (no Booth/Bikerman/adsorption — see
reference/CONSTRAINTS.md §0 and §4 for the OUT-OF-SCOPE list).
"""

# --- Physical constants (SI) ---
EPS0 = 8.8541878128e-12      # F/m, vacuum permittivity
F    = 96485.33212           # C/mol, Faraday constant
R    = 8.31446261815324      # J/(mol K), gas constant
NA   = 6.02214076e23         # 1/mol, Avogadro
E_CHARGE = 1.602176634e-19   # C, elementary charge
K_B  = 1.380649e-23          # J/K, Boltzmann
T    = 298.15                # K, default temperature

# --- Solvent (water at T=298.15 K) ---
EPS_R_WATER = 78.5           # relative permittivity per article SI Table S1 (was 78.4 generic literature)

# --- Default geometry/material ---
C_H_DEFAULT = 50e-6 * 1e4    # 50 µF/cm² → SI: F/m²

__all__ = [
    "EPS0", "F", "R", "NA", "E_CHARGE", "K_B", "T",
    "EPS_R_WATER", "C_H_DEFAULT",
]
