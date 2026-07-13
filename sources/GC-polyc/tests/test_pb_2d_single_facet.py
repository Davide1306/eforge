"""G4 acceptance: 2-D PB single-facet limit recovers G3.

Set E_pzc^1 = E_pzc^2, C_H^1 = C_H^2 → problem is y-independent.
Checks:
  (i)  Var_y(ψ_0(y)) < 1e-6 V²  (lateral variance ~ machine noise)
  (ii) |σ̄_M_2D − σ_GC| / |σ_GC| < 1e-3
  (iii) result.success and ||R|| < 1e-6
"""

from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import pytest
from src.gc_pb_2d import solve_pb_2d
from src.gc_analytic import sigma_GC, solve_psi0

C_B = 1.0        # mol/m³ = 1 mM
C_H_VAL = 0.5   # F/m² = 50 µF/cm²
L = 10e-9        # 10 nm stripe period
E_PZC = 0.0      # [V]

# Single-facet test cases (E_M values to probe)
E_M_VALS = [-0.2, -0.1, 0.1, 0.2]

# Diagnostic mesh — fast
MESH = {'Ny': 8, 'Nz': 64, 'Z_max_factor': 8.0}


@pytest.mark.parametrize("E_M", E_M_VALS)
def test_single_facet_lateral_uniformity(E_M):
    """ψ_0(y) is laterally uniform when both facets are identical."""
    case = {
        'E_M': E_M,
        'E_pzc': [E_PZC, E_PZC],
        'C_H': [C_H_VAL, C_H_VAL],
        'x': 0.5,
        'c_b': C_B,
        'L': L,
    }
    phi_yz, sigma_M, sigma_bar, result = solve_pb_2d(case, MESH)

    assert result.success, f"root failed at E_M={E_M}: {result.message}"

    residual_norm = float(np.linalg.norm(result.fun))
    assert residual_norm < 1e-6, f"||R||={residual_norm:.2e} >= 1e-6 at E_M={E_M}"

    psi_surf = phi_yz[:, 0]
    var_psi = float(np.var(psi_surf))
    assert var_psi < 1e-6, (
        f"E_M={E_M}: Var_y(ψ_0)={var_psi:.2e} >= 1e-6 V²  "
        f"(expected y-independent for single-facet)"
    )


@pytest.mark.parametrize("E_M", E_M_VALS)
def test_single_facet_sigma_vs_gc(E_M):
    """σ̄_M(2D) agrees with Chapman-Grahame σ_GC to 1e-3 relative."""
    case = {
        'E_M': E_M,
        'E_pzc': [E_PZC, E_PZC],
        'C_H': [C_H_VAL, C_H_VAL],
        'x': 0.5,
        'c_b': C_B,
        'L': L,
    }
    phi_yz, sigma_M, sigma_bar, result = solve_pb_2d(case, MESH)

    assert result.success, f"root failed at E_M={E_M}: {result.message}"

    # Reference: GC formula at the solved ψ_0 (from single-facet transcendental)
    psi_0_ref = solve_psi0(E_M, E_PZC, C_H_VAL, C_B)
    sigma_gc = sigma_GC(psi_0_ref, C_B)

    rel_err = abs(sigma_bar - sigma_gc) / abs(sigma_gc)
    assert rel_err < 1e-3, (
        f"E_M={E_M}: σ̄_M={sigma_bar:.4e}, σ_GC={sigma_gc:.4e}, "
        f"rel_err={rel_err:.2e} >= 1e-3"
    )
