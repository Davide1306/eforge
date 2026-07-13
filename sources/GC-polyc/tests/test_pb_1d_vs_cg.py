from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import pytest
from src.gc_pb_1d import solve_pb_1d
from src.gc_analytic import sigma_GC
from src import EPS0, EPS_R_WATER

C_B = 1.0    # mol/m³ = 1 mM
PSI_VALS = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]   # skip 0 (σ_GC=0 → div)


@pytest.mark.parametrize("psi_0", PSI_VALS)
def test_pb_1d_vs_cg(psi_0):
    """G3 acceptance: |σ_num − σ_GC| / |σ_GC| < 1e-3 at each ψ_0."""
    phi, sigma_num, result = solve_pb_1d(psi_0, C_B)

    # Convergence proof — MANDATORY at G3+
    assert result.success, (
        f"scipy.optimize.root failed at psi_0={psi_0} V: {result.message}"
    )
    residual_norm = float(np.linalg.norm(result.fun))
    assert residual_norm < 1e-6, (
        f"||R|| = {residual_norm:.2e} >= 1e-6 at psi_0={psi_0} V"
    )

    sigma_cg = sigma_GC(psi_0, C_B)
    rel_err = abs(sigma_num - sigma_cg) / abs(sigma_cg)
    assert rel_err < 1e-3, (
        f"psi_0={psi_0} V: σ_num={sigma_num:.4e}, σ_GC={sigma_cg:.4e}, "
        f"rel_err={rel_err:.2e} >= 1e-3\n"
        f"scipy result.success={result.success}, ||R||={residual_norm:.2e}"
    )


def test_phi_boundary():
    """ϕ(z=0) = ψ_0 and ϕ(z=Z_max) ≈ 0 (Dirichlet BCs satisfied)."""
    psi_0 = 0.2
    phi, _, result = solve_pb_1d(psi_0, C_B)
    assert abs(phi[0] - psi_0) < 1e-12
    assert abs(phi[-1]) < 1e-12


def test_phi_sign():
    """ϕ(z) has the same sign as ψ_0 everywhere (monotone decay)."""
    for psi_0 in [0.2, -0.2]:
        phi, _, _ = solve_pb_1d(psi_0, C_B)
        if psi_0 > 0:
            assert np.all(phi >= -1e-10), "phi should be >= 0 for psi_0 > 0"
        else:
            assert np.all(phi <= 1e-10), "phi should be <= 0 for psi_0 < 0"
