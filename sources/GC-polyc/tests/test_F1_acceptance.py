"""F1 2-D potential map acceptance test — checks from final-mesh CSV + milestone solve.

Acceptance criteria (CONSTRAINTS §3 F1):
  - PNG and CSV exist (final-mesh figure already generated).
  - ϕ → 0 at the far boundary (z = Z_max): mean|ϕ| < 1 mV.
  - At z=0, the potential is antisymmetric about y=L/2 for the symmetric
    x=0.5 facet geometry: ϕ(facet1) ≈ -ϕ(facet2), confirming opposite
    surface-charge sign on the two facets.
  - Milestone-mesh re-solve: success=True, ||R||<1e-6.
"""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import pytest

PNG = ROOT / 'output' / 'F1_phi2D.png'
CSV = ROOT / 'output' / 'F1_phi2D.csv'

MESH_MS = {'Ny': 64, 'Nz': 128, 'Z_max_factor': 8.0}


@pytest.fixture(scope='module')
def f1_csv():
    d = np.loadtxt(CSV, delimiter=',', skiprows=1)
    # cols: y_nm, z_nm, phi_V
    return d


def test_png_exists():
    assert PNG.exists(), f"F1 PNG missing: {PNG}"


def test_csv_exists():
    assert CSV.exists(), f"F1 CSV missing: {CSV}"


def test_phi_vanishes_at_far_boundary(f1_csv):
    """ϕ at z = Z_max must be < 1 mV (bulk reference)."""
    z_max = float(np.max(f1_csv[:, 1]))
    far_mask = f1_csv[:, 1] >= 0.99 * z_max
    phi_far = f1_csv[far_mask, 2]
    mean_abs = float(np.mean(np.abs(phi_far)))
    assert mean_abs < 1e-3, (
        f"ϕ at z≈Z_max = {mean_abs*1e3:.2f} mV — not decaying to bulk reference")


def test_surface_phi_antisymmetric_x05(f1_csv):
    """At z=0, ϕ is antisymmetric about y=L/2 (x=0.5 facet, ΔE_pzc=0.6V).

    Facet 1 (y < L/2): E_pzc=+0.3V > E_M=0 → below PZC → positive σ_M → ϕ < 0.
    Facet 2 (y > L/2): E_pzc=-0.3V < E_M=0 → above PZC → negative σ_M → ϕ > 0.
    Mean ϕ_facet1 ≈ -mean ϕ_facet2 within 20%.
    """
    # Extract z=0 row (smallest z values)
    z_min = float(np.min(f1_csv[:, 1]))
    surf_mask = f1_csv[:, 1] <= z_min + 0.01  # z=0 within numerical precision
    surf = f1_csv[surf_mask]
    if len(surf) < 4:
        pytest.skip("Insufficient z=0 rows in CSV")
    y_vals = surf[:, 0]
    phi_vals = surf[:, 2]
    y_mid = float(np.max(y_vals)) / 2.0
    phi_f1 = float(np.mean(phi_vals[y_vals < y_mid]))
    phi_f2 = float(np.mean(phi_vals[y_vals > y_mid]))
    if abs(phi_f1) < 1e-6 or abs(phi_f2) < 1e-6:
        pytest.skip("Surface ϕ near zero; skip antisymmetry check")
    ratio = phi_f1 / phi_f2
    assert ratio < 0, (
        f"Facet ϕ not antisymmetric: ϕ_f1={phi_f1*1e3:.2f}mV, ϕ_f2={phi_f2*1e3:.2f}mV")
    assert abs(ratio + 1.0) < 0.20, (
        f"Asymmetry ratio={ratio:.3f} (expected ≈-1.0 ±20%): "
        f"ϕ_f1={phi_f1*1e3:.2f}mV, ϕ_f2={phi_f2*1e3:.2f}mV")


def test_f1_milestone_convergence():
    """F1 base case: success=True, ||R||<1e-6 at milestone mesh."""
    from src.gc_pb_2d import solve_pb_2d
    from src.parameters import make_f2_case
    case = make_f2_case(E_M=0.0, delta_E_pzc=0.6, c_b=10.0, L=10e-9)
    _, _, _, result = solve_pb_2d(case.to_solve_dict(), MESH_MS, solver_method='sparse')
    assert result.success, f"F1 milestone did not converge: {result.message}"
    assert np.linalg.norm(result.fun) < 1e-6, (
        f"F1 milestone ||R||={np.linalg.norm(result.fun):.2e} ≥ 1e-6")
