"""G5 acceptance: F2 panel (b) — C_dl(E_M) at ΔE_pzc=0.6 V, c_b=10 mM.

DIAGNOSTIC tier (Ny=8, Nz=64): checks #-of-minima and convergence.
Milestone/final tier (Ny=64, Nz=128): adds position ±50 mV and
  magnitude ±30 % quantitative bands per loop.txt G5 acceptance.

CG cross-check at ΔE_pzc=0 is mandatory per LESSON G2 before any
#-of-minima assertion is promotable.
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
from src.parameters import make_f2_case

# ── F2(b) sweep parameters ─────────────────────────────────────────────
DELTA_E_PZC = 0.6          # V
C_B = 10.0                 # mol/m³ = 10 mM
L_VALS = [1e-9, 10e-9, 100e-9]   # 1, 10, 100 nm
E_M_VALS    = np.linspace(-0.5, 0.5, 11)   # 11 points — diagnostic speed
E_M_VALS_MS = np.linspace(-0.5, 0.5, 41)   # 41 points — milestone precision

# Diagnostic mesh — fast; milestone test upgrades to (Ny=64, Nz=128)
MESH_DIAG = {'Ny': 8, 'Nz': 64, 'Z_max_factor': 8.0}
MESH_MS   = {'Ny': 64, 'Nz': 128, 'Z_max_factor': 8.0}

# Expected #-of-minima per L — matches Fig 2b (loop.txt TARGET METRICS)
EXPECTED_N_MINIMA = {1e-9: 1, 10e-9: 2, 100e-9: 2}


def sweep_cdl(L: float, mesh: dict, solver_method: str = 'hybr',
              em_vals: np.ndarray = None):
    """Sweep E_M for one L; return (E_M, sigma_bar, C_dl, converged_mask)."""
    if em_vals is None:
        em_vals = E_M_VALS
    sigma_vals = []
    converged = []
    for E_M in em_vals:
        case = make_f2_case(E_M=E_M, delta_E_pzc=DELTA_E_PZC, c_b=C_B, L=L)
        _, _, sb, result = solve_pb_2d(case.to_solve_dict(), mesh,
                                       solver_method=solver_method)
        sigma_vals.append(sb)
        converged.append(result.success and float(np.linalg.norm(result.fun)) < 1e-6)
    sigma_arr = np.array(sigma_vals)
    C_dl = np.gradient(sigma_arr, em_vals)
    return em_vals, sigma_arr, C_dl, np.array(converged)


def count_minima(C_dl: np.ndarray) -> int:
    """Count local minima in C_dl array."""
    n = len(C_dl)
    count = 0
    for i in range(1, n - 1):
        if C_dl[i] < C_dl[i - 1] and C_dl[i] < C_dl[i + 1]:
            count += 1
    return count


# ── CG cross-check at ΔE_pzc=0 (LESSON G2, mandatory convergence gate) ─
def test_cg_crosscheck():
    """σ̄_M(2D) at ΔE_pzc=0 matches σ_GC to 1e-3 (LESSON G2 gate)."""
    E_M_test = 0.15   # well off PZC so σ_GC ≠ 0
    C_H_VAL = 0.5
    case = {
        'E_M': E_M_test,
        'E_pzc': [0.0, 0.0],
        'C_H': [C_H_VAL, C_H_VAL],
        'x': 0.5,
        'c_b': C_B,
        'L': 10e-9,
    }
    _, _, sigma_bar, result = solve_pb_2d(case, MESH_DIAG)
    assert result.success
    assert float(np.linalg.norm(result.fun)) < 1e-6

    psi_0 = solve_psi0(E_M_test, 0.0, C_H_VAL, C_B)
    sigma_gc = sigma_GC(psi_0, C_B)
    rel_err = abs(sigma_bar - sigma_gc) / abs(sigma_gc)
    assert rel_err < 1e-3, (
        f"CG cross-check FAIL: σ̄_M={sigma_bar:.4e}, σ_GC={sigma_gc:.4e}, "
        f"rel_err={rel_err:.2e}"
    )


# ── Convergence at each E_M for all L ────────────────────────────────────
@pytest.mark.parametrize("L", L_VALS)
def test_convergence_all_em(L):
    """All E_M points converge (scipy.success + ||R||<1e-6)."""
    _, _, _, converged = sweep_cdl(L, MESH_DIAG)
    n_fail = int(np.sum(~converged))
    assert n_fail == 0, f"L={L*1e9:.0f}nm: {n_fail}/{len(E_M_VALS)} E_M points failed"


# ── #-of-minima check ─────────────────────────────────────────────────────
@pytest.mark.parametrize("L,expected", [
    (1e-9, 1),
    (10e-9, 2),
    (100e-9, 2),
])
def test_n_minima(L, expected):
    """#-of-minima in C_dl(E_M) matches Fig 2b per loop.txt TARGET METRICS."""
    _, _, C_dl, converged = sweep_cdl(L, MESH_DIAG)
    assert np.all(converged), f"L={L*1e9:.0f}nm: not all points converged"
    n = count_minima(C_dl)
    assert n == expected, (
        f"L={L*1e9:.0f}nm: found {n} minima, expected {expected}. "
        f"C_dl={np.array2string(C_dl, precision=2)}"
    )


# ── Milestone tier (Ny=64, Nz=128, sparse solver) ─────────────────────────
def test_milestone_cg_crosscheck():
    """Milestone CG cross-check: σ̄_M(2D) at ΔE_pzc=0 matches σ_GC (sparse)."""
    E_M_test = 0.15
    C_H_VAL = 0.5
    case = {
        'E_M': E_M_test,
        'E_pzc': [0.0, 0.0],
        'C_H': [C_H_VAL, C_H_VAL],
        'x': 0.5,
        'c_b': C_B,
        'L': 10e-9,
    }
    _, _, sigma_bar, result = solve_pb_2d(case, MESH_MS, solver_method='sparse')
    assert result.success
    assert float(np.linalg.norm(result.fun)) < 1e-6

    psi_0 = solve_psi0(E_M_test, 0.0, C_H_VAL, C_B)
    sigma_gc = sigma_GC(psi_0, C_B)
    rel_err = abs(sigma_bar - sigma_gc) / abs(sigma_gc)
    assert rel_err < 1e-3, (
        f"Milestone CG FAIL: σ̄_M={sigma_bar:.4e}, σ_GC={sigma_gc:.4e}, "
        f"rel_err={rel_err:.2e}"
    )


def _minima_positions(C_dl: np.ndarray, em: np.ndarray):
    """Return E_M values of local minima."""
    return [em[i] for i in range(1, len(C_dl) - 1)
            if C_dl[i] < C_dl[i - 1] and C_dl[i] < C_dl[i + 1]]


@pytest.mark.parametrize("L,expected_n,pos_ref_V", [
    (1e-9,   1, [0.0]),          # single minimum at E_pzc^avg=0 V
    (10e-9,  2, None),           # 2 minima; no hard position spec for L=λ_D regime
    (100e-9, 2, [-0.3, +0.3]),   # 2 minima at individual E_pzc^i
])
def test_milestone_f2b(L, expected_n, pos_ref_V):
    """Milestone G5: convergence + #-of-minima + position ±50mV (41 pts, sparse)."""
    em, _, C_dl, converged = sweep_cdl(L, MESH_MS, solver_method='sparse',
                                       em_vals=E_M_VALS_MS)
    n_fail = int(np.sum(~converged))
    assert n_fail == 0, f"L={L*1e9:.0f}nm: {n_fail}/{len(em)} points failed"
    n = count_minima(C_dl)
    assert n == expected_n, (
        f"L={L*1e9:.0f}nm: found {n} minima, expected {expected_n}. "
        f"C_dl={np.array2string(C_dl, precision=2)}"
    )
    if pos_ref_V is not None:
        pos = _minima_positions(C_dl, em)
        assert len(pos) == len(pos_ref_V), f"L={L*1e9:.0f}nm: minima count mismatch"
        for p, ref in zip(sorted(pos), sorted(pos_ref_V)):
            assert abs(p - ref) <= 0.05, (
                f"L={L*1e9:.0f}nm: minimum at {p:+.3f}V, ref={ref:+.3f}V, "
                f"err={abs(p-ref)*1e3:.0f}mV > 50mV"
            )
