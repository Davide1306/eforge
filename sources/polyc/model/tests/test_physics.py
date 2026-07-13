"""
G2 prerequisite — pointwise constitutive kernel (eqs 2-4). The bulk/low-field
limits are exact falsifiers (worker EDIT RATIONALE iter3): any limit off >1e-9,
eps_eff outside (eps_opt, eps_S], or wrong ion-asymmetry sign means a kernel bug.
"""
import numpy as np

from src import constants as C
from src import physics as PH


def _rel(a, b):
    return abs(a - b) / abs(b)


def test_special_function_limits():
    assert _rel(PH.sinhc(0.0), 1.0) < 1e-12
    assert abs(PH.langevin_L(0.0)) < 1e-12
    assert _rel(PH.langevin_Lc(0.0), 1.0 / 3.0) < 1e-9
    # mid-range vs direct definitions
    assert _rel(PH.sinhc(2.0), np.sinh(2.0) / 2.0) < 1e-10
    assert _rel(PH.langevin_L(2.0), 1.0 / np.tanh(2.0) - 0.5) < 1e-10
    assert _rel(PH.langevin_Lc(2.0), (1.0 / np.tanh(2.0) - 0.5) / 2.0) < 1e-10


def test_denominator_unity_at_bulk_any_v():
    for mM in (0.1, 10.0, 200.0):
        v = C.v_frac(C.mM_to_M(mM))
        assert _rel(PH.denominator_D(0.0, 0.0, v), 1.0) < 1e-12


def test_bulk_limit_densities_water_eps():
    cb = C.mM_to_M(0.1)
    nb, v = C.n_b(cb), C.v_frac(cb)
    npl, nmi = PH.number_densities(0.0, 0.0, nb, v)
    assert _rel(npl, nb) < 1e-12 and _rel(nmi, nb) < 1e-12
    assert _rel(PH.n_water(0.0, 0.0, v), C.n_w_b) < 1e-12
    assert _rel(PH.eps_eff(0.0, 0.0, v), C.eps_S) < 1e-9   # eq-3 closure


def test_eps_eff_saturates_with_field():
    v = C.v_frac(C.mM_to_M(0.1))
    e_lo = PH.eps_eff(0.0, 0.0, v)
    e_hi = PH.eps_eff(0.0, 5e9, v)
    assert e_hi < e_lo                 # dielectric saturation
    assert e_hi > C.eps_opt            # bounded below by optical permittivity
    assert e_lo <= C.eps_S * (1.0 + 1e-9)


def test_charge_neutral_at_zero_potential():
    cb = C.mM_to_M(1.0)
    nb, v = C.n_b(cb), C.v_frac(cb)
    npl, nmi = PH.number_densities(0.0, 3e8, nb, v)
    assert abs(npl - nmi) < 1e-6 * nb


def test_ion_asymmetry_sign():
    # phi > 0 (positive local potential) -> anions enhanced, cations depleted.
    cb = C.mM_to_M(1.0)
    nb, v = C.n_b(cb), C.v_frac(cb)
    npl, nmi = PH.number_densities(+0.1, 1e8, nb, v)
    assert nmi > nb > npl


def test_vectorized_eps_eff():
    v = C.v_frac(C.mM_to_M(0.1))
    E = np.array([0.0, 1e8, 1e9, 5e9])
    e = PH.eps_eff(np.zeros_like(E), E, v)
    assert e.shape == E.shape
    assert _rel(e[0], C.eps_S) < 1e-9
    assert e[-1] < e[0]                # saturates toward smaller eps at high field
    assert np.all(e > C.eps_opt)
