"""
G1 SMOKE — pin the CONSTRAINTS §1 derived quantities to their hand-checked
targets. Falsifier (worker EDIT RATIONALE iter1): any value off target by
>0.5% (lambda_D >1.5%) means a constant/formula is wrong.
"""
import math

from src import constants as C


def _rel(a, b):
    return abs(a - b) / abs(b)


def test_beta():
    assert _rel(C.beta, 2.4299e20) < 5e-3


def test_p_dipole_pin():
    # re-derived formula vs the table value
    assert _rel(C.p_from_table(), 1.580e-29) < 5e-3
    assert _rel(C.p_dip, C.p_from_table()) < 5e-3


def test_N_tot():
    a_Pt = 3.92e-10
    assert _rel(C.N_tot, 4.0 / (math.sqrt(3.0) * a_Pt**2)) < 5e-3


def test_n_w_b_two_forms_agree():
    assert _rel(C.n_w_b, 3.357e28) < 5e-3
    assert _rel(C.n_w_b, C.n_w_b_conc) < 4e-3   # ~0.3% apart (d_t rounding)


def test_lambda_D_four_concentrations():
    # CONSTRAINTS §1: c_b = 0.1/0.3/1/10 mM -> 30.4/17.6/9.6/3.0 nm (Fig 7 canonical)
    targets_nm = {0.1: 30.4, 0.3: 17.6, 1.0: 9.6, 10.0: 3.0}
    for mM, tgt in targets_nm.items():
        got_nm = C.lambda_D(C.mM_to_M(mM)) * 1e9
        assert _rel(got_nm, tgt) < 1.5e-2, (mM, got_nm, tgt)


def test_eps_eff_low_field_limit_is_eps_S():
    # eq-3 bulk limit: eps_opt + n_w_b p^2 beta/3 == eps_S exactly (p-pin).
    eps_eff0 = C.eps_opt + C.n_w_b * C.p_from_table() ** 2 * C.beta / 3.0
    assert _rel(eps_eff0, C.eps_S) < 1e-6


def test_v_frac_dilute():
    # base case 0.1 mM is strongly dilute -> tiny ion volume fraction.
    v = C.v_frac(C.mM_to_M(0.1))
    assert 0.0 < v < 1e-4


def test_C_H_unit_conversion():
    # 50 uF/cm^2 == 0.5 F/m^2
    assert _rel(C.C_H_default, 0.5) < 1e-9
    assert _rel(C.SI_to_uFcm2(C.C_H_default), 50.0) < 1e-9
