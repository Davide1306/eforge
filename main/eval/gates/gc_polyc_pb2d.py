"""Physics fast gates for template gc_polyc_pb2d (FROZEN once hashed).

Verbatim move (2026-06-04) of the certified eval/fast_gates.py gate body —
behavior byte-identical; fast_gates.py is now a per-template dispatcher.
"""

import numpy as np

from eval.reference import cg_oracle

CG_TOL = {'diagnostic': 5.0e-3, 'milestone': 1.5e-3, 'final': 1.0e-3}


def run(exec_result: dict, resolved: dict) -> list[dict]:
    out = []
    r = exec_result

    # 1. convergence proof
    ok = bool(r.get('success')) and r.get('residual_norm', 1.0) < 1e-6
    out.append({'id': 'residual_converged', 'pass': ok,
                'detail': f"success={r.get('success')} |R|={r.get('residual_norm'):.2e}"})

    # 2. single-facet sigma vs Chapman-Grahame oracle (tier-aware tolerance)
    if resolved['E_pzc_1'] == resolved['E_pzc_2'] and resolved['C_H_1'] == resolved['C_H_2']:
        tol = CG_TOL[resolved['tier']]
        psi0 = cg_oracle.solve_psi0(resolved['E_M'], resolved['E_pzc_1'],
                                    resolved['C_H_1'], resolved['c_b'])
        s_cg = cg_oracle.sigma_GC(psi0, resolved['c_b'])
        s = r.get('sigma_bar_C_per_m2', float('nan'))
        if abs(s_cg) < 1e-9:
            ok = abs(s - s_cg) < 1e-6
            detail = f'abs_err={abs(s - s_cg):.2e} (near-zero reference)'
        else:
            rel = abs(s - s_cg) / abs(s_cg)
            ok = rel < tol
            detail = f'rel_err={rel:.2e} tol={tol:.0e} (tier {resolved["tier"]})'
        out.append({'id': 'sigma_vs_cg_single_facet', 'pass': bool(ok), 'detail': detail})

    # 3. sign of sigma_bar vs C_H- and area-weighted drive.
    x = resolved['x']
    d1 = resolved['E_M'] - resolved['E_pzc_1']
    d2 = resolved['E_M'] - resolved['E_pzc_2']
    drive_w = x * resolved['C_H_1'] * d1 + (1.0 - x) * resolved['C_H_2'] * d2
    gross = x * resolved['C_H_1'] * abs(d1) + (1.0 - x) * resolved['C_H_2'] * abs(d2)
    s = r.get('sigma_bar_C_per_m2', float('nan'))
    if gross < 1e-12 or abs(drive_w) < 0.2 * gross or abs(s) < 1e-5:
        ok = True
        detail = (f'near-cancellation exempt: |drive_w|={abs(drive_w):.1e} '
                  f'< 0.2*gross={0.2 * gross:.1e} or sigma~0 ({s:+.1e})')
    else:
        ok = bool(np.sign(s) == np.sign(drive_w))
        detail = f'sign(sigma)={np.sign(s):+.0f} sign(drive_w)={np.sign(drive_w):+.0f}'
    out.append({'id': 'sigma_sign', 'pass': ok, 'detail': detail})

    # 4. C_dl positivity + sweep convergence (only when a sweep ran)
    if 'C_dl_min_F_per_m2' in r:
        ok = r['C_dl_min_F_per_m2'] > 0 and bool(r.get('sweep_all_converged', False))
        out.append({'id': 'cdl_positive', 'pass': ok,
                    'detail': f"C_dl_min={r['C_dl_min_F_per_m2']:.3e} "
                              f"all_conv={r.get('sweep_all_converged')}"})

    # 5. oracle self-consistency (analytic DH limit; no model involvement)
    c_b = resolved['c_b']
    rel = abs(cg_oracle.C_GC(1e-4, c_b) - cg_oracle.C_DH(c_b)) / cg_oracle.C_DH(c_b)
    out.append({'id': 'dh_limit_analytic', 'pass': bool(rel < 1e-4),
                'detail': f'rel={rel:.1e}'})
    return out
