"""Physics fast gates for template polyc_mpb2d (FROZEN once hashed).

The frozen full-physics gate body for the 2-D modified-Poisson-Boltzmann
(steric / Langevin / Booth / Robin / Langmuir-adsorption) template polyc_mpb2d.
Mirrors the contract of eval/gates/gc_polyc_pb2d.py byte-for-byte at the
interface level: `run(exec_result, resolved) -> list[dict]`, each item a
`{'id': str, 'pass': bool, 'detail': str}`. Auto-discovered by the
template-agnostic dispatcher eval/fast_gates.py (no dispatcher edit).

Gate ids are the stable contracts declared in
templates/polyc_mpb2d/gates.frozen.yaml (the integrity-pinned file). Bands are
sourced from the reference implementation's constraints (article-grade):
  B-pos   feature/minimum positions      |dE|  <= 20 mV (0.020 V)  -- vs reference
  B-mag   capacitance magnitudes         rel.  <= 10%   (0.10)     -- vs reference
  B-count number of minima               EXACT (shallow-near-L~lambda_D = "one")
  B-slope Parsons-Zobel                   |d slope| <= 0.05 abs AND R^2 > 0.99
  CG cross-check (convergence oracle)     rel  <  1e-3 (final tier)
  F5 adsorption                           reproduced-with-caveat (NOT B-slope-tight)

Two adjudication regimes, exactly mirroring how gc_polyc's fast gate behaves:

  * Gates with an INTRINSIC ground truth run on every evaluation:
    `residual_converged` (always), `cg_cross_check` (single-facet analytic GCS
    limit, tier-aware), `minima_count_exact` (regime structural rule), and
    `adsorption_shift` (presence/sign, F5 caveat).

  * The article-grade "vs digitized reference" gates -- `feature_position`
    (B-pos), `magnitude` (B-mag), `pz_slope` (B-slope) -- compare the model
    against a REFERENCE that must be supplied in `resolved` (a held-out
    reference fill carries `min_positions_ref` / `magnitude_ref` /
    `pz_slope_ref`). When no reference is wired into `resolved` (the normal
    pipeline case: a novel LLM-filled problem has no digitized curve), these
    gates are SKIPPED -- they append nothing -- precisely as gc_polyc's
    `sigma_vs_cg_single_facet` only runs when both facets coincide. The
    certification battery (the deterministic floor, eval/validate_skeleton.py)
    adjudicates the full reference-curve agreement against polyc_reference.json;
    the fast gate never invents a reference it does not have.

The analytic ground truth is eval/reference/polyc_oracle.py: the
single-facet / dilute Gouy-Chapman-Stern closed forms (eqs 10-11) and the
Langmuir C_ads peak (eq 13). polyc_oracle takes bulk concentration in mol/L (M);
the template slot `c_b` is mol/m^3 (1 mM = 1.0), so c_b_M = resolved['c_b']*1e-3.
"""

import numpy as np

from eval.reference import polyc_oracle

# Tier-aware tolerance for the convergence-oracle cross-check. Matches gc_polyc's
# tiers; the FINAL band is the < 1e-3 convergence-oracle floor (CONSTRAINTS §3).
CG_TOL = {'diagnostic': 5.0e-3, 'milestone': 1.5e-3, 'final': 1.0e-3}

# B-band absolutes (CONSTRAINTS §3 / gates.frozen.yaml).
B_POS = 0.020          # V    feature/minimum-position tolerance (20 mV)
B_MAG = 0.10           # rel  capacitance-magnitude tolerance (10%)
B_SLOPE = 0.05         # abs  Parsons-Zobel slope tolerance
B_SLOPE_R2 = 0.99      #      Parsons-Zobel fit R^2 floor
# F5 adsorption is reproduced-with-caveat (model under paper ~0.05-0.10
# out-of-band); the gate certifies sign/direction reproduction, NOT article-grade.
F5_CAVEAT = "caveat: F5 PZ-magnitude documented out-of-band (reproduced-with-caveat)"


def _single_facet_gcs_min(resolved: dict) -> float:
    """Analytic single-facet GCS capacitance minimum [F/m^2] for the case.

    The C_GC minimum is at E = E_pzc (cosh = 1 -> C_GC = eps_S/lambda_D); the
    series GCS minimum is C_dl_GCS(C_H, eps_S/lambda_D). Valid only when both
    facets coincide (single-facet limit of the full model)."""
    c_b_M = resolved['c_b'] * 1e-3
    lam_D = polyc_oracle.lambda_D(c_b_M)
    c_gc_min = float(polyc_oracle.C_GC(resolved['E_pzc_1'],
                                       resolved['E_pzc_1'], lam_D))
    return polyc_oracle.C_dl_GCS(resolved['C_H_1'], c_gc_min)


def _is_single_facet(resolved: dict) -> bool:
    return (resolved['E_pzc_1'] == resolved['E_pzc_2']
            and resolved['C_H_1'] == resolved['C_H_2'])


def _expected_min_count(resolved: dict) -> int:
    """Expected number of C_dl minima for the case, from physics (E7 / Fig S1).

    One minimum for coincident facets, or for L <= lambda_D (the facets merge);
    two minima for split facet PZCs in the L > lambda_D regime. Shallow extra
    minima near the L ~ lambda_D transition count as "one" (Fig S1 convention)."""
    if resolved.get('n_minima_expected') is not None:
        return int(resolved['n_minima_expected'])
    if resolved['E_pzc_1'] == resolved['E_pzc_2']:
        return 1
    c_b_M = resolved['c_b'] * 1e-3
    lam_D = polyc_oracle.lambda_D(c_b_M)
    return 2 if resolved['L'] > lam_D else 1


def run(exec_result: dict, resolved: dict) -> list[dict]:
    out = []
    r = exec_result

    # 1. convergence proof (solver is inlined + deterministic). Mirrors gc.
    ok = bool(r.get('success')) and r.get('residual_norm', 1.0) < 1e-6
    out.append({'id': 'residual_converged', 'pass': bool(ok),
                'detail': f"success={r.get('success')} "
                          f"|R|={r.get('residual_norm', float('nan')):.2e}"})

    # 2. cg_cross_check: single-facet / dilute analytic GCS minimum vs the model's
    #    reported C_dl minimum, tier-aware tolerance. Only a valid comparison when
    #    both facets coincide (the single-facet limit of the full model) -- exactly
    #    as gc_polyc gates sigma_vs_cg_single_facet on E_pzc_1==E_pzc_2 and
    #    C_H_1==C_H_2. Two-facet cases skip this gate (append nothing).
    if _is_single_facet(resolved):
        tol = CG_TOL[resolved['tier']]
        ref = _single_facet_gcs_min(resolved)
        c_mod = r.get('C_dl_min_F_per_m2', float('nan'))
        if abs(ref) < 1e-9:
            ok = abs(c_mod - ref) < 1e-6
            detail = f'abs_err={abs(c_mod - ref):.2e} (near-zero reference)'
        else:
            rel = abs(c_mod - ref) / abs(ref)
            ok = rel < tol
            detail = (f'rel_err={rel:.2e} tol={tol:.0e} (tier {resolved["tier"]}) '
                      f'C_dl_min_model={c_mod:.4e} C_dl_GCS={ref:.4e}')
        out.append({'id': 'cg_cross_check', 'pass': bool(ok), 'detail': detail})

    # 3. minima_count_exact (B-count): the number of C_dl minima the model reports
    #    must EXACTLY equal the expected count for the regime (intrinsic structural
    #    rule -- robust without a digitized reference). Honors the shallow-minimum
    #    convention via _expected_min_count. Only when the model reported a count.
    n_mod = r.get('n_minima')
    if n_mod is not None:
        exp_n = _expected_min_count(resolved)
        ok = int(n_mod) == exp_n
        out.append({'id': 'minima_count_exact', 'pass': bool(ok),
                    'detail': f'n_minima_model={int(n_mod)} expected={exp_n}'})

    # 4. feature_position (B-pos): worst model minimum-position error vs the
    #    REFERENCE positions <= 20 mV. The article band is "vs digitized reference",
    #    so this needs reference positions supplied in resolved['min_positions_ref']
    #    (a held-out reference fill carries them). Absent a reference (the normal
    #    pipeline case), SKIP -- the full reference-curve agreement is adjudicated by
    #    the certification battery vs polyc_reference.json, not here.
    ref_pos = resolved.get('min_positions_ref')
    minima = r.get('minima')
    if ref_pos is not None and minima:
        ref_pos = sorted(float(p) for p in ref_pos)
        e_mod = sorted(float(m[0]) for m in minima)
        worst = max(min(abs(em - ep) for ep in ref_pos) for em in e_mod)
        ok = worst <= B_POS
        out.append({'id': 'feature_position', 'pass': bool(ok),
                    'detail': f'worst|dE|={worst * 1e3:.1f} mV tol={B_POS * 1e3:.0f} mV '
                              f'(model E_min={e_mod} ref={ref_pos})'})

    # 5. magnitude (B-mag): capacitance-magnitude rel. err <= 10% vs reference. The
    #    article band is pointwise on the digitized curve; supply the reference
    #    minimum capacitance in resolved['magnitude_ref'] (held-out reference fill).
    #    Absent that, fall back to the single-facet analytic GCS minimum when the
    #    case is single-facet (an intrinsic magnitude anchor, mirroring gc's analytic
    #    limit); otherwise SKIP (no reference, certification battery adjudicates).
    mag_ref = resolved.get('magnitude_ref')
    if mag_ref is None and _is_single_facet(resolved):
        mag_ref = _single_facet_gcs_min(resolved)
    if mag_ref is not None and 'C_dl_min_F_per_m2' in r:
        c_mod = r.get('C_dl_min_F_per_m2', float('nan'))
        rel = abs(c_mod - mag_ref) / abs(mag_ref) if abs(mag_ref) > 1e-12 else float('nan')
        ok = rel <= B_MAG
        out.append({'id': 'magnitude', 'pass': bool(ok),
                    'detail': f'rel_err={rel:.2e} tol={B_MAG:.0%} '
                              f'C_dl_min_model={c_mod:.4e} ref={mag_ref:.4e}'})

    # 6. pz_slope (B-slope): |slope - slope_ref| <= 0.05 abs AND R^2 > 0.99. The PZ
    #    slope is regime-dependent (1.0 only in the dilute single-facet limit; a
    #    two-facet split-PZC stripe legitimately gives a different slope), so the
    #    band is vs a REFERENCE slope, not a hardcoded constant. Supply
    #    resolved['pz_slope_ref'] (held-out reference fill). Absent a reference, the
    #    R^2 floor is still an intrinsic linear-fit quality check we can apply when a
    #    fit exists, but the slope band is skipped; to avoid a partial/ambiguous
    #    verdict we SKIP the whole gate unless a reference slope is provided --
    #    mirroring gc, whose reference-curve gates are battery-side, not fast-gate.
    pz_slope = r.get('pz_slope')
    pz_r2 = r.get('pz_r2')
    slope_ref = resolved.get('pz_slope_ref')
    if slope_ref is not None and pz_slope is not None and pz_r2 is not None:
        dslope = abs(float(pz_slope) - float(slope_ref))
        ok = (dslope <= B_SLOPE) and (float(pz_r2) > B_SLOPE_R2)
        out.append({'id': 'pz_slope', 'pass': bool(ok),
                    'detail': f'|d slope|={dslope:.3f} tol={B_SLOPE:.2f} '
                              f'(slope={float(pz_slope):.4f} ref={float(slope_ref):.3f}) '
                              f'R2={float(pz_r2):.4f} (>{B_SLOPE_R2})'})

    # 7. adsorption_shift (F5 caveat): only when adsorption is ON (theta_max > 0).
    #    Reproduced-with-caveat -- certify the adsorption-induced E_min shift is
    #    present and physically sized (theta_max=1% shifts E_min by tens of mV in the
    #    article), NOT the tight article-grade PZ band. A present, finite,
    #    non-negligible shift (|shift| >= 1 mV) PASSES with the F5 caveat in detail.
    #    When theta_max == 0/off, append nothing (no adsorption to check) -- mirrors
    #    gc, where conditional gates only run when their key is present.
    if resolved.get('theta_max', 0.0) > 0.0:
        shift_meV = r.get('adsorption_shift_meV', float('nan'))
        on = bool(r.get('adsorption_on', False))
        finite = shift_meV is not None and np.isfinite(shift_meV)
        present = on and finite and abs(shift_meV) >= 1.0
        shown = shift_meV if finite else float('nan')
        out.append({'id': 'adsorption_shift', 'pass': bool(present),
                    'detail': f'adsorption_on={on} shift={shown:.1f} meV '
                              f'(theta_max={resolved.get("theta_max")}); {F5_CAVEAT}'})

    return out
