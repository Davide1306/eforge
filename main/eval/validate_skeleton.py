"""Step-4 gate: emitted standalone model(s) vs their OWN-solver ground truth (NO LLM).

The deterministic floor. Template-parameterized over a SPECS list so it validates
BOTH templates that ship a Python skeleton + a frozen reference:

  gc_polyc_pb2d  (eval/reference/gc_reference.json, _check_gc)
      For every case: render skeleton with hardcoded fills -> run in subprocess ->
      parse RESULT_JSON -> require success, |R|<1e-6, and rel_err(sigma_bar vs the
      GC-polyc reference) < 1e-3. Single-facet cases also cross-checked against the
      analytic CG oracle (cg_oracle) at < 5e-3.

  polyc_mpb2d    (eval/reference/polyc_reference.json, _check_polyc)
      For every case: render skeleton with the case fills -> run -> compare the
      RESULT_JSON to the stored polyc target values. The skeleton INLINES the
      reference solver, so emitted == reference at machine precision; the
      hard floor is rel < 1e-3 (observed ~1e-12). Minima COUNT must match EXACTLY
      (B-count). The F5 adsorption-shift MAGNITUDE uses the documented
      reproduced-with-caveat band (NOT article-grade) -- everything else is hard
      <1e-3. Single-facet cases cross-checked against polyc_oracle (GCS) at < 5e-3.

Both specs feed one accumulator; selftest_corrections() (a gc-specific G2 guard)
still runs. CLI contract unchanged: `python -m eval.validate_skeleton` -> exit 0
on all-pass.

Run:  cd main && .venv/bin/python -m eval.validate_skeleton
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emitters.resolve import (SlotError, _all_slots, _check_one,    # noqa: E402
                              load_template, resolve_slots)
from emitters.python_emitter import write_python                   # noqa: E402
from eval.reference import cg_oracle                               # noqa: E402
from eval.reference import polyc_oracle                            # noqa: E402

REL_TOL_GC = 1e-3
REL_TOL_POLYC = 1e-3            # emitted-vs-reference equivalence floor (observed ~1e-12)
ABS_TOL_POLYC = 1e-9           # symmetric-cancellation absolute tol for near-zero quantities
F5_CAVEAT_REL = 0.10          # F5 adsorption-shift MAGNITUDE: reproduced-with-caveat band
                              # (NOT article-grade; model PZ theta-dependence ~0.07-0.10
                              #  under paper, EXANG-20260618). The emitted-vs-reference
                              #  rel is still ~1e-12 since both sides are the same skeleton.
PY = str(ROOT / '.venv' / 'bin' / 'python')


def run_emitted(path: Path, cwd: Path, timeout: int = 300) -> dict:
    proc = subprocess.run([PY, str(path)], cwd=str(cwd), timeout=timeout,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'emitted model failed:\n{proc.stderr[-2000:]}')
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith('RESULT_JSON: '):
            return json.loads(line[len('RESULT_JSON: '):])
    raise RuntimeError('no RESULT_JSON line in emitted model stdout')


def selftest_corrections() -> int:
    """G2 regression (2026-06-04): unconstrained cross-checker corrections must
    never bypass type/range/enum validation. Exercises the exact _check_one
    path that pipeline/graph.py fill_group now routes every correction through."""
    specs = _all_slots(load_template('gc_polyc_pb2d'))
    n_fail = 0
    for slot, bad_val, label in [('sweep_var', 'single', 'out-of-enum'),
                                 ('L', 1.0, 'out-of-range'),
                                 ('E_M', 'minus point two', 'ill-typed')]:
        try:
            _check_one(slot, specs[slot][0], bad_val)
            print(f'FAIL corrections-guard: {label} {slot}={bad_val!r} accepted')
            n_fail += 1
        except SlotError:
            print(f'PASS corrections-guard: {label} {slot}={bad_val!r} rejected')
    return n_fail


def _rel(a: float, b: float, abs_tol: float) -> tuple:
    """(rel_err, used_abs_tol). Falls back to absolute error for near-zero |ref|
    (symmetric-cancellation quantities) so a tiny denominator can't blow up rel."""
    if abs(b) < abs_tol:
        return abs(a - b), True       # absolute error in the quantity's own units
    return abs(a - b) / abs(b), False


def _check_gc(template, rec, td: Path) -> tuple:
    """Existing gc_polyc per-case logic, factored out VERBATIM. Renders the gc
    skeleton with the case fills, runs it, checks success/|R|/sigma_bar vs the
    GC reference (with the symmetric-cancellation abs branch) and the single-facet
    cg_oracle cross-check at <5e-3. Returns (ok, line)."""
    fills = dict(rec['fills'])
    fills.update({'tier': 'diagnostic', 'sweep_var': 'none',
                  'sweep_lo': -0.5, 'sweep_hi': 0.5, 'n_points': 5,
                  'panel_L_nm': [fills['L'] * 1e9]})
    resolved = resolve_slots(template, fills)
    mod = write_python('gc_polyc_pb2d', resolved,
                       td / f"{rec['id']}.py", problem_id=rec['id'])
    out = run_emitted(mod, td)

    ok = out['success'] and out['residual_norm'] < 1e-6
    if abs(rec['sigma_bar']) < 1e-6:   # symmetric-cancellation cases
        rel = abs(out['sigma_bar_C_per_m2'] - rec['sigma_bar'])
        ok = ok and rel < 1e-6          # absolute tol [C/m^2]
    else:
        rel = abs(out['sigma_bar_C_per_m2'] - rec['sigma_bar']) / \
            abs(rec['sigma_bar'])
        ok = ok and rel < REL_TOL_GC

    # analytic CG cross-check for single-facet cases
    cg_note = ''
    f = rec['fills']
    if f['E_pzc_1'] == f['E_pzc_2'] and f['C_H_1'] == f['C_H_2']:
        psi0 = cg_oracle.solve_psi0(f['E_M'], f['E_pzc_1'], f['C_H_1'], f['c_b'])
        s_cg = cg_oracle.sigma_GC(psi0, f['c_b'])
        rel_cg = abs(out['sigma_bar_C_per_m2'] - s_cg) / max(abs(s_cg), 1e-30)
        ok = ok and rel_cg < 5e-3   # diagnostic-tier CG tolerance (gates.frozen.yaml)
        cg_note = f'  rel_cg={rel_cg:.2e}'

    status = 'PASS' if ok else 'FAIL'
    return ok, (f"{status} {rec['id']:24s} rel_gc={rel:.2e} "
                f"|R|={out['residual_norm']:.2e}{cg_note}")


def _check_polyc(template, rec, td: Path) -> tuple:
    """polyc_mpb2d per-case: render the skeleton with the case fills, run it, and
    compare the RESULT_JSON to the stored polyc target values. Emitted == reference
    at machine precision (inlined solver) -> hard floor rel < 1e-3 on every stored
    quantity; minima COUNT EXACT (B-count). The F5 adsorption-shift MAGNITUDE uses
    the documented reproduced-with-caveat band. Single-facet cases get an analytic
    polyc_oracle GCS cross-check at <5e-3. Returns (ok, line)."""
    fills = dict(rec['fills'])
    fills.update({'tier': 'diagnostic', 'sweep_var': 'none',
                  'sweep_lo': -0.5, 'sweep_hi': 0.5, 'n_points': 5,
                  'panel_L_nm': [10.0]})
    resolved = resolve_slots(template, fills)
    mod = write_python('polyc_mpb2d', resolved,
                       td / f"{rec['id']}.py", problem_id=rec['id'])
    out = run_emitted(mod, td)

    ok = bool(out['success']) and out['residual_norm'] < 1e-6
    worst = 0.0           # worst HARD-floor rel error (excludes caveated F5 magnitude)

    # (1) minima COUNT must match EXACTLY (B-count)
    if int(out['n_minima']) != int(rec['n_minima']):
        ok = False
    # (2) C_dl_min, sigma, q_free, lam_D : hard <1e-3 (abs branch for near-zero)
    for key in ('C_dl_min_F_per_m2', 'sigma', 'q_free', 'lam_D'):
        rel, _ = _rel(float(out[key]), float(rec[key]), ABS_TOL_POLYC)
        worst = max(worst, rel)
        ok = ok and rel < REL_TOL_POLYC
    # (3) minima positions + magnitudes (paired; count already matched if we got here)
    if int(out['n_minima']) == int(rec['n_minima']):
        for (em_o, cm_o), (em_r, cm_r) in zip(out['minima'], rec['minima']):
            for a, b in ((em_o, em_r), (cm_o, cm_r)):
                rel, _ = _rel(float(a), float(b), ABS_TOL_POLYC)
                worst = max(worst, rel)
                ok = ok and rel < REL_TOL_POLYC
    # (4) PZ slope/intercept/R^2 (only when the reference stored a fit)
    for key in ('pz_slope', 'pz_intercept', 'pz_r2'):
        if rec.get(key) is not None and out.get(key) is not None:
            rel, _ = _rel(float(out[key]), float(rec[key]), ABS_TOL_POLYC)
            worst = max(worst, rel)
            ok = ok and rel < REL_TOL_POLYC

    # (5) F5 adsorption shift: emitted-vs-reference is still ~1e-12 (same skeleton),
    #     but the MAGNITUDE is contractually a reproduced-with-caveat band, so it is
    #     checked against F5_CAVEAT_REL and EXCLUDED from `worst` (not article-grade).
    ads_note = ''
    if rec.get('kind') == 'ads':
        rel_ads, _ = _rel(float(out['adsorption_shift_meV']),
                          float(rec['adsorption_shift_meV']), 1e-6)
        ok = ok and rel_ads < F5_CAVEAT_REL     # caveat band, not the hard floor
        ads_note = f'  ads_rel={rel_ads:.2e}[caveat<{F5_CAVEAT_REL:g}]'

    # analytic polyc_oracle GCS cross-check for single-facet cases (E_pzc_1==E_pzc_2)
    cg_note = ''
    f = rec['fills']
    if f['E_pzc_1'] == f['E_pzc_2'] and f['C_H_1'] == f['C_H_2']:
        cb_M = f['c_b'] * 1e-3                                   # mol/m^3 -> M
        c_gc_min = polyc_oracle.eps_S / polyc_oracle.lambda_D(cb_M)   # C_GC minimum
        cdl_gcs = polyc_oracle.C_dl_GCS(f['C_H_1'], c_gc_min)
        rel_cg = abs(float(out['C_dl_min_F_per_m2']) - cdl_gcs) / max(abs(cdl_gcs), 1e-30)
        ok = ok and rel_cg < 5e-3    # diagnostic-tier GCS tolerance (gates.frozen.yaml)
        cg_note = f'  rel_gcs={rel_cg:.2e}'

    status = 'PASS' if ok else 'FAIL'
    return ok, (f"{status} {rec['id']:24s} rel_max={worst:.2e} "
                f"|R|={out['residual_norm']:.2e} nmin={out['n_minima']}"
                f"{cg_note}{ads_note}")


# Template-parameterized floor specs. gc stays first + byte-identical; polyc added.
SPECS = [
    {'template_id': 'gc_polyc_pb2d', 'ref': 'eval/reference/gc_reference.json',
     'check': _check_gc},
    {'template_id': 'polyc_mpb2d', 'ref': 'eval/reference/polyc_reference.json',
     'check': _check_polyc},
]


def main() -> int:
    n_fail = 0
    with tempfile.TemporaryDirectory(prefix='skelval_') as td:
        td = Path(td)
        for spec in SPECS:
            template = load_template(spec['template_id'])
            ref = json.loads((ROOT / spec['ref']).read_text())
            print(f"--- {spec['template_id']} ({len(ref['cases'])} cases) ---")
            for rec in ref['cases']:
                ok, line = spec['check'](template, rec, td)
                if not ok:
                    n_fail += 1
                print(line)

    n_fail += selftest_corrections()
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
