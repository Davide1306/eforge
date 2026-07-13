"""Battery evaluation harness.

Modes:
  llm           — full pipeline (selector/filler/checker LLM calls via the
                  configured profile; 'mock' exercises graph mechanics only)
  deterministic — NO LLM anywhere: ground-truth fills -> resolve -> emit ->
                  execute -> gates. The no-model deterministic leg;
                  certifies emitters/skeletons/gates on this machine.

Usage:
  .venv/bin/python -m eval.harness --battery problems/seed --mode deterministic
  .venv/bin/python -m eval.harness --battery problems/seed --mode llm --profile mock
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emitters.resolve import SlotError, load_template, resolve_slots   # noqa: E402
from emitters.python_emitter import write_python                       # noqa: E402
from emitters.comsol_emitter import write_comsol                       # noqa: E402
from emitters.validators.comsol_static import validate_comsol, slot_to_param_for  # noqa: E402
from eval import fast_gates                                            # noqa: E402
from eval.metrics import compute_metrics                               # noqa: E402
from eval.scorer import score                                          # noqa: E402

GT_PATH = ROOT / 'eval/reference/ground_truth_fills.yaml'


def _expected(problem_id: str) -> dict:
    gt = yaml.safe_load(GT_PATH.read_text())
    merged = dict(gt.get('problems') or {})
    # library templates add per-template ground truth as gt_<template_id>.yaml
    # (additive; template-1's frozen file above is never modified)
    for p in sorted((ROOT / 'eval/reference').glob('gt_*.yaml')):
        extra = yaml.safe_load(p.read_text()) or {}
        merged.update(extra.get('problems') or {})
    return merged.get(problem_id, {})


def _resolved_expected(rec: dict) -> dict | None:
    if rec.get('template_id') in (None, 'no_match'):
        return None
    template = load_template(rec['template_id'])
    return resolve_slots(template, rec['fills'])


def run_one_deterministic(problem_id: str, out_dir: Path,
                          comsol_exec: bool = False) -> dict:
    """Ground-truth fills through the deterministic chain (no LLM).

    When comsol_exec is True the emitted model_comsol.py is actually run (it needs a
    COMSOL engine + MPh runtime); the resulting fields are diagnostic and NON-gating —
    they do not enter the score or the vetoes. Default False keeps certification runs
    (and the deterministic regression gate) untouched.
    """
    rec = _expected(problem_id)
    r = {'problem_id': problem_id,
         'template_id_expected': rec.get('template_id', 'gc_polyc_pb2d'),
         'template_id_got': rec.get('template_id', 'gc_polyc_pb2d'),
         'hallucination_incidents': 0, 'repair_rounds': 0, 'tokens_total': 0}
    if rec.get('template_id') == 'no_match':
        r.update({'exec_ok': None, 'gates_ok': None, 'comsol_static_ok': None})
        return r
    t0 = time.time()
    try:
        resolved = _resolved_expected(rec)
    except SlotError as e:
        r.update({'exec_ok': False, 'gates_ok': False, 'comsol_static_ok': False,
                  'error': f'ground-truth fills invalid: {e}'})
        return r
    r['resolved_expected'] = resolved
    r['resolved_got'] = resolved
    out_dir.mkdir(parents=True, exist_ok=True)
    py = write_python(rec['template_id'], resolved, out_dir / 'model.py', problem_id)
    co = write_comsol(rec['template_id'], resolved, out_dir / 'model_comsol.py', problem_id)
    r['comsol_static_ok'] = (
        (not validate_comsol(co.read_text(),
                             slot_to_param=slot_to_param_for(rec['template_id'])))
        if co else None)
    if comsol_exec and co:
        # Actually run the emitted COMSOL builder (needs a COMSOL engine + MPh runtime).
        # NON-GATING: these fields are diagnostic only; the .mph is the deliverable.
        cproc = subprocess.run([sys.executable, str(co)], cwd=str(out_dir),
                               capture_output=True, text=True, timeout=1800)
        mph_files = sorted(p.name for p in out_dir.glob('*.mph'))
        r['mph_produced'] = mph_files or []
        r['comsol_exec_ok'] = cproc.returncode == 0 and bool(mph_files)
        err = (cproc.stderr or cproc.stdout).strip()
        r['comsol_error'] = None if r['comsol_exec_ok'] else (
            err.splitlines()[-1] if err else f'exit {cproc.returncode}')
        r['comsol_exec_tail'] = (cproc.stderr or cproc.stdout)[-800:]
    elif co:
        r['comsol_exec_ok'] = None      # not exercised this run
    proc = subprocess.run([sys.executable, str(py)], cwd=str(out_dir),
                          capture_output=True, text=True, timeout=900)
    res = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith('RESULT_JSON: '):
            res = json.loads(line[len('RESULT_JSON: '):])
            break
    r['exec_ok'] = proc.returncode == 0 and res is not None
    if r['exec_ok']:
        gr = fast_gates.run_fast_gates(res, resolved, rec['template_id'])
        r['gates_ok'] = all(x['pass'] for x in gr)
        r['gate_results'] = gr
        r['exec_result'] = res
    else:
        r['gates_ok'] = False
        r['error'] = proc.stderr[-500:]
    r['wall_time_s'] = round(time.time() - t0, 2)
    return r


def run_one_llm(problem_id: str, problem_text: str, out_dir: Path,
                profile: str | None) -> dict:
    from pipeline.graph import run_pipeline
    rec = _expected(problem_id)
    t0 = time.time()
    final = run_pipeline(problem_text, problem_id, out_dir, profile)
    r = {'problem_id': problem_id,
         'template_id_expected': rec.get('template_id', 'gc_polyc_pb2d'),
         'template_id_got': final.get('template_id', '(none)'),
         'hallucination_incidents': final.get('hallucination_incidents', 0),
         'repair_rounds': final.get('repair_rounds', 0),
         'tokens_total': (final.get('llm_usage') or {}).get('tokens_in', 0)
         + (final.get('llm_usage') or {}).get('tokens_out', 0),
         'status': final.get('status'),
         'wall_time_s': round(time.time() - t0, 2)}
    if rec.get('template_id') == 'no_match':
        r.update({'exec_ok': None, 'gates_ok': None, 'comsol_static_ok': None})
        return r
    try:
        r['resolved_expected'] = _resolved_expected(rec)
    except SlotError:
        r['resolved_expected'] = None
    r['resolved_got'] = final.get('resolved')
    r['exec_ok'] = bool(final.get('exec_result')) and 'error' not in (final.get('exec_result') or {})
    r['gates_ok'] = final.get('status') == 'ok' and bool(final.get('gate_results')) \
        and all(x['pass'] for x in final['gate_results'])
    arts = final.get('artifacts') or {}
    cs = arts.get('comsol_static') if arts else False
    r['comsol_static_ok'] = None if cs is None else (cs == [])
    r['failures'] = final.get('failures')
    return r


def run_battery(battery_dir: Path, mode: str, profile: str | None,
                out_root: Path, comsol_exec: bool = False) -> dict:
    out_root = Path(out_root).resolve()
    manifest = yaml.safe_load((battery_dir / 'manifest.yaml').read_text())
    records = []
    for item in manifest['problems']:
        pid = item['id']
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)   # progress countable mid-run
        if mode == 'deterministic':
            rec = run_one_deterministic(pid, out_dir, comsol_exec=comsol_exec)
        else:
            text = (battery_dir / f'{pid}.md').read_text()
            rec = run_one_llm(pid, text, out_dir, profile)
        records.append(rec)
        flag = ('OK' if rec.get('gates_ok')
                else 'ABSTAIN' if rec.get('exec_ok') is None
                and rec['template_id_got'] == 'no_match'
                else 'FAIL')
        print(f"  {pid:28s} sel={rec['template_id_got']:<14s} {flag}", flush=True)
    m = compute_metrics(records)
    m['score'] = score(m)
    m['mode'] = mode
    m['profile'] = profile
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'metrics.json').write_text(json.dumps(m, indent=1))
    (out_root / 'records.json').write_text(json.dumps(records, indent=1, default=str))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--battery', default='problems/seed')
    ap.add_argument('--mode', choices=['llm', 'deterministic'], default='deterministic')
    ap.add_argument('--profile', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--comsol-exec', action='store_true',
                    help='also execute the emitted model_comsol.py (needs a COMSOL '
                         'engine + MPh runtime); non-gating diagnostic fields')
    args = ap.parse_args()
    out = Path(args.out) if args.out else \
        ROOT / 'runs' / f'{args.mode}_{int(time.time())}'
    m = run_battery(ROOT / args.battery, args.mode, args.profile, out,
                    comsol_exec=args.comsol_exec)
    print(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
