"""LangGraph pipeline: problem -> select -> fill (chunked) -> emit -> execute -> gates.

Nodes marked LLM are the served model's agent calls (selector / slot_filler /
cross_checker — same model, different roles). Everything else is deterministic
code. The repair edge routes gate failures back to the offending slot-group
with a feedback string; attempts are capped by policy.repair_max_rounds.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import yaml
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emitters.resolve import (SlotError, _all_slots, _check_one,          # noqa: E402
                              load_template, resolve_slots)
from emitters.python_emitter import write_python                          # noqa: E402
from emitters.comsol_emitter import write_comsol                          # noqa: E402
from emitters.validators.py_static import validate_python                 # noqa: E402
from emitters.validators.comsol_static import validate_comsol, slot_to_param_for  # noqa: E402
from eval import fast_gates                                               # noqa: E402
from pipeline.llm_client import LLMClient                                 # noqa: E402
from pipeline.schemas import CHECKER_SCHEMA, SELECTOR_SCHEMA, group_schemas  # noqa: E402

PROMPTS = Path(__file__).parent / 'prompts'


class PipelineState(TypedDict, total=False):
    problem_id: str
    problem_text: str
    out_dir: str
    template_id: str
    template: dict
    groups: list[dict]
    group_idx: int
    fills: dict[str, Any]
    feedback: dict[str, str]          # group -> repair feedback string
    repair_rounds: int
    failures: list[dict]
    hallucination_incidents: int
    resolved: dict
    artifacts: dict
    exec_result: dict
    gate_results: list[dict]
    status: str                       # ok | abstained | failed_<stage>
    llm_usage: dict


def _read(p: Path) -> str:
    return p.read_text()


def _feedback_templates() -> dict:
    return yaml.safe_load((PROMPTS / 'repair_feedback.yaml').read_text())


def _slot_from_sloterror(msg: str) -> str | None:
    """Extract the offending slot name from a SlotError message.

    Message contract from emitters/resolve.py: per-slot errors lead with the
    slot name ('<name>=...' or '<name>: ...'); cross-field cases special-cased.
    Returns None when no single slot is identifiable (-> terminal failure).
    """
    if msg.startswith('invented field'):
        return None
    if 'sweep_lo must be' in msg:
        return 'sweep_lo'
    for prefix in ('missing required slot ', 'slot '):
        if msg.startswith(prefix):
            return msg[len(prefix):].split()[0]
    head = re.split(r'[=:]', msg, maxsplit=1)[0].strip()
    return head if head.isidentifier() else None


def _group_for_slot(groups: list[dict], slot: str) -> int | None:
    return next((i for i, g in enumerate(groups) if slot in g['slots']), None)


def _fewshots(group: str, k: int) -> str:
    if k <= 0:
        return ''
    lines = (PROMPTS / 'fewshots/parameters.jsonl').read_text().splitlines()
    shots = [json.loads(l) for l in lines if l.strip()]
    shots = [s for s in shots if s['group'] == group or group.startswith(s['group'])][:k]
    if not shots:
        return ''
    blocks = [f"Problem: {s['problem']}\nJSON: {json.dumps(s['fill'])}" for s in shots]
    return 'Examples:\n' + '\n\n'.join(blocks) + '\n\n'


def build_graph(client: LLMClient):
    policy = client.cfg['policy']
    feedback_t = _feedback_templates()

    # ── LLM node: selector ────────────────────────────────────────────
    def selector(state: PipelineState) -> dict:
        registry = (ROOT / 'templates/registry.yaml').read_text()
        out = client.complete_json(
            'selector', _read(PROMPTS / 'selector.system.md'),
            f'TEMPLATE REGISTRY:\n{registry}\n\nPROBLEM:\n{state["problem_text"]}\n',
            SELECTOR_SCHEMA, meta={'problem_id': state['problem_id']})
        tid = out.get('template_id', 'no_match')
        if tid == 'no_match':
            return {'template_id': tid, 'status': 'abstained'}
        try:
            template = load_template(tid)
        except SlotError:
            return {'template_id': tid, 'status': 'failed_selector',
                    'failures': [{'stage': 'selector', 'reason': f'unknown id {tid!r}'}]}
        groups = group_schemas(template, policy.get('chunk_mode', 'per_group'))
        return {'template_id': tid, 'template': template, 'groups': groups,
                'group_idx': 0, 'fills': {}, 'feedback': {}, 'repair_rounds': 0,
                'failures': [], 'hallucination_incidents': 0, 'status': 'ok'}

    # ── LLM node: fill one slot-group (with retries + optional cross-check) ──
    def fill_group(state: PipelineState) -> dict:
        g = state['groups'][state['group_idx']]
        schema = g['schema']
        fb = state['feedback'].get(g['group'], '')
        shots = _fewshots(g['group'], policy.get('fewshot_count', 1))
        user = (f'{shots}SLOT GROUP: {g["group"]}\n'
                f'SCHEMA:\n{json.dumps(schema, indent=1)}\n\n'
                f'PROBLEM:\n{state["problem_text"]}\n'
                + (f'\nPREVIOUS ATTEMPT WAS REJECTED: {fb}\n' if fb else ''))

        fills, halluc, failures = dict(state['fills']), 0, list(state['failures'])
        last_err = None
        for attempt in range(1 + policy.get('retry_budget', 2)):
            try:
                out = client.complete_json('slot_filler',
                                           _read(PROMPTS / 'slot_filler.system.md'),
                                           user, schema,
                                           meta={'problem_id': state['problem_id']})
            except Exception as e:                     # unparseable output
                halluc += 1
                last_err = f'malformed output: {e}'
                user += f'\nYour last output was invalid JSON ({e}). Output ONLY the JSON object.'
                continue
            invented = set(out) - set(g['slots'])
            if invented:
                halluc += 1
                last_err = f'invented fields {sorted(invented)}'
                user += f'\nYou added fields {sorted(invented)} not in the schema. Only fill: {g["slots"]}.'
                continue

            if policy.get('cross_check', True):
                chk = client.complete_json(
                    'cross_checker', _read(PROMPTS / 'cross_checker.system.md'),
                    f'PROBLEM:\n{state["problem_text"]}\n\nPROPOSED FILL ({g["group"]}):\n'
                    f'{json.dumps(out)}\n', CHECKER_SCHEMA,
                    meta={'problem_id': state['problem_id']})
                if not chk.get('ok', False):
                    raw_corr = {k: v for k, v in (chk.get('corrections') or {}).items()
                                if k in g['slots']}
                    # corrections are UNCONSTRAINED model output — validate every
                    # value through the same type/range/enum check as resolve_slots
                    slot_specs = _all_slots(state['template'])
                    corr, bad = {}, []
                    for k, v in raw_corr.items():
                        try:
                            corr[k] = _check_one(k, slot_specs[k][0], v)
                        except SlotError as e:
                            bad.append(str(e))
                    if corr and not bad:
                        out.update(corr)
                    else:
                        if bad:
                            halluc += 1
                            last_err = f'cross-checker correction invalid: {bad}'
                        else:
                            last_err = f'cross-checker rejected: {chk.get("reason", "")}'
                        user += (f'\nA reviewer rejected this fill: {chk.get("reason", "")}. '
                                 'Re-derive carefully, respecting types, ranges and enums.')
                        continue
            fills.update(out)
            return {'fills': fills, 'group_idx': state['group_idx'] + 1,
                    'hallucination_incidents': state['hallucination_incidents'] + halluc,
                    'failures': failures}

        failures.append({'stage': 'fill', 'group': g['group'], 'reason': last_err,
                         'attempts': attempt + 1})
        return {'fills': fills, 'group_idx': state['group_idx'] + 1,
                'hallucination_incidents': state['hallucination_incidents'] + halluc,
                'failures': failures, 'status': 'failed_fill'}

    # ── code node: assemble + emit + static-validate ──────────────────
    def assemble_emit(state: PipelineState) -> dict:
        out_dir = Path(state['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            resolved = resolve_slots(state['template'], state['fills'])
        except SlotError as e:
            failures = state['failures'] + [{'stage': 'assemble', 'reason': str(e)}]
            halluc = state['hallucination_incidents'] + (1 if 'invented' in str(e) else 0)
            # route the failure back to the offending slot-group when identifiable
            slot = _slot_from_sloterror(str(e))
            gi = _group_for_slot(state['groups'], slot) if slot else None
            if gi is not None and state['repair_rounds'] < policy.get('repair_max_rounds', 2):
                gname = state['groups'][gi]['group']
                fb = feedback_t['assemble'].format(slot=slot, error=e)
                return {'status': 'repair_assemble', 'group_idx': gi,
                        'repair_rounds': state['repair_rounds'] + 1,
                        'feedback': {**state['feedback'], gname: fb},
                        'failures': failures, 'hallucination_incidents': halluc}
            return {'status': 'failed_assemble', 'failures': failures,
                    'hallucination_incidents': halluc}
        py_path = write_python(state['template_id'], resolved,
                               out_dir / 'model.py', state['problem_id'])
        co_path = write_comsol(state['template_id'], resolved,
                               out_dir / 'model_comsol.py', state['problem_id'])
        # per-template required-defs (template's outputs.required_defs); None ->
        # py_static falls back to the gc_polyc set, so gc routing is unchanged.
        req_defs = (state['template'].get('outputs') or {}).get('required_defs')
        vp = validate_python(py_path.read_text(), req_defs)
        # per-template slot->COMSOL-param map (gc_polyc -> default -> identical).
        vc = (validate_comsol(co_path.read_text(),
                              slot_to_param=slot_to_param_for(state['template']))
              if co_path else None)  # None = template ships no COMSOL leg
        artifacts = {'python': str(py_path),
                     'comsol': str(co_path) if co_path else None,
                     'py_static': vp, 'comsol_static': vc}
        if vp:
            return {'resolved': resolved, 'artifacts': artifacts, 'status': 'failed_static',
                    'failures': state['failures'] + [{'stage': 'py_static', 'reason': vp}]}
        # reset any stale repair status so the post-assemble router proceeds
        return {'resolved': resolved, 'artifacts': artifacts, 'status': 'ok'}

    # ── code node: execute emitted python at its tier ──────────────────
    def execute(state: PipelineState) -> dict:
        t0 = time.time()
        py = state['artifacts']['python']
        try:
            proc = subprocess.run([sys.executable, py], cwd=str(Path(py).parent),
                                  capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return {'status': 'failed_exec', 'exec_result': {'error': 'timeout'},
                    'failures': state['failures'] + [{'stage': 'exec', 'reason': 'timeout'}]}
        if proc.returncode != 0:
            return {'status': 'failed_exec',
                    'exec_result': {'error': proc.stderr[-1500:]},
                    'failures': state['failures'] + [{'stage': 'exec',
                                                      'reason': proc.stderr[-300:]}]}
        res = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith('RESULT_JSON: '):
                res = json.loads(line[len('RESULT_JSON: '):])
                break
        if res is None:
            return {'status': 'failed_exec', 'exec_result': {'error': 'no RESULT_JSON'},
                    'failures': state['failures'] + [{'stage': 'exec',
                                                      'reason': 'no RESULT_JSON'}]}
        res['wall_time_s'] = round(time.time() - t0, 2)
        # Opt-in COMSOL leg: run the emitted model_comsol.py to produce a .mph. OFF by
        # default (env EFORGE_COMSOL_EXEC=1 to enable) so mock/llm certification is
        # byte-unchanged; needs a COMSOL engine + MPh runtime. Stashed into exec_result
        # only (no status/edge change) — purely diagnostic, never gates the pipeline.
        co = state['artifacts'].get('comsol')
        if os.environ.get('EFORGE_COMSOL_EXEC') == '1' and co:
            try:
                cp = subprocess.run([sys.executable, co], cwd=str(Path(co).parent),
                                    capture_output=True, text=True, timeout=1800)
                mph = sorted(p.name for p in Path(co).parent.glob('*.mph'))
                res['comsol_exec'] = {
                    'ok': cp.returncode == 0 and bool(mph),
                    'mph_produced': mph,
                    'tail': (cp.stderr or cp.stdout)[-500:]}
            except subprocess.TimeoutExpired:
                res['comsol_exec'] = {'ok': False, 'mph_produced': [], 'tail': 'timeout'}
        return {'exec_result': res}

    # ── code node: physics fast gates ──────────────────────────────────
    def gates(state: PipelineState) -> dict:
        results = fast_gates.run_fast_gates(state['exec_result'], state['resolved'],
                                            state['template_id'])
        ok = all(r['pass'] for r in results)
        upd: dict = {'gate_results': results}
        if ok:
            upd['status'] = 'ok'
            return upd
        # map first failing gate to the offending slot-group for repair
        fail = next(r for r in results if not r['pass'])
        target = {'sigma_vs_cg_single_facet': 'parameters_core',
                  'sigma_sign': 'parameters_core',
                  'cdl_positive': 'sweep',
                  # polyc_mpb2d gate ids (disjoint from gc_polyc's above, so .get
                  # cannot change gc routing). residual_converged has no target ->
                  # falls through to failed_gates, mirroring gc's solver-side gates.
                  'cg_cross_check': 'parameters_core',
                  'minima_count_exact': 'parameters_core',
                  'feature_position': 'parameters_core',
                  'magnitude': 'parameters_core',
                  'pz_slope': 'sweep',
                  'adsorption_shift': 'parameters_optional'}.get(fail['id'])
        if target and state['repair_rounds'] < policy.get('repair_max_rounds', 2):
            gi = next(i for i, g in enumerate(state['groups'])
                      if g['group'] == target or g['group'].startswith(target))
            fb = feedback_t['gates'].format(gate_id=fail['id'], detail=fail['detail'])
            upd.update({'status': 'repair', 'group_idx': gi,
                        'feedback': {**state['feedback'], target: fb},
                        'repair_rounds': state['repair_rounds'] + 1})
        else:
            upd['status'] = 'failed_gates'
            upd['failures'] = state['failures'] + [
                {'stage': 'gates', 'reason': [r for r in results if not r['pass']]}]
        return upd

    # ── wiring ─────────────────────────────────────────────────────────
    g = StateGraph(PipelineState)
    g.add_node('selector', selector)
    g.add_node('fill_group', fill_group)
    g.add_node('assemble_emit', assemble_emit)
    g.add_node('execute', execute)
    g.add_node('gates', gates)

    g.set_entry_point('selector')
    g.add_conditional_edges('selector',
                            lambda s: 'end' if s.get('status') in ('abstained', 'failed_selector')
                            else 'fill',
                            {'end': END, 'fill': 'fill_group'})
    g.add_conditional_edges('fill_group',
                            lambda s: 'more' if s['group_idx'] < len(s['groups'])
                            else 'done',
                            {'more': 'fill_group', 'done': 'assemble_emit'})
    g.add_conditional_edges('assemble_emit',
                            lambda s: 'repair' if s.get('status') == 'repair_assemble'
                            else 'end' if s.get('status', '').startswith('failed')
                            else 'exec',
                            {'repair': 'fill_group', 'end': END, 'exec': 'execute'})
    g.add_conditional_edges('execute',
                            lambda s: 'end' if s.get('status', '').startswith('failed')
                            else 'gates',
                            {'end': END, 'gates': 'gates'})
    g.add_conditional_edges('gates',
                            lambda s: 'repair' if s.get('status') == 'repair' else 'end',
                            {'repair': 'fill_group', 'end': END})
    return g.compile()


def run_pipeline(problem_text: str, problem_id: str, out_dir: str | Path,
                 profile: str | None = None) -> dict:
    client = LLMClient(profile)
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    client.trace_path = out_path / 'traces.jsonl'
    app = build_graph(client)
    init: PipelineState = {
        'problem_id': problem_id, 'problem_text': problem_text,
        'out_dir': str(Path(out_dir).resolve()), 'fills': {}, 'feedback': {},
        'repair_rounds': 0, 'failures': [], 'hallucination_incidents': 0,
        'status': 'ok',
    }
    final = app.invoke(init, config={'recursion_limit': 80})
    final['llm_usage'] = client.usage()
    return dict(final)
