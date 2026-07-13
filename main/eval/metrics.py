"""Battery-level metric aggregation."""

REL_TOL_FILL = 1e-9


def _fills_match(got: dict, expected: dict) -> tuple[int, int]:
    """(n_match, n_total) over expected slots, with float tolerance."""
    n_match = 0
    for k, v in expected.items():
        g = got.get(k)
        if isinstance(v, float) and isinstance(g, (int, float)):
            ref = abs(v) if v else 1.0
            if abs(float(g) - v) / ref < REL_TOL_FILL:
                n_match += 1
        elif isinstance(v, list) and isinstance(g, list) and len(v) == len(g):
            if all(abs(float(a) - float(b)) <= REL_TOL_FILL * max(abs(float(b)), 1.0)
                   for a, b in zip(g, v)):
                n_match += 1
        elif g == v:
            n_match += 1
    return n_match, len(expected)


def compute_metrics(records: list[dict]) -> dict:
    """records: one per battery problem (see harness.run_one for shape)."""
    n = len(records)
    sel_ok = sum(1 for r in records
                 if r['template_id_got'] == r['template_id_expected'])
    runnable = [r for r in records if r['template_id_expected'] != 'no_match']

    fill_match = fill_total = 0
    for r in runnable:
        if r.get('resolved_got') and r.get('resolved_expected'):
            m, t = _fills_match(r['resolved_got'], r['resolved_expected'])
            fill_match += m
            fill_total += t
        elif r.get('resolved_expected'):
            fill_total += len(r['resolved_expected'])

    exec_ok = sum(1 for r in runnable if r.get('exec_ok'))
    gates_ok = sum(1 for r in runnable if r.get('gates_ok'))
    # comsol rate over APPLICABLE records only (None = template has no COMSOL leg)
    comsol_app = [r for r in runnable if r.get('comsol_static_ok') is not None]
    comsol_ok = sum(1 for r in comsol_app if r['comsol_static_ok'])
    # NON-GATING: COMSOL execution rate over records where it was actually attempted
    # (comsol_exec_ok not None). Diagnostic only — NOT a scorer weight, NOT a veto.
    cexec_app = [r for r in runnable if r.get('comsol_exec_ok') is not None]
    cexec_ok = sum(1 for r in cexec_app if r['comsol_exec_ok'])
    halluc = sum(r.get('hallucination_incidents', 0) for r in records)
    repair = sum(r.get('repair_rounds', 0) for r in records)
    tokens = sum(r.get('tokens_total', 0) for r in records)
    wall = [r['wall_time_s'] for r in records if r.get('wall_time_s')]

    nr = max(len(runnable), 1)
    return {
        'n_problems': n,
        'selection_accuracy': sel_ok / max(n, 1),
        'slot_fill_accuracy': fill_match / max(fill_total, 1),
        'executability_rate': exec_ok / nr,
        'physics_pass_rate': gates_ok / nr,
        'comsol_static_pass_rate': comsol_ok / len(comsol_app) if comsol_app else 1.0,
        'comsol_exec_pass_rate': (cexec_ok / len(cexec_app)) if cexec_app else None,
        'hallucination_incidents': halluc,
        'repair_rounds_total': repair,
        'tokens_total': tokens,
        'tokens_per_problem': tokens / max(n, 1),
        'mean_wall_time_s': sum(wall) / max(len(wall), 1),
    }
