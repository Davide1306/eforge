"""Aggregate scalar score + hard vetoes for keep/revert decisions."""

WEIGHTS = {
    'physics_pass_rate': 0.35,
    'slot_fill_accuracy': 0.25,
    'executability_rate': 0.15,
    'selection_accuracy': 0.15,
    'comsol_static_pass_rate': 0.10,
}


def score(metrics: dict) -> float:
    return sum(w * metrics.get(k, 0.0) for k, w in WEIGHTS.items())


def vetoes(metrics: dict, best_metrics: dict | None) -> list[str]:
    """Hard vetoes per gates.frozen.yaml `veto:` block (integrity hashes are
    checked by installer/verify.py, not here)."""
    v = []
    if best_metrics:
        if metrics.get('hallucination_incidents', 0) > \
                best_metrics.get('hallucination_incidents', 0):
            v.append('hallucination_incidents regressed')
        if metrics.get('executability_rate', 0) < \
                best_metrics.get('executability_rate', 0):
            v.append('executability_rate regressed')
    return v
