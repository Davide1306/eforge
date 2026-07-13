"""Per-template physics-gate dispatcher.

Gate implementations live in eval/gates/<template_id>.py (each exposing
run(exec_result, resolved) -> [{id, pass, detail}, ...]) and are frozen via
integrity-pinned once a template is consolidated. This dispatcher stays thin and
template-agnostic.
"""

import importlib


def run_fast_gates(exec_result: dict, resolved: dict, template_id: str) -> list[dict]:
    mod = importlib.import_module(f'eval.gates.{template_id}')
    return mod.run(exec_result, resolved)
