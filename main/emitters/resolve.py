"""Slot resolution: merge LLM fills with template defaults; validate types/ranges.

Shared by the assembler node and both emitters. The output of resolve_slots()
is the complete, validated value set the Jinja2 skeletons consume.
"""

from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / 'templates'

FILLABLE_SECTIONS = ('parameters', 'parameters_optional', 'sweep', 'mesh_tier')


class SlotError(ValueError):
    """A fill violates the template contract (type/range/missing/invented)."""


def load_template(template_id: str) -> dict:
    reg = yaml.safe_load((TEMPLATES_DIR / 'registry.yaml').read_text())
    for t in reg['templates']:
        if t['id'] == template_id:
            if t['path'] is None:
                raise SlotError(f'template {template_id} is the abstain sentinel')
            return yaml.safe_load((TEMPLATES_DIR / t['path']).read_text())
    raise SlotError(f'unknown template_id {template_id!r}')


def template_dir(template_id: str) -> str:
    """Registry-relative directory of a template (e.g. 'gc_polyc')."""
    reg = yaml.safe_load((TEMPLATES_DIR / 'registry.yaml').read_text())
    for t in reg['templates']:
        if t['id'] == template_id and t['path']:
            return str(Path(t['path']).parent)
    raise SlotError(f'unknown or abstain template_id {template_id!r}')


def _all_slots(template: dict) -> dict:
    """name -> (slot_spec, section) for every fillable slot."""
    out = {}
    for sec in FILLABLE_SECTIONS:
        block = template.get(sec) or {}
        for name, spec in (block.get('slots') or {}).items():
            out[name] = (spec, sec)
    return out


def _check_one(name: str, spec: dict, val):
    typ = spec.get('type', 'float')
    rng_raw = spec.get('range')
    if rng_raw is not None:                     # PyYAML parses '1.0e4' as str
        spec = dict(spec, range=[float(r) for r in rng_raw])
    if typ == 'float':
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SlotError(f'{name}: expected float, got {type(val).__name__} {val!r}')
        val = float(val)
        rng = spec.get('range')
        if rng and not (rng[0] <= val <= rng[1]):
            raise SlotError(f'{name}={val} outside range {rng}')
    elif typ == 'int':
        if isinstance(val, bool) or not isinstance(val, int):
            raise SlotError(f'{name}: expected int, got {val!r}')
        rng = spec.get('range')
        if rng and not (rng[0] <= val <= rng[1]):
            raise SlotError(f'{name}={val} outside range {rng}')
    elif typ == 'enum':
        if val not in spec['choices']:
            raise SlotError(f'{name}={val!r} not in {spec["choices"]}')
    elif typ == 'list[float]':
        if not isinstance(val, list) or not val or \
                any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in val):
            raise SlotError(f'{name}: expected non-empty list of floats, got {val!r}')
        val = [float(v) for v in val]
    else:
        raise SlotError(f'{name}: unknown slot type {typ!r} in template')
    return val


def resolve_slots(template: dict, fills: dict) -> dict:
    """Merge fills with defaults; validate; return complete render context.

    Raises SlotError on: missing required slot, out-of-range/ill-typed value,
    or any fill key that is not a declared fillable slot (invented field —
    counted upstream as a hallucination_incident).
    """
    slots = _all_slots(template)

    invented = set(fills) - set(slots) - {'use_default'}
    if invented:
        raise SlotError(f'invented field(s): {sorted(invented)}')

    resolved = {}
    for name, (spec, _sec) in slots.items():
        if name in fills and fills[name] is not None:
            resolved[name] = _check_one(name, spec, fills[name])
        elif 'default' in spec:
            resolved[name] = spec['default']
        elif spec.get('required', False):
            raise SlotError(f'missing required slot {name}')
        else:
            raise SlotError(f'slot {name} has no value and no default')

    # Mesh tier -> resolved dict from the frozen table (never free-filled)
    table = template['mesh_tier']['resolved_table']
    resolved['mesh_resolved'] = dict(table[resolved['tier']])

    # Sweep sanity (cross-field)
    if resolved['sweep_var'] == 'E_M' and resolved['sweep_lo'] >= resolved['sweep_hi']:
        raise SlotError('sweep_lo must be < sweep_hi')
    return resolved
