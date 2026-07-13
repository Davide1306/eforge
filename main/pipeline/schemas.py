"""JSON-schema builders derived from template slot specs.

The SAME schema drives constrained decoding (Ollama format= / response_format)
and post-hoc validation, so the contract can never drift between them.
"""

from emitters.resolve import FILLABLE_SECTIONS


def _slot_to_prop(name: str, spec: dict) -> dict:
    typ = spec.get('type', 'float')
    desc = f"{spec.get('desc', '')} Unit: {spec.get('unit', '-')}.".strip()
    if typ == 'float':
        p = {'type': 'number', 'description': desc}
        if spec.get('range'):
            p['minimum'], p['maximum'] = [float(r) for r in spec['range']]
    elif typ == 'int':
        p = {'type': 'integer', 'description': desc}
        if spec.get('range'):
            p['minimum'], p['maximum'] = [int(r) for r in spec['range']]
    elif typ == 'enum':
        p = {'type': 'string', 'enum': list(spec['choices']), 'description': desc}
    elif typ == 'list[float]':
        p = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 1,
             'description': desc}
    else:
        raise ValueError(f'unknown slot type {typ!r} for {name}')
    return p


def group_schemas(template: dict, chunk_mode: str = 'per_group') -> list[dict]:
    """Ordered list of {group, schema, required_slots, optional_slots}.

    chunk_mode: per_group (default) | per_slot (one schema per slot) |
                merged (single schema with every fillable slot).
    """
    groups = []
    for sec in FILLABLE_SECTIONS:
        block = template.get(sec) or {}
        slots = block.get('slots') or {}
        if not slots:
            continue
        props, req = {}, []
        for name, spec in slots.items():
            props[name] = _slot_to_prop(name, spec)
            if spec.get('required', False):
                req.append(name)
        groups.append({
            'group': block.get('group', sec),
            'schema': {'type': 'object', 'properties': props,
                       'required': req, 'additionalProperties': False},
            'slots': list(slots),
        })

    if chunk_mode == 'per_group':
        return groups
    if chunk_mode == 'merged':
        props, req, names = {}, [], []
        for g in groups:
            props.update(g['schema']['properties'])
            req += g['schema']['required']
            names += g['slots']
        return [{'group': 'merged', 'slots': names,
                 'schema': {'type': 'object', 'properties': props, 'required': req,
                            'additionalProperties': False}}]
    if chunk_mode == 'per_slot':
        out = []
        for g in groups:
            for name in g['slots']:
                prop = g['schema']['properties'][name]
                req = [name] if name in g['schema']['required'] else []
                out.append({'group': f"{g['group']}.{name}", 'slots': [name],
                            'schema': {'type': 'object', 'properties': {name: prop},
                                       'required': req, 'additionalProperties': False}})
        return out
    raise ValueError(f'unknown chunk_mode {chunk_mode!r}')


SELECTOR_SCHEMA = {
    'type': 'object',
    'properties': {'template_id': {'type': 'string'}},
    'required': ['template_id'],
    'additionalProperties': False,
}

CHECKER_SCHEMA = {
    'type': 'object',
    'properties': {
        'ok': {'type': 'boolean'},
        'reason': {'type': 'string'},
        'corrections': {'type': 'object'},
    },
    'required': ['ok', 'reason'],
    'additionalProperties': False,
}
