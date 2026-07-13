"""Static validation of emitted COMSOL (MPh) scripts — license-free.

Four checks (cannot execute without COMSOL):
  1. valid Python (ast.parse) + no unrendered markers
  2. `import mph` present; pymodel.<method> calls limited to the introspected
     MPh API surface (api_surface_mph.txt); model.java.* chains are allowed
     structurally (the Java API surface is not lintable without COMSOL)
  3. parameter completeness: every required template slot appears in a
     pymodel.parameter('<name>', ...) call (slot 'x' maps to 'x_frac')
  4. exactly two facet boundary selections referenced
"""

import ast
from pathlib import Path

API_SURFACE_FILE = Path(__file__).with_name('api_surface_mph.txt')

# slot name -> COMSOL parameter name in the gc_polyc skeleton (DEFAULT map).
SLOT_TO_PARAM = {
    'E_M': 'E_M', 'E_pzc_1': 'E_pzc_1', 'E_pzc_2': 'E_pzc_2',
    'C_H_1': 'C_H_1', 'C_H_2': 'C_H_2', 'x': 'x_frac',
    'c_b': 'c_b', 'L': 'L',
}

# slot name -> COMSOL parameter name in the polyc_mpb2d skeleton. Extends the
# gc_polyc map with the full-physics adsorption inputs (E_ads0, theta_max), which
# the polyc COMSOL builder always emits as parameters (resolved from defaults even
# when adsorption is OFF, so parameter-completeness holds either way).
POLYC_SLOT_TO_PARAM = {
    'E_M': 'E_M', 'E_pzc_1': 'E_pzc_1', 'E_pzc_2': 'E_pzc_2',
    'C_H_1': 'C_H_1', 'C_H_2': 'C_H_2', 'x': 'x_frac',
    'c_b': 'c_b', 'L': 'L',
    'E_ads0': 'E_ads0', 'theta_max': 'theta_max',
}

# template_id -> built-in slot->param map (used when template.yaml carries no
# explicit `comsol.slot_to_param` mapping).
_BUILTIN_MAPS = {
    'gc_polyc_pb2d': SLOT_TO_PARAM,
    'polyc_mpb2d': POLYC_SLOT_TO_PARAM,
}


def slot_to_param_for(template) -> dict:
    """Resolve the slot->COMSOL-param map for a template (back-compatible).

    Accepts a loaded template dict (e.g. graph.py's ``state['template']``) or a
    template_id string. Preference order: an explicit ``comsol.slot_to_param``
    mapping in the template.yaml; then the built-in map keyed by template_id;
    then the gc_polyc default. gc_polyc resolves to the default either way, so the
    legacy call ``validate_comsol(src)`` is byte-identical.
    """
    if isinstance(template, dict):
        comsol = template.get('comsol') or {}
        explicit = comsol.get('slot_to_param')
        if isinstance(explicit, dict) and explicit:
            return dict(explicit)
        tid = template.get('template_id')
        return _BUILTIN_MAPS.get(tid, SLOT_TO_PARAM)
    if isinstance(template, str):
        return _BUILTIN_MAPS.get(template, SLOT_TO_PARAM)
    return SLOT_TO_PARAM


def _mph_api() -> set[str]:
    return {l.strip() for l in API_SURFACE_FILE.read_text().splitlines()
            if l.strip() and not l.startswith('#')}


def validate_comsol(source: str, required_slots=None, slot_to_param=None) -> list[str]:
    """Return list of violations (empty = pass).

    ``slot_to_param`` is the per-template slot->COMSOL-param map; it defaults to
    the gc_polyc ``SLOT_TO_PARAM``. ``required_slots`` defaults to every key of
    the resolved map. Both defaults reproduce the original gc_polyc behaviour, so
    ``validate_comsol(src)`` is unchanged.
    """
    if slot_to_param is None:
        slot_to_param = SLOT_TO_PARAM
    if required_slots is None:
        required_slots = tuple(slot_to_param)
    problems = []
    if '{{' in source or '}}' in source:
        problems.append('unrendered Jinja2 markers present')
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return problems + [f'syntax error: {e}']

    has_mph = any(isinstance(n, ast.Import) and any(a.name == 'mph' for a in n.names)
                  for n in ast.walk(tree))
    if not has_mph:
        problems.append('missing `import mph`')

    api = _mph_api()
    param_names = set()
    facet_selections = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            obj, meth = fn.value.id, fn.attr
            # check 2: pymodel/client direct method calls vs MPh surface
            if obj in ('pymodel', 'client') and meth not in api:
                problems.append(f'unknown MPh API call: {obj}.{meth}()')
            # check 3: collect parameter names
            if obj == 'pymodel' and meth == 'parameter' and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    param_names.add(a0.value)
        # check 4: boundary_facetN string literals anywhere
        for arg in ast.walk(node):
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value.startswith('boundary_facet'):
                facet_selections.add(arg.value)

    missing = [s for s in required_slots if slot_to_param[s] not in param_names]
    if missing:
        problems.append(f'slots missing from model.parameter() calls: {missing}')

    if len(facet_selections) != 2:
        problems.append(f'expected exactly 2 facet boundary selections, '
                        f'found {sorted(facet_selections)}')
    return problems
