"""Static validation of emitted Python models (no execution)."""

import ast

IMPORT_ALLOWLIST = {'numpy', 'scipy', 'json', 'csv', 'math', 'sys', 'os'}

# Per-template required-defs fallback. When a caller passes no `required_defs`
# (e.g. gc_polyc_pb2d, whose template.yaml declares no outputs.required_defs),
# this gc_polyc set is enforced -- keeping gc_polyc behaviour byte-identical to
# the pre-generalization validator. Templates that DO declare a per-template
# `required_defs` (e.g. polyc_mpb2d) pass it through the call site, and it is
# enforced instead. `REQUIRED_DEFS` is kept as a back-compat alias of the gc set.
_GC_REQUIRED_DEFS = {'solve_pb_2d', 'residual_2d', 'build_jac_csr', 'main'}
REQUIRED_DEFS = _GC_REQUIRED_DEFS


def validate_python(source: str, required_defs=None) -> list[str]:
    """Return list of violations (empty = pass).

    `required_defs` is the per-template set of stable function names the emitted
    model must define (from the template's `outputs.required_defs`). When None or
    empty, the gc_polyc fallback set `_GC_REQUIRED_DEFS` is enforced -- so the
    gc_polyc call (which passes no list) is unchanged.
    """
    req = set(required_defs) if required_defs else _GC_REQUIRED_DEFS
    problems = []
    if '{{' in source or '}}' in source:
        problems.append('unrendered Jinja2 markers present')
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return problems + [f'syntax error: {e}']

    defs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split('.')[0]
                if root not in IMPORT_ALLOWLIST:
                    problems.append(f'forbidden import: {a.name}')
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or '').split('.')[0]
            if root not in IMPORT_ALLOWLIST:
                problems.append(f'forbidden import-from: {node.module}')
        elif isinstance(node, ast.FunctionDef):
            defs.add(node.name)

    missing = req - defs
    if missing:
        problems.append(f'missing required functions: {sorted(missing)}')

    if 'RESULT_JSON' not in source:
        problems.append('missing RESULT_JSON output contract')
    return problems
