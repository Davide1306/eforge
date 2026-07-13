"""Render the standalone Python model from resolved slot values (deterministic)."""

from pathlib import Path

import jinja2

from emitters.resolve import template_dir

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / 'templates'

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
    undefined=jinja2.StrictUndefined,   # missing slot -> hard error, never silent
    keep_trailing_newline=True,
)


def emit_python(template_id: str, resolved: dict, problem_id: str = 'unknown') -> str:
    """resolved = output of emitters.resolve.resolve_slots(). Returns source text."""
    tpl = _env.get_template(f'{template_dir(template_id)}/python_skeleton.py.j2')
    src = tpl.render(problem_id=problem_id, **resolved)
    if '{{' in src or '}}' in src:
        raise RuntimeError('unrendered Jinja2 markers remain in emitted source')
    return src


def write_python(template_id: str, resolved: dict, out_path: str | Path,
                 problem_id: str = 'unknown') -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    tmp.write_text(emit_python(template_id, resolved, problem_id))
    tmp.rename(out_path)   # atomic
    return out_path
