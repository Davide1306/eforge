"""Render the COMSOL (MPh) model-builder script from resolved slot values."""

from pathlib import Path

import jinja2

from emitters.resolve import template_dir

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / 'templates'

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
)


def emit_comsol(template_id: str, resolved: dict, problem_id: str = 'unknown') -> str | None:
    """Returns None when the template ships no COMSOL skeleton (formula-only models)."""
    tdir = template_dir(template_id)
    if not (TEMPLATES_DIR / tdir / 'comsol_skeleton.py.j2').exists():
        return None
    tpl = _env.get_template(f'{tdir}/comsol_skeleton.py.j2')
    src = tpl.render(problem_id=problem_id, **resolved)
    if '{{' in src or '}}' in src:
        raise RuntimeError('unrendered Jinja2 markers remain in emitted COMSOL script')
    return src


def write_comsol(template_id: str, resolved: dict, out_path: str | Path,
                 problem_id: str = 'unknown') -> Path | None:
    src = emit_comsol(template_id, resolved, problem_id)
    if src is None:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    tmp.write_text(src)
    tmp.rename(out_path)
    return out_path
