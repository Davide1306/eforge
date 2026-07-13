"""Shared matplotlib style + atomic dual PNG/PDF export for PPTX targets.

`apply_pptx_style()` installs rcParams sized for figures embedded into a
16:9 PowerPoint slide deck per `IMG_Refine/QC_CHECKLIST.md` §1 and §3:
single sans-serif family, slide-appropriate sizes, ≥0.5 pt axis lines,
TrueType font embedding for PDF/PS, opaque white background.

`save_figure(fig, basename)` saves `fig` as both `<basename>.png`
(raster, 300 dpi) and `<basename>.pdf` (vector, embedded TrueType) into
`OUTPUT_DIR` (env `GCPOLYC_OUTPUT_DIR`, default `output/`).  Each file
is written `.tmp` then `mv`d into place per the CLAUDE.md atomic-write
rule, so concurrent readers never see a half-written file.
"""

from __future__ import annotations

import os
import matplotlib


OUTPUT_DIR = os.environ.get('GCPOLYC_OUTPUT_DIR', 'output')


def apply_pptx_style() -> None:
    """Install slide-grade rcParams.  Idempotent; safe to call repeatedly."""
    matplotlib.rcParams.update({
        # Single sans-serif family.  Arial / Helvetica preferred when
        # installed; matplotlib falls back through the list if not.
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans',
                              'Liberation Sans', 'Bitstream Vera Sans'],
        # Slide-target sizes per QC §1.
        'font.size':         11.0,
        'axes.titlesize':    14.0,
        'axes.labelsize':    12.0,
        'xtick.labelsize':   11.0,
        'ytick.labelsize':   11.0,
        'legend.fontsize':   10.0,
        'figure.titlesize':  14.0,
        # Embed TrueType subsets so PDF / PS text remains text (not Type 3
        # outlines), per QC §3.
        'pdf.fonttype':      42,
        'ps.fonttype':       42,
        # Line weights ≥ 0.5 pt (axes) / 1 pt (data), QC §1.
        'axes.linewidth':    0.8,
        'lines.linewidth':   1.5,
        'lines.markersize':  5.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        # Opaque white background everywhere — both the matplotlib default
        # 0.95 light gray and any inherited transparency are disallowed by
        # QC §3 for slide embedding.
        'figure.facecolor':  'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'axes.facecolor':    'white',
        # save_figure() passes dpi explicitly for PNGs; this is the rcParam
        # default for any direct fig.savefig() callers that bypass it.
        'figure.dpi':        100,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.05,
    })


def save_figure(fig, basename: str, *, dpi_png: int = 300) -> tuple[str, str, str]:
    """Save `fig` as `<basename>.png`, `<basename>.pdf`, and `<basename>.svg`
    in OUTPUT_DIR.

    `basename` is the output file stem without extension.  Absolute
    paths are honored; relative paths are joined to OUTPUT_DIR.  Each
    file is written to `<final>.tmp` and then `os.replace`d to its
    final name (atomic on POSIX) so partial writes are never visible.

    Returns the (png_path, pdf_path, svg_path) tuple of final paths.
    """
    stem = basename if os.path.isabs(basename) else os.path.join(OUTPUT_DIR, basename)
    parent = os.path.dirname(stem) or '.'
    os.makedirs(parent, exist_ok=True)

    paths: dict[str, str] = {}
    for ext in ('png', 'pdf', 'svg'):
        final = f'{stem}.{ext}'
        tmp = f'{final}.tmp'
        # `format=` is explicit so the `.tmp` suffix doesn't confuse the
        # matplotlib backend dispatcher.
        kwargs: dict = {'facecolor': 'white', 'format': ext}
        # PNG dpi is the raster resolution. PDF/SVG are pure vector — no
        # rasterized=True artists remain — so dpi only affects the 600 dpi
        # default canvas DPI used by matplotlib when laying out hi-res PNG.
        if ext == 'png':
            kwargs['dpi'] = dpi_png
        fig.savefig(tmp, **kwargs)
        os.replace(tmp, final)
        paths[ext] = final
        print(f'  saved {final}')
    return paths['png'], paths['pdf'], paths['svg']


def format_conditions(*, E_M=None, delta_E_pzc=None, c_b_mM=None,
                      L_nm=None, x=None, T_K=298.15) -> str:
    """Compose a one-line LaTeX condition string for figure textboxes."""
    parts: list[str] = []
    if E_M is not None:
        parts.append(rf'$E_\mathrm{{M}}={E_M:g}$ V')
    if delta_E_pzc is not None:
        parts.append(rf'$\Delta E_\mathrm{{pzc}}={delta_E_pzc:g}$ V')
    if c_b_mM is not None:
        parts.append(rf'$c_b={c_b_mM:g}$ mM')
    if L_nm is not None:
        parts.append(rf'$L={L_nm:g}$ nm')
    if x is not None:
        parts.append(rf'$x={x:g}$')
    if T_K is not None:
        parts.append(rf'$T={T_K:g}$ K')
    return ', '.join(parts)
