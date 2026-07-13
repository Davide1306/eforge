"""F1–F5 figure generation routines (GC-polyc tier G5+).

Outputs saved to GCPOLYC_OUTPUT_DIR (default: output/).
Each plot routine uses `src.style.save_figure` to emit both
`<basename>.png` (300 dpi) and `<basename>.pdf` (vector, embedded
TrueType) under PPTX-targeted rcParams (`src.style.apply_pptx_style`).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.style import apply_pptx_style, save_figure, format_conditions

OUTPUT_DIR = os.environ.get('GCPOLYC_OUTPUT_DIR', 'output')

apply_pptx_style()


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── F2 — C_dl(E_M) sweeps, all 4 panels ──────────────────────────────

# Panel definitions (a-d) per CONSTRAINTS §3
F2_PANELS = {
    'a': {'delta_E_pzc': 0.6, 'c_b': 0.1},
    'b': {'delta_E_pzc': 0.6, 'c_b': 10.0},
    'c': {'delta_E_pzc': 0.2, 'c_b': 0.1},
    'd': {'delta_E_pzc': 0.2, 'c_b': 10.0},
}
F2_L_VALS = [1e-9, 10e-9, 100e-9]
F2_PANEL_LS_NM = (1, 10, 100)
# Article ±50 mV tolerance band on predicted-minima positions
# (CONSTRAINTS §3 F2 ACCEPT line).
F2_MIN_TOL_V = 0.05


def run_F2_panel_sweep(delta_E_pzc: float, c_b: float,
                       mesh: dict, solver_method: str = 'sparse',
                       n_em: int = 41, verbose: bool = False):
    """Sweep E_M at 3 L values for one F2 panel; return data dict keyed by L."""
    from src.gc_pb_2d import solve_pb_2d
    from src.parameters import make_f2_case

    E_M_VALS = np.linspace(-0.5, 0.5, n_em)
    data = {}
    for L in F2_L_VALS:
        sigma_vals, conv = [], []
        for E_M in E_M_VALS:
            case = make_f2_case(E_M=E_M, delta_E_pzc=delta_E_pzc, c_b=c_b, L=L)
            _, _, sb, result = solve_pb_2d(case.to_solve_dict(), mesh,
                                           solver_method=solver_method)
            sigma_vals.append(sb)
            ok = result.success and float(np.linalg.norm(result.fun)) < 1e-6
            conv.append(ok)
            if verbose:
                print(f'  L={L*1e9:.0f}nm E_M={E_M:+.3f}: ok={ok}')
        sigma_arr = np.array(sigma_vals)
        C_dl = np.gradient(sigma_arr, E_M_VALS)
        data[L] = {'E_M': E_M_VALS, 'sigma': sigma_arr,
                   'C_dl': C_dl, 'converged': np.array(conv)}
    return data


def run_F2b_sweep(mesh: dict, solver_method: str = 'sparse',
                  n_em: int = 41, verbose: bool = False):
    """Sweep E_M for F2 panel (b) — wrapper for backward compatibility."""
    return run_F2_panel_sweep(0.6, 10.0, mesh, solver_method, n_em, verbose)


def save_F2b_csvs(data: dict):
    """Write output/F2b_L<N>nm.csv for each L in data."""
    _ensure_output()
    for L, d in data.items():
        fname = os.path.join(OUTPUT_DIR, f'F2b_L{round(L*1e9):.0f}nm.csv')
        np.savetxt(fname,
                   np.column_stack([d['E_M'], d['sigma'], d['C_dl']]),
                   header='E_M_V,sigma_bar_Cm2,C_dl_Fm2', delimiter=',',
                   comments='')
        print(f'  saved {fname}')


def plot_F2b_panels(data: dict, fname: str = 'F2b_panels'):
    """Save output/F2b_panels.{png,pdf} — C_dl vs E_M, one panel, 3 L lines."""
    _ensure_output()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {'1': '#1f77b4', '10': '#ff7f0e', '100': '#2ca02c'}
    for L, d in sorted(data.items()):
        label = f'L={round(L*1e9):.0f} nm'
        lkey = f'{round(L*1e9):.0f}'
        C_dl_uF_cm2 = d['C_dl'] * 1e2   # F/m² → µF/cm²  (1 µF/cm²=0.01 F/m²)
        ax.plot(d['E_M'], C_dl_uF_cm2, label=label,
                color=colors.get(lkey), marker='o', ms=3)
    ax.set_xlabel('$E_M$ / V vs SHE')
    ax.set_ylabel('$C_{dl}$ / µF cm$^{-2}$')
    ax.set_title(r'F2(b): $\Delta E_{pzc}=0.6$ V, $c_b=10$ mM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, fname)
    plt.close(fig)


def save_F2_panel_csvs(panel: str, data: dict):
    """Write output/F2<panel>_L<N>nm.csv for each L in data."""
    _ensure_output()
    for L, d in data.items():
        fname = os.path.join(OUTPUT_DIR, f'F2{panel}_L{round(L*1e9):.0f}nm.csv')
        np.savetxt(fname,
                   np.column_stack([d['E_M'], d['sigma'], d['C_dl']]),
                   header='E_M_V,sigma_bar_Cm2,C_dl_Fm2', delimiter=',',
                   comments='')
        print(f'  saved {fname}')


def _load_F2_csvs(csv_dir: str | None = None) -> dict:
    """Read cached `F2{a,b,c,d}_L{1,10,100}nm.csv` back into the dict
    shape expected by `plot_F2_4panels`:
    ``data_by_panel[panel_letter][L_meters] -> {'E_M', 'sigma', 'C_dl'}``.
    """
    csv_dir = csv_dir or OUTPUT_DIR
    data_by_panel: dict = {}
    for panel in 'abcd':
        data_by_panel[panel] = {}
        for L_nm in F2_PANEL_LS_NM:
            path = os.path.join(csv_dir, f'F2{panel}_L{L_nm}nm.csv')
            raw = np.loadtxt(path, delimiter=',', skiprows=1)
            data_by_panel[panel][L_nm * 1e-9] = {
                'E_M': raw[:, 0],
                'sigma': raw[:, 1],
                'C_dl': raw[:, 2],
            }
    return data_by_panel


def _detect_local_minima(y: np.ndarray, edge: int = 1) -> list[int]:
    """Indices of strict local minima of `y`, skipping the first/last
    `edge` samples so endpoint dips do not register as minima."""
    return [i for i in range(edge, len(y) - edge)
            if y[i] < y[i - 1] and y[i] < y[i + 1]]


def plot_F2_4panels(data_by_panel: dict, fname: str = 'F2_4panels'):
    """Save `output/F2_4panels.{png,pdf}` — 2×2 capacitance matrix.

    Slide-polished: no main title, no per-panel titles, no grid, no
    E_M=0 line, no tolerance bands, no minima arrows. Conditions are
    conveyed by row labels (ΔE_pzc) and column labels (c_b). Lines are
    smooth (no markers). Single shared legend placed outside the axes
    on the right, included in the image bbox.
    """
    _ensure_output()
    # Okabe-Ito CVD-safe categorical palette (blue / vermillion / green).
    colors = {'1': '#0072B2', '10': '#D55E00', '100': '#009E73'}
    # Panel layout: row = ΔE_pzc, column = c_b.
    #   (a)  ΔE=0.6, c_b=0.1    (b)  ΔE=0.6, c_b=10
    #   (c)  ΔE=0.2, c_b=0.1    (d)  ΔE=0.2, c_b=10
    panel_grid = [['a', 'b'], ['c', 'd']]
    row_labels = [r'$\Delta E_\mathrm{pzc} = 0.6$ V',
                  r'$\Delta E_\mathrm{pzc} = 0.2$ V']
    col_labels = [r'$c_b = 0.1$ mM', r'$c_b = 10$ mM']

    FS = 18
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             sharex=True, sharey=True)
    for r_idx, row in enumerate(panel_grid):
        for c_idx, panel in enumerate(row):
            ax = axes[r_idx, c_idx]
            data = data_by_panel.get(panel, {})
            for L, d in sorted(data.items()):
                lkey = f'{round(L*1e9):.0f}'
                C_dl_uF_cm2 = d['C_dl'] * 1e2
                ax.plot(d['E_M'], C_dl_uF_cm2,
                        label=f'L = {lkey} nm',
                        color=colors.get(lkey), lw=2.0)
            ax.set_xlabel(r'$E_\mathrm{M}$ (V vs SHE)', fontsize=FS)
            ax.set_ylabel(r'$C_\mathrm{dl}$ (µF cm$^{-2}$)', fontsize=FS)
            ax.tick_params(labelsize=FS)
            ax.label_outer()

    for c_idx, label in enumerate(col_labels):
        axes[0, c_idx].annotate(
            label, xy=(0.5, 1.04), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=FS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(0.99, 0.5), fontsize=FS,
               frameon=True, framealpha=0.95)

    fig.tight_layout(rect=(0, 0, 0.98, 0.96))

    fig.canvas.draw()
    y_top = axes[0, 1].get_position().y0 + axes[0, 1].get_position().height / 2
    y_bot = axes[1, 1].get_position().y0 + axes[1, 1].get_position().height / 2
    fig.text(0.99, y_top, row_labels[0],
             ha='left', va='center', fontsize=FS)
    fig.text(0.99, y_bot, row_labels[1],
             ha='left', va='center', fontsize=FS)
    save_figure(fig, fname)
    plt.close(fig)


# ── F4 — Parsons-Zobel plots ──────────────────────────────────────────

F4_CB_VALS   = [0.1, 0.3, 1.0, 10.0]    # mol/m³ = mM
F4_L_VALS    = [1e-9, 10e-9, 30e-9, 100e-9]
F4_DEPZC_VALS = [0.0, 0.2, 0.4, 0.6]
_F4_DEM = 0.05   # V half-step for C_dl FD at E_M=0


def _cgc_min(c_b: float, T: float = 298.15) -> float:
    """C_GC minimum = ε₀εr/λ_D [F/m²]."""
    from src import EPS0, EPS_R_WATER, R, F as FARADAY
    lam_D = np.sqrt(EPS0 * EPS_R_WATER * R * T / (2.0 * FARADAY**2 * c_b))
    return EPS0 * EPS_R_WATER / lam_D


def _cdl_at_pzc(L: float, c_b: float, delta_E_pzc: float,
                mesh: dict, solver_method: str) -> float:
    """C_dl at E_M=0 via 2-pt central FD over ±_F4_DEM."""
    from src.gc_pb_2d import solve_pb_2d
    from src.parameters import make_f2_case
    sb = []
    for E_M in [-_F4_DEM, +_F4_DEM]:
        case = make_f2_case(E_M=E_M, delta_E_pzc=delta_E_pzc, c_b=c_b, L=L)
        _, _, s, _ = solve_pb_2d(case.to_solve_dict(), mesh,
                                  solver_method=solver_method)
        sb.append(s)
    return (sb[1] - sb[0]) / (2.0 * _F4_DEM)


def compute_F4_data(mesh: dict, solver_method: str = 'sparse', verbose: bool = False):
    """Compute F4 Parsons-Zobel data for panels (a) and (b).

    Returns
    -------
    panel_a : dict[L → dict[c_b → {'C_dl', 'C_GC_min'}]]   (ΔE_pzc=0.2V fixed)
    panel_b : dict[ΔE_pzc → dict[c_b → {'C_dl', 'C_GC_min'}]]  (L=30nm fixed)
    """
    panel_a, panel_b = {}, {}
    for L in F4_L_VALS:
        panel_a[L] = {}
        for c_b in F4_CB_VALS:
            C_dl = _cdl_at_pzc(L, c_b, 0.2, mesh, solver_method)
            C_GC = _cgc_min(c_b)
            panel_a[L][c_b] = {'C_dl': C_dl, 'C_GC_min': C_GC}
            if verbose:
                print(f'  F4a L={L*1e9:.0f}nm cb={c_b}: 1/Cdl={1/C_dl*1e-4:.4f} 1/CGC={1/C_GC*1e-4:.4f} cm²/µF')
    for dEpzc in F4_DEPZC_VALS:
        panel_b[dEpzc] = {}
        for c_b in F4_CB_VALS:
            C_dl = _cdl_at_pzc(30e-9, c_b, dEpzc, mesh, solver_method)
            C_GC = _cgc_min(c_b)
            panel_b[dEpzc][c_b] = {'C_dl': C_dl, 'C_GC_min': C_GC}
            if verbose:
                print(f'  F4b dE={dEpzc}V cb={c_b}: 1/Cdl={1/C_dl*1e-4:.4f} 1/CGC={1/C_GC*1e-4:.4f} cm²/µF')
    return panel_a, panel_b


def pz_slope(points_by_cb: dict) -> float:
    """Linear fit slope of 1/C_dl vs 1/C_GC_min over the 4 c_b points."""
    xs = np.array([1.0 / points_by_cb[cb]['C_GC_min'] for cb in sorted(points_by_cb)])
    ys = np.array([1.0 / points_by_cb[cb]['C_dl']     for cb in sorted(points_by_cb)])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def save_F4_csvs(panel_a: dict, panel_b: dict):
    """Save F4 data as CSV files."""
    _ensure_output()
    for L, d in panel_a.items():
        fname = os.path.join(OUTPUT_DIR, f'F4a_L{round(L*1e9):.0f}nm.csv')
        rows = [(cb, d[cb]['C_dl'], d[cb]['C_GC_min']) for cb in sorted(d)]
        np.savetxt(fname, rows, header='c_b_mM,C_dl_Fm2,C_GC_min_Fm2',
                   delimiter=',', comments='')
        print(f'  saved {fname}')
    for dE, d in panel_b.items():
        fname = os.path.join(OUTPUT_DIR, f'F4b_dEpzc{dE:.1f}V.csv')
        rows = [(cb, d[cb]['C_dl'], d[cb]['C_GC_min']) for cb in sorted(d)]
        np.savetxt(fname, rows, header='c_b_mM,C_dl_Fm2,C_GC_min_Fm2',
                   delimiter=',', comments='')
        print(f'  saved {fname}')


F4_L_NM   = (1, 10, 30, 100)
F4_DE_VAL = (0.0, 0.2, 0.4, 0.6)


def _load_F4_csvs(csv_dir: str | None = None) -> tuple[dict, dict]:
    """Read cached `F4a_L*nm.csv` + `F4b_dEpzc*V.csv` back into the
    `(panel_a, panel_b)` dict shape expected by `plot_F4_panels`:

        panel_a[L_meters][c_b_mM]      -> {'C_dl', 'C_GC_min'}
        panel_b[delta_E_pzc][c_b_mM]   -> {'C_dl', 'C_GC_min'}
    """
    csv_dir = csv_dir or OUTPUT_DIR
    panel_a: dict = {}
    for L_nm in F4_L_NM:
        path = os.path.join(csv_dir, f'F4a_L{L_nm}nm.csv')
        raw = np.loadtxt(path, delimiter=',', skiprows=1)
        L_m = L_nm * 1e-9
        panel_a[L_m] = {float(row[0]): {'C_dl': float(row[1]),
                                          'C_GC_min': float(row[2])}
                        for row in raw}
    panel_b: dict = {}
    for dE in F4_DE_VAL:
        path = os.path.join(csv_dir, f'F4b_dEpzc{dE:.1f}V.csv')
        raw = np.loadtxt(path, delimiter=',', skiprows=1)
        panel_b[dE] = {float(row[0]): {'C_dl': float(row[1]),
                                         'C_GC_min': float(row[2])}
                       for row in raw}
    return panel_a, panel_b


def plot_F4_panels(panel_a: dict, panel_b: dict, fname: str = 'F4_panels'):
    """Save `output/F4_panels.{png,pdf}` — two Parsons-Zobel panels stacked.

    Slide-polished: no titles, no grid, no GC-limit reference line.
    ScalarFormatter ×10⁻³ exponent ticks. Legends placed outside each
    panel on the right, included in the image bbox. The PZ slope is
    quoted numerically in each legend label.

    Units: `1 m²/F = 10⁻² cm² µF⁻¹`.
    """
    _ensure_output()
    # Horizontal layout: panel (a) and (b) side-by-side, shared y-axis.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Okabe-Ito CVD-safe palette (blue, vermillion, green, sky-blue).
    palette  = ['#0072B2', '#D55E00', '#009E73', '#56B4E9']
    colors_L = dict(zip([1e-9, 10e-9, 30e-9, 100e-9], palette))
    colors_dE = dict(zip([0.0, 0.2, 0.4, 0.6], palette))

    # Plot 1/C in SI units m² F⁻¹ — natural scale gives values in 0–30 range
    # so no ×10⁻ⁿ exponent is needed on either axis.
    for L, d in sorted(panel_a.items()):
        xs = np.array([1.0 / d[cb]['C_GC_min'] for cb in sorted(d)])
        ys = np.array([1.0 / d[cb]['C_dl']     for cb in sorted(d)])
        s = pz_slope(d)
        ax1.plot(xs, ys, 'o-', lw=2.0, ms=6,
                 label=rf'$L$ = {round(L*1e9):.0f} nm  ($s$ = {s:.2f})',
                 color=colors_L.get(L))
    for dE, d in sorted(panel_b.items()):
        xs = np.array([1.0 / d[cb]['C_GC_min'] for cb in sorted(d)])
        ys = np.array([1.0 / d[cb]['C_dl']     for cb in sorted(d)])
        s = pz_slope(d)
        ax2.plot(xs, ys, 'o-', lw=2.0, ms=6,
                 label=rf'$\Delta E_\mathrm{{pzc}}$ = {dE:.1f} V  ($s$ = {s:.2f})',
                 color=colors_dE.get(dE))

    ax1.set_ylabel(r'$1/C_\mathrm{dl}$ (m² F$^{-1}$)')
    for ax in (ax1, ax2):
        ax.set_xlabel(r'$1/C_\mathrm{GC,min}$ (m² F$^{-1}$)')
        ax.legend(loc='upper left', fontsize=10,
                  frameon=True, framealpha=0.95)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Panel banners — top center of each axis. Panel (a) sweeps L at
    # fixed ΔE_pzc = 0.2 V; panel (b) sweeps ΔE_pzc at fixed L = 30 nm.
    banner_bbox = dict(boxstyle='round,pad=0.3', fc='white',
                       ec='0.7', lw=0.4, alpha=0.95)
    ax1.text(0.5, 0.97, r"(a)  $\Delta E_\mathrm{pzc}$ = 0.2 V",
             transform=ax1.transAxes, ha='center', va='top',
             fontsize=11, fontweight='bold', bbox=banner_bbox)
    ax2.text(0.5, 0.97, r"(b)  $L$ = 30 nm",
             transform=ax2.transAxes, ha='center', va='top',
             fontsize=11, fontweight='bold', bbox=banner_bbox)

    fig.tight_layout()
    save_figure(fig, fname)
    plt.close(fig)


# ── F5 — Phase diagram of #-of-minima ────────────────────────────────────

F5_L_VALS      = [1e-9, 10e-9, 30e-9, 100e-9]
F5_CB_VALS     = [0.1, 1.0, 10.0]          # mol/m³ = mM
# eΔE_pzc/kT values matching the 5 rows in SI Fig S1
F5_DEPZC_NORM  = [5.0, 10.0, 15.0, 20.0, 25.0]
_F5_NEM        = 41   # E_M sweep points for minima counting (0.025 V step)


def _lam_D(c_b: float, T: float = 298.15) -> float:
    from src import EPS0, EPS_R_WATER, R, F as FARADAY
    return float(np.sqrt(EPS0 * EPS_R_WATER * R * T / (2.0 * FARADAY**2 * c_b)))


def compute_F5_data(mesh: dict, solver_method: str = 'hybr', verbose: bool = False,
                    n_em: int = _F5_NEM):
    """Compute #-of-minima in C_dl(E_M) on (L, c_b, eΔE_pzc/kT) grid.

    Returns list of dicts with keys:
      L, c_b, delta_E_pzc, L_over_lam_D, eDE_over_kT, n_minima, converged
    """
    from src.gc_pb_2d import solve_pb_2d
    from src.parameters import make_f2_case
    from src import R, F as FARADAY
    T = 298.15
    kT_e = R * T / FARADAY  # V

    records = []
    for L in F5_L_VALS:
        for c_b in F5_CB_VALS:
            lam_D_val = _lam_D(c_b)
            ratio = L / lam_D_val
            for norm in F5_DEPZC_NORM:
                dE = norm * kT_e
                E_M_vals = np.linspace(-0.5, 0.5, n_em)
                sb = np.empty(len(E_M_vals))
                n_fail = 0
                for idx, E_M in enumerate(E_M_vals):
                    case = make_f2_case(E_M=E_M, delta_E_pzc=dE, c_b=c_b, L=L)
                    _, _, s, res = solve_pb_2d(
                        case.to_solve_dict(), mesh, solver_method=solver_method)
                    ok = res.success and float(np.linalg.norm(res.fun)) < 1e-6
                    if not ok:
                        n_fail += 1
                    sb[idx] = s
                C_dl = np.gradient(sb, E_M_vals)
                n_min = sum(
                    1 for i in range(1, len(C_dl) - 1)
                    if C_dl[i] < C_dl[i - 1] and C_dl[i] < C_dl[i + 1]
                )
                rec = {
                    'L': L, 'c_b': c_b, 'delta_E_pzc': dE,
                    'L_over_lam_D': ratio, 'eDE_over_kT': norm,
                    'n_minima': n_min, 'converged': (n_fail == 0),
                }
                records.append(rec)
                if verbose:
                    ok_str = 'OK' if n_fail == 0 else f'WARN({n_fail}fail)'
                    print(f'  L={L*1e9:.0f}nm cb={c_b:.1f}mM '
                          f'L/lD={ratio:.2f} eDE={norm:.0f}kT: '
                          f'{n_min}min {ok_str}')
    return records


def save_F5_csv(data: list):
    """Save F5 phase diagram data as CSV."""
    _ensure_output()
    fname = os.path.join(OUTPUT_DIR, 'F5_phase_diagram.csv')
    rows = [(r['L'] * 1e9, r['c_b'], r['delta_E_pzc'],
             r['L_over_lam_D'], r['eDE_over_kT'], r['n_minima'])
            for r in data]
    np.savetxt(fname, rows,
               header='L_nm,c_b_mM,dE_pzc_V,L_over_lamD,eDE_over_kT,n_minima',
               delimiter=',', comments='')
    print(f'  saved {fname}')


def plot_F5_phase_diagram(data: list, fname: str = 'F5_phase_diagram'):
    """Save output/F5_phase_diagram.{png,pdf} — slide-polished phase diagram.

    Slide-polished:
    - No title, no grid, no regime shading (per "less crowded" preference).
    - Okabe-Ito CVD-safe palette (blue / vermillion) for n_minima.
    - Marker shape encodes L (○ △ ▢ ◊), keyed by rounded nm to avoid
      `100 * 1e-9 != 100e-9` float-precision drift.
    - Horizontal jitter (Option A) disambiguates overlapping markers.
    - Secondary y-axis: ΔE_pzc in mV alongside e·ΔE_pzc / k_BT.
    - Legend placed outside the axes on the right, included in image bbox.
    """
    _ensure_output()
    from matplotlib.lines import Line2D
    from src import R as GAS_R, F as FARADAY

    T = 298.15
    kT_e = GAS_R * T / FARADAY  # V

    MARKERS = {1: 'o', 10: '^', 30: 's', 100: 'D'}
    COLORS  = {1: '#0072B2', 2: '#D55E00'}
    JITTER  = {1: -0.06, 10: -0.02, 30: +0.02, 100: +0.06}

    def _L_key(L_m: float) -> int:
        return int(round(L_m * 1e9))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    xs_log = np.log10([r['L_over_lam_D'] for r in data])
    ys     = np.array([r['eDE_over_kT']  for r in data])
    log_x_min, log_x_max = xs_log.min() - 0.2, xs_log.max() + 0.2
    y_min, y_max         = ys.min() - 1.0,    ys.max() + 1.0

    # Data points with horizontal jitter (Option A).
    for rec in data:
        key   = _L_key(rec['L'])
        x_jit = rec['L_over_lam_D'] * 10**JITTER.get(key, 0.0)
        ax.scatter([x_jit], [rec['eDE_over_kT']],
                   c=COLORS.get(rec['n_minima'], 'gray'),
                   marker=MARKERS.get(key, 'x'),
                   s=95, edgecolors='black', linewidths=0.6, zorder=3)

    # Legend: L glyphs + n_minima color swatches.
    handles = []
    for L_nm, m in MARKERS.items():
        handles.append(Line2D([0], [0], marker=m, color='dimgray',
                              linestyle='None', markersize=9,
                              markeredgecolor='black', markeredgewidth=0.6,
                              label=f'$L$ = {L_nm} nm'))
    handles.append(Line2D([0], [0], marker='s', color=COLORS[1],
                          linestyle='None', markersize=10,
                          markeredgecolor='black', markeredgewidth=0.6,
                          label='1 minimum'))
    handles.append(Line2D([0], [0], marker='s', color=COLORS[2],
                          linestyle='None', markersize=10,
                          markeredgecolor='black', markeredgewidth=0.6,
                          label='2 minima'))

    ax.set_xscale('log')
    ax.set_xlim(10**log_x_min, 10**log_x_max)
    ax.set_ylim(y_min, y_max)
    # x and primary-y labels: dimensionless mathematical ratios — the "/"
    # is division between physical quantities, NOT a unit separator.
    ax.set_xlabel(r'$L\,/\,\lambda_D$', fontsize=18)
    ax.set_ylabel(r'$e\,\Delta E_\mathrm{pzc}\,/\,k_B T$', fontsize=18)

    # Secondary y-axis on the right shows the same quantity in physical
    # mV units. Unit in parentheses so it's unambiguously the unit of
    # ΔE_pzc and not a math division. Drawn BEFORE the legend so the
    # legend can be anchored relative to the full right-axis extent.
    secax = ax.secondary_yaxis(
        'right',
        functions=(lambda y: y * kT_e * 1e3, lambda mv: mv / (kT_e * 1e3)),
    )
    secax.set_ylabel(r'$\Delta E_\mathrm{pzc}$ (mV)', fontsize=18)

    # Legend inside the plot, lower-right corner. Opaque so it does not
    # bleed onto the underlying markers.
    ax.legend(handles=handles, loc='lower right',
              fontsize=10, framealpha=1.0, frameon=True)

    ax.grid(False)

    fig.tight_layout()
    save_figure(fig, fname)
    plt.close(fig)


# ── F3 — four 2-D ϕ(y,z) regime maps ─────────────────────────────────

# Four regime cases per CONSTRAINTS §3 F3 (all at c_b = 10 mM, λ_D ≈ 3 nm).
# `descriptor` is the physically meaningful one-line title (QC §4); the
# regime tag is appended in parentheses by the plot routine.
F3_CASES_SPEC = [
    {'tag': 'a', 'regime': 'Regime 1',
     'descriptor': 'Lateral averaging dominates',
     'L': 1e-9,  'dE': 0.4, 'x': 0.5},
    {'tag': 'b', 'regime': 'Regime 2a',
     'descriptor': 'Two distinct field zones',
     'L': 10e-9, 'dE': 0.6, 'x': 0.5},
    {'tag': 'c', 'regime': 'Regime 2b',
     'descriptor': 'Weak PZC contrast — averaged',
     'L': 30e-9, 'dE': 0.2, 'x': 0.5},
    {'tag': 'd', 'regime': 'Regime 2c',
     'descriptor': 'Asymmetric — small facet dominates',
     'L': 30e-9, 'dE': 0.6, 'x': 0.1},
]
F3_C_B_MM = 10.0
F3_T_K = 298.15


def _f3_lam_D() -> float:
    from src import EPS0, F as FARADAY, R as GAS_R, EPS_R_WATER
    return float(np.sqrt(EPS0 * EPS_R_WATER * GAS_R * F3_T_K
                         / (2.0 * FARADAY**2 * F3_C_B_MM)))


def compute_F3_data(mesh: dict, solver_method: str = 'sparse') -> list[dict]:
    """Solve all four F3 regime cases at the given mesh; return records list.

    Each record: {'spec': F3_CASES_SPEC[i], 'phi_yz': (Ny, Nz+1) array,
                  'y_m': (Ny,), 'z_m': (Nz+1,), 'lam_D': float}.
    """
    from src.gc_pb_2d import solve_pb_2d, make_yz_mesh
    from src.parameters import CaseConfig

    Ny = int(mesh['Ny'])
    Nz = int(mesh['Nz'])
    Z_max_factor = float(mesh.get('Z_max_factor', 8.0))
    lam_D = _f3_lam_D()
    Z_max = Z_max_factor * lam_D

    records: list[dict] = []
    for spec in F3_CASES_SPEC:
        case = CaseConfig(E_M=0.0,
                          E_pzc_1=+spec['dE'] / 2.0,
                          E_pzc_2=-spec['dE'] / 2.0,
                          x=spec['x'], c_b=F3_C_B_MM, L=spec['L'])
        phi_yz, _, _, result = solve_pb_2d(
            case.to_solve_dict(), mesh, solver_method=solver_method)
        print(f"  F3{spec['tag']}: success={result.success} "
              f"||R||={np.linalg.norm(result.fun):.2e}")
        y_m, z_m = make_yz_mesh(Ny, Nz, spec['L'], Z_max, alpha=3.0)
        records.append({'spec': spec, 'phi_yz': phi_yz,
                        'y_m': y_m, 'z_m': z_m, 'lam_D': lam_D})
    return records


def save_F3_csvs(records: list[dict]) -> None:
    """Write output/F3_phi2D_{a,b,c,d}.csv (long form: y_nm, z_nm, phi_V)."""
    _ensure_output()
    for rec in records:
        tag = rec['spec']['tag']
        y_m, z_m, phi_yz = rec['y_m'], rec['z_m'], rec['phi_yz']
        fname = os.path.join(OUTPUT_DIR, f'F3_phi2D_{tag}.csv')
        tmp = fname + '.tmp'
        with open(tmp, 'w') as fh:
            fh.write('y_nm,z_nm,phi_V\n')
            for i, y in enumerate(y_m):
                for k, z in enumerate(z_m):
                    fh.write(f'{y*1e9:.4f},{z*1e9:.4f},{phi_yz[i,k]:.6e}\n')
        os.replace(tmp, fname)
        print(f'  saved {fname}')


def load_F3_csvs(csv_dir: str | None = None) -> list[dict]:
    """Read cached F3 CSVs back into the records format used by plot routines."""
    csv_dir = csv_dir or OUTPUT_DIR
    lam_D = _f3_lam_D()
    records: list[dict] = []
    for spec in F3_CASES_SPEC:
        path = os.path.join(csv_dir, f"F3_phi2D_{spec['tag']}.csv")
        raw = np.loadtxt(path, delimiter=',', skiprows=1)
        y_nm, z_nm, phi = raw[:, 0], raw[:, 1], raw[:, 2]
        y_un = np.unique(y_nm)
        z_un = np.unique(z_nm)
        phi_yz = phi.reshape(len(y_un), len(z_un))
        records.append({'spec': spec, 'phi_yz': phi_yz,
                        'y_m': y_un * 1e-9, 'z_m': z_un * 1e-9,
                        'lam_D': lam_D})
    return records


def _build_F3_figure(records: list[dict], vmax_strategy: str = 'per_panel'):
    """Build the four-panel F3 figure.

    vmax_strategy:
      'per_panel'         — Option A: each panel uses its own |φ|_max,
                            value annotated in the panel title.
      'shared_normalized' — Option B: shared global vmax, with the
                            y-axis plotted as y/L so all panels share
                            [0, 1] on the horizontal axis.
    """
    if vmax_strategy not in ('per_panel', 'shared_normalized'):
        raise ValueError(f'unknown vmax_strategy: {vmax_strategy!r}')

    lam_D = records[0]['lam_D']
    vmax_global_mV = max(float(np.max(np.abs(r['phi_yz']))) for r in records) * 1e3

    # 2×2 layout (slide-friendly aspect) instead of 1×4 wide strip.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axflat = axes.flatten()
    last_im = None
    panel_labels = []  # collect per-panel |φ|_max for the legend strip
    for idx, (ax, rec) in enumerate(zip(axflat, records)):
        spec = rec['spec']
        phi_mV = rec['phi_yz'] * 1e3
        z_nm = rec['z_m'] * 1e9
        if vmax_strategy == 'per_panel':
            y_axis = rec['y_m'] * 1e9      # nm
            vmax_mV = float(np.max(np.abs(phi_mV)))
            x_label = '$y$ (nm)'
            x_right = y_axis[-1] + (y_axis[1] - y_axis[0])
        else:  # shared_normalized
            L = spec['L']
            y_axis = rec['y_m'] / L        # dimensionless [0, 1)
            vmax_mV = vmax_global_mV
            x_label = '$y/L$'              # mathematical ratio, no unit
            x_right = y_axis[-1] + (y_axis[1] - y_axis[0])
        panel_labels.append((spec['tag'], vmax_mV))

        Y, Z = np.meshgrid(y_axis, z_nm, indexing='ij')
        im = ax.pcolormesh(Y, Z, phi_mV, cmap='RdBu_r',
                           vmin=-vmax_mV, vmax=vmax_mV, shading='auto',
                           rasterized=False, edgecolors='face', linewidth=0)
        last_im = im
        ax.set_xlim(0, x_right)
        ax.set_ylim(0, 4.0 * lam_D * 1e9)
        ax.set_xlabel(x_label)
        ax.set_ylabel('$z$ (nm)')
        ax.label_outer()

        # Combined panel letter + |φ|_max banner, centered at top of panel.
        if vmax_strategy == 'per_panel':
            banner = rf"({spec['tag']})  $|\varphi|_\mathrm{{max}}={vmax_mV:.0f}$ mV"
        else:
            banner = f"({spec['tag']})"
        ax.text(0.5, 0.96, banner,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='0.7', lw=0.4, alpha=0.9))

    # Single shared colorbar to the right of the 2×2 grid, label on top.
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                        fraction=0.04, pad=0.08)
    cbar.ax.set_title(r'$\varphi_\mathrm{app}$ (mV)', pad=10)
    if vmax_strategy == 'per_panel':
        cbar.set_ticks([cbar.vmin, 0.0, cbar.vmax])
        cbar.set_ticklabels([r'$-|\varphi|_\mathrm{max}$', '0',
                             r'$+|\varphi|_\mathrm{max}$'])

    return fig


def plot_F3_from_csvs(vmax_strategy: str = 'per_panel',
                       fname_stem: str = 'F3_regime_maps') -> None:
    """Re-render F3 from cached `F3_phi2D_{a,b,c,d}.csv` — no solver run."""
    _ensure_output()
    records = load_F3_csvs()
    fig = _build_F3_figure(records, vmax_strategy=vmax_strategy)
    save_figure(fig, fname_stem)
    plt.close(fig)


def run_and_plot_F3(mesh: dict, solver_method: str = 'sparse',
                    vmax_strategy: str = 'per_panel'):
    """Solve, cache four CSVs, and plot F3."""
    _ensure_output()
    records = compute_F3_data(mesh, solver_method=solver_method)
    save_F3_csvs(records)
    fig = _build_F3_figure(records, vmax_strategy=vmax_strategy)
    save_figure(fig, 'F3_regime_maps')
    plt.close(fig)
    return records


# ── F1 — 2-D ϕ map ────────────────────────────────────────────────────


# F1 base case (per CONSTRAINTS §3 F1; mirrored in make_f2_case below).
F1_E_M = 0.0
F1_DELTA_E_PZC = 0.6
F1_C_B_MM = 10.0
F1_L_NM = 10.0
F1_X = 0.5
F1_T_K = 298.15


def _build_F1_figure(phi_yz: np.ndarray, y_m: np.ndarray,
                     z_m: np.ndarray, lam_D: float):
    """Build the F1 matplotlib Figure object.  No I/O.

    Pure-vector: pcolormesh emits one <rect> per cell with edgecolors='face'
    so adjacent cells share their boundary color (no hairline seams).
    """
    FS = 18
    Y, Z = np.meshgrid(y_m * 1e9, z_m * 1e9, indexing='ij')
    fig, ax = plt.subplots(figsize=(9, 6.5))
    vmax_mV = float(np.max(np.abs(phi_yz))) * 1e3
    im = ax.pcolormesh(Y, Z, phi_yz * 1e3, cmap='RdBu_r',
                       vmin=-vmax_mV, vmax=vmax_mV, shading='auto',
                       rasterized=False, edgecolors='face', linewidth=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_title(r'$\varphi$ (mV)', fontsize=FS, pad=10)
    cbar.ax.tick_params(labelsize=FS)

    ax.set_xlabel('$y$ (nm)', fontsize=FS)
    ax.set_ylabel('$z$ (nm)', fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.set_ylim(0, 3 * lam_D * 1e9)

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    boundary_nm = F1_X * F1_L_NM
    epzc_1 = +F1_DELTA_E_PZC / 2.0
    epzc_2 = -F1_DELTA_E_PZC / 2.0
    banner_bbox = dict(boxstyle='round,pad=0.25', fc='white',
                       ec='0.7', lw=0.4, alpha=0.9)
    ax.text(boundary_nm / 2.0, 1.03,
            rf'facet 1: $E_\mathrm{{pzc}}={epzc_1:+.2f}$ V',
            transform=trans, ha='center', va='bottom',
            fontsize=FS, bbox=banner_bbox)
    ax.text((boundary_nm + F1_L_NM) / 2.0, 1.03,
            rf'facet 2: $E_\mathrm{{pzc}}={epzc_2:+.2f}$ V',
            transform=trans, ha='center', va='bottom',
            fontsize=FS, bbox=banner_bbox)

    fig.tight_layout()
    return fig


def plot_F1_from_csv(csv_path: str | None = None,
                     fname_stem: str = 'F1_phi2D'):
    """Re-render F1 from the cached `F1_phi2D.csv` — no solver needed."""
    from src import EPS0, F as FARADAY, R as GAS_R, EPS_R_WATER

    _ensure_output()
    csv_path = csv_path or os.path.join(OUTPUT_DIR, 'F1_phi2D.csv')
    raw = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    y_nm, z_nm, phi = raw[:, 0], raw[:, 1], raw[:, 2]
    y_un = np.unique(y_nm)
    z_un = np.unique(z_nm)
    phi_yz = phi.reshape(len(y_un), len(z_un))
    y_m = y_un * 1e-9
    z_m = z_un * 1e-9
    lam_D = float(np.sqrt(EPS0 * EPS_R_WATER * GAS_R * 298.15
                          / (2.0 * FARADAY**2 * 10.0)))

    fig = _build_F1_figure(phi_yz, y_m, z_m, lam_D)
    save_figure(fig, fname_stem)
    plt.close(fig)


def run_and_plot_F1(mesh: dict, solver_method: str = 'sparse'):
    """Solve and save F1: 2-D ϕ map at Liu Fig 1 base case."""
    from src.gc_pb_2d import solve_pb_2d, make_yz_mesh
    from src import EPS0, F as FARADAY, R as GAS_R, EPS_R_WATER
    from src.parameters import make_f2_case

    _ensure_output()
    case = make_f2_case(E_M=0.0, delta_E_pzc=0.6, c_b=10.0, L=10e-9)
    phi_yz, sigma_M, sigma_bar, result = solve_pb_2d(
        case.to_solve_dict(), mesh, solver_method=solver_method)

    Ny = int(mesh['Ny'])
    Nz = int(mesh['Nz'])
    lam_D = float(np.sqrt(EPS0 * EPS_R_WATER * GAS_R * 298.15
                          / (2.0 * FARADAY**2 * 10.0)))
    y_m = np.linspace(0, 10e-9, Ny, endpoint=False)
    z_m = np.linspace(0, 8.0 * lam_D, Nz + 1)

    csv_path = os.path.join(OUTPUT_DIR, 'F1_phi2D.csv')
    with open(csv_path, 'w') as fh:
        fh.write('y_nm,z_nm,phi_V\n')
        for i, y in enumerate(y_m):
            for k, z in enumerate(z_m):
                fh.write(f'{y*1e9:.4f},{z*1e9:.4f},{phi_yz[i,k]:.6e}\n')
    print(f'  saved {csv_path}')

    fig = _build_F1_figure(phi_yz, y_m, z_m, lam_D)
    save_figure(fig, 'F1_phi2D')
    plt.close(fig)
    print(f'  F1 result.success={result.success}  '
          f'||R||={np.linalg.norm(result.fun):.2e}')
    return phi_yz, sigma_M, result
