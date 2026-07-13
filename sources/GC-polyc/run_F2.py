"""Generate F2 figure panels. Run one panel at a time (each ~6 min at final mesh)."""
import os, sys
os.environ['GCPOLYC_OUTPUT_DIR'] = 'output'

from src.plotting import (run_F2_panel_sweep, run_F2b_sweep,
                          save_F2b_csvs, plot_F2b_panels,
                          save_F2_panel_csvs, plot_F2_4panels)

MESH_FINAL = {'Ny': 128, 'Nz': 256, 'Z_max_factor': 8.0}

# Panel spec: (panel_label, delta_E_pzc, c_b)
PANELS = {
    'a': (0.6, 0.1),
    'b': (0.6, 10.0),
    'c': (0.2, 0.1),
    'd': (0.2, 10.0),
}

panel = sys.argv[1] if len(sys.argv) > 1 else 'b'

if panel not in PANELS:
    print(f'Usage: python run_F2.py [a|b|c|d]')
    sys.exit(1)

dEpzc, c_b = PANELS[panel]
print(f'=== F2 panel ({panel}) at final mesh: dEpzc={dEpzc}V c_b={c_b}mM ===')
data = run_F2_panel_sweep(dEpzc, c_b, MESH_FINAL, solver_method='sparse', verbose=True)

n_conv = sum(1 for d in data.values() for ok in d['converged'] if ok)
n_total = sum(len(d['converged']) for d in data.values())
print(f'Converged: {n_conv}/{n_total}')

save_F2_panel_csvs(panel, data)

if panel == 'b':
    plot_F2b_panels(data)

print('DONE')
