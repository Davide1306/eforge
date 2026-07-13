"""Generate F4 Parsons-Zobel plots at final mesh."""
import os
os.environ['GCPOLYC_OUTPUT_DIR'] = 'output'

from src.plotting import (compute_F4_data, save_F4_csvs, plot_F4_panels, pz_slope)

MESH_FINAL = {'Ny': 128, 'Nz': 256, 'Z_max_factor': 8.0}

print('=== F4 Parsons-Zobel data ===')
panel_a, panel_b = compute_F4_data(MESH_FINAL, solver_method='sparse', verbose=True)

print('\n=== PZ slopes ===')
print('Panel (a) — ΔE_pzc=0.2V:')
for L, d in sorted(panel_a.items()):
    s = pz_slope(d)
    print(f'  L={L*1e9:.0f}nm: slope={s:.3f}')

print('Panel (b) — L=30nm:')
for dE, d in sorted(panel_b.items()):
    s = pz_slope(d)
    print(f'  ΔE_pzc={dE:.1f}V: slope={s:.3f}')

save_F4_csvs(panel_a, panel_b)
plot_F4_panels(panel_a, panel_b)
print('DONE')
