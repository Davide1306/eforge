"""Generate F5 phase diagram at final mesh."""
import os
os.environ['GCPOLYC_OUTPUT_DIR'] = 'output'

from src.plotting import compute_F5_data, save_F5_csv, plot_F5_phase_diagram

MESH_FINAL = {'Ny': 128, 'Nz': 256, 'Z_max_factor': 8.0}

print('=== F5 Phase diagram (final mesh 128x256, 21-pt E_M sweep) ===')
data = compute_F5_data(MESH_FINAL, solver_method='sparse', verbose=True, n_em=21)
save_F5_csv(data)
plot_F5_phase_diagram(data)
print('DONE')
