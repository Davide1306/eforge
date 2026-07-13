"""Generate F1 2-D potential map at final mesh."""
import os
os.environ['GCPOLYC_OUTPUT_DIR'] = 'output'

from src.plotting import run_and_plot_F1

MESH_FINAL = {'Ny': 128, 'Nz': 256, 'Z_max_factor': 8.0}

print('=== F1 2-D potential map (final mesh 128x256) ===')
phi_yz, sigma_M, result = run_and_plot_F1(MESH_FINAL, solver_method='sparse')
print('DONE')
