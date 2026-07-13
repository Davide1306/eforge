"""Generate F3 regime potential maps."""
import os
os.environ['GCPOLYC_OUTPUT_DIR'] = 'output'

from src.plotting import run_and_plot_F3

MESH_FINAL = {'Ny': 128, 'Nz': 256, 'Z_max_factor': 8.0}

print('=== F3 Regime potential maps (final mesh 128x256) ===')
run_and_plot_F3(MESH_FINAL, solver_method='sparse')
print('DONE')
