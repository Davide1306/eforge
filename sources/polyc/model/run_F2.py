"""
run_F2.py — deliverable F2 (Fig 5): facet ratio x and Helmholtz C_H effects.

Three Fig-5 cases at L = 10 nm, c_b = 10 mM (lambda_D = 3 nm), Delta_E_pzc = 0.6 V
(facet PZCs at -0.3 / +0.3 V; facet 1 = the x / C_H^1 facet at -Delta_E_pzc/2),
C_H^2 = 50 uF/cm^2:

  blue   : x = 0.5, C_H^1 = C_H^2 = 50            (symmetric reference)
  orange : x = 0.5, C_H^1 = 10  (= C_H^2 / 5)     facet-1 Helmholtz suppressed
  yellow : x = 0.2, C_H^1 = C_H^2 = 50            facet-1 (minority) suppressed

Reproduces the 2nd-minimum suppression and the full feature pattern (maxima /
minima). Supervisor-confirmed model article-grade-quality (B-mag 0.5-2%,
B-count/suppression exact, clean-feature B-pos <= 15 mV; A9-F2-20260618).

Run:  python model/run_F2.py
Out:  model/output/F2.json
"""
import json
import pathlib
import sys
import warnings

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from src import constants as C  # noqa: E402
from src import pb2d  # noqa: E402

CB = C.mM_to_M(10.0)
C50 = C.uFcm2_to_SI(50.0)
C10 = C.uFcm2_to_SI(10.0)
E = np.linspace(-0.6, 0.6, 41)
# facet 1 at -0.3 (= -dE_pzc/2); facet 2 at +0.3
CASES = [("blue_x0.5_sym", 0.5, C50, C50),
         ("orange_x0.5_CH1=10", 0.5, C10, C50),
         ("yellow_x0.2_sym", 0.2, C50, C50)]


def main():
    out = {}
    print("=== F2 (Fig 5): C_dl(E), L=10nm, c_b=10mM, dE_pzc=0.6, C_H2=50 ===",
          flush=True)
    for lbl, x, CH1, CH2 in CASES:
        q, _ = pb2d.C_dl_2facet(E, CB, 10e-9, x, -0.3, 0.3, CH1, CH2, Ny=64, Nz=96)
        Ef, Cf, _ = pb2d.minima_from_qfree(E, q)
        Cu = C.SI_to_uFcm2(Cf)
        rng = Cu.max() - Cu.min()
        mx, _ = find_peaks(Cu, prominence=0.03 * rng)
        mn, _ = find_peaks(-Cu, prominence=0.03 * rng)
        maxs = [(round(float(Ef[i]), 3), round(float(Cu[i]), 1)) for i in mx]
        mins = [(round(float(Ef[i]), 3), round(float(Cu[i]), 1)) for i in mn]
        out[lbl] = dict(maxima=maxs, minima=mins, nmin=len(mn))
        print(f" {lbl:20s}: maxima={maxs} minima={mins}", flush=True)
    pathlib.Path("model/output").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("model/output/F2.json", "w"))
    print("saved model/output/F2.json", flush=True)


if __name__ == "__main__":
    main()
