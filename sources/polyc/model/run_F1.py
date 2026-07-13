"""
run_F1.py — deliverable F1 (Fig 2a): C_dl(E) one/two-minimum transition.

Two-facet differential capacitance C_dl(E_M) at the Fig-2a panel conditions
(main.pdf Fig 2a / SI): c_b = 0.1 mM (lambda_D ~ 30 nm), Delta_E_pzc = 0.6 V
(facet PZCs at -0.3 / +0.3 V; facet 1 at -Delta_E_pzc/2), x = 0.5, C_H = 50
uF/cm^2 on both facets, for L in {1, 10, 100} nm. The acceptance feature is the
#minima transition {1, 1, 2} (single averaged minimum at small L/lambda_D ->
two facet-resolved minima at L >> lambda_D), with magnitudes/positions within
the CONSTRAINTS §3 bands. Nonuniform z-grid stretch=7.

Run:  python model/run_F1.py
Out:  model/output/F1.json
"""
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from src import constants as C  # noqa: E402
from src import pb2d  # noqa: E402

CB = C.mM_to_M(0.1)
C_H = C.uFcm2_to_SI(50.0)
LS_NM = [1.0, 10.0, 100.0]
E = np.linspace(-0.6, 0.6, 41)


def main():
    out = {}
    print("=== F1 (Fig 2a): C_dl(E), c_b=0.1 mM, dE_pzc=0.6 V, x=0.5, C_H=50 ===",
          flush=True)
    for Lnm in LS_NM:
        # facet 1 at -dE_pzc/2 (corrected convention); x=0.5 -> symmetric anyway
        q, _ = pb2d.C_dl_2facet(E, CB, Lnm * 1e-9, 0.5, -0.3, 0.3, C_H, C_H,
                                Ny=48, Nz=128)
        Ef, Cf, m = pb2d.minima_from_qfree(E, q, prom_frac=0.04)
        Cu = C.SI_to_uFcm2(Cf)
        mins = [(round(float(p[0]), 3), round(float(C.SI_to_uFcm2(p[1])), 2))
                for p in m]
        out[f"L{Lnm:.0f}nm"] = dict(L_over_lamD=Lnm * 1e-9 / C.lambda_D(CB),
                                    nmin=len(m), minima=mins,
                                    Cdl_min=round(float(Cu.min()), 2))
        print(f" L={Lnm:6.0f} nm (L/lamD={Lnm*1e-9/C.lambda_D(CB):5.2f}): "
              f"#min={len(m)} at {mins}", flush=True)
    pathlib.Path("model/output").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("model/output/F1.json", "w"))
    nmin_seq = [out[f"L{L:.0f}nm"]["nmin"] for L in LS_NM]
    print(f"#minima sequence {{L=1,10,100}} = {nmin_seq}  (Fig 2a target {{1,1,2}})",
          flush=True)
    print("saved model/output/F1.json", flush=True)


if __name__ == "__main__":
    main()
