"""
run_F5.py — deliverable F5 (Figs 8-9): adsorption, total C = C_dl + C_ads.

(a) Fig 8 E_min shift. x = 0.2, L = 10 nm, c_b = 0.1 mM, Delta_E_pzc = 0.2 V
    (facet 1 at -0.1), C_H = 50 uF/cm^2. Sweeping the adsorption equilibrium
    potential E_ads^0 (theta_max = 1%), the total-capacitance minimum E_min
    shifts by > 70 mV when the C_ads peak (at E_ads^0 - 0.237 V; E4-resolved
    c_b^(A-) = 1/c_b) overlaps the C_dl minimum -- and the minimum splits
    (two minima at overlap). Reproduces the Fig 8b anchor (g6-002).

(b) Fig 9 Parsons-Zobel slope. x = 0.5, L = 30 nm, Delta_E_pzc = 0.2 V (E6:
    Fig 9 base = Fig 7 base = L=30/x=0.5), E_ads^0 = 0.40 V. PZ slope of the
    total-C minima vs 1/C_GC, with the paper's Fig 9b point selection (theta=1%
    all four c_b; theta=8% drops the 10 mM / lowest-1/C_GC two-min point):
    REPORTS model ~0.70 (theta=1%) / ~0.54 (theta=8%) vs paper 0.76 / 0.48.
    The ~0.07-0.10 magnitude gap is a DOCUMENTED real model limitation (the
    adsorption-PZ theta-dependence is too weak; NOT procedural -- confirmed
    iter 44; EXANG-20260618-f5pz-thetadep). Reported, not re-derived; no fitting.

C_ads via analytics (eqs 12-13). gamma_+ = gamma_- = 3 (frozen).

Run:  python model/run_F5.py
Out:  model/output/F5.json
"""
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from src import analytics as A  # noqa: E402
from src import constants as C  # noqa: E402
from src import pb2d  # noqa: E402

C_H = C.uFcm2_to_SI(50.0)


def fig8_emin_shift():
    """E_min shift sweep over E_ads^0 at theta_max=1% (Fig 8b)."""
    cb = C.mM_to_M(0.1)
    E = np.linspace(-0.4, 0.5, 31)
    q, _ = pb2d.C_dl_2facet(E, cb, 10e-9, 0.2, -0.1, 0.1, C_H, C_H, Ny=48, Nz=128)
    Ef, Cdl_f, _ = pb2d.minima_from_qfree(E, q)
    Emin0 = float(Ef[np.argmin(Cdl_f)])
    sweep = []
    for E0 in np.linspace(0.0, 0.5, 26):
        Ctot = Cdl_f + A.C_ads(Ef, E0, 1.0 / cb, 0.01)
        Emin = float(Ef[np.argmin(Ctot)])
        sweep.append([round(float(E0), 3), round((Emin - Emin0) * 1000.0, 1)])
    maxshift = max(abs(s[1]) for s in sweep)
    return Emin0, maxshift, sweep


def fig9_pz():
    """PZ slope of total-C minima (Fig 9b point selection)."""
    cbs = [0.1, 0.3, 1.0, 10.0]
    E_ads0 = 0.40
    E = np.linspace(-0.4, 0.5, 41)
    out = {}
    for tmax, keep in [(0.01, [0, 1, 2, 3]), (0.08, [0, 1, 2])]:
        ig, ic = [], []
        for cb in cbs:
            q, _ = pb2d.C_dl_2facet(E, C.mM_to_M(cb), 30e-9, 0.5, -0.1, 0.1,
                                    C_H, C_H, Ny=48, Nz=128)
            Ef, Cdl_f, _ = pb2d.minima_from_qfree(E, q)
            Ctot = Cdl_f + A.C_ads(Ef, E_ads0, 1.0 / C.mM_to_M(cb), tmax)
            ig.append(1.0 / C.SI_to_uFcm2(C.eps_S / C.lambda_D(C.mM_to_M(cb))))
            ic.append(1.0 / C.SI_to_uFcm2(float(np.min(Ctot))))
        ig, ic = np.array(ig), np.array(ic)
        s, _ = np.polyfit(ig[keep], ic[keep], 1)
        out[tmax] = round(float(s), 3)
    return out


def main():
    print("=== F5 (Figs 8-9): adsorption, C = C_dl + C_ads ===", flush=True)
    Emin0, maxshift, sweep = fig8_emin_shift()
    print(f"(a) Fig 8 E_min shift (x=0.2): C_dl-only E_min0={Emin0:+.3f} V; "
          f"max|shift| over E_ads0 = {maxshift:.0f} meV  (anchor: >70 meV)",
          flush=True)
    pz = fig9_pz()
    print(f"(b) Fig 9 PZ slope (x=0.5, L=30, E_ads0=0.40): "
          f"theta=1% -> {pz[0.01]} (paper 0.76); theta=8% -> {pz[0.08]} "
          f"(paper 0.48)  [~0.06 over B-slope = documented limitation, EXANG]",
          flush=True)
    out = dict(fig8_Emin0=Emin0, fig8_max_shift_meV=maxshift, fig8_sweep=sweep,
               fig9_pz_slopes={str(k): v for k, v in pz.items()},
               fig9_pz_anchors={"0.01": 0.76, "0.08": 0.48},
               fig9_note="model PZ theta-dependence ~0.07-0.10 under paper; "
                         "documented limitation (EXANG-20260618), no fitting")
    pathlib.Path("model/output").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("model/output/F5.json", "w"))
    print("saved model/output/F5.json", flush=True)


if __name__ == "__main__":
    main()
