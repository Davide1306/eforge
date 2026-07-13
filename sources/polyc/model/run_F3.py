"""
run_F3.py — deliverable F3 (Fig 6): E_dl,min - E_PZC deviation vs L/lambda_D.

(a) One-minimum regime (Fig 6a). x = 0.2 (asymmetric facets) — main.pdf p5:
    "the facet ratio was set to x = 0.2, as symmetry dictates a perfect match
    between E_dl,min and the global E_pzc for x = 0.5". Delta_E_pzc = 0.2 V
    (facet 1 at -0.1, facet 2 at +0.1), C_H = 50 uF/cm^2, L = 100 nm; vary c_b
    -> L/lambda_D. The deviation GROWS with L/lambda_D (facet-asymmetry effect),
    reaching the ~45-50 meV plateau.

(b) Two-minimum regime (Fig 6b). x = 0.5, Delta_E_pzc = 0.6 V (facets -+0.3),
    L = 100 nm; vary c_b. Each facet minimum deviates from its facet PZC; the
    deviation SHRINKS as L/lambda_D grows (facets decouple, minima -> facet PZCs).

E_PZC = potential of zero free charge (q_free spline crossing); E_dl,min =
capacitance-minimum potential. gamma_+ = gamma_- = 3 (frozen, symmetric).

Run:  python model/run_F3.py
Out:  model/output/F3.json
"""
import json
import pathlib
import sys
import warnings

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from src import constants as C  # noqa: E402
from src import pb2d  # noqa: E402

C_H = C.uFcm2_to_SI(50.0)
CBS = [0.1, 0.3, 1.0, 10.0]
L = 100e-9


def _deviation_onemin(cb_mM, x, ep1, ep2):
    """E_dl,min - E_PZC [meV] in the one-minimum regime."""
    E = np.linspace(-0.25, 0.25, 51)
    q, _ = pb2d.C_dl_2facet(E, C.mM_to_M(cb_mM), L, x, ep1, ep2, C_H, C_H,
                            Ny=64, Nz=128)
    cs = CubicSpline(E, q)
    Ef2 = np.linspace(E[0], E[-1], 8001)
    E_pzc = float(Ef2[np.argmin(np.abs(cs(Ef2)))])
    Ef, Cf, _ = pb2d.minima_from_qfree(E, q)
    E_dlmin = float(Ef[np.argmin(Cf)])
    return (E_dlmin - E_pzc) * 1000.0


def main():
    out = {"fig6a_onemin_x0.2": {}, "fig6b_twomin_x0.5": {}}
    print("=== F3 (Fig 6): E_dl,min - E_PZC deviation ===", flush=True)
    print("(a) one-minimum, x=0.2, dE_pzc=0.2 (facet1@-0.1):", flush=True)
    for cb in CBS:
        dev = _deviation_onemin(cb, 0.2, -0.1, 0.1)
        LlD = L / C.lambda_D(C.mM_to_M(cb))
        out["fig6a_onemin_x0.2"][f"{cb}mM"] = dict(LlD=round(LlD, 2),
                                                   dev_meV=round(dev, 1))
        print(f"   c_b={cb:5.1f} mM  L/lamD={LlD:5.2f}: dev={dev:+6.1f} meV",
              flush=True)
    print("(b) two-minimum, x=0.5, dE_pzc=0.6 (facets -+0.3):", flush=True)
    for cb in CBS:
        E = np.linspace(-0.6, 0.6, 41)
        q, _ = pb2d.C_dl_2facet(E, C.mM_to_M(cb), L, 0.5, -0.3, 0.3, C_H, C_H,
                                Ny=64, Nz=128)
        Ef, Cf, m = pb2d.minima_from_qfree(E, q, prom_frac=0.04)
        LlD = L / C.lambda_D(C.mM_to_M(cb))
        dev = round((max(p[0] for p in m) - 0.3) * 1000.0, 1) if len(m) >= 2 else None
        out["fig6b_twomin_x0.5"][f"{cb}mM"] = dict(LlD=round(LlD, 2),
                                                   nmin=len(m),
                                                   plus_branch_dev_meV=dev)
        print(f"   c_b={cb:5.1f} mM  L/lamD={LlD:5.2f}: #min={len(m)} "
              f"+branch_dev={dev} meV", flush=True)
    pathlib.Path("model/output").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("model/output/F3.json", "w"))
    print("saved model/output/F3.json", flush=True)


if __name__ == "__main__":
    main()
