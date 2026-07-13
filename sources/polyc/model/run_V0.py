"""
run_V0.py — validation V0 (Fig S3): single-facet C_dl(E) vs bulk concentration.

Single facet (one E_pzc, one C_H), C_H = 25 uF/cm^2, c_b in {0.1,1,10,100,400} mM,
E - E_pzc in [-1, 1] V. Pass criterion: GC minimum at E_pzc for dilute c_b;
camel (central minimum) -> bell (central maximum) transition with rising c_b.

This also serves as the HF-4 high-field cross-check for the F1 edge: the model's
single-facet high-field C_dl (E - E_pzc ~ 0.9 V) is compared to Fig S3 (~7-8
uF/cm^2 where the curves converge at the edges).
"""
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from scipy.signal import find_peaks  # noqa: E402

from src import constants as C  # noqa: E402
from src import pb1d  # noqa: E402

C_H = C.uFcm2_to_SI(25.0)
CBS = [0.1, 1.0, 10.0, 100.0, 400.0]


def single_facet_Cdl(c_b_mM, E):
    """C_dl(E) [uF/cm^2] for a single facet (E_pzc=0, C_H=25), warm-started
    center-out so no extreme E is cold-started."""
    cb = C.mM_to_M(c_b_mM)
    q = np.empty(E.size)
    c0 = int(np.argmin(np.abs(E)))
    s = pb1d.solve_robin(E[c0], 0.0, C_H, cb)
    q[c0] = s["q_free"]
    phi_c = s["phi"]
    prev = phi_c
    for k in range(c0 + 1, E.size):           # march up from center
        s = pb1d.solve_robin(E[k], 0.0, C_H, cb, phi_guess=prev)
        q[k] = s["q_free"]
        prev = s["phi"]
    prev = phi_c
    for k in range(c0 - 1, -1, -1):            # march down from center
        s = pb1d.solve_robin(E[k], 0.0, C_H, cb, phi_guess=prev)
        q[k] = s["q_free"]
        prev = s["phi"]
    return pb1d_curve(E, q)


def pb1d_curve(E, q):
    from src import pb2d  # reuse the spline extractor
    return pb2d.minima_from_qfree(E, q, prom_frac=0.04)


def main():
    E = np.linspace(-1.0, 1.0, 25)
    out = {}
    print("=== V0 / Fig S3: single-facet C_dl(E), C_H=25 ===", flush=True)
    print(" c_b(mM)  C_dl(0)  C_dl(0.9)  shape       max(uFcm2)", flush=True)
    for cb in CBS:
        Ef, Cf, mins = single_facet_Cdl(cb, E)
        Cu = C.SI_to_uFcm2(Cf)
        c_center = float(Cu[np.argmin(np.abs(Ef))])
        i09 = int(np.argmin(np.abs(Ef - 0.9)))
        c_hf = float(Cu[i09])
        nmin = len(mins)
        shape = "camel(min@0)" if nmin >= 1 and abs(Ef[np.argmin(Cu)]) < 0.1 else "bell(max@0)"
        out[f"{cb}mM"] = dict(C0=c_center, C_hf09=c_hf, nmin=nmin, cmax=float(Cu.max()),
                              E=Ef.tolist(), C=Cu.tolist())
        print(f" {cb:6.1f}   {c_center:6.2f}   {c_hf:6.2f}    {shape:14s} {Cu.max():.2f}", flush=True)
    pathlib.Path("model/output").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("model/output/V0.json", "w"))
    print("saved model/output/V0.json", flush=True)


if __name__ == "__main__":
    main()
