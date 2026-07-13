"""
run_F4.py — deliverable F4 (Fig 7, Parsons-Zobel plots).

PZ plot: 1/C_dl,min vs 1/C_GC across bulk concentration c_b (which sets lambda_D,
hence C_GC = eps_S/lambda_D, eq 10 at the PZC). A linear fit gives the PZ slope
(1 = ideal GCS; < 1 from facet heterogeneity) and intercept (~1/C_H).

Per the article (Fig 7 caption), PZ points where the C_dl(E) curve has TWO minima
are dropped ("missing point at low 1/C_GC where two minima") -- the PZ relation
assumes a single capacitance minimum. We detect #minima per point and drop the
two-minimum points before fitting.

  Fig 7a: base dE_pzc = 0.2 V, x = 0.5, C_H = 50; PZ slope vs L. Anchors 0.99->0.42.
  Fig 7b: base L = 30 nm, x = 0.5, C_H = 50; PZ slope vs dE_pzc. Anchors 1.00->0.47.
  c_b in {0.1, 0.3, 1, 10} mM -> lambda_D 30.4/17.6/9.6/3.0 nm.
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

CBS = [0.1, 0.3, 1.0, 10.0]
C_H = C.uFcm2_to_SI(50.0)


def pz_point(L, dEpzc, cb_mM):
    """Return (1/C_GC, 1/C_dl,min [cm^2/uF], n_minima) for one (L, dEpzc, c_b)."""
    ep = dEpzc / 2.0
    win = 0.20 + dEpzc                      # window widens with the facet split
    E = np.linspace(-win, win, 15)
    q, _ = pb2d.C_dl_2facet(E, C.mM_to_M(cb_mM), L, 0.5, ep, -ep, C_H, C_H,
                            Ny=48, Nz=128)
    Ef, Cf, mins = pb2d.minima_from_qfree(E, q, prom_frac=0.05)
    lam = C.lambda_D(C.mM_to_M(cb_mM))
    inv_cgc = 1.0 / C.SI_to_uFcm2(C.eps_S / lam)
    inv_cdl = 1.0 / C.SI_to_uFcm2(float(np.min(Cf)))
    return inv_cgc, inv_cdl, len(mins)


def pz_fit(L, dEpzc):
    xs, ys, dropped = [], [], []
    for cb in CBS:
        x, y, nmin = pz_point(L, dEpzc, cb)
        if nmin >= 2:                       # paper drops two-minimum points
            dropped.append(cb)
            continue
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    s, b = np.polyfit(xs, ys, 1)
    R2 = 1.0 - np.sum((ys - (s * xs + b))**2) / np.sum((ys - ys.mean())**2)
    return dict(slope=float(s), R2=float(R2), intercept=float(b),
                n_pts=len(xs), dropped=dropped)


def main():
    out = {"fig7a": {}, "fig7b": {}}
    print("=== Fig 7a: PZ slope vs L (dE_pzc=0.2); anchors 0.99->0.42 ===", flush=True)
    for L in (1.0, 10.0, 30.0, 100.0):
        r = pz_fit(L * 1e-9, 0.2)
        out["fig7a"][f"L{L}nm"] = r
        anc = {1.0: 0.99, 100.0: 0.42}.get(L)
        print(f" L={L:5.0f}nm: slope={r['slope']:.3f}{' (anchor %.2f)'%anc if anc else ''} "
              f"R2={r['R2']:.4f} intercept={r['intercept']:.4f} n={r['n_pts']} dropped={r['dropped']}", flush=True)
    print("=== Fig 7b: PZ slope vs dE_pzc (L=30nm); anchors 1.00->0.47 ===", flush=True)
    for dE in (0.0, 0.1, 0.2, 0.3):
        r = pz_fit(30e-9, dE)
        out["fig7b"][f"dE{dE}"] = r
        anc = {0.0: 1.00, 0.3: 0.47}.get(dE)
        print(f" dE_pzc={dE}: slope={r['slope']:.3f}{' (anchor %.2f)'%anc if anc else ''} "
              f"R2={r['R2']:.4f} intercept={r['intercept']:.4f} n={r['n_pts']} dropped={r['dropped']}", flush=True)
    pathlib.Path("model/output").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("model/output/F4.json", "w"), indent=1)
    print("saved model/output/F4.json", flush=True)


if __name__ == "__main__":
    main()
