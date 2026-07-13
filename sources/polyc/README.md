# polyc — reference implementation of Liu, Doblhoff-Dier & Koper (*ACS Electrochem.* 2026, 2, 995–1004)

A Python reproduction of "Modelling the Double Layer of Polycrystalline Electrodes:
Capacitance, Potential of Zero Charge, and Parsons–Zobel Plot"
(DOI 10.1021/acselectrochem.5c00544). The model (`model/`) implements the article's full
mean-field EDL model — a 2-D modified Poisson–Boltzmann equation with asymmetric ion sizes
and Langevin solvent polarization on a two-facet electrode (eqs 1–13) — and computes its
figures F1–F5. `model/reference/CONSTRAINTS.md` is the frozen ground truth (parameters +
equations). This is a read-only reference copy vendored into eforge for provenance; the
`polyc_mpb2d` template's physics is derived from it.

## Layout

```
polyc/model/
├─ src/         the paper's solver modules
├─ tests/       pytest suite
├─ reference/   CONSTRAINTS.md (params + equations)
├─ scripts/     post-solve analysis / re-render helpers
└─ run_<fig>.py figure entry points
```

## Run the science (from `model/`)

```bash
cd polyc/model
python -m pytest tests/ -q
python run_<fig>.py          # regenerate a figure
```
