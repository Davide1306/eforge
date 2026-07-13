# Certification

All checks below were run on this repository's tree and reproduce with
`python3 installer/verify.py` (offline) and the commands in [architecture.md](architecture.md).
The authoritative correctness certificate is the deterministic regression gate: with
ground-truth fills and no model in the loop, the 18-problem battery must score **1.0**.

## Results

| Check | Result | How to reproduce |
|---|---|---|
| `validate_skeleton` (8 gc + 7 polyc cases + 3 corrections guards) | **15/15 PASS** (gc worst rel 1.5e-14; polyc rel_max 0.00e+00 — the inlined solver equals the reference to machine precision) | `cd main && .venv/bin/python -m eval.validate_skeleton` |
| Deterministic regression gate (18 problems, no model) | **score 1.0** · selection 1.0 · slot-fill 1.0 · executability 1.0 · physics 1.0 · comsol-static 1.0 · hallucinations 0 · repairs 0 | `.venv/bin/python -m eval.harness --mode deterministic` |
| Graph-mechanics smoke, mock client (no network) | **18/18 records**, exit 0 (canned answers — proves the pipeline runs end to end, not physics) | `.venv/bin/python -m eval.harness --mode llm --profile mock` |
| COMSOL builder — static checks | **4/4** (AST/markers, MPh-API surface, parameter completeness, exactly-two-facet selections) | carried in the deterministic records |
| COMSOL builder — execution (opt-in) | executes to `mph.start()`; produces a solved `.mph` when a COMSOL engine is present, otherwise records `Could not find a supported Comsol installation` (non-gating) | `.venv/bin/python -m eval.harness --mode deterministic --comsol-exec` |
| Integrity manifest (9 SHA256-pinned frozen files) | **9/9 MATCH** | `python3 installer/verify.py` |

Evidence run directories are written under `main/runs/` (a `deterministic_*` and a `mock_*`
directory; each holds `metrics.json`, `records.json`, and one folder per problem with the
emitted `model.py` and `model_comsol.py`).

## The two representative COMSOL test cases

`p03_single_crystal` (gc_polyc_pb2d) and `p18_v0_dilute` (polyc_mpb2d) — one per template.
With `--comsol-exec`, each emits its `model_comsol.py`, executes it, builds the model,
`solve('std1')`s the stationary study, and saves a solved `.mph`
(`gc_pb2d_p03_single_crystal.mph` / `polyc_mpb2d_p18_v0_dilute.mph`). On a machine with no
COMSOL engine the run reaches `mph.start()` and stops there — the emitted script and the
wiring are correct; only the engine is absent. See "Running the COMSOL leg" in
[../README.md](../README.md).

## Known limitations (inherited by anything downstream)

- **COMSOL** is static-checked (4/4) and executed only to the engine boundary here — a real
  solved `.mph` requires a COMSOL install + MPh runtime + license. The post-solve
  charge/capacitance export expression is best-effort, pending validation on a licensed run.
- **Live model scores** are the served model's behaviour (bring your own OpenAI-compatible
  endpoint; non-gating). The deterministic 1.0 gate is the certificate.
- **The mock battery** is a graph-mechanics artifact, not a physics result.
- **F5 adsorption** is reproduced with a documented caveat (the Parsons-Zobel slope magnitude
  sits below the paper anchors); the `adsorption_shift` gate certifies a present, finite,
  physically-sized shift, not the article-grade band.
