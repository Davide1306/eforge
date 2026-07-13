# Handoff

This repository is a working, certified, self-contained snapshot of
eforge — a two-template model factory that turns a short natural-language electrochemistry
problem into a validated standalone Python solver plus a statically-checked COMSOL builder
script. The language model never writes code: it emits schema-constrained JSON slot-fills,
deterministic Jinja2 emitters render the artifacts, and a deterministic physics-gate battery
decides correctness.

- Physics provenance: Liu, Doblhoff-Dier & Koper, *Modelling the Double Layer of Polycrystalline
  Electrodes*, ACS Electrochemistry 2026, 2, 995–1004 (DOI 10.1021/acselectrochem.5c00544). The
  two templates reproduce the paper's two model levels via reference implementations vendored
  under [sources/](sources/README.md).
- Start here, then read [README.md](README.md) (overview and commands) →
  [docs/architecture.md](docs/architecture.md) (the deep specification) →
  [docs/certification.md](docs/certification.md) (the certification report).

## 1. What is certified

| Check | Result | Evidence |
|---|---|---|
| Emitted skeletons ≡ reference solvers (8 gc + 7 polyc cases + 3 guards) | 15/15 PASS (gc worst ~1e-14; polyc rel_max 0.00e+00) | `docs/certification.md`, Check 1 |
| Deterministic regression gate (18 problems, no model) | score 1.0, 0 hallucinations | the deterministic run dir under `main/runs/` |
| Graph mechanics, mock client (no network) | 18/18 records, exit 0 | the mock run dir under `main/runs/` |
| COMSOL builders | static 4/4; opt-in execution verified to the COMSOL-engine boundary | carried in the deterministic records |
| Integrity manifest (9 SHA256-pinned files) | 9/9 MATCH | `integrity.json` |

Full evidence, with the exact command outputs, is in [docs/certification.md](docs/certification.md).

## 2. Reproduce

All runs from `main/` (its `.venv` is a symlink to the project virtual environment):

```bash
cd main
.venv/bin/python -m eval.validate_skeleton                   # expect: ALL PASS (15/15), exit 0
.venv/bin/python -m eval.harness --mode deterministic        # expect: "score": 1.0 (18 problems; p08/p16 abstain)
.venv/bin/python -m eval.harness --mode llm --profile mock   # expect: exit 0, 18/18 records (mechanics only)
```

The integrity check (`python3 installer/verify.py`) recomputes the nine hashes — expect
`9/9 MATCH`.

## 3. The two templates

| Template | Physics | Reference implementation | Problems |
|---|---|---|---|
| `gc_polyc_pb2d` | classical 2-D Gouy-Chapman, two-facet stripe, point ions, constant ε | `sources/GC-polyc/` | p01–p10 (p08 abstains) |
| `polyc_mpb2d` | full modified PB: Bikerman steric, Langevin/Booth field-dependent ε, facet-resolved Robin BCs, Langmuir adsorption | `sources/polyc/` | p11–p18 (p16 abstains) |

Routing is registry-only (`main/templates/registry.yaml`); out-of-scope physics (more than two
facets, kinetics/currents, time dependence) must route `no_match`.

## 4. Integrity model (please keep it intact)

- **The deterministic regression gate** — `--mode deterministic` must always score 1.0; any
  change that breaks it is wrong by definition, regardless of live scores.
- **The integrity manifest** — nine files (both templates' frozen physics narratives, frozen
  gates, and Python skeletons, plus the two reference JSONs and the held-out ground-truth fills)
  are SHA256-pinned in `integrity.json`. Never edit them; recompute 9/9 after any work session
  (`python3 installer/verify.py`). Any of them can be restored from git with
  `git show <commit>:<path>`.
- **Held-out ground truth** — `main/eval/reference/ground_truth_fills.yaml` is for scoring only;
  never quote it into prompts or few-shots.

## 5. External dependencies (what does not travel with the repository)

| Dependency | Role | Portability note |
|---|---|---|
| An OpenAI-compatible serving endpoint | serves the live model for `--mode llm` runs | add your `base_url`, `model`, and `api_key_env` to the `openai_example` profile in `main/config.yaml`. Keep `json_mode: plain` (see docs/architecture.md). |
| COMSOL + MPh license | executing the emitted `model_comsol.py` builders | absent here — the top finalization item |
| Python 3.11+ virtual environment | runtime | rebuilt by the installer |

**Setup on a new machine.** Copy or clone the repository anywhere and run
`python3 installer/install.py` — it builds the virtual environment from the exact tested versions
(`installer/requirements.lock.txt`), recreates the `main/.venv` symlink, and re-runs the full
offline certification. See [installer/INSTALL.md](installer/INSTALL.md).

## 6. Vendored sources (`sources/`)

Read-only code snapshots of the two reference implementations, hash-verified file-by-file:

```bash
cd sources && sha256sum -c MANIFEST.sha256    # expect: clean, exit 0
```

`sources/README.md` carries the full include/exclude contract. The runtime cannot see this
directory (discovery is registry- and manifest-driven) — it is provenance, not a code path.

## 7. Finalization checklist

1. **Run the COMSOL leg against a real COMSOL engine.** The pipeline *executes* the emitted
   `model_comsol.py` (opt-in: `harness --comsol-exec`, or `EFORGE_COMSOL_EXEC=1` in the LangGraph
   path); each builder builds, `solve('std1')`s the stationary study, and saves a solved `.mph`
   (`gc_pb2d_<pid>.mph` / `polyc_mpb2d_<pid>.mph`). It is verified end-to-end on the two test
   cases (`p03_single_crystal`, `p18_v0_dilute`) up to `mph.start()`, which on an engine-less
   machine reports `Could not find a supported Comsol installation` (recorded, non-gating). To
   produce real `.mph` files, run it where a COMSOL install + MPh runtime + license is present (or
   point `mph.start(host=…)` at a `comsolmphserver`). Then validate the post-solve
   charge/capacitance export (currently best-effort) against the Python models' `RESULT_JSON`.
2. **Stand up your own serving** and run the live battery on your infrastructure (section 5);
   judge against the deterministic gate, which must stay 1.0. Live scores are non-gating.
3. **F5 adsorption** is reproduced with a caveat (Parsons-Zobel slopes ≈0.699/0.536 versus paper
   anchors 0.76/0.48) — close it only if the article-grade band is required.

## 8. Known limitations (inherited by anything you claim downstream)

- COMSOL: static-checked by default; opt-in execution reaches the COMSOL-engine boundary (no
  license here).
- Mock scores are graph-mechanics artifacts, not physics.
- Live scores are the served model's behaviour — run-to-run variable, non-gating; the
  deterministic gate is the only correctness certificate.
- F5 adsorption is reproduced with a caveat (see item 3 above).
