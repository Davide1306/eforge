# eforge

**Natural-language electrochemistry problem → validated Python + COMSOL double-layer model.**

eforge is a model factory for electrochemical double-layer simulations: given a short
natural-language problem, it produces a validated standalone Python solver (numpy/scipy) and a
statically-checked COMSOL builder script — without human intervention. The language model never
writes code; it fills typed template slots as schema-constrained JSON, and deterministic Jinja2
emitters render the artifacts.

## Quickstart

**Requirements:** Python ≥ 3.11 and network access for `pip`; Linux, macOS, or WSL.

```bash
python3 installer/install.py     # builds the venv, installs pinned deps, verifies (~5 min)
```

Run it from the repository root on Linux, macOS, or WSL. The installer builds the virtual
environment, installs the exact tested dependencies, and runs the full offline certification.
Expected final line:

```
integrity 9/9 · skeleton ALL PASS (15/15) · deterministic score 1.0 (18 problems) · mock 18/18
```

For flags, live-endpoint setup, and troubleshooting see [installer/INSTALL.md](installer/INSTALL.md).
Then read [HANDOFF.md](HANDOFF.md) for what is certified and what remains to finalize.

## What it does

Two physics templates carry the work, selected by a registry
(`main/templates/registry.yaml`, three entries including the `no_match` abstention label):

- **`gc_polyc_pb2d`** — the classical Gouy-Chapman 2-D Poisson-Boltzmann model on a two-facet
  stripe (point ions, constant permittivity). Its emitted solver matches a vendored reference
  implementation to ~1e-14 on eight frozen cases.
- **`polyc_mpb2d`** — the full modified Poisson-Boltzmann model: steric/Bikerman ion crowding,
  Langevin/Booth field-dependent permittivity, facet-resolved Robin boundary conditions, and
  Langmuir specific adsorption. Its emitted solver matches its vendored reference implementation
  at rel_max 0.00e+00 on seven frozen cases.

Both templates reproduce the two model levels of Liu, Doblhoff-Dier & Koper, *Modelling the
Double Layer of Polycrystalline Electrodes*, ACS Electrochemistry 2026, 2, 995–1004
(DOI 10.1021/acselectrochem.5c00544). The two reference implementations are vendored as
read-only code snapshots under `sources/` (with `sources/MANIFEST.sha256`); the runtime never
reads them. The full design is in [docs/architecture.md](docs/architecture.md); the file-by-file
layout is in [docs/repository-map.md](docs/repository-map.md).

## Certification

The correctness certificate is the deterministic regression gate: with ground-truth fills and no
model in the loop, the eighteen-problem battery must score **1.0**. The current results — with
evidence under `main/runs/`, reported in [docs/certification.md](docs/certification.md):

| Check | Result |
|---|---|
| `validate_skeleton` (8 gc + 7 polyc cases + 3 guards) | 15/15 PASS (gc worst ~1e-14; polyc rel_max 0.00e+00) |
| deterministic battery (18 problems, no model) | score 1.0 |
| mock graph-mechanics smoke | 18/18 records |
| COMSOL builders | static 4/4; opt-in execution verified to the COMSOL-engine boundary |
| integrity manifest | 9/9 |

## Commands

All commands run from `main/` with the pinned interpreter — its `.venv` is a symlink to the
project virtual environment, and `python -m` package resolution requires `main/` as the working
directory. Always use `.venv/bin/python`, never a bare `python3`:

```bash
cd main

# skeletons ≡ references, 15 cases, both templates (no model)
.venv/bin/python -m eval.validate_skeleton

# the deterministic regression gate — must stay 1.0 (18 problems)
.venv/bin/python -m eval.harness --mode deterministic

# full pipeline with the no-network mock client (graph mechanics)
.venv/bin/python -m eval.harness --mode llm --profile mock
```

For a live run, always pass `--profile`. The shipped `active_profile` (`openai_example`) is a
placeholder pointing at `api.your-provider.example`, so a bare live run errors out until it is
pointed at a real endpoint. Set your OpenAI-compatible `base_url`, `model`, and `api_key_env` in
`main/config.yaml` (keep `json_mode: plain`) and run
`.venv/bin/python -m eval.harness --mode llm --profile openai_example`
(see [installer/INSTALL.md](installer/INSTALL.md), section 5).

## Running the COMSOL leg

The pipeline emits a COMSOL builder (`model_comsol.py`) for each problem and, by default, only
static-checks it (four checks). To execute it and produce a solved `.mph`, opt in:

```bash
cd main
# deterministic path (no model): a two-problem battery, one per template
.venv/bin/python -m eval.harness --mode deterministic --comsol-exec \
  --battery <dir-with-a-manifest-listing-p03_single_crystal-and-p18_v0_dilute>
# or the LangGraph path:
EFORGE_COMSOL_EXEC=1 .venv/bin/python -m eval.harness --mode llm --profile openai_example
```

Each run builds the model, `solve('std1')`s the stationary study, and saves a **solved** COMSOL
model into the problem's output directory (`gc_pb2d_<pid>.mph` / `polyc_mpb2d_<pid>.mph`). This
needs a COMSOL install + MPh runtime + license (or a reachable `comsolmphserver` via MPh's
`mph.start(host=…)`); on a machine without one, the run reaches `mph.start()` and records
`Could not find a supported Comsol installation` in the per-problem `comsol_error` field
(non-gating — the score is unaffected). The two representative test cases are `p03_single_crystal`
(gc) and `p18_v0_dilute` (polyc).

## Serving

Model serving is configured in `main/config.yaml`, which ships two profiles:

- **`mock`** — an in-process, no-network fixture client that answers from
  `ground_truth_fills.yaml`; it exercises the graph mechanics offline (`--profile mock`).
- **`openai_example`** *(active)* — a template for any OpenAI-compatible endpoint. Set its
  `base_url`, `model`, and `api_key_env` (the name of the environment variable holding your key).

`json_mode: plain` is load-bearing: grammar-constrained decoding was measured to drop optional
fields and snap floats to schema bounds, so plain mode is kept; do not re-enable `openai_schema`
without re-testing.

## Integrity model

- **The deterministic regression gate** must score 1.0 at all times; any regression is rejected.
  This — not any live score — is the authoritative correctness certificate.
- **The integrity manifest** (`integrity.json`) SHA256-pins nine frozen template and reference
  files; `installer/verify.py` recomputes all nine and requires 9/9.
- Any frozen file can be restored from git with `git show <commit>:<path>`.

## Known limitations

- **COMSOL execution needs a COMSOL engine.** The builder is emitted and static-checked (4/4) by
  default, and can be executed opt-in to build, solve, and save a `.mph`. A real `.mph` requires a
  COMSOL install + MPh runtime + license (or a reachable `comsolmphserver`); without one the run
  reaches `mph.start()` and records `Could not find a supported Comsol installation` (non-gating).
  The post-solve charge/capacitance export is best-effort, pending a licensed run. See
  [Running the COMSOL leg](#running-the-comsol-leg).
- **Live scores do not certify physics** — they are the served model's behaviour and vary run to
  run; the deterministic 1.0 gate is the certificate.
- **The mock battery is a mechanics artifact**, not a physics result.
- **F5 adsorption is reproduced with a caveat** — the Parsons-Zobel slope magnitudes
  (≈0.699/0.536) sit below the paper anchors (0.76/0.48); the `adsorption_shift` gate certifies a
  present, finite, physically-sized shift, not the article-grade band.
- **Windows support is via WSL only.**

## License

eforge's own code and documentation are released under the BSD-3-Clause-LBNL license
(SPDX: `BSD-3-Clause-LBNL`); see [LICENSE](LICENSE). The `sources/` directory contains
read-only vendored snapshots of third-party reference implementations, which remain under
their own upstream terms.
