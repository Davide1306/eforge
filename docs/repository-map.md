# Repository map

The role of each part of the repository. Markers used below: **frozen** = integrity-pinned or
otherwise not to be edited; **mutable** = the intended editing surface; **generated** = build or
evaluation artifacts. For the design and pipeline see [architecture.md](architecture.md).

```
eforge/
├── README.md                 quickstart and system overview
├── integrity.json            the nine SHA256 anchors (keys main/-relative)
├── requirements.txt          top-level Python dependencies (installer uses the pinned lock)
├── .gitignore                venv, caches, install-verify run dirs
├── installer/                receiver setup (install / verify + lock + INSTALL.md)
├── main/                     the working system (frozen except the documented mutation surface)
├── sources/                  vendored read-only code snapshots of the two reference implementations
└── docs/                     this documentation set
    ├── architecture.md
    ├── repository-map.md
    ├── certification.md
    └── LICENSE                BSD-3-Clause-LBNL (eforge's own code and docs)
```

## Top-level files

| Path | Role |
|---|---|
| `README.md` | Quickstart (`python3 installer/install.py`), two-template overview, the operating commands, and known limitations. |
| `integrity.json` | frozen. The SHA256 manifest of the nine pinned template and reference files (keys are `main/`-relative). Recomputed by `installer/verify.py`; required to be 9/9. |
| `requirements.txt` | The top-level dependency list. The exact tested versions live in `installer/requirements.lock.txt`. |
| `.gitignore` | Ignores `.venv/`, `main/.venv`, `__pycache__/`, `.pytest_cache/`, and the install-verify run dirs (`main/runs/install_verify_*/`). |

## `installer/`

| Path | Role |
|---|---|
| `install.py` | One-command setup: checks OS and Python (≥ 3.11), creates the virtual environment, installs the pinned dependencies, recreates the `main/.venv` symlink, and runs the verification suite. |
| `verify.py` | The four offline checks — integrity 9/9 (against `integrity.json`), skeleton, deterministic 1.0, mock 18/18 — re-runnable standalone. |
| `requirements.lock.txt` | The exact tested dependency versions. The scipy pin is load-bearing (the `polyc_mpb2d` template imports a private scipy API). |
| `INSTALL.md` | Receiver instructions, live-endpoint setup, and troubleshooting. |

## `main/` — configuration

`main/config.yaml` (mutable) defines the two serving profiles (`mock`, `openai_example`) and the
policy block; see [architecture.md](architecture.md), "Serving profiles and policy".

## `main/pipeline/` — the LangGraph application

| Path | Role |
|---|---|
| `graph.py` | frozen. The five-node state machine (`selector`, `fill_group`, `assemble_emit`, `execute`, `gates`), conditional routing, repair loop, the gate→group repair table (`graph.py:291-302`, covering both templates), and the `run_pipeline()` entry point. |
| `llm_client.py` | frozen. `LLMClient.complete_json()` over two backends: OpenAI-compatible (`json_mode: plain` ⇒ no `response_format`, parsed by `_parse_json_loose` at `llm_client.py:140`; profile `temperature`/`max_tokens` passed presence-checked at `llm_client.py:101-104`) and an in-process mock. `api_key_env` names the environment variable holding the key; empty ⇒ `'none'` (`llm_client.py:45`). Logs every call to `traces.jsonl`. |
| `schemas.py` | frozen. Builds per-group JSON Schemas from `template.yaml`; the selector and checker schemas. |
| `prompts/` | mutable. `selector.system.md`, `slot_filler.system.md`, `cross_checker.system.md` (disabled), `repair_feedback.yaml`, `fewshots/parameters.jsonl`. |

## `main/emitters/` — deterministic rendering and static checks (frozen)

| Path | Role |
|---|---|
| `resolve.py` | `resolve_slots(template, fills)`: merge defaults, validate types/ranges/enums, reject invented fields. Template resolution is registry-only (`resolve.py:11,21`) — the runtime cannot see `sources/`. |
| `python_emitter.py` / `comsol_emitter.py` | Render `templates/<id>/python_skeleton.py.j2` / `comsol_skeleton.py.j2` with resolved slots; fail on unrendered markers. |
| `validators/py_static.py` | AST parse; import allowlist `{numpy, scipy, json, csv, math, sys, os}`; per-template required definitions — enforced when a template declares `outputs.required_defs` (polyc_mpb2d: `[solve_2d, C_dl_2facet, minima_from_qfree, main]`), with the gc set `{solve_pb_2d, residual_2d, build_jac_csr, main}` as fallback; the `RESULT_JSON` contract. |
| `validators/comsol_static.py` | AST plus `import mph`; calls checked against the MPh API surface (`api_surface_mph.txt`); one `parameter()` per required slot; exactly two facet-boundary references. |

## `main/templates/` — the typed-slot template library

| Path | Role |
|---|---|
| `registry.yaml` | Three entries: `gc_polyc_pb2d`, `polyc_mpb2d`, `no_match`. The only file the selector reads; discriminative descriptions route steric/Booth/adsorption physics to `polyc_mpb2d` and out-of-scope physics to `no_match`. |
| `gc_polyc/template.yaml` | Slot spec. Fillable groups: `parameters_core`, `parameters_optional`, `sweep`, `mesh_tier`. Only the `desc:` strings are mutable. |
| `gc_polyc/physics.frozen.md` | frozen (integrity-pinned). Governing PDE, boundary conditions, dimensionless form. |
| `gc_polyc/gates.frozen.yaml` | frozen (integrity-pinned). Acceptance criteria and regression block. |
| `gc_polyc/python_skeleton.py.j2` | frozen (integrity-pinned). The 2-D solver skeleton. |
| `gc_polyc/comsol_skeleton.py.j2` | The MPh builder skeleton. Not integrity-pinned. |
| `polyc_mpb2d/template.yaml` | Template 2's slot spec (same four groups; declares `outputs.required_defs`). Only the `desc:` strings are mutable. |
| `polyc_mpb2d/physics.frozen.md` | frozen (integrity-pinned). Modified-PB narrative: steric/Bikerman, Langevin/Booth permittivity, facet-resolved Robin BCs, Langmuir adsorption. |
| `polyc_mpb2d/gates.frozen.yaml` | frozen (integrity-pinned). Template 2's acceptance criteria. |
| `polyc_mpb2d/python_skeleton.py.j2` | frozen (integrity-pinned). Template 2's inlined-solver skeleton; reproduces `polyc_reference.json` at rel_max 0.00e+00. |
| `polyc_mpb2d/comsol_skeleton.py.j2` | Template 2's MPh builder. Not integrity-pinned. |

## `main/eval/` — harness and ground truth (frozen)

| Path | Role |
|---|---|
| `harness.py` | Entry point: `--battery` (default `problems/seed`), `--mode deterministic\|llm`, `--profile`, `--comsol-exec`, `--out` (working-directory-relative — run from `main/`). Deterministic mode = ground-truth fills through the pipeline with no model (the regression gate). Problems are manifest-only (`harness.py:159`); the only reference glob is confined to `eval/reference/` (`harness.py:43`). Writes `metrics.json` + `records.json`. |
| `metrics.py` / `scorer.py` | Aggregation and the weighted score (0.35 physics / 0.25 slot-fill / 0.15 exec / 0.15 selection / 0.10 comsol, `scorer.py:3-9`) plus the regression vetoes (`scorer.py:16-27`). |
| `fast_gates.py` → `gates/` | Template-agnostic dispatcher (`eval/gates/<template_id>.py`, auto-discovered) over two gate modules: `gates/gc_polyc_pb2d.py` and `gates/polyc_mpb2d.py`. |
| `validate_skeleton.py` | Proves emitted skeletons ≡ their references: eight gc cases (~1e-14) + seven polyc cases (rel_max 0.00e+00) + three corrections-guard checks. |
| `reference/cg_oracle.py` | frozen. Analytical Chapman-Grahame oracle. |
| `reference/gc_reference.json` | frozen (integrity-pinned). Eight full-solver gc reference cases. |
| `reference/polyc_oracle.py` | Analytic single-facet GCS closed forms plus the Langmuir C_ads peak. |
| `reference/polyc_reference.json` | frozen (integrity-pinned). Template 2's reference cases. |
| `reference/ground_truth_fills.yaml` | frozen (integrity-pinned). Per-problem expected template and fills for all eighteen problems. Held out — scoring only, never in prompts. |

## `main/problems/` — the battery

| Path | Role |
|---|---|
| `seed/manifest.yaml` + `p01…p18.md` | Eighteen frozen problem narratives: p01–p10 for `gc_polyc_pb2d` (`p08_abstain_stern` → `no_match`) and p11–p18 for `polyc_mpb2d` (`p16_abstain_polyfacet` → `no_match`). Traps include `p04_molar_trap`, `p06_unit_trap_CH`, `p05_minimal_defaults`, `p15_unit_trap`, `p12_booth_single`, `p13_adsorption_f5`, `p17_pz_slope`, `p18_v0_dilute`. |

## `main/runs/` — evaluation outputs (generated)

Certification and verification artifacts are written here, each a directory of `metrics.json`,
`records.json`, and one folder per problem (the emitted `model.py`, `model_comsol.py`, output
CSVs, and `traces.jsonl` for model runs). A certification produces a deterministic battery dir, a
mock dir, and — when serving is configured — a live dir; refer to them generically (for example,
the deterministic run dir under `main/runs/`). The install-verify dirs
(`main/runs/install_verify_*/`) are gitignored.

## `sources/` — vendored reference-implementation snapshots (frozen)

| Path | Role |
|---|---|
| `GC-polyc/` | Read-only code snapshot of the reference implementation for `gc_polyc_pb2d`. |
| `polyc/` | Read-only code snapshot of the reference implementation for `polyc_mpb2d`. |
| `README.md` | Provenance: include/exclude contract, copy date, template mapping, the runtime-safety proof (registry-only discovery, manifest-only problems), and MANIFEST usage. |
| `MANIFEST.sha256` | Hashes over both code trees, each verified equal to its original at copy time. Check with `cd sources && sha256sum -c MANIFEST.sha256`. |

## `docs/`

| Path | Role |
|---|---|
| `architecture.md` | The system: deliverable, design principles, pipeline, roles, serving, evaluation, integrity model, operating commands, the COMSOL leg, and known limitations. |
| `repository-map.md` | This file. |
| `certification.md` | The certification report: the four offline checks and the COMSOL leg, each with the exact command output. |

## External dependencies

- **An OpenAI-compatible serving endpoint** — needed for live `--mode llm` runs. Add your
  `base_url`, `model`, and `api_key_env` to the `openai_example` profile in `main/config.yaml`
  (see [architecture.md](architecture.md)).
- **A COMSOL / MPh license** — required to execute the emitted COMSOL builders (they are
  static-checked by default here).
- **The reference implementations** (GC-polyc, polyc) — the physics ground truth the templates
  reproduce, vendored read-only under `sources/`.
