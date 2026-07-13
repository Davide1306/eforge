# Architecture

eforge is a model factory: given a short natural-language electrochemistry problem, it produces
a validated simulation model without human intervention. This document describes what it
produces, the design principles behind it, the pipeline, the evaluation system, and the
integrity model. For the file-by-file layout see [repository-map.md](repository-map.md); for the
certification evidence see [certification.md](certification.md).

## Overview

Given a problem such as *"a polycrystalline electrode with two facet types in 0.1 mM salt, sweep
the potential −0.5…+0.5 V"*, eforge produces:

1. **An executable Python model** — a standalone 2-D finite-element / finite-difference solver
   (numpy/scipy only) that converges and passes physics acceptance gates.
2. **A statically-valid COMSOL builder script** (MPh API) for the same case.

Two physics templates carry this, selected by a registry (`main/templates/registry.yaml`, three
entries including the `no_match` abstention label):

- **`gc_polyc_pb2d`** — the classical Gouy-Chapman 2-D Poisson-Boltzmann model on a two-facet
  stripe. The emitted skeleton reproduces its reference solver to ~1e-14 on eight frozen cases.
- **`polyc_mpb2d`** — the full modified Poisson-Boltzmann model: steric/Bikerman ion crowding,
  Langevin/Booth field-dependent permittivity, facet-resolved Robin boundary conditions, and
  Langmuir specific adsorption. The emitted skeleton reproduces its reference solver at
  rel_max 0.00e+00 on seven frozen cases.

Both templates reproduce the two model levels of Liu, Doblhoff-Dier & Koper, *Modelling the
Double Layer of Polycrystalline Electrodes*, ACS Electrochemistry 2026, 2, 995–1004
(DOI 10.1021/acselectrochem.5c00544). The two reference implementations are vendored as
read-only code snapshots under `sources/` — `GC-polyc/` → `gc_polyc_pb2d`, `polyc/` →
`polyc_mpb2d` — verified against their originals in `sources/MANIFEST.sha256`. `sources/` is
reference material only; the runtime never reads it (template discovery is registry-only, problem
discovery is manifest-only).

## Design principles

**1. The language model never writes code.** Templates are typed-slot specifications
(`main/templates/<id>/template.yaml`). Three model roles — the same served model under three
system prompts — emit only small, schema-constrained JSON objects:

| Role | Responsibility |
|---|---|
| `selector` | map the problem to a `template_id`, or `no_match` to abstain |
| `slot_filler` | fill one slot-group (parameters, sweep, mesh tier) as typed JSON |
| `cross_checker` | optional adversarial second pass on units and values (currently disabled) |

Deterministic Jinja2 emitters render the actual Python and COMSOL artifacts from the resolved
slots. The design asks the model only for typed transcription (for example, turning `"10 nm"`
into `10e-9`), never for code.

**2. Minimal context per call.** Each fill request sees one slot-group's schema, a small number
of few-shot examples, and the problem statement — nothing else. Invented fields, malformed JSON,
and bad corrections are counted as `hallucination_incidents` and rejected if they regress a
result. This narrow context is the primary defence against hallucination.

**3. First-answer correctness over self-repair.** Repair rounds are not treated as a convergence
mechanism; the served model is not asked to fix its own output. Instead the prompts and slot
descriptions are engineered so the first answer is correct. Every deterministic and mock battery
to date records `repair_rounds_total: 0`.

**4. Templates are added additively, never edited in place.** `polyc_mpb2d` was added without
touching any `gc_polyc_pb2d` template, gate, or reference file — all verified byte-identical by
the integrity manifest — and extended the shared
`main/eval/reference/ground_truth_fills.yaml` additively (the p01–p10 entries stayed
byte-unchanged), leaving the `gc_polyc_pb2d` battery subset scoring 1.0 throughout. New physics
arrives as new files plus one registry entry; template discovery is registry-only.

## The pipeline

The inner pipeline is a LangGraph state machine (`main/pipeline/graph.py`):

```
problem text
     │
     ▼
 ┌──────────┐    ┌────────────┐    ┌───────────────┐    ┌─────────┐    ┌───────┐
 │ selector │──▶│ fill_group │──▶│ assemble_emit │──▶│ execute │──▶│ gates │──▶ END
 └──────────┘    └────────────┘    └───────────────┘    └─────────┘    └───────┘
   (model)        (model, loops       (deterministic)   (deterministic) (deterministic)
      │            per group)               │                                 │
      ▼                ▲                     │      repair loop                │
  abstain /            └─────────────────────┴─────────────────────────────────┘
  no_match → END        a slot or gate failure is routed back to the offending group,
                        capped at policy.repair_max_rounds (2)
```

| Node | Kind | Function |
|---|---|---|
| `selector` | model | problem + registry → one `template_id`, or `no_match`. Matches on physics, not vocabulary. |
| `fill_group` | model | fills one slot-group at a time (`parameters_core`, `parameters_optional`, `sweep`, `mesh_tier` — the same four groups in both templates). Up to `retry_budget` (2) re-asks on validation failure (`graph.py:137`). |
| `assemble_emit` | deterministic | `resolve_slots()` merges fills with defaults, validates types/ranges, rejects invented fields (`SlotError`); Jinja2 emits `model.py` + `model_comsol.py`; static validators check both. |
| `execute` | deterministic | `subprocess.run(python model.py)` (900 s timeout); parses the final `RESULT_JSON:` line. |
| `gates` | deterministic | runs the template's physics gates (a per-template module, auto-discovered). |

**Repair routing.** A slot error is routed back to its group; a gate failure is routed by a fixed
table (`graph.py:291-302`): the `gc_polyc_pb2d` gates `sigma_vs_cg_single_facet`/`sigma_sign`
route to `parameters_core` and `cdl_positive` to `sweep`; the `polyc_mpb2d` gates
`cg_cross_check`/`minima_count_exact`/`feature_position`/`magnitude` route to `parameters_core`,
`pz_slope` to `sweep`, and `adsorption_shift` to `parameters_optional`. `residual_converged` has
no repair target — a convergence failure falls through to `failed_gates`. The re-asked group
receives feedback rendered from `main/pipeline/prompts/repair_feedback.yaml`.

**State.** A `PipelineState` TypedDict carries `fills`, `feedback`, `repair_rounds`,
`hallucination_incidents`, `failures`, `status`, `artifacts`, `exec_result`, `gate_results`, and
`llm_usage`. Every model call is appended to `traces.jsonl` in the run's output directory
(`graph.py:353`).

## Model roles and prompts

Prompts live in `main/pipeline/prompts/`:

| Role | Prompt | Output | Key rules |
|---|---|---|---|
| selector | `selector.system.md` | `{"template_id": "<id>"}` | classical Gouy-Chapman → `gc_polyc_pb2d`; steric/Bikerman, Booth/Langevin permittivity, or Langmuir adsorption → `polyc_mpb2d`; kinetics/faradaic currents, more than two facets/grain boundaries, non-aqueous, or time dependence → `no_match`. |
| slot_filler | `slot_filler.system.md` | one JSON object | unit recipes: `c_b` in mol/m³ (1 mM = 1.0), `L` in m (10 nm = 10e-9), `panel_L_nm` in nm, `C_H` in F/m² (50 µF/cm² = 0.5); omit unstated optionals; single-point problem ⇒ `sweep_var='none'`. |
| cross_checker | `cross_checker.system.md` | `{"ok", "reason", "corrections"}` | disabled (`policy.cross_check: false`); in testing its corrections reduced fill accuracy rather than improving it, so it is left off. |

Few-shots come from `main/pipeline/prompts/fewshots/parameters.jsonl` (`fewshot_count: 1`). On
OpenAI-compatible profiles the client prepends a `Reasoning: <effort>` line from the profile's
per-node `reasoning_effort` map (`llm_client.py:67-69`).

## Serving profiles and policy

Profiles are defined in `main/config.yaml`. Two ship:

| Profile | Backend | Use |
|---|---|---|
| `openai_example` *(active)* | any OpenAI-compatible endpoint — set `base_url`, `model`, `api_key_env`; `json_mode: plain`, temperature 0, max_tokens 4096, 900 s timeout | the live target (a placeholder until pointed at a real endpoint) |
| `mock` | in-process, no network | graph-mechanics testing (answers from `ground_truth_fills.yaml`) |

Three facts about the live profile are load-bearing:

- **`json_mode: plain` is required.** Grammar-constrained decoding (`response_format json_schema
  strict`) was measured to drop non-required fields and snap floats to schema bounds; plain mode
  passed the same tests. Plain mode sends no `response_format`; content is parsed by
  `_parse_json_loose` (`llm_client.py:140`). Do not re-enable `openai_schema` without re-testing.
- **The API key comes from the environment.** `api_key_env` names the environment variable that
  holds the key; the client reads it and falls back to `'none'` when the variable is empty or
  unset (`llm_client.py:45`), so no key sits in the repository. Point `api_key_env` at the
  variable that holds your key.
- **Decode kwargs.** The OpenAI branch forwards `temperature`/`max_tokens` from the profile,
  presence-checked so temperature 0 survives (`llm_client.py:101-104`). No seed; no client-side
  retry.

The policy block (`config.yaml`) sets `retry_budget: 2`, `fewshot_count: 1`, `chunk_mode:
per_group`, `cross_check: false`, `repair_max_rounds: 2`.

## Evaluation

**Battery.** Eighteen frozen problems in `main/problems/seed/` (`manifest.yaml`): p01–p10 for
`gc_polyc_pb2d` (nine runnable plus `p08_abstain_stern`, which must route `no_match`) and p11–p18
for `polyc_mpb2d` (seven runnable plus `p16_abstain_polyfacet`, which must route `no_match` —
more than two facets / grain boundaries is out of scope for both templates). Deliberate traps
include `p04_molar_trap` (molar units), `p06_unit_trap_CH` (µF/cm²), `p05_minimal_defaults` (omit
optionals), `p15_unit_trap`, `p12_booth_single` (field-dependent permittivity routing), and
`p13_adsorption_f5` (Langmuir adsorption).

**Physics gates — one module per template**, auto-discovered by the dispatcher
`main/eval/fast_gates.py` (`eval/gates/<template_id>.py`):

`gates/gc_polyc_pb2d.py` (criteria frozen in `templates/gc_polyc/gates.frozen.yaml`):

| Gate | Checks |
|---|---|
| `residual_converged` | solver success and \|R\| < 1e-6 |
| `sigma_vs_cg_single_facet` | σ vs the Chapman-Grahame oracle, tier-aware tolerance; single-facet cases |
| `sigma_sign` | sign of σ̄ matches the C_H-weighted potential drive |
| `cdl_positive` | differential capacitance > 0 across the sweep |
| `dh_limit_analytic` | oracle self-consistency in the Debye-Hückel limit |

`gates/polyc_mpb2d.py` (criteria frozen in `templates/polyc_mpb2d/gates.frozen.yaml`):

| Gate | Checks |
|---|---|
| `residual_converged` | solver success and \|R\| < 1e-6 |
| `cg_cross_check` | model C_dl minimum vs the analytic GCS closed form (`eval/reference/polyc_oracle.py`); single-facet cases |
| `minima_count_exact` | the number of C_dl minima exactly equals the regime's expected count (1 or 2) |
| `feature_position` | worst minimum-position error ≤ 20 mV vs a supplied reference — battery-side; skipped without one |
| `magnitude` | capacitance magnitude within 10% of reference; falls back to the analytic GCS minimum on single-facet cases |
| `pz_slope` | \|Δslope\| ≤ 0.05 and R² > 0.99 vs a supplied reference slope — battery-side; skipped without one |
| `adsorption_shift` | when adsorption is on: the induced E_min shift is present, finite, \|shift\| ≥ 1 mV |

The reference-curve gates (`feature_position`, `magnitude` against a digitized curve, `pz_slope`)
are battery-side by design — a novel model-filled problem has no digitized curve. Full
reference-curve agreement is adjudicated by `eval/validate_skeleton.py` against
`polyc_reference.json` (rel_max 0.00e+00 at certification).

**Static validators** run before execution: `emitters/validators/py_static.py` (AST, an import
allowlist `{numpy, scipy, json, csv, math, sys, os}`, and per-template required definitions — a
template's `outputs.required_defs` is enforced when declared, with the gc set as fallback) and
`comsol_static.py` (MPh API surface, one `parameter()` call per slot, exactly two facet
boundaries).

**Score** (`main/eval/scorer.py:3-9`), a weighted sum with maximum 1.0:

```
0.35·physics_pass_rate + 0.25·slot_fill_accuracy + 0.15·executability_rate
+ 0.15·selection_accuracy + 0.10·comsol_static_pass_rate
```

**Regression vetoes.** Two hard vetoes force a revert regardless of score (`scorer.py:16-27`):
`hallucination_incidents` regressed versus best, or `executability_rate` regressed versus best.
Two further invariants sit outside the scorer: any changed integrity hash (checked by
`installer/verify.py`) and the deterministic battery dropping below 1.0.

Each run writes `metrics.json` (aggregates plus tokens and wall time) and `records.json`
(per-problem expected-versus-got, gate results, execution result).

## Integrity model

Correctness rests on three anchors:

- **The deterministic regression gate.** `--mode deterministic` pushes ground-truth fills through
  the same resolve→emit→execute→gates path with no model in the loop. It must score **1.0** at all
  times (eighteen problems, both templates, two abstain traps); any regression is rejected. This —
  not any live score — is the authoritative correctness certificate.
- **The integrity manifest.** Nine files (both templates' frozen physics narratives, frozen gates,
  and Python skeletons, plus `gc_reference.json`, `polyc_reference.json`, and the held-out
  `ground_truth_fills.yaml`) are SHA256-pinned in `integrity.json` (keys are `main/`-relative).
  `installer/verify.py` recomputes all nine and requires 9/9. Any file can be restored from git
  with `git show <commit>:<path>`.
- **Held-out ground truth.** `main/eval/reference/ground_truth_fills.yaml` is used only for
  scoring; it is never copied into prompts or few-shots.

## Operating commands

All commands run from `main/` with the pinned interpreter — its `.venv` is a symlink to the
project virtual environment, and `python -m` package resolution requires `main/` as the working
directory. Always use `.venv/bin/python`, never a bare `python3`:

```bash
cd main
.venv/bin/python -m eval.validate_skeleton                    # skeletons ≡ references, 15 cases (no model)
.venv/bin/python -m eval.harness --mode deterministic         # the regression gate — must stay 1.0 (18 problems)
.venv/bin/python -m eval.harness --mode llm --profile mock    # graph mechanics, no network
```

For a live run, always pass `--profile`. The shipped `active_profile` (`openai_example`) is a
placeholder, so a bare live run errors out until it is pointed at a real endpoint. Set its
`base_url`, `model`, and `api_key_env` in `main/config.yaml`, export the key, and run:

```bash
.venv/bin/python -m eval.harness --mode llm --profile openai_example
```

Keep `json_mode: plain` on any new profile (see [Serving profiles](#serving-profiles-and-policy)).
The one-command setup that builds the virtual environment and runs the four offline checks is
`python3 installer/install.py` (see [../installer/INSTALL.md](../installer/INSTALL.md)).

## Running the COMSOL leg

For every problem the pipeline emits a COMSOL builder (`model_comsol.py`) and static-checks it by
default (four checks: AST/markers, MPh-API surface, parameter completeness, exactly-two-facet
selections). It can also *execute* that builder — opt-in via `harness --comsol-exec` or the
environment variable `EFORGE_COMSOL_EXEC=1` in the LangGraph path — which builds, `solve('std1')`s
the stationary study, and saves a solved `.mph` (`gc_pb2d_<pid>.mph` / `polyc_mpb2d_<pid>.mph`)
into the problem's output directory. That last step requires a COMSOL install + MPh runtime +
license (or a reachable `comsolmphserver` via MPh's `mph.start(host=…)`); without one the run
reaches `mph.start()` and records `Could not find a supported Comsol installation` in a non-gating
`comsol_error` field (the score is unaffected). Two representative test cases exercise the leg,
one per template: `p03_single_crystal` (gc) and `p18_v0_dilute` (polyc). The post-solve
charge/capacitance export expression is best-effort, pending validation on a licensed run.

## Known limitations

- **COMSOL execution needs a COMSOL engine.** The emit-and-static-check path passes by default;
  opt-in execution reaches `mph.start()` and produces a solved `.mph` only where a COMSOL install +
  MPh runtime + license (or a reachable `comsolmphserver`) is present. The emitted script and the
  wiring are correct; only the engine is absent here. See "Running the COMSOL leg" above.
- **Live scores do not certify physics.** They are the served model's behaviour and vary run to
  run; the deterministic 1.0 gate is the certificate.
- **The mock battery is a mechanics artifact.** Its answers are canned; it proves only that the
  pipeline runs end to end.
- **F5 adsorption is reproduced with a caveat.** The Parsons-Zobel slope magnitudes (≈0.699/0.536)
  sit below the paper anchors (0.76/0.48); the `adsorption_shift` gate certifies a present, finite,
  physically-sized shift, not the article-grade band.
- **Windows support is via WSL only.** The installer refuses to run under native Windows.
