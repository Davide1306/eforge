# Live model reference run — local Gemma (8B)

A concrete end-to-end run of the full pipeline (selector → slot-filler → executor
→ physics gates) driven by a **real served language model**, as opposed to the
deterministic gate (no model) or the mock client (canned answers). It demonstrates
that a modest local model can drive the factory; it is **illustrative, not part of
the correctness certificate**.

> Live scores are the served model's behaviour and vary with model, quantization,
> runtime, and decoding. They are **non-gating**. The authoritative correctness
> certificate remains the deterministic regression gate (`--mode deterministic`,
> score 1.0), which uses no model at all.

## Setup

| | |
|---|---|
| Model | Gemma-family, 8.0B params, Q4_K_M, 128K context, Apache-2.0 (Ollama tag `gemma4:e4b`, the Gemma 3n E4B class) |
| Served tag | `gemma4-temp0:latest` — a temperature-0 build of the above |
| Endpoint | Ollama OpenAI-compatible API (`http://localhost:11434/v1`) |
| Decoding | `temperature: 0`, `json_mode: plain` (profile `gemma_local` in `main/config.yaml`) |
| Battery | `problems/seed` (all 18 problems) |

Reproduce (needs the model pulled in a local Ollama and the `gemma_local` profile):

```bash
cd main
.venv/bin/python -m eval.harness --mode llm --profile gemma_local \
  --battery problems/seed --out runs/gemma_live
```

## Result — composite score 0.976

| Metric | Value |
|---|---|
| **Physics-gate pass rate** | **1.0** (all 16 non-abstain problems pass every physics gate) |
| Executability | 1.0 |
| Hallucination incidents | 0 |
| Repair rounds | 0 |
| Template selection | 0.889 (16/18) |
| Slot-fill accuracy | 0.972 |
| COMSOL static-check | 1.0 |
| **Composite score** | **0.976** |
| Cost | 122,789 tokens · ~35 s/problem · ~11 min total |

The two abstain traps (`p08`, `p16`) are correctly routed to `no_match`.

### The two imperfections (both benign)

- **Routing (16/18).** `p01_f2a_dilute` and `p09_custom_sweep` were routed to the
  full `polyc_mpb2d` model where ground truth expects the simpler `gc_polyc_pb2d`.
  Because the modified-PB model is a physical **superset** of Gouy-Chapman, both
  still pass every physics gate in these dilute/point-ion regimes — it is
  over-selection, not a wrong answer.
- **Slot-fill (97.2%).** Every genuine miss on a correctly-routed problem is on a
  physically-inconsequential slot — the stripe period `panel_L_nm` on single-facet
  cases where the prompt states the period is irrelevant, plus one adsorption-energy
  default (`E_ads0`) on `p13`. None breaks a physics gate, which is why the physics
  pass rate stays 1.0.

## Files

`metrics.json` (the summary above), `records.json` (per-problem outcomes), and one
folder per problem holding the emitted `model.py` / `model_comsol.py`, any sweep
CSVs, and `traces.jsonl` — the full per-call log (system/user prompts, raw model
output, token counts, latencies) that evidences the served-model calls.
