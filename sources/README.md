# sources/ — vendored reference-implementation snapshots (read-only)

**Provenance.** This directory holds vendored, read-only code snapshots of the two
upstream reference implementations the eforge templates reproduce. It exists for
portability and provenance only — it is **NOT part of the runtime**; the templates that
run live in `main/templates/`. Runtime safety: the harness cannot see this directory —
template discovery is registry-only (`main/emitters/resolve.py:11` pins `TEMPLATES_DIR`;
`:21` resolves ids by reading `templates/registry.yaml`), problems are manifest-only
(`main/eval/harness.py:159` reads `battery_dir / 'manifest.yaml'`), and the only
filesystem glob is confined to `main/eval/reference/` (`main/eval/harness.py:43`,
`(ROOT / 'eval/reference').glob('gt_*.yaml')`).

**Source paths + copy date.** Copied 2026-07-12 from the upstream project directories:

- `polyc` → `sources/polyc/`
- `GC-polyc` → `sources/GC-polyc/`

**Which template each project is the reference for:**

- `polyc` → `polyc_mpb2d` (full modified Poisson-Boltzmann)
- `GC-polyc` → `gc_polyc_pb2d` (classical Gouy-Chapman)

## What is included

Each snapshot is **code only** — the source, tests, run scripts, and frozen specification
sheets of the upstream project:

- `sources/polyc/` ← `polyc`: `model/src/**`, `model/run_*.py`
  (`run_F1`…`run_F5` + `run_V0`), `model/tests/**`, `model/conftest.py`,
  `model/requirements.txt`, `model/scripts/**`, `model/reference/CONSTRAINTS.md` plus the
  loose top-level files in `model/reference/`, and the root `README.md`.
- `sources/GC-polyc/` ← `GC-polyc`: `src/**`, `run_F*.py`, `tests/**`,
  `scripts/**`, `requirements.txt`, and `reference/CONSTRAINTS.md` plus the loose top-level
  `reference/` files.

The projects' generated `output/` figure trees are **dropped** — this is a code-only
snapshot. Virtual environments (`.venv`, which protects the `model/.venv -> ../.venv`
symlink), `.git`, `__pycache__`, and `.pytest_cache` are not copied. The runtime never
reads any of it.

## MANIFEST usage

```
cd sources && sha256sum -c MANIFEST.sha256
```

Every hash in `MANIFEST.sha256` was verified equal to its original file's hash at copy
time. The MANIFEST covers the two copied code trees (`polyc/`, `GC-polyc/`) — not itself,
and not this README (they have no original to verify against).
