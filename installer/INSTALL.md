# eforge — installation

`eforge` is shared as a plain folder (copy-paste, USB stick, zip, git — anything
that moves files). Everything machine-specific is rebuilt on your machine by one
command. No files outside this folder are ever created or modified.

## 1. Requirements

- **OS:** Linux, macOS, or Windows-via-WSL. (Windows-native Python is not
  supported — on Windows, install Ubuntu from the Microsoft Store and copy this
  folder INTO the Linux filesystem, e.g. `~/eforge`, not `C:\...`/`/mnt/c/...`.)
- **Python ≥ 3.11** with the `venv` module (tested on 3.14; 3.12+ recommended).
  Debian/Ubuntu ship venv separately: `sudo apt install python3.12-venv` (match
  your minor version).
- **Network access to PyPI** for `pip install` (~500 MB downloaded during
  install). Behind a proxy: `export HTTPS_PROXY=...` first.

## 2. Install (one command)

```bash
cd <wherever-you-copied>/eforge
python3 installer/install.py
```

That's it. The installer: checks your OS/Python → creates `.venv` → installs
the **exact tested dependency versions** (`installer/requirements.lock.txt`) →
recreates the `main/.venv` symlink that plain copies drop → runs the 4-check
verification suite (~5 min). You should end with:

```
  PASS  integrity       9/9 MATCH
  PASS  skeleton        exit 0, ALL PASS
  PASS  deterministic   score 1.0, 18 problems ...
  PASS  mock            18/18 records ...
ALL CHECKS PASSED — eforge is certified-green on this machine.
```

Re-running the installer is safe (a healthy venv is reused; `--force` rebuilds
it). Other flags: `--skip-verify`, `--latest` (unpinned deps — not recommended:
the scipy pin is load-bearing), `--offline` (only if this copy ships
`installer/wheels/`).

## 3. What just got verified

| Check | Meaning |
|---|---|
| integrity | the 9 SHA256-pinned frozen files (physics, gates, skeletons, reference data) are byte-identical to the certified originals (`integrity.json`) |
| skeleton | the emitted solvers reproduce the reference solutions (15 cases, both templates) |
| deterministic | ground-truth slot fills → emit → execute → physics gates = score **1.0** on all 18 problems (the regression gate; no model involved) |
| mock | the full LangGraph pipeline runs green end-to-end with a canned no-network client (mechanics only, not physics) |

## 4. Verify again anytime

```bash
python3 installer/verify.py
```

(Each run writes its artifacts to `main/runs/install_verify_*` — they are yours,
and gitignored.)

## 5. Live LLM runs (bring your own endpoint)

The pipeline talks to any **OpenAI-compatible** endpoint. The shipped
`active_profile` (`openai_example`) is a placeholder — its `base_url` points at
`api.your-provider.example`, so a bare `--mode llm` run errors out until you set
a real endpoint. **Always pass `--profile`**:

1. Edit `main/config.yaml` → set the `openai_example` profile's `base_url`,
   `model`, and `api_key_env` (or add your own `kind: openai` profile):

   ```yaml
     openai_example:
       kind: openai
       base_url: https://api.your-provider.example/v1
       model: your-model-name
       api_key_env: OPENAI_API_KEY     # name of the env var holding your key
       temperature: 0
       max_tokens: 4096
       timeout_s: 900
       json_mode: plain                # KEEP plain — schema-constrained decoding
                                        # was measured to drop fields and snap floats;
                                        # see README "Serving" notes
       reasoning_effort: {selector: low, slot_filler: high, cross_checker: low}
   ```

2. `export OPENAI_API_KEY=...`
3. From `main/`:

   ```bash
   .venv/bin/python -m eval.harness --mode llm --profile openai_example
   ```

   (or `PIPELINE_PROFILE=openai_example` in the environment).

The deterministic floor — not any live score — is the correctness certificate;
judge any live run against the deterministic gate, which must stay 1.0.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ensurepip`/venv creation fails (Debian/Ubuntu) | `sudo apt install python3.X-venv` for your X, re-run |
| `Python >= 3.11 required` | install a newer python, run `python3.12 installer/install.py` |
| pip: network/proxy errors | `export HTTPS_PROXY=...`; or corporate mirror via `PIP_INDEX_URL` |
| pip: `No matching distribution` on your Python | try `--latest` (unpinned; scipy risk documented above) |
| symlink error on `/mnt/c/...` (WSL) | move the folder into the Linux FS (`~/eforge`), re-run |
| live `--mode llm` run errors before reaching the model | the shipped `openai_example` is a placeholder — set your endpoint in `config.yaml` (§5) |
| `ModuleNotFoundError: eval` | run harness commands from the `main/` directory (its package root) |
| COMSOL (`model_comsol.py`) questions | the COMSOL leg is emitted + statically checked by default; executing it needs your own COMSOL/MPh license (see "Running the COMSOL leg" in README.md) |

## 7. Integrity notes

- The 9 integrity-pinned files are the certified truth anchors — don't edit them; the
  verify suite recomputes their hashes (`integrity.json`) on every run.
- Any frozen file can be restored from git with `git show <commit>:<path>`.
