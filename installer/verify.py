#!/usr/bin/env python3
"""verify.py — run eforge's four offline certification checks and report a table.

Stdlib-only; runs under ANY python (it locates the repo venv itself and uses it
for the harness subprocesses — it never trusts the interpreter it was started
with). Zero network required: all four checks are offline by construction.

Checks (fail-fast, cheap first):
  1. integrity     9 SHA256-pinned files vs integrity.json  (~1 s)
  2. skeleton      python -m eval.validate_skeleton  -> 'ALL PASS' + exit 0
  3. deterministic python -m eval.harness --mode deterministic      -> score 1.0, 18 problems
  4. mock          python -m eval.harness --mode llm --profile mock -> 18 records

Expected wall time ~4-6 minutes (the deterministic battery solves 16 FEM models).
Exit codes: 0 all pass · 1 any check failed · 2 layout/venv error.

Usage:  python3 installer/verify.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main"
VENV_PY = ROOT / ".venv" / "bin" / "python"
TIMEOUT = 1800


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(MAIN), capture_output=True, text=True, timeout=TIMEOUT
    )


def check_integrity() -> tuple[bool, str]:
    fw = json.loads((ROOT / "integrity.json").read_text())["files"]
    bad = [
        rel
        for rel, want in fw.items()
        if hashlib.sha256((MAIN / rel).read_bytes()).hexdigest() != want
    ]
    return (not bad, f"{len(fw) - len(bad)}/{len(fw)} MATCH" + (f" — MISMATCH: {bad}" if bad else ""))


def check_skeleton() -> tuple[bool, str]:
    proc = run([str(VENV_PY), "-m", "eval.validate_skeleton"])
    ok = proc.returncode == 0 and "ALL PASS" in proc.stdout
    detail = f"exit {proc.returncode}, {'ALL PASS' if 'ALL PASS' in proc.stdout else 'no ALL PASS'}"
    if not ok:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        detail += " | tail: " + " / ".join(tail)
    return ok, detail


def check_deterministic(ts: int) -> tuple[bool, str]:
    out_rel = f"runs/install_verify_det_{ts}"
    proc = run([str(VENV_PY), "-m", "eval.harness", "--mode", "deterministic", "--out", out_rel])
    metrics_path = MAIN / out_rel / "metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        return False, f"exit {proc.returncode}, metrics {'missing' if not metrics_path.exists() else 'present'} | " + " / ".join(tail)
    m = json.loads(metrics_path.read_text())
    ok = m.get("score") == 1.0 and m.get("n_problems") == 18
    return ok, f"score {m.get('score')}, {m.get('n_problems')} problems, halluc {m.get('hallucination_incidents')} ({out_rel})"


def check_mock(ts: int) -> tuple[bool, str]:
    out_rel = f"runs/install_verify_mock_{ts}"
    proc = run([str(VENV_PY), "-m", "eval.harness", "--mode", "llm", "--profile", "mock", "--out", out_rel])
    records_path = MAIN / out_rel / "records.json"
    if proc.returncode != 0 or not records_path.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        return False, f"exit {proc.returncode}, records {'missing' if not records_path.exists() else 'present'} | " + " / ".join(tail)
    n = len(json.loads(records_path.read_text()))
    return n == 18, f"{n}/18 records (mock score is a mechanics artifact, not physics) ({out_rel})"


def main() -> int:
    if not (MAIN / "eval" / "harness.py").exists():
        print("verify: repo layout not found (expected main/eval/harness.py next to installer/)", file=sys.stderr)
        return 2
    if not VENV_PY.exists():
        print(f"verify: venv python not found at {VENV_PY} — run installer/install.py first", file=sys.stderr)
        return 2

    print("eforge verify — 4 offline checks, expect ~4-6 minutes total")
    ts = int(time.time())
    results: list[tuple[str, bool, str, float]] = []
    checks = [
        ("integrity", check_integrity),
        ("skeleton", check_skeleton),
        ("deterministic", lambda: check_deterministic(ts)),
        ("mock", lambda: check_mock(ts)),
    ]
    all_ok = True
    for name, fn in checks:
        t0 = time.time()
        try:
            ok, detail = fn()
        except subprocess.TimeoutExpired:
            ok, detail = False, f"TIMEOUT after {TIMEOUT}s"
        except Exception as exc:  # layout/JSON errors -> report, don't crash the table
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        dt = time.time() - t0
        results.append((name, ok, detail, dt))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<14} ({dt:5.1f}s)  {detail}")
        all_ok &= ok
        if not ok and name in ("integrity", "skeleton"):
            print("  (fail-fast: skipping remaining checks)")
            break

    print()
    print("=" * 72)
    for name, ok, detail, dt in results:
        print(f"  {'PASS' if ok else 'FAIL':<4}  {name:<14} {detail}")
    print("=" * 72)
    if all_ok and len(results) == len(checks):
        print("ALL CHECKS PASSED — eforge is certified-green on this machine.")
        return 0
    print("VERIFICATION FAILED — see details above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
