#!/usr/bin/env python3
"""install.py — one-command setup for a freshly copied eforge folder.

Run it with any system Python >= 3.11 (stdlib only, no pip needed to start):

    python3 installer/install.py

What it does, in order:
  1. sanity-check the folder layout and OS (Linux/macOS/WSL supported)
  2. create the .venv (or reuse a healthy one; --force rebuilds)
  3. install the exact tested dependency versions (installer/requirements.lock.txt)
  4. create/repair the main/.venv -> ../.venv symlink (plain copies drop it)
  5. run the four offline certification checks (verify.py; ~4-6 min)

Flags:
  --force        delete and rebuild the venv even if it looks healthy
  --skip-verify  do everything except the final verification suite
  --latest       install unpinned deps from requirements.txt instead of the lock
                 (NOT recommended: the scipy pin is load-bearing)
  --offline      install from installer/wheels/ without network (only if this
                 copy ships a wheels/ dir built for your platform)

Exit codes: 0 ok · 2 unsupported OS · 3 python too old · 4 bad layout ·
5 venv creation failed · 6 venv failed (missing python3.X-venv) ·
7 pip install failed · 8 symlink failed · 9 verify failed
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "bin" / "python"
MAIN_LINK = ROOT / "main" / ".venv"


def step(n: int, msg: str) -> None:
    print(f"\n[{n}/5] {msg}")


def fail(code: int, *lines: str) -> int:
    for ln in lines:
        print(f"ERROR: {ln}", file=sys.stderr)
    return code


def venv_healthy() -> bool:
    if not VENV_PY.exists():
        return False
    try:
        proc = subprocess.run([str(VENV_PY), "-c", "import sys"], capture_output=True, timeout=30)
        return proc.returncode == 0
    except Exception:
        return False


def rmtree_venv_guarded() -> bool:
    """Delete ROOT/.venv only if it really looks like a venv."""
    if not (VENV / "pyvenv.cfg").exists():
        print(f"ERROR: {VENV} exists but has no pyvenv.cfg — refusing to delete it. "
              "Remove it manually and re-run.", file=sys.stderr)
        return False
    shutil.rmtree(VENV)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="eforge installer")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    print(f"eforge installer — root: {ROOT}")

    # ---- 1. layout + OS + python version -------------------------------
    step(1, "checking layout, OS, and Python version")
    if not (ROOT / "main" / "eval" / "harness.py").exists():
        return fail(4, "this does not look like an eforge folder "
                       "(main/eval/harness.py missing next to installer/)")
    if os.name != "posix":
        return fail(2,
                    "Windows-native Python is not supported (the pipeline is developed and",
                    "verified on Linux/WSL; the venv layout and symlink differ on Windows).",
                    "Use WSL: install Ubuntu from the Microsoft Store, copy this folder INTO",
                    "the Linux filesystem (e.g. ~/eforge, not /mnt/c/...), then run:",
                    "    python3 installer/install.py")
    if str(ROOT).startswith("/mnt/"):
        print("WARNING: this folder sits on a Windows-mounted drive (/mnt/...). Symlinks and")
        print("         file performance are unreliable there — consider moving it into the")
        print("         Linux filesystem (e.g. ~/eforge) if anything fails below.")
    if sys.version_info < (3, 11):
        return fail(3,
                    f"Python >= 3.11 required, found {sys.version.split()[0]} at {sys.executable}.",
                    "The pipeline is tested on Python 3.14 (3.12+ recommended).",
                    "Debian/Ubuntu:  sudo apt install python3.12 python3.12-venv   (or newer)",
                    "macOS:          brew install python@3.12",
                    "then re-run with that interpreter, e.g.:  python3.12 installer/install.py")
    print(f"  OK — {sys.version.split()[0]} at {sys.executable} "
          f"({'tested version' if sys.version_info[:2] == (3, 14) else 'tested on 3.14; this version is untested but should work'})")

    # ---- 2. venv --------------------------------------------------------
    step(2, "creating the virtual environment (.venv)")
    if venv_healthy() and not args.force:
        print("  existing healthy venv reused (use --force to rebuild)")
    else:
        if VENV.exists():
            why = "--force" if args.force else "existing venv is broken (its interpreter does not run — typical after a plain copy)"
            print(f"  rebuilding: {why}")
            if not rmtree_venv_guarded():
                return 5
        proc = subprocess.run([sys.executable, "-m", "venv", str(VENV)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            err = proc.stderr + proc.stdout
            if "ensurepip" in err:
                minor = sys.version_info.minor
                return fail(6,
                            "venv creation failed because ensurepip/pip is not bundled.",
                            f"Debian/Ubuntu ships it separately:  sudo apt install python3.{minor}-venv",
                            "then re-run this installer.")
            return fail(5, "venv creation failed:", err.strip()[-800:])
        print(f"  created {VENV}")

    # ---- 3. dependencies -------------------------------------------------
    step(3, "installing dependencies")
    lock = ROOT / "installer" / "requirements.lock.txt"
    reqs = ROOT / "requirements.txt"
    if args.latest:
        req_file = reqs
        print("  --latest: installing UNPINNED requirements.txt")
        print("  WARNING: the tested scipy pin (1.17.1) is load-bearing — the polyc template")
        print("           imports the private API scipy.optimize._numdiff.approx_derivative.")
    elif lock.exists():
        req_file = lock
        print(f"  installing exact tested versions from installer/{lock.name}")
    else:
        req_file = reqs
        print("  WARNING: installer/requirements.lock.txt missing — falling back to the")
        print("           UNPINNED requirements.txt (scipy version risk, see INSTALL.md).")
    cmd = [str(VENV_PY), "-m", "pip", "install", "-r", str(req_file)]
    if args.offline:
        wheels = ROOT / "installer" / "wheels"
        if not wheels.is_dir():
            return fail(7, "--offline requested but installer/wheels/ does not exist in this copy.")
        cmd += ["--no-index", "--find-links", str(wheels)]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        return fail(7,
                    "pip install failed (see pip output above).",
                    "* no network / behind a proxy?  export HTTPS_PROXY=... and re-run,",
                    "  or use a copy that ships installer/wheels/ with --offline",
                    "* version resolution error on your Python? try --latest (unpinned)",
                    "* corporate index? set PIP_INDEX_URL to your mirror")
    print("  dependencies installed")

    # ---- 4. main/.venv symlink -------------------------------------------
    step(4, "creating the main/.venv -> ../.venv symlink")
    try:
        if MAIN_LINK.is_symlink():
            if os.readlink(MAIN_LINK) == "../.venv":
                print("  already correct")
            else:
                MAIN_LINK.unlink()
                os.symlink("../.venv", MAIN_LINK)
                print("  repaired (was pointing elsewhere)")
        elif MAIN_LINK.is_dir():
            # a copy tool materialized the old symlink into a real directory
            if not (MAIN_LINK / "pyvenv.cfg").exists():
                return fail(8, f"{MAIN_LINK} is a real directory that does not look like a venv — "
                               "refusing to delete it; remove it manually and re-run.")
            shutil.rmtree(MAIN_LINK)
            os.symlink("../.venv", MAIN_LINK)
            print("  replaced a materialized copy with the symlink")
        elif MAIN_LINK.exists():
            return fail(8, f"{MAIN_LINK} exists and is neither a symlink nor a directory — "
                           "remove it manually and re-run.")
        else:
            os.symlink("../.venv", MAIN_LINK)
            print("  created")
    except OSError as exc:
        return fail(8,
                    f"could not create the symlink: {exc}",
                    "This usually means the folder sits on a filesystem without symlink",
                    "support (e.g. /mnt/c on WSL without Developer Mode). Move the folder",
                    "into the Linux filesystem (e.g. ~/eforge) and re-run.")

    # ---- 5. verification ---------------------------------------------------
    step(5, "verification suite (4 offline checks, ~4-6 min)")
    if args.skip_verify:
        print("  skipped (--skip-verify). Run later with:  python3 installer/verify.py")
    else:
        proc = subprocess.run([str(VENV_PY), str(ROOT / "installer" / "verify.py")], text=True)
        if proc.returncode != 0:
            return fail(9, "verification failed — eforge is installed but NOT certified-green",
                            "on this machine. See the table above and INSTALL.md troubleshooting.")

    print("\n" + "=" * 72)
    print("eforge is installed" + ("" if args.skip_verify else " and verified"))
    print("Next steps (all commands from the main/ directory):")
    print(f"    cd {ROOT / 'main'}")
    print("    .venv/bin/python -m eval.validate_skeleton         # skeleton floor")
    print("    .venv/bin/python -m eval.harness --mode deterministic   # the 1.0 regression gate")
    print("    .venv/bin/python -m eval.harness --mode llm --profile mock   # graph mechanics")
    print("Live LLM runs (your own endpoint): see installer/INSTALL.md, section 5.")
    print("Docs: README.md -> HANDOFF.md -> docs/")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
