from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GCPOLYC_OUTPUT_DIR', str(ROOT / 'output'))

import numpy as np
import scipy
import matplotlib
import fitz
import pytest
import jax
import jax.numpy as jnp


EXPECTED_VENV = ROOT / ".venv"


def test_imports():
    assert np.__version__
    assert scipy.__version__
    assert matplotlib.__version__
    assert fitz.__version__
    assert pytest.__version__


def test_jax_float64():
    jax.config.update("jax_enable_x64", True)
    arr = jnp.array([1.0])
    assert arr.dtype == jnp.float64, f"Expected float64, got {arr.dtype}"


def test_constraints_md_exists():
    p = ROOT / "reference" / "CONSTRAINTS.md"
    assert p.exists(), f"reference/CONSTRAINTS.md not found at {p}"
    assert p.stat().st_size > 0, "reference/CONSTRAINTS.md is empty"


def test_venv_path():
    # Python 3.14 resolves sys.executable through symlinks; use sys.prefix instead.
    assert Path(sys.prefix).resolve() == EXPECTED_VENV.resolve(), (
        f"sys.prefix={sys.prefix} is not {EXPECTED_VENV}"
    )


def test_src_constants():
    from src import EPS0, F, R, NA, E_CHARGE, K_B, T, EPS_R_WATER
    assert abs(EPS_R_WATER - 78.5) < 1e-9, "EPS_R_WATER must be 78.5 (Table S1)"
    assert abs(T - 298.15) < 1e-9
    assert F > 96000
