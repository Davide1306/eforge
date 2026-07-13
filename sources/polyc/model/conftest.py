"""pytest path bootstrap — make `from src import ...` resolve from anywhere.

conftest.py at model/ is an ancestor of model/tests/, so pytest auto-loads it
regardless of the invocation cwd, putting model/ on sys.path. run_F*.py scripts
add the same path themselves.
"""
import pathlib
import sys

_MODEL_DIR = str(pathlib.Path(__file__).resolve().parent)
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
