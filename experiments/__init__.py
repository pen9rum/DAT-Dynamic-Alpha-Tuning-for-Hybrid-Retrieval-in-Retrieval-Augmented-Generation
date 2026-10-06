"""Experiment entry points. Run from the project root, e.g.
    python -m experiments.oracle_gap --config configs/squad.yaml
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(_SRC))
