"""Reproducibility tests for stable hashing used in cache signatures/seeds."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def _run_python(code: str) -> str:
    """Execute Python code in a fresh interpreter and return stdout."""
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()


def test_stable_int_seed_identical_across_processes():
    code = textwrap.dedent("""
        from hash_utils import stable_int_seed
        print(stable_int_seed(("scenario_simulation", "ERCOT", 95, 14.2)))
    """)
    first = _run_python(code)
    second = _run_python(code)
    assert first == second


def test_callbacks_data_hash_identical_across_processes():
    code = textwrap.dedent("""
        import json
        import pandas as pd
        from components.callbacks import _compute_data_hash

        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        demand = pd.DataFrame({"timestamp": ts, "demand_mw": range(48)})
        weather = pd.DataFrame({"timestamp": ts, "temperature_2m": range(48)})
        print(_compute_data_hash(demand, weather, "PJM"))
    """)
    first = _run_python(code)
    second = _run_python(code)
    assert first == second


def test_simulated_forecasts_identical_across_processes():
    code = textwrap.dedent("""
        import json
        import numpy as np
        from models.model_service import _simulate_forecasts

        actual = np.linspace(30000, 32000, 96)
        result = _simulate_forecasts("CAISO", actual, models_shown=None)
        payload = {
            "prophet": [round(float(v), 6) for v in result["prophet"][:6]],
            "arima": [round(float(v), 6) for v in result["arima"][:6]],
            "xgboost": [round(float(v), 6) for v in result["xgboost"][:6]],
            "ensemble": [round(float(v), 6) for v in result["ensemble"][:6]],
        }
        print(json.dumps(payload, sort_keys=True))
    """)
    first = json.loads(_run_python(code))
    second = json.loads(_run_python(code))
    assert first == second
