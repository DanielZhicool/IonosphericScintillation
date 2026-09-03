"""Pytest fixtures for automated Qt GUI testing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.config import ProcessingConfig
from core.synthetic_generator import generate_synthetic_scintillation


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """Provide a synthetic DataFrame representing a 600-sample 4-channel observation."""
    N = 600
    fs = 1.0
    sig1, sig2 = generate_synthetic_scintillation(length=N, fs=fs, seed=42)
    start_dt = pd.to_datetime("2026-01-01 00:00:00")
    datetimes = [start_dt + pd.Timedelta(seconds=i) for i in range(N)]

    df = pd.DataFrame(
        {
            "MJD": np.linspace(56000.0, 56000.1, N),
            "Datetime": datetimes,
            "Time_sec": np.arange(N, dtype=float),
            "P1_20A": sig1 + 1.0,
            "M1_20A": np.ones(N),
            "P2_20B": sig1 + 1.0,
            "M2_20B": np.ones(N),
            "P3_25A": sig2 + 1.0,
            "M3_25A": np.ones(N),
            "P4_25B": sig2 + 1.0,
            "M4_25B": np.ones(N),
        }
    )
    df.attrs["fs"] = fs
    return df


@pytest.fixture
def mock_sessions() -> list[dict[str, Any]]:
    """Provide mock session dictionary list."""
    return [
        {"target": "3C144", "start": 50, "end": 250},
        {"target": "3C273", "start": 300, "end": 550},
    ]


@pytest.fixture
def default_config() -> ProcessingConfig:
    """Provide default immutable ProcessingConfig."""
    return ProcessingConfig()
