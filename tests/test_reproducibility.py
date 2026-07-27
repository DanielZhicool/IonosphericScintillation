"""
Unit tests for deterministic seed reproducibility across gap-filling and spectral analysis pipelines.
"""

import numpy as np

from core.signal_processing import fill_gap_with_red_noise
from core.spectral_analysis import run_spectral_pipeline


def test_red_noise_seed_reproducibility():
    """Verify that fill_gap_with_red_noise produces bit-for-bit identical output with fixed seed."""
    np.random.seed(123)
    data = np.random.randn(1000)
    gap_start, gap_end = 200, 300

    run1 = fill_gap_with_red_noise(data, gap_start, gap_end, seed=42)
    run2 = fill_gap_with_red_noise(data, gap_start, gap_end, seed=42)

    np.testing.assert_array_equal(run1, run2)
    assert not np.array_equal(run1[gap_start:gap_end], data[gap_start:gap_end])


def test_spectral_pipeline_reproducibility():
    """Verify that run_spectral_pipeline is 100% deterministic."""
    fs = 1.0
    t = np.linspace(0, 1000, 1000)
    s1 = np.sin(2 * np.pi * 0.02 * t) + 0.1 * np.random.randn(1000)
    s2 = np.sin(2 * np.pi * 0.02 * t + 0.5) + 0.1 * np.random.randn(1000)

    pm_signals = {
        "20 MHz Pol A": s1,
        "20 MHz Pol B": s1,
        "25 MHz Pol A": s2,
        "25 MHz Pol B": s2,
    }

    res1 = run_spectral_pipeline(pm_signals, fs, 0.01, 0.1, 15, 3.0, True)
    res2 = run_spectral_pipeline(pm_signals, fs, 0.01, 0.1, 15, 3.0, True)

    np.testing.assert_array_equal(res1["freqs"], res2["freqs"])
    np.testing.assert_array_equal(res1["cross"]["Pol A"]["power"], res2["cross"]["Pol A"]["power"])
