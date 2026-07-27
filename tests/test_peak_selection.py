"""
Unit tests for peak selection and IDVE velocity table formatting logic.
"""

import numpy as np
from scipy.signal import find_peaks

from core.spectral_analysis import format_velocity_table


def test_distinct_peak_selection_no_clustering():
    """Verify that find_peaks selects distinct physical local maxima rather than adjacent bin clusters."""
    # Synthetic spectrum with two distinct peaks at idx 100 (period 100s) and idx 300 (period 33s)
    n_pts = 1000
    vals_pow = np.ones(n_pts) * 10.0
    vals_pow[100] = 500.0
    vals_pow[99] = 490.0
    vals_pow[101] = 495.0

    vals_pow[300] = 400.0
    vals_pow[299] = 380.0
    vals_pow[301] = 385.0

    dist = max(1, len(vals_pow) // 50)
    peaks, _ = find_peaks(vals_pow, distance=dist, prominence=0.01 * np.max(vals_pow))
    top_indices = sorted(peaks, key=lambda i: vals_pow[i], reverse=True)[:2]

    # Should select idx 100 and idx 300, not idx 100 and idx 101
    assert 100 in top_indices
    assert 300 in top_indices
    assert 101 not in top_indices


def test_velocity_table_formatting_in_phase():
    """Verify velocity table formatting displays in-phase / fast drift tags cleanly."""
    mock_velocities = {
        "Pol A": [
            {
                "period": 10.0,
                "phase_deg": 0.5,
                "dt": 0.014,
                "velocity": 178571.4,
                "mean_coherence": 0.98,
                "is_valid": True,
                "gating_reason": None,
            },
            {
                "period": 35.3,
                "phase_deg": -16.9,
                "dt": -1.66,
                "velocity": -1504.3,
                "mean_coherence": 0.25,
                "is_valid": False,
                "gating_reason": "Mean coherence (0.25) below threshold (0.70)",
            },
        ]
    }

    formatted_txt = format_velocity_table(mock_velocities, "Small (5-150 s)")

    assert "[Pol A (20 MHz vs 25 MHz)]" in formatted_txt
    assert "Peak 1: Period =  10.0 s" in formatted_txt
    assert ">10,000 m/s (in-phase)" in formatted_txt
    assert "Coh = 0.98" in formatted_txt
    assert "(low coh)" in formatted_txt
