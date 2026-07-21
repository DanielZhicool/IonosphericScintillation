"""
Unit tests for statistical precision & advanced phase regression features.
"""

import numpy as np
import pytest

from core.spectral_analysis import (
    compute_ftest,
    compute_multitaper_psd,
    estimate_velocities,
)
from core.types import MultitaperPSDResult, VelocityEstimate


@pytest.mark.parametrize("n_tapers", [5, 7, 9])
def test_multitaper_jackknife_ci(n_tapers: int) -> None:
    """Verify Thomson Multitaper PSD computes valid Jackknife 95% confidence intervals across tapers."""
    fs = 100.0  # Hz
    t = np.arange(1000) / fs
    f_sine = 10.0  # Hz

    rng = np.random.default_rng(42)
    signal = np.sin(2 * np.pi * f_sine * t) + rng.normal(0, 0.2, len(t))

    res = compute_multitaper_psd(signal, fs=fs, n_tapers=n_tapers, nw=4.0, compute_ci=True)

    assert isinstance(res, MultitaperPSDResult)
    assert res.log_psd_se is not None
    assert res.ci95_low is not None
    assert res.ci95_high is not None
    assert len(res.psd) == len(res.ci95_low) == len(res.ci95_high)

    # Low CI < PSD < High CI everywhere (except edge numerical zeros)
    valid_mask = res.psd > 1e-12
    assert np.all(res.ci95_low[valid_mask] <= res.psd[valid_mask] + 1e-12)
    assert np.all(res.ci95_high[valid_mask] >= res.psd[valid_mask] - 1e-12)

    # Verify tuple unpacking compatibility
    freqs_unpacked, psd_unpacked = res
    np.testing.assert_array_equal(freqs_unpacked, res.freqs)
    np.testing.assert_array_equal(psd_unpacked, res.psd)


@pytest.mark.parametrize("fdr_alpha", [0.01, 0.05, 0.10])
def test_fdr_multiple_testing_threshold(fdr_alpha: float) -> None:
    """Verify Thomson F-test applies Benjamini-Hochberg FDR control across significance levels."""
    fs = 50.0  # Hz
    t = np.arange(500) / fs
    rng = np.random.default_rng(123)

    # Pure noise signal (no deterministic line)
    noise_signal = rng.normal(0, 1.0, len(t))

    freqs, f_stat, threshold, T0 = compute_ftest(
        noise_signal, fs=fs, n_tapers=7, nw=4.0, confidence=0.95, fdr_alpha=fdr_alpha
    )

    assert threshold > 0.0
    assert len(f_stat) == len(freqs)


@pytest.mark.parametrize("dt_true", [1.5, 2.0, -2.5])
def test_phase_regression_velocity_estimation(dt_true: float) -> None:
    """Verify weighted phase regression estimates delay, velocity, and CIs accurately across delays."""
    freqs = np.linspace(0.01, 1.0, 100)

    dx = 2500.0  # m

    # Linear phase spectrum: phi(f) = 2*pi * f * dt_true (in degrees)
    phase_rad = 2 * np.pi * freqs * dt_true
    cross_phase_deg = np.degrees(phase_rad)
    coherence = np.ones_like(freqs) * 0.95

    peak_indices = np.array([40])  # f ~ 0.41 Hz

    estimates = estimate_velocities(
        cross_phase_deg,
        freqs,
        peak_indices,
        coherence=coherence,
        dx=dx,
        min_coherence=0.7,
        bandwidth_hz=0.05,
        enable_phase_regression=True,
    )

    assert len(estimates) == 1
    v_est = estimates[0]

    assert isinstance(v_est, VelocityEstimate)
    assert v_est.is_valid is True
    assert v_est.gating_reason is None
    assert v_est.mean_coherence >= 0.7

    # Delay tau should match true delay
    assert v_est.dt == pytest.approx(dt_true, rel=0.01)
    # Velocity should match dx / dt
    expected_v = dx / dt_true
    assert v_est.velocity == pytest.approx(expected_v, rel=0.01)

    # 95% CIs should contain true delay and velocity
    lower_dt, upper_dt = min(v_est.dt_ci95), max(v_est.dt_ci95)
    assert lower_dt <= dt_true <= upper_dt

    lower_v, upper_v = min(v_est.velocity_ci95), max(v_est.velocity_ci95)
    assert lower_v <= expected_v <= upper_v

    # Backward compatibility dict indexing
    assert v_est["period"] == pytest.approx(v_est.period)
    assert v_est["velocity"] == pytest.approx(v_est.velocity)
    assert v_est["dt"] == pytest.approx(v_est.dt)


@pytest.mark.parametrize("low_coherence", [0.1, 0.3, 0.5])
def test_coherence_gating(low_coherence: float) -> None:
    """Verify velocity estimation is suppressed (is_valid=False) when mean coherence
    falls below min_coherence threshold.
    """
    freqs = np.linspace(0.01, 0.5, 50)

    # Linear phase corresponding to true delay dt = 2.0s
    cross_phase_deg = np.degrees(2 * np.pi * freqs * 2.0)
    coherence = np.ones_like(freqs) * low_coherence
    peak_indices = np.array([10, 20])

    estimates = estimate_velocities(
        cross_phase_deg, freqs, peak_indices, coherence=coherence, dx=2500.0, min_coherence=0.7, bandwidth_hz=0.02
    )

    for v_est in estimates:
        assert isinstance(v_est, VelocityEstimate)
        assert v_est.is_valid is False
        assert v_est.gating_reason is not None
        assert "below threshold" in v_est.gating_reason
