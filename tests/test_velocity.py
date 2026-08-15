import numpy as np
import pytest

from core.spectral_analysis import compute_cross_spectrum, estimate_velocities, find_spectral_peaks


@pytest.mark.parametrize(
    "target_velocity, dt_sign",
    [
        (500.0, 1.0),
        (1000.0, 1.0),
        (1500.0, 1.0),
        (2500.0, 1.0),
        (1000.0, -1.0),
        (1666.67, -1.0),
    ],
)
def test_velocity_estimation_speed_scaling(target_velocity: float, dt_sign: float) -> None:
    """Test drift velocity estimation across scaling speeds and positive/negative delays."""
    fs = 1.0  # Hz
    t = np.arange(1000)
    f_target = 0.05  # Hz
    dx = 2500.0  # meters baseline

    # Compute expected delay dt = dt_sign * (dx / target_velocity)
    dt_true = dt_sign * (dx / target_velocity)

    sig20 = np.sin(2 * np.pi * f_target * t)
    sig25 = np.sin(2 * np.pi * f_target * (t - dt_true))

    freqs, power, phase, coherence, _, _ = compute_cross_spectrum(sig20, sig25, fs, n_tapers=7, nw=4.0)
    peaks = find_spectral_peaks(power, freqs, 0.01, 0.1, n_peaks=1)

    assert len(peaks) > 0
    results = estimate_velocities(phase, freqs, peaks, dx=dx)
    assert len(results) == 1

    est = results[0]

    # Verify delay accuracy within 1% relative error
    assert est["dt"] == pytest.approx(dt_true, rel=0.01)

    # Verify physical velocity accuracy: dx / |dt|
    expected_velocity = dx / abs(dt_true)
    assert abs(est["velocity"]) == pytest.approx(expected_velocity, rel=0.01)


@pytest.mark.parametrize("f_target", [0.02, 0.05, 0.08])
def test_velocity_estimation_frequency_independence(f_target: float) -> None:
    """Test that drift velocity estimation is independent of target harmonic frequency in scintillation band."""
    fs = 1.0
    t = np.arange(1000)
    dt_true = 2.0  # seconds
    dx = 2500.0  # meters

    sig20 = np.sin(2 * np.pi * f_target * t)
    sig25 = np.sin(2 * np.pi * f_target * (t - dt_true))

    freqs, power, phase, _, _, _ = compute_cross_spectrum(sig20, sig25, fs, n_tapers=7, nw=4.0)
    peaks = find_spectral_peaks(power, freqs, 0.005, 0.15, n_peaks=1)

    assert len(peaks) > 0
    results = estimate_velocities(phase, freqs, peaks, dx=dx)
    assert len(results) == 1

    assert results[0]["dt"] == pytest.approx(dt_true, rel=0.02)
    assert results[0]["velocity"] == pytest.approx(dx / dt_true, rel=0.02)


def test_velocity_estimation_theoretical_confidence_intervals() -> None:
    """Verify that theoretical phase variance produces narrower CIs for high coherence than low coherence."""
    freqs = np.linspace(0.001, 0.5, 500)
    idx = 50  # peak at ~0.05 Hz
    peak_freq = freqs[idx]
    period = 1.0 / peak_freq

    # Phase corresponding to 2s delay
    dt_target = 2.0
    phase_deg = np.zeros_like(freqs)
    phase_deg[idx] = (dt_target / period) * 360.0

    # High coherence
    coh_high = np.full_like(freqs, 0.95)
    res_high = estimate_velocities(phase_deg, freqs, [idx], coherence=coh_high, dx=2500.0)

    # Low coherence
    coh_low = np.full_like(freqs, 0.35)
    res_low = estimate_velocities(phase_deg, freqs, [idx], coherence=coh_low, dx=2500.0)

    assert len(res_high) == 1 and len(res_low) == 1
    ci_high_width = res_high[0]["dt_ci95"][1] - res_high[0]["dt_ci95"][0]
    ci_low_width = res_low[0]["dt_ci95"][1] - res_low[0]["dt_ci95"][0]

    # High coherence must have tighter (narrower) confidence intervals than low coherence
    assert ci_high_width < ci_low_width
    assert ci_high_width > 0.0
