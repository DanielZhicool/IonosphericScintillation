import numpy as np
import pytest

from core.signal_processing import (
    clean_and_smooth_signal,
    fill_gap_with_red_noise,
)
from core.spectral_analysis import run_spectral_pipeline
from core.synthetic_generator import (
    apply_delay,
    generate_power_law_noise,
    generate_synthetic_scintillation,
)


def test_generate_power_law_noise() -> None:
    """Test that generated power-law noise has correct shape, zero mean, and unit variance."""
    length = 1000
    fs = 10.0
    noise = generate_power_law_noise(length, fs, seed=42)

    assert len(noise) == length
    assert np.mean(noise) == pytest.approx(0.0, abs=1e-12)
    assert np.std(noise) == pytest.approx(1.0, abs=1e-12)


def test_apply_delay() -> None:
    """Test that applying a delay in the frequency domain shifts the signal correctly."""
    fs = 10.0  # Hz
    t = np.arange(1000) / fs
    f_sine = 0.5  # Hz
    # Pure sine wave
    sig = np.sin(2 * np.pi * f_sine * t)

    delay_sec = 2.3
    delayed = apply_delay(sig, delay_sec, fs)

    # Expected shifted sine wave
    expected = np.sin(2 * np.pi * f_sine * (t - delay_sec))

    # Avoid boundary effects of circular convolution by checking center region
    start, end = 200, 800
    rmse = np.sqrt(np.mean((delayed[start:end] - expected[start:end]) ** 2))
    assert rmse < 0.05


def test_generate_synthetic_scintillation_stats() -> None:
    """Test that synthetic scintillation outputs have correct lengths, stats, and coherence."""
    length = 5000
    fs = 1.0
    coherence = 0.8
    sig1, sig2 = generate_synthetic_scintillation(
        length=length,
        fs=fs,
        coherence=coherence,
        delay_sec=0.0,
        white_noise_std=0.0,
        seed=123,
    )

    assert len(sig1) == length
    assert len(sig2) == length
    assert np.mean(sig1) == pytest.approx(0.0, abs=0.1)
    assert np.std(sig1) == pytest.approx(1.0, abs=0.1)

    # Pearson correlation coefficient should approximate the target coherence
    corr = np.corrcoef(sig1, sig2)[0, 1]
    assert corr == pytest.approx(coherence, abs=0.08)


def test_end_to_end_pipeline_validation() -> None:
    """
    Validate the entire processing pipeline using synthetic data.

    Generates synthetic signals with:
    - Injected delay = 1.5 seconds
    - Fundamental period = 25 seconds (frequency = 0.04 Hz)
    - Injected outliers (spikes)
    - Injected gaps (calibration block)

    Verifies:
    1. Hampel filter outlier removal.
    2. Red noise gap filling.
    3. Bandpass filtering.
    4. Cross-spectral delay & drift velocity estimation.
    """
    length = 2000
    fs = 1.0  # 1 Hz sampling rate
    dx = 2500.0  # 2500 meters separation
    true_delay = 1.5  # seconds
    expected_velocity = dx / true_delay  # 1666.67 m/s
    true_period = 25.0  # seconds

    # Define a gap from index 800 to 950
    gap_start, gap_end = 800, 950

    # 1. Generate baseline clean scintillation signals
    # We add a strong periodic component (harmonic) to easily locate it in cross-spectrum
    sig1_clean, sig2_clean = generate_synthetic_scintillation(
        length=length,
        fs=fs,
        f_fresnel=0.1,
        spectral_index=8.0 / 3.0,
        harmonics=[(true_period, 2.5)],  # strong period
        coherence=0.9,
        delay_sec=true_delay,
        white_noise_std=0.05,
        seed=42,
    )

    # Save copies of the clean signals for reference comparison
    sig1_ref = sig1_clean.copy()

    # 2. Inject outlier spikes into Channel 1
    # Replace index 500 with a large spike
    sig1_noisy = sig1_clean.copy()
    sig1_noisy[500] = 50.0

    # 3. Inject gaps
    sig1_noisy[gap_start:gap_end] = 0.0
    sig2_noisy = sig2_clean.copy()
    sig2_noisy[gap_start:gap_end] = 0.0

    # --- Start Pipeline Execution ---

    # Step A: Hampel filter outlier removal on noisy signal
    # We clean the signal with a window size of 15, and 3 sigma limit
    cleaned_sig1 = clean_and_smooth_signal(sig1_noisy, window_size=15, n_sigmas=3.0, apply_smoothing=False)
    # Check that the outlier at 500 was successfully suppressed (replaced by rolling median)
    assert abs(cleaned_sig1[500] - sig1_ref[500]) < 1.0

    # Step B: Gap filling with red noise
    filled_sig1 = fill_gap_with_red_noise(cleaned_sig1, start_idx=gap_start, end_idx=gap_end, seed=99)
    filled_sig2 = fill_gap_with_red_noise(sig2_noisy, start_idx=gap_start, end_idx=gap_end, seed=100)

    # Check that the gap is filled (no longer all zeros) and finite
    assert not np.all(filled_sig1[gap_start:gap_end] == 0.0)
    assert np.all(np.isfinite(filled_sig1))
    assert np.all(np.isfinite(filled_sig2))

    # Step C: Prepare dictionary structure to feed to the spectral pipeline
    pm_signals = {
        "20 MHz Pol A": filled_sig1,
        "25 MHz Pol A": filled_sig2,
        "20 MHz Pol B": filled_sig1,  # Duplicate for pipeline format
        "25 MHz Pol B": filled_sig2,
    }

    # Step D: Run spectral-correlation pipeline
    # The bandpass range will be 0.01 to 0.1 Hz, which covers our 0.04 Hz target frequency
    lowcut, highcut = 0.01, 0.1
    pipeline_results = run_spectral_pipeline(
        pm_signals=pm_signals,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=False,
    )

    # Step E: Validate cross-spectral peak detection
    velocities_a = pipeline_results["velocities"]["Pol A"]

    assert len(velocities_a) > 0
    # The primary peak (highest power/coherence) should represent the injected harmonic
    best_est = velocities_a[0]

    # Assert estimated period is close to true period
    assert best_est["period"] == pytest.approx(true_period, abs=2.5)

    # Assert delay dt matches injected delay
    assert best_est["dt"] == pytest.approx(true_delay, abs=0.2)

    # Assert estimated velocity matches expected physical velocity
    assert best_est["velocity"] == pytest.approx(expected_velocity, abs=250.0)
