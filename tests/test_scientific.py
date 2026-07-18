import numpy as np
import pytest

from core.signal_processing import bandpass_filter, clean_and_smooth_signal, fill_gap_with_red_noise
from core.spectral_analysis import (
    compute_cross_spectrum,
    compute_multitaper_psd,
    estimate_velocities,
    find_spectral_peaks,
)


def test_clean_and_smooth_signal_outliers() -> None:
    """Test that Hampel outlier detection successfully removes large spike noise."""
    t = np.linspace(0, 10, 100)
    # A clean sine wave
    clean_signal = np.sin(t)

    # Inject a large single-sample spike outlier
    noisy_signal = clean_signal.copy()
    noisy_signal[50] = 50.0

    # Clean the signal without applying Savitzky-Golay smoothing first
    cleaned = clean_and_smooth_signal(noisy_signal, window_size=7, n_sigmas=3.0, apply_smoothing=False)

    # Verify that the spike at index 50 has been replaced by something close to the true value
    assert abs(cleaned[50] - clean_signal[50]) < 0.5

    # Verify that non-outlier samples remain mostly unchanged
    assert cleaned[10] == pytest.approx(clean_signal[10])


def test_clean_and_smooth_signal_smoothing() -> None:
    """Test that Savitzky-Golay smoothing reduces noise variance."""
    t = np.linspace(0, 10, 200)
    clean_signal = np.sin(t)

    # Add small Gaussian noise using local generator
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, size=len(clean_signal))
    noisy_signal = clean_signal + noise

    # Clean and apply Savitzky-Golay smoothing
    cleaned = clean_and_smooth_signal(noisy_signal, window_size=15, n_sigmas=4.0, apply_smoothing=True, polyorder=2)

    # The cleaned/smoothed signal should have a lower root-mean-square error to the true signal
    # compared to the raw noisy signal
    rmse_noisy = np.sqrt(np.mean((noisy_signal - clean_signal) ** 2))
    rmse_cleaned = np.sqrt(np.mean((cleaned - clean_signal) ** 2))

    assert rmse_cleaned < rmse_noisy
    assert rmse_cleaned < 0.08  # significantly reduced noise


def test_bandpass_filter() -> None:
    """Test that bandpass filter isolates target frequency component and attenuates others in the frequency domain."""
    fs = 100.0  # Hz
    t = np.arange(1000) / fs

    # Signal composed of a low frequency (0.2 Hz) and high frequency (5.0 Hz) sine wave
    f_low = 0.2
    f_high = 5.0
    signal = np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t)

    # Filter to isolate the high frequency (5.0 Hz), suppressing the low frequency (0.2 Hz)
    # We set passband from 3.0 Hz to 7.0 Hz
    filtered = bandpass_filter(signal, lowcut=3.0, highcut=7.0, fs=fs, order=4)

    # Validate in the frequency domain using FFT
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    fft_val = np.abs(np.fft.rfft(filtered))

    idx_low = np.argmin(np.abs(freqs - f_low))
    idx_high = np.argmin(np.abs(freqs - f_high))

    # Verify that target frequency (5 Hz) is strong and unwanted frequency (0.2 Hz) is heavily attenuated
    assert fft_val[idx_high] > 10.0  # Dominant component should be strong
    assert fft_val[idx_low] / fft_val[idx_high] < 0.02  # Strongly attenuated (at least -34 dB)

    # Verify peak frequency in FFT is near 5.0 Hz
    peak_freq = freqs[np.argmax(fft_val)]
    assert peak_freq == pytest.approx(f_high, abs=0.2)


def test_fill_gap_with_red_noise() -> None:
    """Test gap filling interpolates realistic noise without leaving zeros or steps, and verifies reproducibility."""
    rng = np.random.default_rng(123)
    target_mean = 10.0
    target_std = 2.0
    signal = rng.normal(target_mean, target_std, 1000)

    # Inject a gap of zero values from index 400 to 600
    signal[400:600] = 0.0

    # Fill the gap
    filled = fill_gap_with_red_noise(signal, start_idx=400, end_idx=600, context_window=150, seed=42)

    # Assert that all filled values are finite
    assert np.all(np.isfinite(filled))

    # Assert that the filled portion is no longer all zero
    assert not np.all(filled[400:600] == 0.0)

    # Assert that the statistics of the filled portion are reasonably close to the surrounding context
    filled_portion = filled[400:600]
    assert np.mean(filled_portion) == pytest.approx(target_mean, abs=1.0)
    assert np.std(filled_portion) == pytest.approx(target_std, abs=1.0)

    # Verify no massive discontinuities at the boundaries
    assert abs(filled[400] - filled[399]) < 3.5 * target_std
    assert abs(filled[600] - filled[599]) < 3.5 * target_std

    # Verify reproducibility with the same seed
    filled_first = fill_gap_with_red_noise(signal, start_idx=400, end_idx=600, context_window=150, seed=42)
    filled_second = fill_gap_with_red_noise(signal, start_idx=400, end_idx=600, context_window=150, seed=42)
    np.testing.assert_array_equal(filled_first, filled_second)


def test_compute_multitaper_psd() -> None:
    """Test that multitaper PSD accurately detects the dominant frequency and peak prominence."""
    fs = 50.0  # Hz
    t = np.arange(500) / fs
    f_sine = 6.5  # Hz
    # Pure sine wave at 6.5 Hz
    signal = np.sin(2 * np.pi * f_sine * t)

    # Compute multitaper PSD (NW=4, 7 tapers)
    freqs, psd = compute_multitaper_psd(signal, fs, n_tapers=7, nw=4.0)

    # Find the frequency with the maximum power spectral density
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]

    # The peak frequency should be close to 6.5 Hz
    assert peak_freq == pytest.approx(f_sine, abs=0.2)

    # Verify that the peak stands clearly above the background (frequencies away from the peak)
    # Frequencies away from the peak are defined as |f - 6.5| > 1.5 Hz
    background_mask = np.abs(freqs - f_sine) > 1.5
    background_mean_power = np.mean(psd[background_mask])
    peak_power = psd[peak_idx]

    # The peak power should be at least 100 times larger than the average background power
    assert peak_power > 100 * background_mean_power


def test_velocity_estimation_scientific_validation() -> None:
    """Test velocity estimation with known synthetic delays to validate physical correctness."""
    fs = 1.0  # Hz
    t = np.arange(1000)
    f_target = 0.05  # Hz
    dx = 2500.0  # meters

    # Case 1: Positive delay (2.0s)
    dt_true_pos = 2.0
    sig20_pos = np.sin(2 * np.pi * f_target * t)
    sig25_pos = np.sin(2 * np.pi * f_target * (t - dt_true_pos))

    freqs, power, phase, coherence, real, imag = compute_cross_spectrum(sig20_pos, sig25_pos, fs, n_tapers=7, nw=4.0)
    peaks_pos = find_spectral_peaks(power, freqs, 0.01, 0.1, n_peaks=1)

    assert len(peaks_pos) > 0
    results_pos = estimate_velocities(phase, freqs, peaks_pos, dx=dx)
    assert len(results_pos) == 1

    # Verify recovered delay is within 1% of true delay (2.0s)
    assert results_pos[0]["dt"] == pytest.approx(dt_true_pos, rel=0.01)
    # Verify recovered velocity is physically correct: dx / dt
    expected_vel_pos = dx / dt_true_pos
    assert results_pos[0]["velocity"] == pytest.approx(expected_vel_pos, rel=0.01)

    # Case 2: Negative delay (-1.5s)
    dt_true_neg = -1.5
    sig20_neg = np.sin(2 * np.pi * f_target * t)
    sig25_neg = np.sin(2 * np.pi * f_target * (t - dt_true_neg))

    _, _, phase_neg, _, _, _ = compute_cross_spectrum(sig20_neg, sig25_neg, fs, n_tapers=7, nw=4.0)
    peaks_neg = find_spectral_peaks(power, freqs, 0.01, 0.1, n_peaks=1)

    assert len(peaks_neg) > 0
    results_neg = estimate_velocities(phase_neg, freqs, peaks_neg, dx=dx)
    assert len(results_neg) == 1

    # Verify recovered delay is within 1% of true delay (-1.5s)
    assert results_neg[0]["dt"] == pytest.approx(dt_true_neg, rel=0.01)
    # Verify recovered velocity is physically correct: dx / |dt|
    expected_vel_neg = dx / abs(dt_true_neg)
    assert results_neg[0]["velocity"] == pytest.approx(expected_vel_neg, rel=0.01)


def test_edge_case_empty_signals() -> None:
    """Verify and document behavior for empty arrays.

    Hampel outlier detection/Savitzky-Golay smoothing and bandpass filtering raise ValueError,
    while Multitaper PSD handles empty inputs gracefully by returning empty arrays.
    """
    with pytest.raises(
        ValueError, match="window_length must be less than or equal to the size of x|must be.*less than"
    ):
        clean_and_smooth_signal(np.array([]))

    with pytest.raises(ValueError, match="cannot reshape array of size 0 into shape"):
        bandpass_filter(np.array([]), lowcut=1.0, highcut=5.0, fs=10.0)

    # MTM PSD on empty signal returns zero arrays gracefully
    freqs, psd = compute_multitaper_psd(np.array([]), fs=10.0)
    assert len(freqs) == 1
    assert freqs[0] == 0.0
    assert psd[0] == 0.0


def test_edge_case_constant_signals() -> None:
    """Verify and document behavior for constant signals.

    Bandpass filter detrends the signal (subtracting baseline trend/DC offset), returning all zeros.
    Hampel cleaning preserves constant signals without modification as there are no outliers.
    """
    const_signal = np.ones(100) * 5.0

    # Bandpass filter removes the linear trend/DC offset, resulting in a zero signal
    filtered_const = bandpass_filter(const_signal, lowcut=1.0, highcut=4.0, fs=10.0)
    assert np.allclose(filtered_const, 0.0)

    # Hampel cleaning should preserve constant signal
    cleaned_const = clean_and_smooth_signal(const_signal, window_size=5, apply_smoothing=False)
    np.testing.assert_array_equal(cleaned_const, const_signal)


def test_edge_case_nan_signals() -> None:
    """Verify and document behavior for signals containing NaN values.

    Hampel cleaning handles and replaces NaNs using pandas bfill/ffill propagation.
    Bandpass filtering raises ValueError because underlying SciPy filtering routines require finite data.
    """
    nan_signal = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])

    # Hampel cleaning recovers NaNs via backfill/forwardfill
    cleaned_nan = clean_and_smooth_signal(nan_signal, window_size=5, apply_smoothing=True)
    assert np.all(np.isfinite(cleaned_nan))
    assert len(cleaned_nan) == len(nan_signal)

    # Bandpass filtering does not support NaNs and raises ValueError
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        bandpass_filter(nan_signal, lowcut=1.0, highcut=3.0, fs=10.0)


def test_edge_case_short_signals() -> None:
    """Verify and document behavior for very short signals.

    Hampel cleaning and Savitzky-Golay filters raise ValueError if window length exceeds the signal size.
    Bandpass filter raises ValueError if the signal length is smaller than the required pad length.
    """
    short_signal = np.ones(5)

    with pytest.raises(ValueError, match="window_length must be less than or equal to the size of x"):
        clean_and_smooth_signal(short_signal, window_size=15)

    with pytest.raises(ValueError, match="length of the input vector x must be greater than padlen"):
        bandpass_filter(short_signal, lowcut=1.0, highcut=3.0, fs=10.0)


def test_edge_case_invalid_filter_parameters() -> None:
    """Verify and document behavior for invalid bandpass filter parameters.

    Verify that ValueError is raised for:
      - lowcut <= 0
      - highcut >= Nyquist (fs / 2)
      - lowcut >= highcut
    """
    long_signal = np.ones(100)

    # 1. lowcut <= 0
    with pytest.raises(ValueError, match="filter critical frequencies must be greater than 0"):
        bandpass_filter(long_signal, lowcut=-1.0, highcut=4.0, fs=10.0)

    # 2. highcut >= Nyquist
    with pytest.raises(
        ValueError, match="critical frequencies must be 0 < Wn < 1|frequencies must be.*between 0 and 1"
    ):
        bandpass_filter(long_signal, lowcut=1.0, highcut=6.0, fs=10.0)  # Nyquist is 5.0 Hz

    # 3. lowcut >= highcut
    with pytest.raises(ValueError, match="Wn\\[0\\] must be less than Wn\\[1\\]"):
        bandpass_filter(long_signal, lowcut=4.0, highcut=2.0, fs=10.0)


def test_preprocessing_round_trip_reproducibility() -> None:
    """Verify that preprocessing (Hampel outlier removal, smoothing, and bandpass) preserves signal frequency."""
    fs = 1.0  # Hz
    t = np.arange(1000)
    f_target = 0.05  # Hz

    rng = np.random.default_rng(999)
    clean_signal = np.sin(2 * np.pi * f_target * t)
    # Add noise and large outliers
    noisy_signal = clean_signal + rng.normal(0, 0.2, size=len(t))
    noisy_signal[150] = 40.0
    noisy_signal[850] = -40.0

    # 1. Clean outliers and apply smoothing
    cleaned = clean_and_smooth_signal(noisy_signal, window_size=15, n_sigmas=3.0, apply_smoothing=True)

    # 2. Apply bandpass filter
    filtered = bandpass_filter(cleaned, lowcut=0.01, highcut=0.1, fs=fs)

    # 3. Compute Multitaper PSD
    freqs, psd = compute_multitaper_psd(filtered, fs=fs, n_tapers=7, nw=4.0)

    # Verify that the peak frequency recovered matches the target frequency
    peak_freq = freqs[np.argmax(psd)]
    assert peak_freq == pytest.approx(f_target, abs=0.005)
