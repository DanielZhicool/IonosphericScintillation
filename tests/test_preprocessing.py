import numpy as np
import pytest

from core.signal_processing import bandpass_filter, clean_and_smooth_signal, fill_gap_with_red_noise


@pytest.mark.parametrize(
    "spike_idx, spike_amplitude, window_size",
    [
        (50, 50.0, 7),
        (25, -40.0, 9),
        (80, 100.0, 15),
    ],
)
def test_clean_and_smooth_signal_outliers(spike_idx: int, spike_amplitude: float, window_size: int) -> None:
    """Test that Hampel outlier detection successfully removes spikes across different locations and sizes."""
    t = np.linspace(0, 10, 100)
    clean_signal = np.sin(t)

    noisy_signal = clean_signal.copy()
    noisy_signal[spike_idx] = spike_amplitude

    cleaned = clean_and_smooth_signal(noisy_signal, window_size=window_size, n_sigmas=3.0, apply_smoothing=False)

    # Verify that the spike has been replaced near the true value
    assert abs(cleaned[spike_idx] - clean_signal[spike_idx]) < 0.5
    # Verify non-outlier sample integrity
    safe_idx = 10 if spike_idx != 10 else 20
    assert cleaned[safe_idx] == pytest.approx(clean_signal[safe_idx])


@pytest.mark.parametrize(
    "noise_std, polyorder",
    [
        (0.05, 2),
        (0.10, 2),
        (0.15, 3),
    ],
)
def test_clean_and_smooth_signal_smoothing(noise_std: float, polyorder: int) -> None:
    """Test that Savitzky-Golay smoothing reduces noise variance for various noise levels."""
    t = np.linspace(0, 10, 200)
    clean_signal = np.sin(t)

    rng = np.random.default_rng(42)
    noisy_signal = clean_signal + rng.normal(0, noise_std, size=len(clean_signal))

    cleaned = clean_and_smooth_signal(
        noisy_signal, window_size=15, n_sigmas=4.0, apply_smoothing=True, polyorder=polyorder
    )

    rmse_noisy = np.sqrt(np.mean((noisy_signal - clean_signal) ** 2))
    rmse_cleaned = np.sqrt(np.mean((cleaned - clean_signal) ** 2))

    assert rmse_cleaned < rmse_noisy


@pytest.mark.parametrize(
    "f_target, f_unwanted, lowcut, highcut, fs",
    [
        (0.2, 5.0, 0.05, 0.5, 50.0),
        (5.0, 0.2, 3.0, 7.0, 100.0),
        (15.0, 2.0, 10.0, 20.0, 100.0),
    ],
)
def test_bandpass_filter(f_target: float, f_unwanted: float, lowcut: float, highcut: float, fs: float) -> None:
    """Test that bandpass filter isolates target frequency component and attenuates others in the frequency domain."""
    t = np.arange(1000) / fs
    signal = np.sin(2 * np.pi * f_target * t) + np.sin(2 * np.pi * f_unwanted * t)

    filtered = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs, order=4)

    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    fft_val = np.abs(np.fft.rfft(filtered))

    idx_target = np.argmin(np.abs(freqs - f_target))
    idx_unwanted = np.argmin(np.abs(freqs - f_unwanted))

    assert fft_val[idx_target] > 10.0
    assert fft_val[idx_unwanted] / fft_val[idx_target] < 0.05

    peak_freq = freqs[np.argmax(fft_val)]
    assert peak_freq == pytest.approx(f_target, abs=0.3)


@pytest.mark.parametrize(
    "gap_start, gap_end, context_window",
    [
        (400, 600, 150),
        (100, 200, 80),
        (700, 850, 200),
    ],
)
def test_fill_gap_with_red_noise(gap_start: int, gap_end: int, context_window: int) -> None:
    """Test gap filling interpolates realistic noise without leaving zeros or steps, and verifies reproducibility."""
    rng = np.random.default_rng(123)
    target_mean = 10.0
    target_std = 2.0
    signal = rng.normal(target_mean, target_std, 1000)

    signal[gap_start:gap_end] = 0.0

    filled = fill_gap_with_red_noise(
        signal, start_idx=gap_start, end_idx=gap_end, context_window=context_window, seed=42
    )

    assert np.all(np.isfinite(filled))
    assert not np.all(filled[gap_start:gap_end] == 0.0)

    filled_portion = filled[gap_start:gap_end]
    assert np.mean(filled_portion) == pytest.approx(target_mean, abs=1.0)
    assert np.std(filled_portion) == pytest.approx(target_std, abs=1.0)

    # Verify reproducibility with the same seed
    filled_second = fill_gap_with_red_noise(
        signal, start_idx=gap_start, end_idx=gap_end, context_window=context_window, seed=42
    )
    np.testing.assert_array_equal(filled, filled_second)


def test_edge_case_empty_signals() -> None:
    """Verify behavior for empty arrays."""
    with pytest.raises(ValueError, match="empty|window_length must be less than or equal to the size of x"):
        clean_and_smooth_signal(np.array([]))

    with pytest.raises(ValueError, match="empty|cannot reshape array of size 0 into shape"):
        bandpass_filter(np.array([]), lowcut=1.0, highcut=5.0, fs=10.0)


def test_edge_case_constant_signals() -> None:
    """Verify behavior for constant signals."""
    const_signal = np.ones(100) * 5.0

    filtered_const = bandpass_filter(const_signal, lowcut=1.0, highcut=4.0, fs=10.0)
    assert np.allclose(filtered_const, 0.0)

    cleaned_const = clean_and_smooth_signal(const_signal, window_size=5, apply_smoothing=False)
    np.testing.assert_array_equal(cleaned_const, const_signal)


def test_edge_case_nan_signals() -> None:
    """Verify behavior for signals containing NaN values."""
    nan_signal = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])

    with pytest.raises(ValueError, match="non-finite"):
        clean_and_smooth_signal(nan_signal, window_size=5, apply_smoothing=True)

    with pytest.raises(ValueError, match="non-finite"):
        bandpass_filter(nan_signal, lowcut=1.0, highcut=3.0, fs=10.0)


def test_edge_case_short_signals() -> None:
    """Verify behavior for very short signals."""
    short_signal = np.ones(5)

    with pytest.raises(ValueError, match="window_size.*must be less than or equal to signal length|window_length"):
        clean_and_smooth_signal(short_signal, window_size=15)

    with pytest.raises(ValueError, match="length of the input vector x must be greater than padlen"):
        bandpass_filter(short_signal, lowcut=1.0, highcut=3.0, fs=10.0)


def test_edge_case_invalid_filter_parameters() -> None:
    """Verify behavior for invalid bandpass filter parameters."""
    long_signal = np.ones(100)

    with pytest.raises(ValueError, match="lowcut must be > 0|filter critical frequencies must be greater than 0"):
        bandpass_filter(long_signal, lowcut=-1.0, highcut=4.0, fs=10.0)

    with pytest.raises(
        ValueError, match="strictly less than Nyquist|critical frequencies must be 0 < Wn < 1|frequencies must be"
    ):
        bandpass_filter(long_signal, lowcut=1.0, highcut=6.0, fs=10.0)

    with pytest.raises(ValueError, match="must be greater than lowcut|Wn\\[0\\] must be less than Wn\\[1\\]"):
        bandpass_filter(long_signal, lowcut=4.0, highcut=2.0, fs=10.0)


def test_processing_config_overrides() -> None:
    """Verify that ProcessingConfig container custom settings take precedence."""
    from core.config import ProcessingConfig

    config = ProcessingConfig(
        sampling_rate=1.0,
        tukey_alpha=0.05,
        pchip_factor=2,
        window_size=5,
        n_sigmas=2.5,
    )

    t = np.linspace(0, 10, 100)
    sig = np.sin(t)
    sig[50] = 50.0

    cleaned = clean_and_smooth_signal(sig, apply_smoothing=False, config=config)
    assert abs(cleaned[50] - np.sin(t[50])) < 0.5
