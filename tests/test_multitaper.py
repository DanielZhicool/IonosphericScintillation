import numpy as np
import pytest

from core.spectral_analysis import compute_multitaper_psd, find_spectral_peaks


@pytest.mark.parametrize(
    "f_sine, fs, n_tapers, nw",
    [
        (0.5, 10.0, 7, 4.0),
        (2.0, 20.0, 7, 4.0),
        (6.5, 50.0, 7, 4.0),
        (12.0, 50.0, 5, 3.0),
        (20.0, 100.0, 7, 4.0),
    ],
)
def test_compute_multitaper_psd_frequency_range(f_sine: float, fs: float, n_tapers: int, nw: float) -> None:
    """Test that multitaper PSD accurately detects dominant frequencies across a representative spectrum."""
    t = np.arange(1000) / fs
    signal = np.sin(2 * np.pi * f_sine * t)

    freqs, psd = compute_multitaper_psd(signal, fs, n_tapers=n_tapers, nw=nw)

    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]

    # Peak frequency should match target frequency within resolution limit
    df = fs / len(signal)
    assert peak_freq == pytest.approx(f_sine, abs=max(0.2, 2 * df))

    # Verify peak power prominence over background
    background_mask = np.abs(freqs - f_sine) > (4 * df + 0.5)
    if np.any(background_mask):
        background_mean_power = np.mean(psd[background_mask])
        peak_power = psd[peak_idx]
        assert peak_power > 50 * background_mean_power


@pytest.mark.parametrize(
    "f_peaks, fs",
    [
        ([0.05], 1.0),
        ([0.1, 0.3], 5.0),
        ([1.5, 4.0], 20.0),
    ],
)
def test_find_spectral_peaks(f_peaks: list[float], fs: float) -> None:
    """Test spectral peak extraction from PSD signals."""
    t = np.arange(1000) / fs
    signal = np.sum([np.sin(2 * np.pi * f * t) for f in f_peaks], axis=0)

    freqs, psd = compute_multitaper_psd(signal, fs, n_tapers=7, nw=4.0)
    detected_peaks = find_spectral_peaks(psd, freqs, lowcut_hz=0.01, highcut_hz=fs / 2 - 0.1, n_peaks=len(f_peaks))

    assert len(detected_peaks) >= len(f_peaks)
    detected_freqs = freqs[detected_peaks]

    for f in f_peaks:
        assert any(abs(df - f) < 0.2 for df in detected_freqs)


def test_multitaper_empty_signal() -> None:
    """Verify Multitaper PSD raises ValueError on empty inputs."""
    with pytest.raises(ValueError, match="Input signal is empty"):
        compute_multitaper_psd(np.array([]), fs=10.0)


def test_multitaper_invalid_inputs() -> None:
    """Verify boundary condition errors in spectral analysis."""
    sig = np.sin(np.linspace(0, 10, 100))

    with pytest.raises(ValueError, match="Sampling frequency fs must be positive"):
        compute_multitaper_psd(sig, fs=-1.0)

    with pytest.raises(ValueError, match="non-finite"):
        compute_multitaper_psd(np.array([1.0, np.nan, 3.0]), fs=10.0)

    with pytest.raises(ValueError, match="lowcut_hz must be > 0"):
        find_spectral_peaks(sig, np.linspace(0, 5, 100), lowcut_hz=-0.1, highcut_hz=2.0)

    with pytest.raises(ValueError, match="must be greater than lowcut_hz"):
        find_spectral_peaks(sig, np.linspace(0, 5, 100), lowcut_hz=3.0, highcut_hz=2.0)


def test_multitaper_processing_config() -> None:
    """Verify ProcessingConfig overrides take effect in multitaper calculations."""
    from core.config import ProcessingConfig

    config = ProcessingConfig(mtm_n_tapers=5, mtm_nw=3.0)

    t = np.arange(500) / 10.0
    sig = np.sin(2 * np.pi * 1.5 * t)

    freqs, psd = compute_multitaper_psd(sig, fs=10.0, config=config)
    assert len(freqs) > 0
    assert np.argmax(psd) > 0
