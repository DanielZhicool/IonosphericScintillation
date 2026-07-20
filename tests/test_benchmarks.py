import numpy as np
import pytest
from typing import Any

from core.signal_processing import (
    clean_and_smooth_signal,
    compute_cwt_spectrogram,
    upsample_pchip,
)
from core.spectral_analysis import run_spectral_pipeline
from core.synthetic_generator import generate_synthetic_scintillation


@pytest.fixture(scope="module")
def benchmark_signals() -> tuple[np.ndarray, np.ndarray]:
    """Provide synthetic dual-frequency scintillation signals for benchmarking."""
    fs = 1.0  # Hz
    length = 2000
    # Generate stable synthetic signals
    sig1, sig2 = generate_synthetic_scintillation(
        length=length,
        fs=fs,
        f_fresnel=0.1,
        spectral_index=8.0 / 3.0,
        harmonics=[(30.0, 1.5)],
        coherence=0.8,
        delay_sec=1.5,
        white_noise_std=0.05,
        seed=101,
    )
    # Add outlier spikes to make Hampel filter work
    sig1[100] = 50.0
    sig1[500] = -40.0
    return sig1, sig2


def test_benchmark_hampel_and_savgol(benchmark: Any, benchmark_signals: tuple[np.ndarray, np.ndarray]) -> None:
    """Benchmark Hampel cleaning and Savitzky-Golay smoothing."""
    sig1, _ = benchmark_signals

    def run_cleaning() -> np.ndarray:
        return clean_and_smooth_signal(sig1, window_size=15, n_sigmas=3.0, apply_smoothing=True)

    benchmark(run_cleaning)


def test_benchmark_pchip_upsampling(benchmark: Any, benchmark_signals: tuple[np.ndarray, np.ndarray]) -> None:
    """Benchmark PCHIP upsampling of the signal."""
    sig1, _ = benchmark_signals

    def run_upsampling() -> tuple[np.ndarray, float]:
        return upsample_pchip(sig1, fs=1.0, factor=3)

    benchmark(run_upsampling)


def test_benchmark_cwt_spectrogram(benchmark: Any, benchmark_signals: tuple[np.ndarray, np.ndarray]) -> None:
    """Benchmark Continuous Wavelet Transform (CWT) without Synchrosqueezing."""
    sig1, _ = benchmark_signals
    # Truncate to 500 samples to keep benchmark run-time reasonable
    sig_trunc = sig1[:500]

    def run_cwt() -> np.ndarray:
        return compute_cwt_spectrogram(sig_trunc, fs=1.0, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=False)

    benchmark(run_cwt)


def test_benchmark_sst_spectrogram(benchmark: Any, benchmark_signals: tuple[np.ndarray, np.ndarray]) -> None:
    """Benchmark Synchrosqueezed Wavelet Transform (SST)."""
    sig1, _ = benchmark_signals
    # Truncate to 500 samples to keep benchmark run-time reasonable
    sig_trunc = sig1[:500]

    def run_sst() -> np.ndarray:
        return compute_cwt_spectrogram(sig_trunc, fs=1.0, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=True)

    benchmark(run_sst)


def test_benchmark_spectral_pipeline(benchmark: Any, benchmark_signals: tuple[np.ndarray, np.ndarray]) -> None:
    """Benchmark the full spectral analysis pipeline (PSD, F-Test, Cross-Spectrum, IDVE)."""
    sig1, sig2 = benchmark_signals
    pm_signals = {
        "20 MHz Pol A": sig1,
        "25 MHz Pol A": sig2,
        "20 MHz Pol B": sig1,
        "25 MHz Pol B": sig2,
    }

    def run_pipeline() -> dict:
        return run_spectral_pipeline(
            pm_signals=pm_signals,
            fs=1.0,
            lowcut=0.01,
            highcut=0.1,
            window_size=15,
            n_sigmas=3.0,
            apply_smoothing=True,
        )

    benchmark(run_pipeline)
