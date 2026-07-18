"""
Spectral-Correlation Analysis for URAN-4 ionospheric scintillation data.

Implements the Thomson Multitaper spectral analysis pipeline:
  Step 10: Multitaper PSD (power spectral density)
  Step 11: Thomson F-Test (deterministic harmonic detection)
  Step 12: Cross-spectral analysis (20 MHz vs 25 MHz)
  Step 13: Ionospheric drift velocity estimation
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.signal import detrend, find_peaks
from scipy.signal.windows import dpss
from scipy.stats import f as f_dist

import core.config as cfg
from core.signal_processing import (
    bandpass_filter,
    clean_and_smooth_signal,
)


def _prepare_signal(
    raw_signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    window_size: int,
    n_sigmas: float,
    apply_smoothing: bool,
) -> tuple[np.ndarray, float]:
    """Clean and bandpass-filter a signal for spectral analysis.

    Note: No PCHIP upsampling is applied here — unlike CWT, FFT-based
    spectral analysis works best at the original sampling rate for correct
    frequency bin placement and maximum period resolution.
    """
    cleaned = clean_and_smooth_signal(
        raw_signal,
        window_size=window_size,
        n_sigmas=n_sigmas,
        apply_smoothing=apply_smoothing,
    )
    filtered = bandpass_filter(cleaned, lowcut, highcut, fs)
    return filtered, fs


def compute_multitaper_psd(
    signal: np.ndarray, fs: float, n_tapers: int | None = None, nw: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Thomson's Multitaper method.

    Uses K DPSS (Slepian) tapers to produce a low-variance, low-bias
    spectral estimate.

    Args:
        signal: 1-D array, bandpass-filtered signal.
        fs: sampling frequency (Hz).
        n_tapers: number of DPSS tapers.
        nw: time-bandwidth product.

    Returns:
        freqs: frequency array (Hz).
        psd: one-sided power spectral density.
    """
    if n_tapers is None:
        n_tapers = cfg.MTM_N_TAPERS
    if nw is None:
        nw = cfg.MTM_NW

    N = len(signal)
    if max(2 * nw, n_tapers) >= N:
        freqs = np.fft.rfftfreq(N, d=1.0 / fs) if N > 0 else np.array([0.0])
        return freqs, np.zeros_like(freqs)

    sig = detrend(signal)

    tapers, eigenvalues = dpss(N, nw, n_tapers, return_ratios=True)

    nfreqs = N // 2 + 1
    Sk = np.zeros((n_tapers, nfreqs))

    for k in range(n_tapers):
        Yk = np.fft.rfft(sig * tapers[k])
        Sk[k] = np.abs(Yk) ** 2

    # Eigenvalue-weighted average across tapers
    psd = np.average(Sk, axis=0, weights=eigenvalues)

    # Normalize to one-sided PSD
    psd *= 2.0 / (fs * N)
    psd[0] /= 2.0  # DC not doubled
    if N % 2 == 0:
        psd[-1] /= 2.0  # Nyquist not doubled

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, psd


def compute_ftest(
    signal: np.ndarray,
    fs: float,
    n_tapers: int | None = None,
    nw: float | None = None,
    confidence: float | None = None,
    lowcut: float | None = None,
    highcut: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    """
    Thomson F-test for detecting deterministic harmonic components.

    At each frequency the test checks whether a sinusoidal component
    rises above the stochastic noise floor.

    Args:
        signal: 1-D array, bandpass-filtered signal.
        fs: sampling frequency (Hz).
        n_tapers: number of DPSS tapers.
        nw: time-bandwidth product.
        confidence: significance level (0-1).

    Returns:
        freqs: frequency array (Hz).
        f_stat: F-statistic at each frequency.
        threshold: critical value of F(2, 2K-2) at *confidence*.
        T0: dominant oscillation period (s), or None if no
            significant peak is found.
    """
    if n_tapers is None:
        n_tapers = cfg.MTM_N_TAPERS
    if nw is None:
        nw = cfg.MTM_NW
    if confidence is None:
        confidence = cfg.FTEST_CONFIDENCE
    N = len(signal)

    if max(2 * nw, n_tapers) >= N:
        freqs = np.fft.rfftfreq(N, d=1.0 / fs) if N > 0 else np.array([0.0])
        return freqs, np.zeros_like(freqs), 1.0, None

    sig = detrend(signal)

    tapers = dpss(N, nw, n_tapers)

    nfreqs = N // 2 + 1
    Yk = np.zeros((n_tapers, nfreqs), dtype=complex)
    for k in range(n_tapers):
        Yk[k] = np.fft.rfft(sig * tapers[k])

    # DC response of each taper (zeroth Fourier coefficient)
    Hk = np.sum(tapers, axis=1)  # (K,)
    H2_sum = np.sum(Hk**2)  # scalar

    # Estimated deterministic line amplitude
    mu_hat = np.sum(Hk[:, None] * Yk, axis=0) / H2_sum  # (nfreqs,)

    # Deterministic power
    det_power = np.abs(mu_hat) ** 2 * H2_sum

    # Stochastic power (residuals after removing the estimated line)
    residuals = Yk - mu_hat[None, :] * Hk[:, None]
    stoch_power = np.sum(np.abs(residuals) ** 2, axis=0)

    # F-statistic  ~  F(2, 2K-2) under null hypothesis
    f_stat = (n_tapers - 1) * det_power / (stoch_power + 1e-30)

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    threshold = f_dist.ppf(confidence, 2, 2 * n_tapers - 2)

    # Find the dominant period T0 (highest F-stat peak above threshold)
    # Restrict search to the analysis band so leakage outside doesn't win
    band_mask = (freqs >= lowcut) & (freqs <= highcut) if lowcut is not None and highcut is not None else freqs > 0
    f_stat_band = np.where(band_mask, f_stat, 0.0)
    dominant_idx = int(np.argmax(f_stat_band))
    T0 = 1.0 / freqs[dominant_idx] if band_mask[dominant_idx] and f_stat[dominant_idx] > threshold else None

    return freqs, f_stat, threshold, T0


def compute_cross_spectrum(
    sig1: np.ndarray, sig2: np.ndarray, fs: float, n_tapers: int | None = None, nw: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multitaper cross-spectral analysis between two signals.

    Computes cross-power, phase difference, and magnitude-squared
    coherence between *sig1* (20 MHz) and *sig2* (25 MHz).

    Args:
        sig1: 1-D array (e.g. 20 MHz P-M).
        sig2: 1-D array (e.g. 25 MHz P-M), same length as *sig1*.
        fs: sampling frequency (Hz).

    Returns:
        freqs: frequency array (Hz).
        cross_power: |S_xy| cross-spectral magnitude.
        phase: angle(S_xy) in **degrees**.
        coherence: magnitude-squared coherence (0-1).
        real_part: Re(S_xy)
        imag_part: Im(S_xy)
    """
    if n_tapers is None:
        n_tapers = cfg.MTM_N_TAPERS
    if nw is None:
        nw = cfg.MTM_NW
    N = len(sig1)

    if max(2 * nw, n_tapers) >= N:
        freqs = np.fft.rfftfreq(N, d=1.0 / fs) if N > 0 else np.array([0.0])
        z = np.zeros_like(freqs)
        return freqs, z, z, z, z, z

    assert len(sig2) == N, "Signals must have the same length"

    s1 = detrend(sig1)
    s2 = detrend(sig2)

    tapers = dpss(N, nw, n_tapers)

    nfreqs = N // 2 + 1
    Xk = np.zeros((n_tapers, nfreqs), dtype=complex)
    Yk_arr = np.zeros((n_tapers, nfreqs), dtype=complex)

    for k in range(n_tapers):
        Xk[k] = np.fft.rfft(s1 * tapers[k])
        Yk_arr[k] = np.fft.rfft(s2 * tapers[k])

    # Cross-spectrum averaged over tapers
    Sxy = np.mean(Xk * np.conj(Yk_arr), axis=0)

    # Auto-spectra for coherence
    Sxx = np.mean(np.abs(Xk) ** 2, axis=0)
    Syy = np.mean(np.abs(Yk_arr) ** 2, axis=0)

    cross_power = np.abs(Sxy)
    phase = np.angle(Sxy, deg=True)
    coherence = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-30)
    real_part = np.real(Sxy)
    imag_part = np.imag(Sxy)

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, cross_power, phase, coherence, real_part, imag_part


def find_spectral_peaks(
    power: np.ndarray, freqs: np.ndarray, lowcut_hz: float, highcut_hz: float, n_peaks: int | None = None
) -> np.ndarray:
    """
    Find the *n_peaks* strongest spectral peaks within a frequency band.

    Args:
        power: spectral power array (PSD or cross-power).
        freqs: corresponding frequency array (Hz).
        lowcut_hz, highcut_hz: band boundaries (Hz).
        n_peaks: number of peaks to return.

    Returns:
        peak_indices: 1-D array of indices into *freqs* / *power*.
    """
    if n_peaks is None:
        n_peaks = cfg.VELOCITY_N_PEAKS
    band_mask = (freqs >= lowcut_hz) & (freqs <= highcut_hz)

    masked_power = np.zeros_like(power)
    masked_power[band_mask] = power[band_mask]

    peaks, _ = find_peaks(masked_power, distance=3)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Sort by power descending, return top N
    sorted_idx = np.argsort(masked_power[peaks])[::-1]
    return peaks[sorted_idx[:n_peaks]]


def estimate_velocities(
    cross_phase_deg: np.ndarray, freqs: np.ndarray, peak_indices: np.ndarray, dx: float | None = None
) -> list[dict[str, Any]]:
    """
    Convert cross-spectral phase at peak frequencies to horizontal
    ionospheric drift velocities.

    Model:  radio waves pass through ionospheric irregularities; the
    beams at 20 MHz and 25 MHz are separated by *dx* metres.
        phase_shift -> time_delay  =  phase * period / 360
        velocity    =  dx / |time_delay|

    Args:
        cross_phase_deg: phase-difference array (degrees).
        freqs: frequency array (Hz).
        peak_indices: indices of spectral peaks to analyse.
        dx: beam separation (metres).

    Returns:
        list of dicts with keys: period, phase_deg, dt, velocity.
    """
    if dx is None:
        dx = cfg.CROSS_SPECTRUM_DX
    results = []
    for idx in peak_indices:
        freq = freqs[idx]
        if freq <= 0:
            continue

        period = 1.0 / freq
        phase_deg = cross_phase_deg[idx]
        dt = phase_deg * period / 360.0

        velocity = dx / abs(dt) if abs(dt) > 1e-6 else np.inf

        results.append(
            {
                "period": period,
                "phase_deg": phase_deg,
                "dt": dt,
                "velocity": velocity,
            }
        )
    return results


def run_spectral_pipeline(
    pm_signals: dict[str, np.ndarray],
    fs: float,
    lowcut: float,
    highcut: float,
    window_size: int,
    n_sigmas: float,
    apply_smoothing: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """
    Full spectral-correlation analysis pipeline for **one** frequency band.

    Orchestrates:
        preprocessing → Multitaper PSD → F-Test →
        Cross-Spectra → Peak Detection → Velocity Estimation

    Args:
        pm_signals: dict mapping channel labels to raw P-M arrays, e.g.
            {'20 MHz Pol A': array, '20 MHz Pol B': array,
             '25 MHz Pol A': array, '25 MHz Pol B': array}
        fs: original sampling frequency (Hz).
        lowcut, highcut: bandpass boundaries (Hz).
        window_size, n_sigmas, apply_smoothing: cleaning params from UI.
        progress_callback: callable taking int 0-100 to report progress.

    Returns:
        dict with keys: freqs, psd, ftest, cross, velocities, lowcut, highcut.
    """
    if progress_callback:
        progress_callback(5)

    filtered: dict[str, np.ndarray] = {}
    new_fs: float = fs
    for ch_name, raw in pm_signals.items():
        sig, nfs = _prepare_signal(
            raw,
            fs,
            lowcut,
            highcut,
            window_size,
            n_sigmas,
            apply_smoothing,
        )
        filtered[ch_name] = sig
        new_fs = nfs

    if progress_callback:
        progress_callback(25)

    # 2. Multitaper PSD for each channel
    psd_results: dict[str, np.ndarray] = {}
    freqs: np.ndarray = np.array([], dtype=np.float64)
    for ch_name, sig in filtered.items():
        f, psd = compute_multitaper_psd(sig, new_fs)
        psd_results[ch_name] = psd
        freqs = f

    if progress_callback:
        progress_callback(45)

    # 3. F-test for each channel
    ftest_results: dict[str, Any] = {}
    threshold = None
    for ch_name, sig in filtered.items():
        _, fstat, thresh, T0 = compute_ftest(sig, new_fs, lowcut=lowcut, highcut=highcut)
        ftest_results[ch_name] = fstat
        ftest_results[ch_name + "_T0"] = T0
        threshold = thresh
    ftest_results["threshold"] = threshold
    ftest_results["confidence"] = cfg.FTEST_CONFIDENCE

    if progress_callback:
        progress_callback(75)

    # 4. Cross-spectra (20 MHz vs 25 MHz per polarisation)
    cross_results = {}
    cross_pairs = {
        "Pol A": ("20 MHz Pol A", "25 MHz Pol A"),
        "Pol B": ("20 MHz Pol B", "25 MHz Pol B"),
    }
    for pol_name, (ch20, ch25) in cross_pairs.items():
        _, power, phase, coh, real_p, imag_p = compute_cross_spectrum(
            filtered[ch20],
            filtered[ch25],
            new_fs,
        )
        cross_results[pol_name] = {"power": power, "phase": phase, "coherence": coh, "real": real_p, "imag": imag_p}

    if progress_callback:
        progress_callback(90)

    # 5. Peak detection + velocity estimation
    vel_results = {}
    for pol_name, cross_data in cross_results.items():
        peaks = find_spectral_peaks(
            cross_data["power"],
            freqs,
            lowcut,
            highcut,
        )
        velocities = estimate_velocities(cross_data["phase"], freqs, peaks)
        vel_results[pol_name] = velocities

    if progress_callback:
        progress_callback(100)

    return {
        "freqs": freqs,
        "psd": psd_results,
        "ftest": ftest_results,
        "cross": cross_results,
        "velocities": vel_results,
        "lowcut": lowcut,
        "highcut": highcut,
    }
