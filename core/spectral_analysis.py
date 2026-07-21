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
from core.types import MultitaperPSDResult, VelocityEstimate


def _prepare_signal(
    raw_signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    window_size: int,
    n_sigmas: float,
    apply_smoothing: bool,
    config: cfg.ProcessingConfig | None = None,
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
        config=config,
    )
    filtered = bandpass_filter(cleaned, lowcut, highcut, fs, config=config)
    return filtered, fs


def compute_multitaper_psd(
    signal: np.ndarray,
    fs: float,
    n_tapers: int | None = None,
    nw: float | None = None,
    compute_ci: bool | None = None,
    config: cfg.ProcessingConfig | None = None,
) -> MultitaperPSDResult:
    """
    Compute power spectral density using Thomson's Multitaper method.

    Uses K DPSS (Slepian) tapers to produce a low-variance, low-bias
    spectral estimate with optional Jackknife 95% confidence intervals.

    Args:
        signal: 1-D array, bandpass-filtered signal.
        fs: sampling frequency (Hz).
        n_tapers: number of DPSS tapers.
        nw: time-bandwidth product.
        compute_ci: Whether to compute Jackknife 95% confidence intervals.
        config: Optional ProcessingConfig container override.

    Returns:
        MultitaperPSDResult object holding freqs, psd, log_psd_se, ci95_low, ci95_high.
        Unpacks natively as (freqs, psd).
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}")

    arr = np.asarray(signal, dtype=float)
    if len(arr) == 0:
        raise ValueError("Input signal is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input signal contains non-finite values (NaN or Inf).")

    if n_tapers is None:
        n_tapers = config.mtm_n_tapers if config is not None else cfg.MTM_N_TAPERS
    if nw is None:
        nw = config.mtm_nw if config is not None else cfg.MTM_NW
    if compute_ci is None:
        compute_ci = config.compute_jackknife_ci if config is not None else cfg.COMPUTE_JACKKNIFE_CI

    if n_tapers < 1:
        raise ValueError(f"n_tapers must be >= 1, got {n_tapers}")
    if nw <= 0:
        raise ValueError(f"nw must be > 0, got {nw}")

    N = len(arr)
    if max(2 * nw, n_tapers) >= N:
        freqs = np.fft.rfftfreq(N, d=1.0 / fs) if N > 0 else np.array([0.0])
        z = np.zeros_like(freqs)
        return MultitaperPSDResult(freqs=freqs, psd=z, log_psd_se=z, ci95_low=z, ci95_high=z)

    sig = detrend(arr)

    tapers, eigenvalues = dpss(N, nw, n_tapers, return_ratios=True)

    nfreqs = N // 2 + 1
    Sk = np.zeros((n_tapers, nfreqs))

    for k in range(n_tapers):
        Yk = np.fft.rfft(sig * tapers[k])
        Sk[k] = np.abs(Yk) ** 2

    # Eigenvalue-weighted average across tapers
    psd = np.average(Sk, axis=0, weights=eigenvalues)

    # Normalize to one-sided PSD
    norm_factor = 2.0 / (fs * N)
    psd *= norm_factor
    psd[0] /= 2.0  # DC not doubled
    if N % 2 == 0:
        psd[-1] /= 2.0  # Nyquist not doubled

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Jackknife 95% Confidence Intervals over DPSS tapers
    if compute_ci and n_tapers > 1:
        Sk_scaled = Sk * norm_factor
        Sk_scaled[:, 0] /= 2.0
        if N % 2 == 0:
            Sk_scaled[:, -1] /= 2.0

        sum_Sk = np.sum(Sk_scaled, axis=0)
        leave_one_out = (sum_Sk[None, :] - Sk_scaled) / (n_tapers - 1)
        log_loo = np.log(leave_one_out + 1e-30)
        mean_log_loo = np.mean(log_loo, axis=0)

        var_log_psd = ((n_tapers - 1) / n_tapers) * np.sum((log_loo - mean_log_loo[None, :]) ** 2, axis=0)
        log_psd_se = np.sqrt(np.maximum(var_log_psd, 0.0))

        ci95_low = psd * np.exp(-1.96 * log_psd_se)
        ci95_high = psd * np.exp(+1.96 * log_psd_se)
    else:
        log_psd_se = None
        ci95_low = None
        ci95_high = None

    return MultitaperPSDResult(
        freqs=freqs,
        psd=psd,
        log_psd_se=log_psd_se,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
    )


def compute_ftest(
    signal: np.ndarray,
    fs: float,
    n_tapers: int | None = None,
    nw: float | None = None,
    confidence: float | None = None,
    fdr_alpha: float | None = None,
    lowcut: float | None = None,
    highcut: float | None = None,
    config: cfg.ProcessingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    """
    Thomson F-test for detecting deterministic harmonic components with Benjamini-Hochberg FDR control.

    At each frequency the test checks whether a sinusoidal component
    rises above the stochastic noise floor.

    Args:
        signal: 1-D array, bandpass-filtered signal.
        fs: sampling frequency (Hz).
        n_tapers: number of DPSS tapers.
        nw: time-bandwidth product.
        confidence: significance level (0-1).
        fdr_alpha: False Discovery Rate alpha (e.g. 0.05).
        lowcut, highcut: optional frequency band limits.
        config: Optional ProcessingConfig container override.

    Returns:
        freqs: frequency array (Hz).
        f_stat: F-statistic at each frequency.
        threshold: critical value of F(2, 2K-2) at *confidence*.
        T0: dominant oscillation period (s), or None if no
            significant peak is found.
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}")

    arr = np.asarray(signal, dtype=float)
    if len(arr) == 0:
        raise ValueError("Input signal is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input signal contains non-finite values (NaN or Inf).")

    if n_tapers is None:
        n_tapers = config.mtm_n_tapers if config is not None else cfg.MTM_N_TAPERS
    if nw is None:
        nw = config.mtm_nw if config is not None else cfg.MTM_NW
    if confidence is None:
        confidence = config.ftest_confidence if config is not None else cfg.FTEST_CONFIDENCE
    if fdr_alpha is None:
        fdr_alpha = config.fdr_alpha if config is not None else cfg.FDR_ALPHA

    if n_tapers < 1:
        raise ValueError(f"n_tapers must be >= 1, got {n_tapers}")
    if nw <= 0:
        raise ValueError(f"nw must be > 0, got {nw}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    N = len(arr)

    if max(2 * nw, n_tapers) >= N:
        freqs = np.fft.rfftfreq(N, d=1.0 / fs) if N > 0 else np.array([0.0])
        return freqs, np.zeros_like(freqs), 1.0, None

    sig = detrend(arr)

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
    d1, d2 = 2, 2 * n_tapers - 2
    raw_threshold = f_dist.ppf(confidence, d1, d2)

    # Benjamini-Hochberg FDR Procedure
    band_mask = (freqs >= lowcut) & (freqs <= highcut) if lowcut is not None and highcut is not None else freqs > 0
    p_vals_band = f_dist.sf(f_stat[band_mask], d1, d2)

    if len(p_vals_band) > 0:
        sorted_idx = np.argsort(p_vals_band)
        sorted_p = p_vals_band[sorted_idx]
        m = len(sorted_p)
        k_indices = np.arange(1, m + 1)
        valid_k = np.where(sorted_p <= (k_indices / m) * fdr_alpha)[0]

        if len(valid_k) > 0:
            k_max = valid_k[-1] + 1
            fdr_p_cutoff = (k_max / m) * fdr_alpha
            effective_threshold = float(f_dist.ppf(1.0 - fdr_p_cutoff, d1, d2))
        else:
            effective_threshold = float(raw_threshold)
    else:
        effective_threshold = float(raw_threshold)

    # Find the dominant period T0 (highest F-stat peak above effective threshold)
    f_stat_band = np.where(band_mask, f_stat, 0.0)
    dominant_idx = int(np.argmax(f_stat_band))
    T0 = 1.0 / freqs[dominant_idx] if band_mask[dominant_idx] and f_stat[dominant_idx] > effective_threshold else None

    return freqs, f_stat, effective_threshold, T0


def compute_cross_spectrum(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: float,
    n_tapers: int | None = None,
    nw: float | None = None,
    config: cfg.ProcessingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multitaper cross-spectral analysis between two signals.

    Computes cross-power, phase difference, and magnitude-squared
    coherence between *sig1* (20 MHz) and *sig2* (25 MHz).

    Args:
        sig1: 1-D array (e.g. 20 MHz P-M).
        sig2: 1-D array (e.g. 25 MHz P-M), same length as *sig1*.
        fs: sampling frequency (Hz).
        n_tapers: number of DPSS tapers.
        nw: time-bandwidth product.
        config: Optional ProcessingConfig container override.

    Returns:
        freqs: frequency array (Hz).
        cross_power: |S_xy| cross-spectral magnitude.
        phase: angle(S_xy) in **degrees**.
        coherence: magnitude-squared coherence (0-1).
        real_part: Re(S_xy)
        imag_part: Im(S_xy)
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}")

    arr1 = np.asarray(sig1, dtype=float)
    arr2 = np.asarray(sig2, dtype=float)

    if len(arr1) == 0 or len(arr2) == 0:
        raise ValueError("Input signal(s) for cross spectrum are empty.")
    if len(arr1) != len(arr2):
        raise ValueError(f"Signals must have the same length, got {len(arr1)} and {len(arr2)}")
    if not np.all(np.isfinite(arr1)) or not np.all(np.isfinite(arr2)):
        raise ValueError("Input signal(s) contain non-finite values (NaN or Inf).")

    if n_tapers is None:
        n_tapers = config.mtm_n_tapers if config is not None else cfg.MTM_N_TAPERS
    if nw is None:
        nw = config.mtm_nw if config is not None else cfg.MTM_NW

    if n_tapers < 1:
        raise ValueError(f"n_tapers must be >= 1, got {n_tapers}")
    if nw <= 0:
        raise ValueError(f"nw must be > 0, got {nw}")

    N = len(arr1)

    if max(2 * nw, n_tapers) >= N:
        freqs = np.fft.rfftfreq(N, d=1.0 / fs) if N > 0 else np.array([0.0])
        z = np.zeros_like(freqs)
        return freqs, z, z, z, z, z

    s1 = detrend(arr1)
    s2 = detrend(arr2)

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
    power: np.ndarray,
    freqs: np.ndarray,
    lowcut_hz: float,
    highcut_hz: float,
    n_peaks: int | None = None,
    config: cfg.ProcessingConfig | None = None,
) -> np.ndarray:
    """
    Find the *n_peaks* strongest spectral peaks within a frequency band.

    Args:
        power: spectral power array (PSD or cross-power).
        freqs: corresponding frequency array (Hz).
        lowcut_hz, highcut_hz: band boundaries (Hz).
        n_peaks: number of peaks to return.
        config: Optional ProcessingConfig container override.

    Returns:
        peak_indices: 1-D array of indices into *freqs* / *power*.
    """
    if lowcut_hz <= 0:
        raise ValueError(f"lowcut_hz must be > 0, got {lowcut_hz}")
    if highcut_hz <= lowcut_hz:
        raise ValueError(f"highcut_hz ({highcut_hz} Hz) must be greater than lowcut_hz ({lowcut_hz} Hz)")

    if n_peaks is None:
        n_peaks = config.velocity_n_peaks if config is not None else cfg.VELOCITY_N_PEAKS

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
    cross_phase_deg: np.ndarray,
    freqs: np.ndarray,
    peak_indices: np.ndarray,
    coherence: np.ndarray | None = None,
    dx: float | None = None,
    min_coherence: float | None = None,
    bandwidth_hz: float | None = None,
    enable_phase_regression: bool | None = None,
    config: cfg.ProcessingConfig | None = None,
) -> list[VelocityEstimate]:
    """
    Convert cross-spectral phase at peak frequencies to horizontal
    ionospheric drift velocities.

    Model:  radio waves pass through ionospheric irregularities; the
    beams at 20 MHz and 25 MHz are separated by *dx* metres.
        unwrapped phase phi(f) = a * f + b  (if enable_phase_regression=True)
        time_delay tau  =  phase_deg * period / 360  (or a / 2pi)
        velocity    =  sign(tau) * (dx / |tau|)

    Args:
        cross_phase_deg: phase-difference array (degrees).
        freqs: frequency array (Hz).
        peak_indices: indices of spectral peaks to analyse.
        coherence: optional magnitude-squared coherence array (0-1).
        dx: beam separation (metres).
        min_coherence: minimum coherence threshold for valid estimate (default 0.7).
        bandwidth_hz: bandwidth for phase regression around peak (default 0.02 Hz).
        enable_phase_regression: whether to fit linear regression over broadband high-coherence bins.
        config: Optional ProcessingConfig container override.

    Returns:
        list of VelocityEstimate objects with analytical 95% CIs and gating status.
    """
    if dx is None:
        dx = config.cross_spectrum_dx if config is not None else cfg.CROSS_SPECTRUM_DX
    if dx <= 0:
        raise ValueError(f"Beam separation dx must be positive, got {dx}")

    if min_coherence is None:
        min_coherence = config.coherence_threshold if config is not None else cfg.COHERENCE_THRESHOLD
    if bandwidth_hz is None:
        bandwidth_hz = config.phase_regression_bandwidth_hz if config is not None else cfg.PHASE_REGRESSION_BANDWIDTH_HZ
    if enable_phase_regression is None:
        enable_phase_regression = config.enable_phase_regression if config is not None else False

    results = []
    for idx in peak_indices:
        freq0 = freqs[idx]
        if freq0 <= 0:
            continue

        period = 1.0 / freq0

        # Select sub-band around peak for weighted phase regression if enabled
        f_min = max(freqs[1], freq0 - bandwidth_hz) if len(freqs) > 1 else freq0
        f_max = freq0 + bandwidth_hz
        band_mask = (freqs >= f_min) & (freqs <= f_max) & (freqs > 0)

        if enable_phase_regression and coherence is not None:
            high_coh_mask = band_mask & (coherence >= min_coherence)
            fit_mask = high_coh_mask if np.sum(high_coh_mask) >= 5 else np.array([idx])
        else:
            fit_mask = np.array([idx])

        sub_freqs = freqs[fit_mask]
        sub_phase_deg = cross_phase_deg[fit_mask]
        sub_phase_rad = np.unwrap(np.radians(sub_phase_deg))

        # Weights from coherence
        if coherence is not None:
            sub_coh = coherence[fit_mask]
            mean_coh = float(np.mean(sub_coh))
        else:
            sub_coh = np.ones_like(sub_freqs)
            mean_coh = 1.0

        # Coherence Gating
        is_valid = True
        gating_reason = None
        if mean_coh < min_coherence:
            is_valid = False
            gating_reason = f"Mean coherence ({mean_coh:.3f}) below threshold ({min_coherence:.2f})"

        n_pts = len(sub_freqs)
        if enable_phase_regression and n_pts >= 5 and coherence is not None:
            # Weighted Linear Regression: phase_rad(f) = a * f + b
            weights = np.maximum(sub_coh, 1e-4)
            W = np.sum(weights)

            f_bar = np.sum(weights * sub_freqs) / W
            y_bar = np.sum(weights * sub_phase_rad) / W

            num = np.sum(weights * (sub_freqs - f_bar) * (sub_phase_rad - y_bar))
            den = np.sum(weights * (sub_freqs - f_bar) ** 2)

            if abs(den) > 1e-12:
                a = float(num / den)
                dt = a / (2.0 * np.pi)
            else:
                single_phase_deg = float(cross_phase_deg[idx])
                dt = single_phase_deg * period / 360.0
                a = 2.0 * np.pi * dt

            # Standard error of slope a
            residuals = sub_phase_rad - (a * sub_freqs + (y_bar - a * f_bar))
            s_sq = np.sum(weights * residuals**2) / ((n_pts - 2) * W)
            var_a = (s_sq * W) / den
            se_a = float(np.sqrt(max(var_a, 0.0)))
        else:
            # Single-point phase conversion
            single_phase_deg = float(cross_phase_deg[idx])
            dt = single_phase_deg * period / 360.0
            a = 2.0 * np.pi * dt
            se_a = 0.05 * abs(a) if a != 0 else 0.05

        se_dt = se_a / (2.0 * np.pi)
        dt_ci95 = (dt - 1.96 * se_dt, dt + 1.96 * se_dt)

        # Velocity v = sign(dt) * (dx / |dt|)
        if abs(dt) > 0.5:
            velocity = float(np.sign(dt) * (dx / abs(dt)))
            se_v = abs(dx / (dt**2)) * se_dt
            vel_ci95 = (velocity - 1.96 * se_v, velocity + 1.96 * se_v)
            if abs(velocity) > 5000.0:
                is_valid = False
                gating_reason = f"Velocity ({velocity:.1f} m/s) exceeds physical limit (dt ~ 0)"
        else:
            velocity = np.inf if dt >= 0 else -np.inf
            vel_ci95 = (np.inf, np.inf)
            is_valid = False
            gating_reason = f"Time delay ({dt:.2f} s) too small (in-phase signals)"

        results.append(
            VelocityEstimate(
                peak_freq=float(freq0),
                period=float(period),
                mean_coherence=mean_coh,
                phase_slope=a,
                dt=dt,
                dt_ci95=dt_ci95,
                velocity=velocity,
                velocity_ci95=vel_ci95,
                is_valid=is_valid,
                gating_reason=gating_reason,
            )
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
    config: cfg.ProcessingConfig | None = None,
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
        config: Optional ProcessingConfig container override.

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
            config=config,
        )
        filtered[ch_name] = sig
        new_fs = nfs

    if progress_callback:
        progress_callback(25)

    # 2. Multitaper PSD for each channel
    psd_results: dict[str, np.ndarray] = {}
    freqs: np.ndarray = np.array([], dtype=np.float64)
    for ch_name, sig in filtered.items():
        res = compute_multitaper_psd(sig, new_fs, config=config)
        psd_results[ch_name] = res.psd
        freqs = res.freqs

    if progress_callback:
        progress_callback(45)

    # 3. F-test for each channel
    ftest_results: dict[str, Any] = {}
    threshold = None
    for ch_name, sig in filtered.items():
        _, fstat, thresh, T0 = compute_ftest(sig, new_fs, lowcut=lowcut, highcut=highcut, config=config)
        ftest_results[ch_name] = fstat
        ftest_results[ch_name + "_T0"] = T0
        threshold = thresh
    ftest_results["threshold"] = threshold
    ftest_results["confidence"] = config.ftest_confidence if config is not None else cfg.FTEST_CONFIDENCE

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
            config=config,
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
            config=config,
        )
        velocities = estimate_velocities(
            cross_data["phase"], freqs, peaks, coherence=cross_data["coherence"], config=config
        )
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
