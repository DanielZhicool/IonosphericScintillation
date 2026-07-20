"""
Synthetic scintillation signal generator for URAN-4 data validation.

Provides physical-statistical models for simulating power-law scintillation,
deterministic harmonic components, multi-channel delays, and realistic noise/outliers.
"""

import numpy as np


def generate_power_law_noise(
    length: int,
    fs: float,
    f_fresnel: float = 0.1,
    spectral_index: float = 8.0 / 3.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate 1D power-law noise using FFT-based spectral coloring.

    The power spectral density follows a flat spectrum below the Fresnel frequency,
    and decays as a power-law above it:
    PSD(f) = 1 / (1 + (f / f_fresnel)^2) ^ (spectral_index / 2)

    Args:
        length: Number of signal samples to generate.
        fs: Sampling frequency in Hz.
        f_fresnel: Fresnel frequency in Hz (transition from flat to power-law).
        spectral_index: Spectral decay power (e.g., 8/3 for Kolmogorov turbulence).
        seed: Random seed for reproducibility.

    Returns:
        1D numpy array containing normalized power-law noise (mean=0, std=1).
    """
    if length <= 0:
        return np.array([], dtype=np.float64)

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(length)

    # Compute FFT
    white_fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, d=1.0 / fs)

    # Compute the transfer function H(f) = sqrt(PSD(f))
    # Denominator is always >= 1.0, so no division by zero.
    h = 1.0 / (1.0 + (freqs / f_fresnel) ** 2) ** (spectral_index / 4.0)

    # Apply transfer function to white noise spectrum
    colored_fft = white_fft * h

    # Transform back to time domain
    colored = np.fft.irfft(colored_fft, n=length)

    # Normalize to zero mean and unit variance
    std_val = np.std(colored)
    if std_val > 0:
        colored = (colored - np.mean(colored)) / std_val

    return colored


def apply_delay(signal: np.ndarray, delay_sec: float, fs: float) -> np.ndarray:
    """
    Apply a fractional time delay to a signal in the frequency domain.

    Using the Fourier shift theorem:
    y(t) = x(t - tau) <=> Y(f) = X(f) * e^(-j * 2 * pi * f * tau)

    Args:
        signal: 1D numpy array representing the input signal.
        delay_sec: Delay to apply in seconds (positive = delay, negative = advance).
        fs: Sampling frequency in Hz.

    Returns:
        1D numpy array of the delayed signal.
    """
    length = len(signal)
    if length == 0 or delay_sec == 0.0:
        return signal.copy()

    sig_fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(length, d=1.0 / fs)

    # Construct the phase shift term
    phase_shift = np.exp(-2j * np.pi * freqs * delay_sec)

    # Apply shift and transform back
    delayed_fft = sig_fft * phase_shift
    delayed = np.fft.irfft(delayed_fft, n=length)

    return delayed


def generate_synthetic_scintillation(
    length: int,
    fs: float,
    f_fresnel: float = 0.1,
    spectral_index: float = 8.0 / 3.0,
    harmonics: list[tuple[float, float]] | None = None,
    coherence: float = 0.8,
    delay_sec: float = 0.0,
    white_noise_std: float = 0.1,
    spike_prob: float = 0.0,
    spike_std: float = 10.0,
    gaps: list[tuple[int, int]] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dual-frequency scintillation signals with configurable
    statistics, delay, coherence, periodic components, and observational artifacts.

    Args:
        length: Number of signal samples to generate.
        fs: Sampling frequency in Hz.
        f_fresnel: Fresnel transition frequency in Hz.
        spectral_index: Spectral index of the decay power.
        harmonics: List of (period_sec, amplitude) tuples for periodic components.
        coherence: Target coherence fraction C in [0, 1] between the two channels.
        delay_sec: Time delay in seconds between channel 1 and channel 2 (S2 = S1(t - delay)).
        white_noise_std: Standard deviation of additive white Gaussian noise.
        spike_prob: Probability of replacing a sample with an outlier spike.
        spike_std: Standard deviation of outlier spikes.
        gaps: List of (start_idx, end_idx) indices to zero out.
        seed: Random seed for reproducibility.

    Returns:
        sig1: 1D numpy array representing Channel 1.
        sig2: 1D numpy array representing Channel 2.
    """
    if length <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if not (0.0 <= coherence <= 1.0):
        raise ValueError("Coherence must be between 0.0 and 1.0")

    # Construct secondary seeds from the primary seed to keep generation deterministic
    seed_common = None if seed is None else seed
    seed_indep1 = None if seed is None else seed + 100_000
    seed_indep2 = None if seed is None else seed + 200_000
    seed_noise = None if seed is None else seed + 300_000

    rng = np.random.default_rng(seed_noise)

    # 1. Generate common and independent power-law noise components
    common = generate_power_law_noise(length, fs, f_fresnel, spectral_index, seed_common)
    indep1 = generate_power_law_noise(length, fs, f_fresnel, spectral_index, seed_indep1)
    indep2 = generate_power_law_noise(length, fs, f_fresnel, spectral_index, seed_indep2)

    # 2. Blend common and independent components to enforce coherence
    # S1 = sqrt(C)*common + sqrt(1-C)*indep1
    # S2 = sqrt(C)*delay(common) + sqrt(1-C)*indep2
    c_weight = np.sqrt(coherence)
    i_weight = np.sqrt(1.0 - coherence)

    sig1 = c_weight * common + i_weight * indep1
    sig2 = c_weight * apply_delay(common, delay_sec, fs) + i_weight * indep2

    # 3. Add periodic components (harmonics)
    t = np.arange(length, dtype=np.float64) / fs
    if harmonics is not None:
        for period, amp in harmonics:
            if period <= 0.0:
                continue
            freq = 1.0 / period
            # Phase is chosen randomly but deterministically
            phi = rng.uniform(0, 2.0 * np.pi)

            h1 = amp * np.sin(2.0 * np.pi * freq * t + phi)
            h2 = amp * np.sin(2.0 * np.pi * freq * (t - delay_sec) + phi)

            sig1 += h1
            sig2 += h2

    # Standardize core signals to ensure unit variance before adding noise
    std1 = np.std(sig1)
    if std1 > 0:
        sig1 = (sig1 - np.mean(sig1)) / std1
    std2 = np.std(sig2)
    if std2 > 0:
        sig2 = (sig2 - np.mean(sig2)) / std2

    # 4. Add white Gaussian noise
    if white_noise_std > 0:
        sig1 += rng.normal(0.0, white_noise_std, length)
        sig2 += rng.normal(0.0, white_noise_std, length)

    # 5. Inject spike outliers
    if spike_prob > 0.0:
        mask1 = rng.random(length) < spike_prob
        mask2 = rng.random(length) < spike_prob
        sig1[mask1] = rng.normal(0.0, spike_std, int(np.sum(mask1)))
        sig2[mask2] = rng.normal(0.0, spike_std, int(np.sum(mask2)))

    # 6. Inject gaps (calibration/observation blocks)
    if gaps is not None:
        for s_idx, e_idx in gaps:
            s_idx = max(0, s_idx)
            e_idx = min(length, e_idx)
            if s_idx < e_idx:
                sig1[s_idx:e_idx] = 0.0
                sig2[s_idx:e_idx] = 0.0

    return sig1, sig2
