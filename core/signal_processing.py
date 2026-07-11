import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd

import core.config as cfg


def clean_and_smooth_signal(signal: np.ndarray, window_size: int = None, n_sigmas: float = None, apply_smoothing: bool = True, polyorder: int = None) -> np.ndarray:
    """
    Clean a radio-astronomy signal by removing spikes/clusters and applying smoothing.

    Args:
        signal: 1D numpy array representing the signal data.
        window_size: Odd window length for rolling operations. Defaults to cfg.DEFAULT_WINDOW_SIZE.
        n_sigmas: Outlier threshold in sigma units. Defaults to cfg.DEFAULT_N_SIGMAS.
        apply_smoothing: Whether to apply Savitzky-Golay smoothing.
        polyorder: Polynomial order for Savitzky-Golay filter. Defaults to cfg.SAVGOL_POLYORDER.

    Returns:
        1D numpy array of the cleaned and smoothed signal.
    """
    if window_size is None: window_size = cfg.DEFAULT_WINDOW_SIZE
    if n_sigmas is None: n_sigmas = cfg.DEFAULT_N_SIGMAS
    if polyorder is None: polyorder = cfg.SAVGOL_POLYORDER
    s = pd.Series(signal)
    
    # Step 1: Outlier detection (Hampel-like)
    # min_periods=1 ensures edge samples are still tested (otherwise the rolling
    # window is NaN at both ends, silently missing outliers there).
    rolling_median = s.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = 1.4826 * (s - rolling_median).abs().rolling(window=window_size, center=True, min_periods=1).median()
    outliers = (s - rolling_median).abs() > (n_sigmas * rolling_mad)
    
    cleaned_s = s.copy()
    cleaned_s[outliers] = rolling_median[outliers]
    
    cleaned_s = cleaned_s.bfill().ffill()
    cleaned_signal = cleaned_s.values
    
    # Step 2: Smoothing (Savitzky-Golay)
    if apply_smoothing:
        from scipy.signal import savgol_filter
        smooth_window = window_size if window_size % 2 != 0 else window_size + 1
        if smooth_window > 3:
            cleaned_signal = savgol_filter(cleaned_signal, window_length=smooth_window, polyorder=polyorder)

    return cleaned_signal


def fill_gap_with_red_noise(
    signal: np.ndarray,
    start_idx: int,
    end_idx: int,
    context_window: int = 200,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Replaces a segment of the signal with generated red noise.

    The generated noise matches the median and MAD-estimated standard deviation of
    the surrounding context.  The edges are blended with a cosine cross-fade so the
    join is seamless *and* the interior retains the correct f^-2 spectral shape
    (a linear ramp would inject low-frequency power and bias the spectrum).

    Args:
        signal: 1D numpy array representing the signal.
        start_idx: Start index of the gap.
        end_idx: End index of the gap (exclusive).
        context_window: Number of samples on each side to use for statistics.
        seed: Optional integer seed for the random-number generator.  Pass an
              explicit value to make gap-filling reproducible across runs.

    Returns:
        1D numpy array with the gap filled.
    """
    from scipy.signal import lfilter

    signal_cleaned = signal.copy()
    N = len(signal_cleaned)

    if start_idx >= end_idx or start_idx < 0 or end_idx > N:
        return signal_cleaned

    left_context = signal_cleaned[max(0, start_idx - context_window) : start_idx]
    right_context = signal_cleaned[end_idx : min(N, end_idx + context_window)]

    means = []
    mads = []

    if len(left_context) > 0:
        l_mean = np.median(left_context)
        means.append(l_mean)
        mads.append(np.median(np.abs(left_context - l_mean)))

    if len(right_context) > 0:
        r_mean = np.median(right_context)
        means.append(r_mean)
        mads.append(np.median(np.abs(right_context - r_mean)))

    if len(means) == 0:
        # Fall back to the gap itself if no surrounding context is available
        gap_data = signal_cleaned[start_idx:end_idx]
        if len(gap_data) == 0:
            return signal_cleaned
        target_mean = np.median(gap_data)
        mad = np.median(np.abs(gap_data - target_mean))
        target_std = 1.4826 * mad if mad > 0 else 1.0
    else:
        target_mean = np.mean(means)
        mad = np.mean(mads)
        target_std = 1.4826 * mad if mad > 0 else 1.0

    length = end_idx - start_idx
    r = 0.95
    red_noise = np.zeros(length)

    # Use a local Generator so we don't pollute / depend on global random state.
    # Passing seed=<int> makes gap-filling fully reproducible.
    rng = np.random.default_rng(seed)

    # Generate AR(1) red noise in chunks to bound memory on very large gaps.
    chunk_size = 1_000_000
    zi = np.array([0.0])  # initial filter state

    for chunk_start in range(0, length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, length)
        c_len = chunk_end - chunk_start

        white_noise = rng.standard_normal(c_len) * np.sqrt(1 - r**2)
        if chunk_start == 0:
            # Pre-scale the very first sample to the stationary std so the
            # AR(1) chain starts at its steady-state variance, not at 0.
            white_noise[0] = white_noise[0] / np.sqrt(1 - r**2)

        c_red, zf = lfilter([1], [1, -r], white_noise, zi=zi)
        red_noise[chunk_start:chunk_end] = c_red
        zi = zf

    # Standardise then rescale to match surrounding context statistics.
    rn_std = np.std(red_noise)
    if rn_std > 0:
        red_noise = (red_noise - np.mean(red_noise)) / rn_std
    red_noise = red_noise * target_std + target_mean

    # Boundary blend: cosine cross-fade instead of a linear ramp.
    # A linear ramp injects a deterministic trend whose power concentrates at
    # low frequencies, distorting the f^-2 spectral shape for long gaps.
    # The cosine fade limits the correction to a short fringe at each edge,
    # leaving the interior noise statistics intact.
    left_val = signal_cleaned[start_idx - 1] if start_idx > 0 else red_noise[0]
    right_val = signal_cleaned[end_idx] if end_idx < N else red_noise[-1]

    blend_len = min(32, length // 4)  # fade zone: up to 32 samples or 25% of gap

    if blend_len > 0:
        # Left fade: smoothly force red_noise[0] -> left_val over blend_len samples
        t_left = np.linspace(0.0, 1.0, blend_len)
        left_weight = 0.5 * (1.0 + np.cos(np.pi * t_left))  # 1 -> 0
        red_noise[:blend_len] = (
            left_weight * left_val + (1.0 - left_weight) * red_noise[:blend_len]
        )

        # Right fade: smoothly force red_noise[-1] -> right_val
        t_right = np.linspace(0.0, 1.0, blend_len)
        right_weight = 0.5 * (1.0 + np.cos(np.pi * t_right[::-1]))  # 0 -> 1
        red_noise[-blend_len:] = (
            right_weight * right_val + (1.0 - right_weight) * red_noise[-blend_len:]
        )

    signal_cleaned[start_idx:end_idx] = red_noise
    return signal_cleaned


def upsample_pchip(signal: np.ndarray, fs: float, factor: int = None) -> tuple[np.ndarray, float]:
    """
    Upsamples a signal using PCHIP interpolation.

    PCHIP is chosen to prevent overshoot (ringing artifacts) common in splines.

    Args:
        signal: 1D numpy array of the input signal.
        fs: Original sampling frequency in Hz.
        factor: Upsampling multiplier. Defaults to cfg.PCHIP_FACTOR.

    Returns:
        A tuple of (upsampled_signal, new_fs).
    """
    if factor is None:
        factor = cfg.PCHIP_FACTOR
    N = len(signal)
    t = np.arange(N) / fs
    
    new_N = N * factor
    t_new = np.linspace(t[0], t[-1], new_N)
    
    from scipy.interpolate import pchip_interpolate

    new_signal = pchip_interpolate(t, signal, t_new)
    new_fs = fs * factor
    
    return new_signal, new_fs


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Applies a stable bandpass filter with edge effect protection.

    Args:
        data: 1D numpy array of the input signal.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order. Defaults to 4.

    Returns:
        1D numpy array of the filtered signal.
    """
    from scipy.signal import butter, sosfiltfilt, detrend
    from scipy.signal.windows import tukey

    # 1. Remove DC offset first, then remove linear trend (baseline drift).
    #    Doing constant-detrend before linear-detrend avoids a residual step at
    #    the edges when the signal has a nonzero mean after slope removal.
    data_detrended = detrend(detrend(data, type='constant'), type='linear')

    # 2. Smoothly taper 5% at the edges to zero to avoid filter shock
    window = tukey(len(data_detrended), alpha=0.05)
    data_ready = data_detrended * window

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data_ready)


def compute_cwt_spectrogram(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    nv: int = None,
    use_ssq: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """
    Computes a Continuous Wavelet Transform (CWT) spectrogram based on the Morse wavelet.

    Executes at 1-second resolution without decimation and uses intelligent chunking
    to bound RAM usage. Applies a local Gaussian blur and converts magnitude to log scale (dB).

    Args:
        signal: 1D numpy array of the input signal.
        fs: Sampling frequency in Hz.
        lowcut: Lower frequency bound of interest in Hz.
        highcut: Upper frequency bound of interest in Hz.
        nv: Number of voices per octave. Defaults to cfg.CWT_NV.
        use_ssq: Whether to use Synchrosqueezing for enhanced time-frequency resolution.
        cancel_check: Optional callback to check for cancellation request.

    Returns:
        2D numpy array representing the spectrogram image in (time, frequency) orientation.
    """
    from ssqueezepy import ssq_cwt, cwt, Wavelet
    from ssqueezepy.utils import make_scales
    from ssqueezepy.wavelets import center_frequency
    from scipy.signal.windows import tukey
    from scipy.ndimage import gaussian_filter

    if nv is None:
        nv = cfg.CWT_NV_BUBBLES if lowcut >= 1.0/150.0 - 1e-6 else cfg.CWT_NV_CLOUDS
        
    N = len(signal)
    if N == 0:
        return np.zeros((1, 1))

    # Tukey window to suppress edge artifacts from abrupt signal ends
    window = tukey(N, alpha=cfg.TUKEY_ALPHA)
    processing_signal = signal * window

    # Setup Morse wavelet core with parameters from config
    morse_wavelet = Wavelet(('gmw', {'gamma': cfg.MORSE_GAMMA, 'beta': cfg.MORSE_BETA}))


    chunk_size = 32768
    overlap = 16384  # Heavily increased overlap to completely eliminate CWT boundary artifacts
    
    # Aggressive UI compression to prevent memory errors. 
    # Downsample time axis to max ~8000 pixels (plenty for any monitor).
    pool_size = max(1, N // 8000)

    def process_chunk(chunk_mag):
        """Applies blur, contrast mapping, and pooling to a single chunk."""
        # Local 2D Gaussian blur (boundary effects are tiny compared to overlap)
        chunk_mag = gaussian_filter(chunk_mag, sigma=(cfg.GAUSSIAN_SIGMA_FREQ, cfg.GAUSSIAN_SIGMA_TIME))
        if cfg.CWT_SHOW_LINEAR_AMP:
            return chunk_mag
        else:
            # Log contrast
            return 20 * np.log10(np.maximum(chunk_mag, 1e-12))

    if N <= chunk_size:
        # Run on full signal
        if use_ssq:
            # ssq_cwt always computes both the plain CWT (Wx) and the
            # synchrosqueezed transform (Tw).  We only use Tw for the
            # sharpened time-frequency image; Wx is intentionally discarded.
            Tw, Wx, freqs, _ = ssq_cwt(
                processing_signal,
                wavelet=morse_wavelet,
                nv=nv,
                fs=fs
            )
            Wx_abs = np.abs(Tw)
            valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
            if len(valid_idx) > 0:
                Wx_abs = Wx_abs[valid_idx, :]
            else:
                Wx_abs = np.zeros((1, N))
        else:
            scales = make_scales(N, wavelet=morse_wavelet, nv=nv)
            fc = center_frequency(morse_wavelet)
            freqs = fc / scales * fs
            valid_idx = (freqs >= lowcut) & (freqs <= highcut)
            if not np.any(valid_idx):
                Wx_abs = np.zeros((1, N))
            else:
                scales = scales[valid_idx].astype(np.float64)
                Wx, _scales = cwt(
                    processing_signal,
                    wavelet=morse_wavelet,
                    scales=scales,
                    fs=fs
                )
                Wx_abs = np.abs(Wx)
            
        Wx_db = process_chunk(Wx_abs)
        if pool_size > 1:
            pad_len = (pool_size - (Wx_db.shape[1] % pool_size)) % pool_size
            if pad_len > 0:
                pad_val = 0.0 if cfg.CWT_SHOW_LINEAR_AMP else -np.inf
                Wx_db = np.pad(Wx_db, ((0, 0), (0, pad_len)), constant_values=pad_val)
            Wx_db = Wx_db.reshape(Wx_db.shape[0], -1, pool_size).max(axis=2)
    else:
        # Chunked processing to bound RAM usage
        Wx_db_chunks = []
        step = chunk_size - overlap
        num_chunks = int(np.ceil((N - overlap) / step))
        
        valid_idx = None
        scales = None
        
        unpooled_buffer = None
        
        for i in range(num_chunks):
            if cancel_check and cancel_check():
                raise RuntimeError("Cancelled")
            start = i * step
            end = start + chunk_size
            
            chunk = processing_signal[start:end]
            actual_chunk_len = len(chunk)
            
            if actual_chunk_len < chunk_size:
                padded_chunk = np.zeros(chunk_size)
                padded_chunk[:actual_chunk_len] = chunk
                chunk = padded_chunk
            
            if use_ssq:
                # Wx (plain CWT) is discarded; only the synchrosqueezed Tw is used.
                Tw, _, freqs, _ = ssq_cwt(
                    chunk,
                    wavelet=morse_wavelet,
                    nv=nv,
                    fs=fs
                )
                Tw_abs = np.abs(Tw)
                
                if valid_idx is None:
                    valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
                
                if len(valid_idx) == 0:
                    chunk_mag = np.zeros((1, chunk_size))
                else:
                    chunk_mag = Tw_abs[valid_idx, :]
            else:
                if scales is None:
                    scales_all = make_scales(chunk_size, wavelet=morse_wavelet, nv=nv)
                    fc = center_frequency(morse_wavelet)
                    freqs = fc / scales_all * fs
                    valid_idx = (freqs >= lowcut) & (freqs <= highcut)
                    
                    if np.any(valid_idx):
                        scales = scales_all[valid_idx].astype(np.float64)
                    else:
                        scales = np.array([])
                
                if len(scales) == 0:
                    chunk_mag = np.zeros((1, chunk_size))
                else:
                    Wx, _scales = cwt(
                        chunk,
                        wavelet=morse_wavelet,
                        scales=scales,
                        fs=fs
                    )
                    chunk_mag = np.abs(Wx)
            
            # Process fully inside chunk!
            chunk_db = process_chunk(chunk_mag)
            
            keep_start = 0 if i == 0 else overlap // 2
            keep_end = actual_chunk_len if i == num_chunks - 1 else chunk_size - overlap // 2
            
            sliced_chunk = chunk_db[:, keep_start:keep_end]
            
            if pool_size > 1:
                if unpooled_buffer is not None:
                    sliced_chunk = np.concatenate([unpooled_buffer, sliced_chunk], axis=1)
                
                n_cols = sliced_chunk.shape[1]
                n_poolable = (n_cols // pool_size) * pool_size
                
                if n_poolable > 0:
                    poolable_db = sliced_chunk[:, :n_poolable]
                    unpooled_buffer = sliced_chunk[:, n_poolable:]
                    pooled = poolable_db.reshape(poolable_db.shape[0], -1, pool_size).max(axis=2)
                    Wx_db_chunks.append(pooled)
                else:
                    unpooled_buffer = sliced_chunk
            else:
                Wx_db_chunks.append(sliced_chunk)
                
        # Flush the final unpooled buffer seamlessly
        if pool_size > 1 and unpooled_buffer is not None and unpooled_buffer.shape[1] > 0:
            pad_len = pool_size - unpooled_buffer.shape[1]
            pad_val = 0.0 if cfg.CWT_SHOW_LINEAR_AMP else -np.inf
            padded_buffer = np.pad(unpooled_buffer, ((0, 0), (0, pad_len)), constant_values=pad_val)
            pooled = padded_buffer.reshape(padded_buffer.shape[0], -1, pool_size).max(axis=2)
            Wx_db_chunks.append(pooled)
            
        Wx_db = np.concatenate(Wx_db_chunks, axis=1)

    if cfg.CWT_SHOW_LINEAR_AMP:
        # Return transposed (time, frequency) directly
        return Wx_db.T
    else:
        # Dynamic Range Clipping in dB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if np.all(np.isnan(Wx_db)):
                max_db = 0.0
            else:
                if use_ssq:
                    # nanpercentile is extremely fast now since Wx_db is highly compressed
                    max_db = np.nanpercentile(Wx_db, 99.9)
                else:
                    max_db = np.nanmax(Wx_db)
            
        Wx_contrast = np.clip(Wx_db, a_min=max_db - cfg.CWT_DYNAMIC_RANGE_DB, a_max=max_db)
        
        # Return transposed (time, frequency) for pyqtgraph
        return Wx_contrast.T


def process_signal_pipeline(
    raw_signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    window_size: int = None,
    n_sigmas: float = None,
    apply_smoothing: bool = True,
    progress_callback: Optional[Callable[[int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    nv: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full processing pipeline for the interactive UI.

    Performs spike removal, Savitzky-Golay smoothing, PCHIP upsampling, bandpass filtering, 
    and CWT spectrogram computation with max pooling for UI performance.

    Args:
        raw_signal: 1D numpy array of the raw signal data.
        fs: Original sampling frequency in Hz.
        lowcut: Lower frequency bound in Hz.
        highcut: Upper frequency bound in Hz.
        window_size: Outlier window length. Defaults to cfg.DEFAULT_WINDOW_SIZE.
        n_sigmas: Outlier threshold. Defaults to cfg.DEFAULT_N_SIGMAS.
        apply_smoothing: Whether to apply Savitzky-Golay smoothing.
        progress_callback: Optional callback taking an integer percentage (0-100).

    Returns:
        A tuple of (downsampled_filtered_signal, spectrogram_image_data).
    """
    if progress_callback: progress_callback(5)
    
    # 1. Remove spikes (Hampel + Savitzky-Golay)
    cleaned_sig = clean_and_smooth_signal(
        raw_signal,
        window_size=window_size,
        n_sigmas=n_sigmas,
        apply_smoothing=apply_smoothing,
    )
    if progress_callback: progress_callback(20)
    
    # Performance shortcut for very long signals (e.g. Global View).
    # PCHIP upsampling is skipped above cfg.PCHIP_LONG_SIGNAL_THRESHOLD because
    # it is memory-intensive and the bandpass filter's effective resolution
    # degrades only modestly at the original fs for long observations.
    N_orig = len(raw_signal)
    actual_pchip_factor = cfg.PCHIP_FACTOR if N_orig <= cfg.PCHIP_LONG_SIGNAL_THRESHOLD else 1
    
    should_use_ssq = True
    actual_nv = nv if nv is not None else (cfg.CWT_NV_BUBBLES if lowcut >= 1.0/150.0 - 1e-6 else cfg.CWT_NV_CLOUDS)
    
    if cancel_check and cancel_check():
        raise RuntimeError("Cancelled")

    # 2. PCHIP Upsampling (skipped if actual_pchip_factor == 1)
    if actual_pchip_factor > 1:
        upsampled_sig, new_fs = upsample_pchip(cleaned_sig, fs, factor=actual_pchip_factor)
    else:
        upsampled_sig, new_fs = cleaned_sig, fs
        
    if progress_callback: progress_callback(30)
    if cancel_check and cancel_check():
        raise RuntimeError("Cancelled")
    
    # 3. Bandpass filtering at HIGH frequency (new_fs)
    filtered_sig = bandpass_filter(upsampled_sig, lowcut, highcut, new_fs)
    
    # Downsample back for 1D plot so time axis in UI doesn't stretch
    filtered_sig_downsampled = filtered_sig[::actual_pchip_factor]
    if progress_callback: progress_callback(40)
    if cancel_check and cancel_check():
        raise RuntimeError("Cancelled")
    
    # 4. Compute CWT at HIGH frequency (perfect phase detail)
    img_data = compute_cwt_spectrogram(filtered_sig, new_fs, lowcut, highcut, nv=actual_nv, use_ssq=should_use_ssq, cancel_check=cancel_check)
    if progress_callback: progress_callback(100)
    
    # Return directly, img_data is already optimally pooled internally!
    return filtered_sig_downsampled, img_data