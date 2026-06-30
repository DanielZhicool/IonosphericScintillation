import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt, savgol_filter, detrend
from scipy.signal.windows import tukey
from scipy.interpolate import pchip_interpolate
from scipy.ndimage import gaussian_filter
from ssqueezepy import ssq_cwt, cwt, Wavelet
from ssqueezepy.utils import make_scales
from ssqueezepy.wavelets import center_frequency  

import core.config as cfg


def clean_and_smooth_signal(signal, window_size=None, n_sigmas=None, apply_smoothing=True, polyorder=None):
    """
    Clean radio-astronomy signal:
    - Remove spikes/clusters (Hampel-like)
    - Optional Savitzky-Golay smoothing

    Params:
    signal (array): 1D numpy array
    window_size (int): odd window length for rolling operations
    n_sigmas (float): outlier threshold in sigma units
    apply_smoothing (bool): apply Savitzky-Golay smoothing
    """
    if window_size is None: window_size = cfg.DEFAULT_WINDOW_SIZE
    if n_sigmas is None: n_sigmas = cfg.DEFAULT_N_SIGMAS
    if polyorder is None: polyorder = cfg.SAVGOL_POLYORDER
    s = pd.Series(signal)
    
    # Step 1: Outlier detection (Hampel-like)
    rolling_median = s.rolling(window=window_size, center=True).median()
    rolling_mad = 1.4826 * (s - rolling_median).abs().rolling(window=window_size, center=True).median()
    outliers = (s - rolling_median).abs() > (n_sigmas * rolling_mad)
    
    cleaned_s = s.copy()
    cleaned_s[outliers] = rolling_median[outliers]
    
    cleaned_s = cleaned_s.bfill().ffill()
    cleaned_signal = cleaned_s.values
    
    # Step 2: Smoothing (Savitzky-Golay)
    if apply_smoothing:
        smooth_window = window_size if window_size % 2 != 0 else window_size + 1
        if smooth_window > 3:
            cleaned_signal = savgol_filter(cleaned_signal, window_length=smooth_window, polyorder=polyorder)

    return cleaned_signal


def fill_gap_with_red_noise(signal, start_idx, end_idx, context_window=200):
    """
    Replaces a segment of the signal [start_idx:end_idx] with generated red noise.
    The generated noise matches the mean and standard deviation of the surrounding context.
    It uses a linear correction (Brownian bridge) to perfectly seamlessly connect 
    both edges of the gap to the surrounding valid signal, preventing any visual jumps.
    """
    signal_cleaned = signal.copy()
    N = len(signal_cleaned)
    
    if start_idx >= end_idx or start_idx < 0 or end_idx > N:
        return signal_cleaned
        
    left_context = signal_cleaned[max(0, start_idx - context_window) : start_idx]
    right_context = signal_cleaned[end_idx : min(N, end_idx + context_window)]
    
    valid_context = np.concatenate([left_context, right_context])
    if len(valid_context) == 0:
        return signal_cleaned 
        
    target_mean = np.mean(valid_context)
    target_std = np.std(valid_context)
    
    length = end_idx - start_idx
    r = 0.95
    white_noise = np.random.normal(0, 1, length)
    red_noise = np.zeros(length)
    
    red_noise[0] = white_noise[0]
    for i in range(1, length):
        red_noise[i] = r * red_noise[i-1] + np.sqrt(1 - r**2) * white_noise[i]
        
    # Scale to target standard deviation and mean
    red_noise = (red_noise * target_std) + target_mean
    
    # Boundary constraints to make it perfectly seamless
    left_val = signal_cleaned[start_idx - 1] if start_idx > 0 else red_noise[0]
    right_val = signal_cleaned[end_idx] if end_idx < N else red_noise[-1]
    
    left_err = left_val - red_noise[0]
    right_err = right_val - red_noise[-1]
    
    # Linearly interpolate the offset so the edges lock in perfectly
    correction = np.linspace(left_err, right_err, length)
    red_noise += correction
    
    signal_cleaned[start_idx:end_idx] = red_noise
    return signal_cleaned


def upsample_pchip(signal, fs, factor=None):
    """
    Upsamples the signal using PCHIP interpolation.
    PCHIP is chosen to prevent overshoot (ringing artifacts) common in splines.
    Returns:
    upsampled_signal, new_fs
    """
    if factor is None:
        factor = cfg.PCHIP_FACTOR
    N = len(signal)
    t = np.arange(N) / fs
    
    new_N = N * factor
    t_new = np.linspace(t[0], t[-1], new_N)
    
    new_signal = pchip_interpolate(t, signal, t_new)
    new_fs = fs * factor
    
    return new_signal, new_fs


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Stable bandpass filter with edge effect protection.
    """
    # 1. Remove linear trend (baseline drift)
    data_detrended = detrend(data)
    
    # 2. Smoothly taper 5% at the edges to zero to avoid filter shock
    window = tukey(len(data_detrended), alpha=0.05)
    data_ready = data_detrended * window
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data_ready)


def compute_cwt_spectrogram(signal, fs, lowcut, highcut, nv=None, use_ssq=True):
    """
    Computes a Continuous Wavelet Transform (CWT) spectrogram
    based on the Morse wavelet (built into ssqueezepy).
    Executes at 1-second resolution without decimation.
    """
    if nv is None:
        nv = cfg.CWT_NV
        
    N = len(signal)
    if N == 0:
        return np.zeros((1, 1))

    # Tukey window to suppress edge artifacts from abrupt signal ends
    window = tukey(N, alpha=cfg.TUKEY_ALPHA)
    processing_signal = signal * window

    # Setup Morse wavelet core with parameters from config
    morse_wavelet = Wavelet(('gmw', {'gamma': cfg.MORSE_GAMMA, 'beta': cfg.MORSE_BETA}))

    print(f"[DEBUG] Running CWT with nv={nv}, gamma={cfg.MORSE_GAMMA}, beta={cfg.MORSE_BETA}, blur=({cfg.GAUSSIAN_SIGMA_FREQ}, {cfg.GAUSSIAN_SIGMA_TIME}), use_ssq={use_ssq}")

    if use_ssq:
        # Perform Synchrosqueezing (SWT) for sharp, high-res frequency representation
        # We MUST compute all scales for synchrosqueezing to correctly reassign energy
        Tw, Wx, freqs, _ = ssq_cwt(
            processing_signal,
            wavelet=morse_wavelet,
            nv=nv,
            fs=fs
        )
        Wx_abs = np.abs(Tw) # Use Synchrosqueezing matrix (Tw)
        
        # Extract requested frequency range after computation
        valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
        if len(valid_idx) > 0:
            Wx_abs = Wx_abs[valid_idx, :]
    else:
        # Generate all scales
        scales = make_scales(N, wavelet=morse_wavelet, nv=nv)
        
        # Filter scales to ONLY the frequency band of interest to save MASSIVE amounts of memory
        fc = center_frequency(morse_wavelet)
        freqs = fc / scales * fs
        valid_idx = (freqs >= lowcut) & (freqs <= highcut)
        scales = scales[valid_idx].astype(np.float64) # Ensure float64
        
        # Run standard CWT (skipping Synchrosqueezing step for stability on huge arrays)
        Wx, _scales = cwt(
            processing_signal,
            wavelet=morse_wavelet,
            scales=scales,
            fs=fs
        )
        Wx_abs = np.abs(Wx) # Use standard CWT magnitude

    # 2D Gaussian blur for visual smoothing
    Wx_abs = gaussian_filter(Wx_abs, sigma=(cfg.GAUSSIAN_SIGMA_FREQ, cfg.GAUSSIAN_SIGMA_TIME))

    # Apply Visual Contrast (Logarithmic Scale)
    Wx_abs = np.maximum(Wx_abs, 1e-12) # avoid log(0)
    Wx_db = 20 * np.log10(Wx_abs)
    
    if use_ssq:
        # Synchrosqueezing produces extreme edge artifacts. Use 99.9th percentile to prevent the dynamic range from being dragged up.
        max_db = np.nanpercentile(Wx_db, 99.9)
    else:
        # Standard CWT has natural variance. Absolute max preserves the gradient (greenish lines instead of saturated yellow).
        max_db = np.nanmax(Wx_db)
        
    Wx_contrast = np.clip(Wx_db, a_min=max_db - cfg.CWT_DYNAMIC_RANGE_DB, a_max=max_db)
    
    # Return transposed (time, frequency) for pyqtgraph
    return Wx_contrast.T


def process_signal_pipeline(raw_signal, fs, lowcut, highcut, window_size=None, n_sigmas=None, apply_smoothing=True, progress_callback=None):
    """
    Full processing pipeline for the UI:
    1. Clean and smooth
    2. Upsample
    3. Bandpass filter
    4. Compute CWT spectrogram
    5. Max pooling for UI performance
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
    
    # Performance shortcut for very long signals (e.g. Global View)
    N_orig = len(raw_signal)
    actual_pchip_factor = cfg.PCHIP_FACTOR if N_orig <= 50000 else 1
    
    # Cap nv to prevent MemoryError on huge datasets (like Global View)
    actual_nv = cfg.CWT_NV
    if N_orig > 50000:
        max_nv_for_ram = max(16, int(1e9 / (8 * N_orig * 5))) # rough estimate assuming 5 octaves
        actual_nv = min(cfg.CWT_NV, max_nv_for_ram)
    
    # 2. PCHIP Upsampling (skipped if actual_pchip_factor == 1)
    if actual_pchip_factor > 1:
        upsampled_sig, new_fs = upsample_pchip(cleaned_sig, fs, factor=actual_pchip_factor)
    else:
        upsampled_sig, new_fs = cleaned_sig, fs
        
    if progress_callback: progress_callback(30)
    
    # 3. Bandpass filtering at HIGH frequency (new_fs)
    filtered_sig = bandpass_filter(upsampled_sig, lowcut, highcut, new_fs)
    
    # Downsample back for 1D plot so time axis in UI doesn't stretch
    filtered_sig_downsampled = filtered_sig[::actual_pchip_factor]
    if progress_callback: progress_callback(40)
    
    # 4. Compute CWT at HIGH frequency (perfect phase detail)
    # Use Synchrosqueezing only for short signals, it's too unstable and memory intensive for the global view
    should_use_ssq = (N_orig <= 50000)
    img_data = compute_cwt_spectrogram(filtered_sig, new_fs, lowcut, highcut, nv=actual_nv, use_ssq=should_use_ssq)
    if progress_callback: progress_callback(85)
    
    # 5. Smart image compression (Max Pooling)
    # Protect against dimension errors: take exactly N_orig * actual_pchip_factor points
    img_data_exact = img_data[: N_orig * actual_pchip_factor, :]
    
    if actual_pchip_factor > 1:
        # Collapse (N_orig * actual_pchip_factor) rows into N_orig rows, taking the maximum pixel
        img_data_pooled = img_data_exact.reshape(N_orig, actual_pchip_factor, img_data_exact.shape[1]).max(axis=1)
    else:
        img_data_pooled = img_data_exact
    
    if progress_callback: progress_callback(100)
    
    return filtered_sig_downsampled, img_data_pooled