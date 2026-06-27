import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt, savgol_filter, detrend
from scipy.signal.windows import tukey
from scipy.interpolate import pchip_interpolate
from scipy.ndimage import gaussian_filter
from ssqueezepy import ssq_cwt  

from core.config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_N_SIGMAS,
    SAVGOL_POLYORDER,
    TUKEY_ALPHA,
    PCHIP_FACTOR,
    CWT_NV,
    MORSE_GAMMA,
    MORSE_BETA,
    GAUSSIAN_SIGMA_FREQ,
    GAUSSIAN_SIGMA_TIME,
    CWT_DYNAMIC_RANGE_DB
)


def clean_and_smooth_signal(signal, window_size=DEFAULT_WINDOW_SIZE, n_sigmas=DEFAULT_N_SIGMAS, apply_smoothing=True, polyorder=SAVGOL_POLYORDER):
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


def fill_gap_with_red_noise(signal, start_idx, end_idx, context_window=50):
    """Replaces the segment with red noise based on surrounding context."""
    left_context = signal[max(0, start_idx - context_window) : start_idx]
    right_context = signal[end_idx : min(len(signal), end_idx + context_window)]
    
    valid_context = np.concatenate([left_context, right_context])
    if len(valid_context) == 0:
        return signal 
        
    target_mean = np.mean(valid_context)
    target_std = np.std(valid_context)
    
    length = end_idx - start_idx
    r = 0.95
    white_noise = np.random.normal(0, 1, length)
    red_noise = np.zeros(length)
    
    red_noise[0] = white_noise[0]
    for i in range(1, length):
        red_noise[i] = r * red_noise[i-1] + np.sqrt(1 - r**2) * white_noise[i]
        
    red_noise = red_noise - np.mean(red_noise)
    red_noise = (red_noise / (np.std(red_noise) + 1e-9)) * target_std
    red_noise = red_noise + target_mean
    
    signal_cleaned = signal.copy()
    signal_cleaned[start_idx:end_idx] = red_noise
    return signal_cleaned


def upsample_pchip(signal, fs, factor=PCHIP_FACTOR):
    """
    Increases the sampling rate (upsampling) using PCHIP interpolation.
    Preserves monotonicity (unlike cubic splines), preventing artificial overshoots on sharp signal drops.
    """
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


def compute_cwt_spectrogram(signal, fs, lowcut, highcut):
    """
    Computes a spectrogram using Continuous Wavelet Transform (CWT)
    based on the Morse wavelet (built into ssqueezepy).
    Executes at 1-second resolution without decimation.
    """
    N = len(signal)
    if N == 0:
        return np.zeros((1, 1))

    # Tukey window to suppress edge artifacts from abrupt signal ends
    window = tukey(N, alpha=TUKEY_ALPHA)
    processing_signal = signal * window

    # Setup Morse wavelet core with parameters from config
    morse_wavelet = ('gmw', {'gamma': MORSE_GAMMA, 'beta': MORSE_BETA})

    # Perform Synchrosqueezing (SWT)
    # Returns: Tw (Synchrosqueezing), Wx (Standard CWT), freqs
    Tw, Wx, freqs, _ = ssq_cwt(
        processing_signal, 
        wavelet=morse_wavelet, 
        nv=CWT_NV, 
        fs=fs
    )
    
    # Use Tw matrix (Synchrosqueezing / SWT) for rendering
    Tw_abs = np.abs(Tw)

    # Gaussian filter smoothly blends pixels vertically to heal broken lines
    Wx_abs = gaussian_filter(Tw_abs, sigma=(GAUSSIAN_SIGMA_FREQ, GAUSSIAN_SIGMA_TIME))
    
    # Extract requested frequency range
    valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
    
    if len(valid_idx) > 0:
        Wx_abs = Wx_abs[valid_idx, :]
        
    # Apply Visual Contrast (Logarithmic Scale)
    Wx_abs = np.maximum(Wx_abs, 1e-12)
    Wx_db = 20 * np.log10(Wx_abs)
    
    max_db = np.max(Wx_db)
    Wx_contrast = np.clip(Wx_db, a_min=max_db - CWT_DYNAMIC_RANGE_DB, a_max=max_db)
    
    return Wx_contrast.T


def process_signal_pipeline(raw_signal, fs, lowcut, highcut, window_size, n_sigmas, apply_smoothing):
    """
    Full processing pipeline for the UI:
    1. Clean and smooth
    2. Upsample
    3. Bandpass filter
    4. Compute CWT spectrogram
    5. Max pooling for UI performance
    """
    # 1. Remove spikes (Hampel + Savitzky-Golay)
    cleaned_sig = clean_and_smooth_signal(
        raw_signal,
        window_size=window_size,
        n_sigmas=n_sigmas,
        apply_smoothing=apply_smoothing,
    )
    
    # 2. PCHIP Upsampling
    upsampled_sig, new_fs = upsample_pchip(cleaned_sig, fs, factor=PCHIP_FACTOR)
    
    # 3. Bandpass filtering at HIGH frequency (new_fs)
    filtered_sig = bandpass_filter(upsampled_sig, lowcut, highcut, new_fs)
    
    # Downsample back for 1D plot so time axis in UI doesn't stretch
    filtered_sig_downsampled = filtered_sig[::PCHIP_FACTOR]
    
    # 4. Compute CWT at HIGH frequency (perfect phase detail)
    img_data = compute_cwt_spectrogram(filtered_sig, new_fs, lowcut, highcut)
    
    # 5. Smart image compression (Max Pooling)
    N_orig = len(raw_signal)
    
    # Protect against dimension errors: take exactly N_orig * PCHIP_FACTOR points
    img_data_exact = img_data[: N_orig * PCHIP_FACTOR, :]
    
    # Collapse (N_orig * PCHIP_FACTOR) rows into N_orig rows, taking the maximum pixel
    img_data_pooled = img_data_exact.reshape(N_orig, PCHIP_FACTOR, img_data_exact.shape[1]).max(axis=1)
    
    return filtered_sig_downsampled, img_data_pooled