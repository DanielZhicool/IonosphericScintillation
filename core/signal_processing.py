import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.signal.windows import tukey
from ssqueezepy import ssq_stft

from core.config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_N_SIGMAS,
    SAVGOL_POLYORDER,
    DECIMATION_Q,
    LOW_FREQ_HIGHCUT,
    TUKEY_ALPHA,
    NFFT_CAP,
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
    # Convert to pandas Series for convenient rolling operations
    s = pd.Series(signal)
    
    # Step 1: Outlier detection (Hampel-like)
    # Compute rolling median
    rolling_median = s.rolling(window=window_size, center=True).median()
    
    # MAD scaled to std (factor 1.4826)
    rolling_mad = 1.4826 * (s - rolling_median).abs().rolling(window=window_size, center=True).median()
    
    # Identify points deviating from median by more than n_sigmas
    outliers = (s - rolling_median).abs() > (n_sigmas * rolling_mad)
    
    # Replace outliers with local median (leave other points untouched)
    cleaned_s = s.copy()
    cleaned_s[outliers] = rolling_median[outliers]
    
    # Fill edge NaNs from rolling by backward/forward fill
    cleaned_s = cleaned_s.bfill().ffill()
    cleaned_signal = cleaned_s.values
    
    # Step 2: Smoothing (Savitzky-Golay)
    if apply_smoothing:
        # Apply Savitzky-Golay smoothing
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




def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Applies a Butterworth bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)




def compute_fsst_spectrogram(signal, fs, lowcut, highcut):
    """
    Compute FSST spectrogram.
    For ultra-low frequencies uses decimation and a Tukey window to reduce edge effects.
    """
    N = len(signal)
    if N == 0:
        return np.zeros((1, 1))

    is_large_clouds = (highcut < LOW_FREQ_HIGHCUT)

    if is_large_clouds:
        # Downsampling (decimation)
        q = DECIMATION_Q
        if N > q * 4:  
            processing_signal = signal[::q]
            processing_fs = fs / q
        else:
            processing_signal = signal
            processing_fs = fs

        # Apply Tukey window to suppress edge artifacts (alpha=TUKEY_ALPHA)
        window = tukey(len(processing_signal), alpha=TUKEY_ALPHA)
        processing_signal = processing_signal * window
        # --------------------------------------------------

        # Choose n_fft with zero-padding (capped later)
        base_n_fft = int(2 ** np.ceil(np.log2(len(processing_signal))))
        if base_n_fft <= 1024:
            n_fft = base_n_fft * 8
        elif base_n_fft <= 4096:
            n_fft = base_n_fft * 4
        else:
            n_fft = base_n_fft 
            
        n_fft = min(n_fft, NFFT_CAP)
        Tx, _, freqs, _ = ssq_stft(processing_signal, fs=processing_fs, n_fft=n_fft)
    else:
        # For "Small bubbles", compute at original sampling rate
        Tx, _, freqs, _ = ssq_stft(signal, fs=fs)
        
    Tx_abs = np.abs(Tx)

    # Trim to requested frequency range
    valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
    
    if len(valid_idx) > 0:
        Tx_abs = Tx_abs[valid_idx, :]
    
    # Transpose for correct drawing in PyQtGraph
    return Tx_abs.T