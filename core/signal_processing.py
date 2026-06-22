import numpy as np
from scipy.signal import butter, filtfilt
from ssqueezepy import ssq_stft




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




def compute_fsst_spectrogram(signal, fs):
    """
    Computes the FSST spectrogram (Synchrosqueezing) for the signal.
    Returns a matrix ready for rendering in PyQtGraph.
    """
    
    # Calculate the spectrogram
    Tx, _, freqs, _ = ssq_stft(signal, fs=fs)
    
    Tx_abs = np.abs(Tx)
    
    # Transpose the matrix (PyQtGraph expects data in the format X, Y)
    img_data = Tx_abs.T 
    
    return img_data