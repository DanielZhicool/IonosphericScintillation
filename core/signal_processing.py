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
    TUKEY_ALPHA
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


def upsample_pchip(signal, fs, factor=3):
    """
    Увеличивает частоту дискретизации (апсемплинг) с использованием PCHIP интерполяции.
    Сохраняет монотонность (в отличие от кубических сплайнов), 
    предотвращая появление искусственных выбросов на резких скачках сигнала.
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
    Стабильный полосовой фильтр с защитой от краевых эффектов.
    """
    # 1. Удаляем линейный тренд (спуск/подъем базовой линии)
    data_detrended = detrend(data)
    
    # 2. Мягко сводим к нулю 5% по краям, чтобы избежать "удара" по фильтру
    window = tukey(len(data_detrended), alpha=0.05)
    data_ready = data_detrended * window
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data_ready)




def compute_cwt_spectrogram(signal, fs, lowcut, highcut):
    """
    Вычисляет спектрограмму с помощью Непрерывного вейвлет-преобразования (CWT)
    на базе вейвлета Морзе (встроен в ssqueezepy).
    Выполняется на секундном разрешении без прореживания.
    """
    N = len(signal)
    if N == 0:
        return np.zeros((1, 1))

    # Окно Тьюки для подавления краевых артефактов обрыва сигнала
    window = tukey(N, alpha=TUKEY_ALPHA)
    processing_signal = signal * window

    # Выполняем CWT. 
    # Формируем ядро Морзе с параметрами из лога профессора:
    # Gamma = 3, Time-Bandwidth (Beta) = 60
    morse_wavelet = ('gmw', {'gamma': 3, 'beta': 60})

    # Выполняем Синхросквизинг (SWT) с 12 голосами на октаву (nv=12)
    # Возвращает: Tw (Синхросквизинг), Wx (Обычный CWT), freqs
    Tw, Wx, freqs, _ = ssq_cwt(
        processing_signal, 
        wavelet=morse_wavelet, 
        nv=128, 
        fs=fs
    )
    
    # Берем матрицу Tw (Синхросквизинг / SWT) для отрисовки
    Tw_abs = np.abs(Tw)

    # 2. Имитируем движок отрисовки MATLAB (создаем тот самый эффект "посередине").
    # Gaussian filter мягко сплавляет 1-пиксельные ступеньки между собой.
    # sigma=(по вертикали, по горизонтали). 
    # Размываем по частоте (вертикали) сильнее, чтобы "залечить" разорванные лесенки.
    Wx_abs = gaussian_filter(Tw_abs, sigma=(2.5, 1.0))
    
    # Вырезаем запрошенный частотный диапазон
    valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
    
    if len(valid_idx) > 0:
        Wx_abs = Wx_abs[valid_idx, :]
        
    # --- Визуальный контраст (Логарифмическая шкала) ---
    Wx_abs = np.maximum(Wx_abs, 1e-12)
    Wx_db = 20 * np.log10(Wx_abs)
    
    dynamic_range = 40.0
    max_db = np.max(Wx_db)
    Wx_contrast = np.clip(Wx_db, a_min=max_db - dynamic_range, a_max=max_db)
    
    return Wx_contrast.T