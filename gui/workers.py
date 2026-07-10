import traceback
from PySide6.QtCore import QThread, Signal

class SpectralAnalysisWorker(QThread):
    """
    Worker for running the full spectral-correlation analysis pipeline in a background thread.
    
    Emits:
        progress (int): Pipeline progress percentage (0-100).
        finished (dict): Completed spectral analysis results.
        error (str): Formatted traceback string on failure.
    """
    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, pm_signals, fs, signal_duration, bands, window_size, n_sigmas, apply_smoothing):
        super().__init__()
        self.pm_signals = pm_signals
        self.fs = fs
        self.signal_duration = signal_duration
        self.bands = bands
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.apply_smoothing = apply_smoothing

    def run(self):
        try:
            from core.spectral_analysis import run_spectral_pipeline
            band_results = {}
            for i, (band_key, lowcut, highcut) in enumerate(self.bands):
                min_period = 1.0 / lowcut
                if self.signal_duration < min_period:
                    band_results[band_key] = None
                    continue
                
                def prog_cb(val):
                    self.progress.emit((i * 100 + val) // len(self.bands))
                    
                res = run_spectral_pipeline(
                    self.pm_signals,
                    self.fs,
                    lowcut,
                    highcut,
                    self.window_size,
                    self.n_sigmas,
                    self.apply_smoothing,
                    progress_callback=prog_cb
                )
                band_results[band_key] = res
                
            self.progress.emit(100)
            self.finished.emit(band_results)
        except Exception:
            self.error.emit(traceback.format_exc())


class SignalAnalysisWorker(QThread):
    """
    Worker for running the standard CWT analysis pipeline in a background thread.

    Emits:
        progress (int): Pipeline progress percentage (0-100).
        finished (tuple): Returns (filtered_sig, img_data).
        error (str): Formatted traceback string on failure.
    """
    progress = Signal(int)
    finished = Signal(tuple)
    error = Signal(str)

    def __init__(self, raw_signal, fs, lowcut, highcut, window_size, n_sigmas, apply_smoothing):
        super().__init__()
        self.raw_signal = raw_signal
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.apply_smoothing = apply_smoothing

    def run(self):
        try:
            from core.signal_processing import process_signal_pipeline
            filtered_sig, img_data = process_signal_pipeline(
                self.raw_signal,
                self.fs,
                self.lowcut,
                self.highcut,
                self.window_size,
                self.n_sigmas,
                self.apply_smoothing,
                progress_callback=self.progress.emit
            )
            self.finished.emit((filtered_sig, img_data))
        except Exception:
            self.error.emit(traceback.format_exc())
