import pandas as pd
import pyqtgraph as pg
import numpy as np
from core.config import NFFT_CAP

class TimeAxisItem(pg.AxisItem):
    """Custom X axis: show absolute time (HH:MM:SS) instead of seconds."""
    def __init__(self, start_datetime, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store PM6 start time as timestamp (seconds since 1970)
        self.start_timestamp = start_datetime.timestamp()

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            try:
                dt = pd.to_datetime(self.start_timestamp + v, unit='s')
                strings.append(dt.strftime('%H:%M:%S'))
            except Exception:
                strings.append("")
        return strings


def choose_nfft(length):
    """Choose n_fft with zero-padding heuristics (capped at NFFT_CAP)."""
    if length <= 0:
        return 1024
    base_n_fft = int(2 ** np.ceil(np.log2(length)))
    if base_n_fft <= 1024:
        n_fft = base_n_fft * 8
    elif base_n_fft <= 4096:
        n_fft = base_n_fft * 4
    else:
        n_fft = base_n_fft
    return min(n_fft, NFFT_CAP)
