import pandas as pd
import pyqtgraph as pg
import numpy as np

class TimeAxisItem(pg.AxisItem):
    """Custom X axis: show absolute time (HH:MM:SS) instead of seconds."""
    def __init__(self, start_datetime, datetime_series=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store PM6 start time as timestamp (seconds since 1970)
        self.start_timestamp = start_datetime.timestamp()
        self.datetime_series = datetime_series

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            try:
                if self.datetime_series is not None and len(self.datetime_series) > 0:
                    idx = int(round(v))
                    if 0 <= idx < len(self.datetime_series):
                        dt = pd.to_datetime(self.datetime_series[idx])
                    else:
                        # Extrapolate: find average sampling rate or use start_timestamp + v
                        dt = pd.to_datetime(self.start_timestamp + v, unit='s')
                else:
                    dt = pd.to_datetime(self.start_timestamp + v, unit='s')
                strings.append(dt.strftime('%H:%M:%S'))
            except (ValueError, OSError, IndexError, pd.errors.OutOfBoundsDatetime):
                strings.append("")
        return strings
