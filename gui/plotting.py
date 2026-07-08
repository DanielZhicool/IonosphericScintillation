import pandas as pd
import pyqtgraph as pg
import numpy as np

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
            except (ValueError, OSError, pd.errors.OutOfBoundsDatetime):
                strings.append("")
        return strings
