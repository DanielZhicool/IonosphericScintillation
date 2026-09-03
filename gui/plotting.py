from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg


class TimeAxisItem(pg.AxisItem):
    """Custom X axis: show absolute time (HH:MM:SS) instead of seconds."""

    def __init__(
        self,
        start_datetime: datetime,
        datetime_series: Sequence[Any] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Store PM6 start time as timestamp (seconds since 1970)
        self.start_timestamp: float = start_datetime.timestamp()
        self.datetime_series: Sequence[Any] | np.ndarray | None = datetime_series

    def tickStrings(self, values: Sequence[float], scale: float, spacing: float) -> list[str]:
        strings: list[str] = []
        for v in values:
            try:
                if self.datetime_series is not None and len(self.datetime_series) > 0:
                    idx = int(round(v))
                    if 0 <= idx < len(self.datetime_series):
                        dt = pd.to_datetime(self.datetime_series[idx])
                    else:
                        # Extrapolate: find average sampling rate or use start_timestamp + v
                        dt = pd.to_datetime(self.start_timestamp + v, unit="s")
                else:
                    dt = pd.to_datetime(self.start_timestamp + v, unit="s")
                strings.append(dt.strftime("%H:%M:%S"))
            except (ValueError, OSError, IndexError, pd.errors.OutOfBoundsDatetime):
                strings.append("")
        return strings
