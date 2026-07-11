import pyqtgraph as pg
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QRectF

import core.config as cfg

from gui.constants import MAIN_TAB_NAME, RAW_DATA_TITLE_TEMPLATE, IONOSPHERIC_TITLE, SPECTROGRAM_TITLE
from gui.plotting import TimeAxisItem

class SignalTab(QWidget):
    """Widget for a signal tab; holds a DataFrame for a target."""
    def __init__(self, df_slice, start_datetime, tab_name=MAIN_TAB_NAME, fs=1.0, full_datetime_series=None):
        super().__init__()
        self.df_slice = df_slice.copy()
        self.time_sec = self.df_slice['Time_sec'].values
        self.fs = fs
        self.full_datetime_series = full_datetime_series if full_datetime_series is not None else self.df_slice['Datetime'].values
        self.start_datetime = start_datetime
        self.tab_name = tab_name  # remember tab name
        self.session_markers = []  # store session markers and labels
        
        self.current_channel = 'P1_20A'
        self.raw_signal = self.df_slice[self.current_channel].values

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        # Global metadata title above all graphs
        self.title_label = pg.LabelItem(justify='center', size='11pt')
        self.graph_widget.ci.addItem(self.title_label, row=0, col=0)
        self._update_title()

        # Raw data plot (tab name in title)
        axis_p1 = TimeAxisItem(self.start_datetime, datetime_series=self.full_datetime_series, orientation='bottom')
        self.p1 = self.graph_widget.addPlot(title=RAW_DATA_TITLE_TEMPLATE.format(tab=self.tab_name, channel=self.current_channel), axisItems={'bottom': axis_p1}, row=1, col=0)
        self.p1.showGrid(x=True, y=True)
        self.p1.setLabel('left', 'Amplitude')
        self.curve_raw = self.p1.plot(self.time_sec, self.raw_signal, pen='b')
        
        self.region = pg.LinearRegionItem()
        region_end = min(self.time_sec[0] + 100, self.time_sec[-1])
        self.region.setRegion([self.time_sec[0], region_end])
        self.region.setZValue(10)
        self.p1.addItem(self.region, ignoreBounds=True)

        # Ionospheric scintillations
        axis_p2 = TimeAxisItem(self.start_datetime, datetime_series=self.full_datetime_series, orientation='bottom')
        self.p2 = self.graph_widget.addPlot(title=IONOSPHERIC_TITLE, axisItems={'bottom': axis_p2}, row=2, col=0)
        self.p2.showGrid(x=True, y=True)
        self.p2.setXLink(self.p1)
        self.p2.setLabel('left', 'Amplitude')
        self.curve_filtered = self.p2.plot(pen='g')

        # Spectrogram
        axis_p3 = TimeAxisItem(self.start_datetime, datetime_series=self.full_datetime_series, orientation='bottom')
        self.p3 = self.graph_widget.addPlot(title=SPECTROGRAM_TITLE, axisItems={'bottom': axis_p3}, row=3, col=0)
        self.p3.setXLink(self.p1)
        if cfg.CWT_SHOW_PERIOD:
            self.p3.setLabel('left', 'Period', units='s')
        else:
            self.p3.setLabel('left', 'Frequency', units='Hz')
        
        self.img_spec = pg.ImageItem()
        self.p3.addItem(self.img_spec)
        
        cmap = pg.colormap.get('viridis')
        self.img_spec.setColorMap(cmap)
        
        self.cbar = pg.ColorBarItem(colorMap=cmap)
        self.cbar.setImageItem(self.img_spec, insert_in=self.p3)
        
        if cfg.CWT_SHOW_LINEAR_AMP:
            self.cbar.getAxis('right').setLabel('Amplitude')
        else:
            self.cbar.getAxis('right').setLabel('Power', units='dB')

    def _update_title(self):
        """Refresh the global metadata title at the top of the plot area."""
        date_str = self.start_datetime.strftime('%Y-%m-%d') if self.start_datetime else ''
        source = self.tab_name if self.tab_name != MAIN_TAB_NAME else 'Full Overview'
        ch = self.current_channel
        self.title_label.setText(
            f'<b>{source}</b>  |  {date_str}  |  Channel: {ch}',
            color='#CCCCCC'
        )

    def set_channel(self, channel_name, force=False):
        if not force and getattr(self, 'current_channel', None) == channel_name and self.curve_filtered.xData is not None and len(self.curve_filtered.xData) > 0:
            return

        self.current_channel = channel_name
        self.raw_signal = self.df_slice[channel_name].values

        # Update sub-plot title and global metadata title
        self.p1.setTitle(RAW_DATA_TITLE_TEMPLATE.format(tab=self.tab_name, channel=channel_name))
        self._update_title()
        self.curve_raw.setData(self.time_sec, self.raw_signal)
        
        # Clear the old analysis so it doesn't look mismatched while the worker calculates
        self.curve_filtered.setData([], [])
        self.img_spec.setImage(np.zeros((1, 1)), autoLevels=False)
        self.last_analysis_state = None

    def update_raw(self, df_updated):
        self.df_slice = df_updated.copy()
        self.time_sec = self.df_slice['Time_sec'].values
        self.full_datetime_series = self.df_slice['Datetime'].values
        # Update TimeAxisItem references
        for p in [self.p1, self.p2, self.p3]:
            axis = p.getAxis('bottom')
            if isinstance(axis, TimeAxisItem):
                axis.datetime_series = self.full_datetime_series
        self.set_channel(self.current_channel, force=True)

    def update_filtered(self, filtered_signal):
        self.curve_filtered.setData(self.time_sec, filtered_signal)
        self.p2.enableAutoRange(axis='y')
        self.p2.autoRange()

    def update_spectrogram(self, img_data, lowcut, highcut):
        img_min = np.nanmin(img_data)
        if cfg.CWT_SHOW_LINEAR_AMP:
            img_max = np.nanpercentile(img_data, 99.5) if not np.all(np.isnan(img_data)) else 1.0
        else:
            img_max = np.nanmax(img_data)
        
        if np.isnan(img_min) or np.isnan(img_max):
            img_min, img_max = 0.0, 1.0
            img_data = np.zeros_like(img_data)
            
        if img_max == img_min:
            img_max = img_min + 1e-6  # Prevent division by zero in colormap
        
        # Update Spectrogram Title with signal name
        source = self.tab_name if self.tab_name != 'Full Overview' else 'Full Overview'
        self.p3.setTitle(f"CWT Spectrogram - {source} ({self.current_channel})")
        
        # CRITICAL: The ColorBarItem owns the level state and overrides setImage(levels=...).
        # We must set levels on the colorbar explicitly or the first render will use its
        # default [0,1] range, causing solid purple (or yellow) until recalculate is pressed.
        self.cbar.setLevels((img_min, img_max))
        
        # Map image to its real range (Frequency vs. Period)
        if cfg.CWT_SHOW_PERIOD:
            img_data_to_plot = img_data  # No flip needed: short period is at bottom, long period at top
            y_min, y_max = 1.0 / highcut, 1.0 / lowcut
        else:
            img_data_to_plot = img_data[:, ::-1]  # Flip to put lowest frequency at bottom

        # Set image WITHOUT autoLevels so pyqtgraph doesn't interfere
        self.img_spec.setImage(img_data_to_plot, autoLevels=False)
        
        y_height = y_max - y_min
        self.img_spec.setRect(QRectF(self.time_sec[0], y_min, self.time_sec[-1]-self.time_sec[0], y_height))
        self.p3.setYRange(y_min, y_max)
