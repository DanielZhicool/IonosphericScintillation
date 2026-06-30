import pyqtgraph as pg
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QRectF

from gui.constants import MAIN_TAB_NAME, RAW_DATA_TITLE_TEMPLATE, IONOSPHERIC_TITLE, SPECTROGRAM_TITLE
from gui.plotting import TimeAxisItem

class SignalTab(QWidget):
    """Widget for a signal tab; holds a DataFrame for a target."""
    def __init__(self, df_slice, start_datetime, tab_name=MAIN_TAB_NAME, fs=1.0):
        super().__init__()
        self.df_slice = df_slice.copy()
        self.time_sec = self.df_slice['Time_sec'].values
        self.fs = fs
        self.start_datetime = start_datetime
        self.tab_name = tab_name  # remember tab name
        self.session_markers = []  # store session markers and labels
        
        self.current_channel = 'P1_20A'
        self.raw_signal = self.df_slice[self.current_channel].values

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        # Raw data plot (tab name in title)
        axis_p1 = TimeAxisItem(self.start_datetime, orientation='bottom')
        self.p1 = self.graph_widget.addPlot(title=RAW_DATA_TITLE_TEMPLATE.format(tab=self.tab_name, channel=self.current_channel), axisItems={'bottom': axis_p1})
        self.p1.showGrid(x=True, y=True)
        self.p1.setLabel('left', 'Amplitude')
        self.curve_raw = self.p1.plot(self.time_sec, self.raw_signal, pen='b')
        
        self.region = pg.LinearRegionItem()
        region_end = min(self.time_sec[0] + 100, self.time_sec[-1])
        self.region.setRegion([self.time_sec[0], region_end])
        self.region.setZValue(10)
        self.p1.addItem(self.region, ignoreBounds=True)

        self.graph_widget.nextRow()
        
        # Ionospheric scintillations
        axis_p2 = TimeAxisItem(self.start_datetime, orientation='bottom')
        self.p2 = self.graph_widget.addPlot(title=IONOSPHERIC_TITLE, axisItems={'bottom': axis_p2})
        self.p2.showGrid(x=True, y=True)
        self.p2.setXLink(self.p1)
        self.p2.setLabel('left', 'Amplitude')
        self.curve_filtered = self.p2.plot(pen='g')

        self.graph_widget.nextRow()

        # Spectrogram
        axis_p3 = TimeAxisItem(self.start_datetime, orientation='bottom')
        self.p3 = self.graph_widget.addPlot(title=SPECTROGRAM_TITLE, axisItems={'bottom': axis_p3})
        self.p3.setXLink(self.p1)
        self.p3.setLabel('left', 'Frequency', units='Hz')
        
        self.img_spec = pg.ImageItem()
        self.p3.addItem(self.img_spec)
        
        cmap = pg.colormap.get('viridis')
        self.img_spec.setColorMap(cmap)
        
        self.cbar = pg.ColorBarItem(colorMap=cmap)
        self.cbar.setImageItem(self.img_spec, insert_in=self.p3)
        self.cbar.getAxis('right').setLabel('Power', units='dB')

    def set_channel(self, channel_name):
        self.current_channel = channel_name
        self.raw_signal = self.df_slice[channel_name].values
            
        # Update title keeping tab name
        self.p1.setTitle(RAW_DATA_TITLE_TEMPLATE.format(tab=self.tab_name, channel=channel_name))
        self.curve_raw.setData(self.time_sec, self.raw_signal)

    def update_raw(self, df_updated):
        self.df_slice = df_updated.copy()
        self.set_channel(self.current_channel)

    def update_filtered(self, filtered_signal):
        self.curve_filtered.setData(self.time_sec, filtered_signal)

    def update_spectrogram(self, img_data, lowcut, highcut):
        img_min = np.nanmin(img_data)
        img_max = np.nanmax(img_data)
        
        if np.isnan(img_min) or np.isnan(img_max):
            img_min, img_max = 0.0, 1.0
            img_data = np.zeros_like(img_data)
            
        if img_max == img_min:
            img_max = img_min + 1e-6  # Prevent division by zero in colormap
        
        # Set image WITHOUT autoLevels so pyqtgraph doesn't interfere
        self.img_spec.setImage(img_data, autoLevels=False)
        
        # CRITICAL: The ColorBarItem owns the level state and overrides setImage(levels=...).
        # We must set levels on the colorbar explicitly or the first render will use its
        # default [0,1] range, causing solid purple (or yellow) until recalculate is pressed.
        self.cbar.setLevels((img_min, img_max))
        
        # Map image to its real frequency range
        freq_height = highcut - lowcut
        self.img_spec.setRect(QRectF(self.time_sec[0], lowcut, self.time_sec[-1]-self.time_sec[0], freq_height))
        self.p3.setYRange(lowcut, highcut)
