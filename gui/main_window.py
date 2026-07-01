import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QComboBox, QSplitter, QMessageBox, QTabWidget,
                               QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QProgressBar)
from PySide6.QtCore import Qt
from gui.constants import (
    APP_TITLE, BTN_APPLY_NOISE, BTN_ANALYZE, BTN_EXPORT, BTN_LOAD_LOGS, BTN_LOAD_PM6,
    BTN_SPECTRAL, BTN_GLOBAL_SPECTRAL, GLOBAL_SPECTRAL_TAB_NAME,
    CHECK_MARKERS, CHECK_SMOOTH, COMBO_BAND_ITEMS, FILTER_PM6,
    FILTER_TEXT_FILES, LABEL_BAND, LABEL_CLEANING_HEADER,
    LABEL_DISPLAY_CHANNEL, LABEL_SIGMA, LABEL_WINDOW,
    LBL_LOADED_SAMPLES, LBL_PM6_NOT_LOADED, MAIN_TAB_NAME,
    MSG_ANALYSIS_DATA_TOO_SHORT_TEXT,
    MSG_ANALYSIS_DATA_TOO_SHORT_TITLE, MSG_ANALYSIS_ERROR_TITLE,
    MSG_LOG_PROCESSING_ERROR_TITLE, MSG_LOG_PROCESSING_RESULT_TITLE,
    MSG_NO_LOG_EVENTS_TEXT, MSG_NO_LOG_EVENTS_TITLE,
    MSG_NO_PM6_SELECTED_TEXT, MSG_NO_PM6_SELECTED_TITLE,
    MSG_OPEN_ERROR_TITLE, MSG_SAVE_ERROR_TITLE,
    MSG_SKIPPED_TABS_TEMPLATE, MSG_TAB_CREATION_NONE,
    MSG_SPECTRAL_NO_SOURCE_TITLE, MSG_SPECTRAL_NO_SOURCE_TEXT,
    MSG_SPECTRAL_ERROR_TITLE, SPECTRAL_TAB_SUFFIX,
    FILE_DIALOG_PM6_TITLE, FILE_DIALOG_REGI_TITLE,
    SAVE_FILE_DEFAULT_NAME, SAVE_FILE_FILTER,
    MSG_CREATED_TABS_TEMPLATE, CHANNELS
)

# Core parsers and signal processing
from core.parsers import load_pm6_data, parse_regi_with_time, build_observation_sessions
from core.signal_processing import fill_gap_with_red_noise, process_signal_pipeline

from gui.plotting import TimeAxisItem
from gui.tabs import SignalTab
from gui.spectral_tab import SpectralTab
from gui.settings_tab import SettingsDialog
from gui.workers import SpectralAnalysisWorker, SignalAnalysisWorker
from core.spectral_analysis import run_spectral_pipeline


class Uran4App(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1300, 850)
        self.fs = 1.0 
        
        self.df_pm6 = None
        self.full_time = None
        self.pm6_start_dt = None
        self.sessions = []
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 1. Load data
        self.btn_load_pm6 = QPushButton(BTN_LOAD_PM6)
        self.btn_load_pm6.clicked.connect(self.load_pm6)
        self.lbl_status = QLabel(LBL_PM6_NOT_LOADED)

        self.combo_channel = QComboBox()
        self.combo_channel.addItems(CHANNELS)
        self.combo_channel.currentTextChanged.connect(self.change_active_channel)
        self.combo_channel.setEnabled(False)

        self.btn_load_logs = QPushButton(BTN_LOAD_LOGS)
        self.btn_load_logs.clicked.connect(self.auto_clean_and_split)
        self.btn_load_logs.setEnabled(False)

        self.check_markers = QCheckBox(CHECK_MARKERS)
        self.check_markers.setChecked(True)
        self.check_markers.stateChanged.connect(self.toggle_markers)
        self.check_markers.setEnabled(False)
        
        self.check_day_markers = QCheckBox("Show Day Markers")
        self.check_day_markers.setChecked(True)
        self.check_day_markers.stateChanged.connect(self.toggle_markers)
        self.check_day_markers.setEnabled(False)
        
        # Noise cleaning controls
        self.spin_window = QSpinBox()
        self.spin_window.setRange(3, 99)
        self.spin_window.setValue(15)
        self.spin_window.setSingleStep(2)  # step=2 to favor odd values
        self.spin_window.valueChanged.connect(self.run_analysis)
        self.spin_window.setEnabled(False)

        self.spin_sigmas = QDoubleSpinBox()
        self.spin_sigmas.setRange(1.0, 10.0)
        self.spin_sigmas.setValue(3.0)
        self.spin_sigmas.setSingleStep(0.5)
        self.spin_sigmas.valueChanged.connect(self.run_analysis)
        self.spin_sigmas.setEnabled(False)

        self.check_smooth = QCheckBox(CHECK_SMOOTH)
        self.check_smooth.setChecked(True)
        self.check_smooth.stateChanged.connect(self.run_analysis)
        self.check_smooth.setEnabled(False)

        self.btn_apply_noise = QPushButton(BTN_APPLY_NOISE)
        self.btn_apply_noise.clicked.connect(self.manual_clean_region)
        self.btn_apply_noise.setEnabled(False)

        self.combo_band = QComboBox()
        self.combo_band.addItems(COMBO_BAND_ITEMS)
        self.combo_band.currentIndexChanged.connect(self.run_analysis)

        self.btn_analyze = QPushButton(BTN_ANALYZE)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setEnabled(False)
        
        self.btn_export = QPushButton(BTN_EXPORT)
        self.btn_export.clicked.connect(self.export_plots)
        self.btn_export.setEnabled(False)

        self.btn_batch_export = QPushButton("Batch Export...")
        self.btn_batch_export.clicked.connect(self.open_batch_export)
        self.btn_batch_export.setEnabled(False)

        self.btn_spectral = QPushButton(BTN_SPECTRAL)
        self.btn_spectral.clicked.connect(self.run_spectral_analysis)
        self.btn_spectral.setEnabled(False)

        self.btn_global_spectral = QPushButton(BTN_GLOBAL_SPECTRAL)
        self.btn_global_spectral.clicked.connect(self.run_global_spectral_analysis)
        self.btn_global_spectral.setEnabled(False)

        # --- Tooltips ---
        self.btn_load_pm6.setToolTip('Load the PM6 binary data file (.pm6)')
        self.btn_load_logs.setToolTip('Load REGI log files to auto-detect observation sessions and create source tabs')
        self.combo_channel.setToolTip('Select which receiver channel to display and analyze')
        self.check_markers.setToolTip('Show/hide observation session markers on the Full Overview')
        self.spin_window.setToolTip('Rolling window size (samples) for outlier detection and smoothing')
        self.spin_sigmas.setToolTip('Outlier rejection threshold in standard deviations (sigma)')
        self.check_smooth.setToolTip('Apply Savitzky-Golay smoothing after outlier removal')
        self.btn_apply_noise.setToolTip('Replace the selected region (shaded area) with synthetic red noise')
        self.combo_band.setToolTip('Frequency band for bandpass filtering and CWT spectrogram')
        self.btn_analyze.setToolTip('Recalculate bandpass filter and spectrogram for the current tab')
        self.btn_export.setToolTip('Export the current tab plots to a PNG image file')
        self.btn_batch_export.setToolTip('Export multiple sources, dates, and channels to a folder')
        self.btn_spectral.setToolTip('Run full spectral-correlation analysis on the current source transit')
        self.btn_global_spectral.setToolTip('Run spectral analysis across all loaded source transits')

        # --- Layout Assembly with Group Boxes ---
        
        # 1. Data & Sessions
        group_data = QGroupBox("1. Data & Sessions")
        layout_data = QVBoxLayout()
        layout_data.addWidget(self.btn_load_pm6)
        layout_data.addWidget(self.lbl_status)
        layout_data.addWidget(self.btn_load_logs)
        group_data.setLayout(layout_data)
        
        # 2. View Options
        group_view = QGroupBox("2. View Options")
        layout_view = QVBoxLayout()
        layout_view.addWidget(QLabel(LABEL_DISPLAY_CHANNEL))
        layout_view.addWidget(self.combo_channel)
        layout_view.addWidget(self.check_markers)
        layout_view.addWidget(self.check_day_markers)
        group_view.setLayout(layout_view)
        
        # 3. Signal Processing
        group_proc = QGroupBox("3. Signal Processing")
        layout_proc = QVBoxLayout()
        layout_proc.addWidget(QLabel("Filter window size (samples):"))
        layout_proc.addWidget(self.spin_window)
        layout_proc.addWidget(QLabel("Outlier threshold (sigma):"))
        layout_proc.addWidget(self.spin_sigmas)
        layout_proc.addWidget(self.check_smooth)
        layout_proc.addWidget(self.btn_apply_noise)
        layout_proc.addSpacing(10)
        layout_proc.addWidget(QLabel(LABEL_BAND))
        layout_proc.addWidget(self.combo_band)
        layout_proc.addWidget(self.btn_analyze)
        group_proc.setLayout(layout_proc)
        
        # 4. Spectral Analysis
        group_spectral = QGroupBox("4. Spectral Analysis")
        layout_spectral = QVBoxLayout()
        layout_spectral.addWidget(self.btn_spectral)
        layout_spectral.addWidget(self.btn_global_spectral)
        group_spectral.setLayout(layout_spectral)
        
        # 5. Export
        group_export = QGroupBox("5. Export")
        layout_export = QVBoxLayout()
        layout_export.addWidget(self.btn_export)
        layout_export.addWidget(self.btn_batch_export)
        group_export.setLayout(layout_export)
        
        # Add groups to main control layout
        control_layout.addWidget(group_data)
        control_layout.addWidget(group_view)
        control_layout.addWidget(group_proc)
        control_layout.addWidget(group_spectral)
        control_layout.addWidget(group_export)
        control_layout.addStretch()

        # 6. Settings & Status
        group_settings = QGroupBox("6. Settings & Status")
        layout_settings = QVBoxLayout()
        self.btn_settings = QPushButton("⚙ Settings")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_settings.setToolTip('Edit CWT, spectral, and processing hyperparameters')
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        layout_settings.addWidget(self.btn_settings)
        layout_settings.addWidget(self.progress_bar)
        group_settings.setLayout(layout_settings)
        control_layout.addWidget(group_settings)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        splitter.addWidget(control_panel)
        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 1000])

    def get_active_tab(self):
        current_widget = self.tabs.currentWidget()
        if isinstance(current_widget, SignalTab):
            # This is the main tab (Full overview)
            return current_widget
        elif isinstance(current_widget, QTabWidget):
            # This is a specific day's tab. Get the active plot inside it.
            inner = current_widget.currentWidget()
            if isinstance(inner, SignalTab):
                return inner
        return None

    def on_tab_changed(self, index):
        if index >= 0 and self.df_pm6 is not None:
            active_tab = self.get_active_tab()
            if active_tab:
                active_tab.set_channel(self.combo_channel.currentText())
            self.run_analysis()

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec()

    def change_active_channel(self, channel_name):
        active_tab = self.get_active_tab()
        if active_tab:
            active_tab.set_channel(channel_name)
            self.run_analysis()

    def set_widgets_enabled(self, enabled=True):
        """Enable or disable the cleaning and export controls."""
        self.btn_load_pm6.setEnabled(enabled)
        self.combo_channel.setEnabled(enabled)
        self.btn_load_logs.setEnabled(enabled)
        self.check_markers.setEnabled(enabled)
        self.check_day_markers.setEnabled(enabled)
        self.btn_apply_noise.setEnabled(enabled)
        self.btn_analyze.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.btn_batch_export.setEnabled(enabled)
        self.spin_window.setEnabled(enabled)
        self.spin_sigmas.setEnabled(enabled)
        self.check_smooth.setEnabled(enabled)
        self.btn_spectral.setEnabled(enabled)
        self.btn_global_spectral.setEnabled(enabled)

    def load_pm6(self):
        filepath, _ = QFileDialog.getOpenFileName(self, FILE_DIALOG_PM6_TITLE, "", FILTER_PM6)
        if not filepath:
            return
            
        try:
            self.df_pm6 = load_pm6_data(filepath)
            self.full_time = self.df_pm6['Time_sec'].values
            
            # Extract exact PM6 start datetime
            self.pm6_start_dt = self.df_pm6['Datetime'].iloc[0]
            
            self.tabs.clear()
            
            # Pass pm6_start_dt into the tab
            main_tab = SignalTab(self.df_pm6, self.pm6_start_dt, tab_name=MAIN_TAB_NAME, fs=self.fs)
            self.tabs.addTab(main_tab, MAIN_TAB_NAME)
            self.tabs.setCurrentWidget(main_tab)
            
            self.lbl_status.setText(LBL_LOADED_SAMPLES.format(len(self.df_pm6)))
            self.set_widgets_enabled(True)
            self.run_analysis()
        except Exception as e:
            QMessageBox.critical(self, MSG_OPEN_ERROR_TITLE, str(e))

    def auto_clean_and_split(self):
        if self.df_pm6 is None:
            QMessageBox.warning(self, MSG_NO_PM6_SELECTED_TITLE, MSG_NO_PM6_SELECTED_TEXT)
            return

        filepath, _ = QFileDialog.getOpenFileName(self, FILE_DIALOG_REGI_TITLE, "", FILTER_TEXT_FILES)
        if not filepath:
            return

        try:
            pm6_start_dt = self.df_pm6['Datetime'].iloc[0]
            df_logs = parse_regi_with_time(filepath, pm6_start_dt)
            
            if df_logs.empty:
                QMessageBox.warning(self, MSG_NO_LOG_EVENTS_TITLE, MSG_NO_LOG_EVENTS_TEXT)
                return

            pm6_max_sec = self.full_time[-1]
            df_logs, calibrations, self.sessions = build_observation_sessions(df_logs, pm6_max_sec)
            
            for _, row in calibrations.iterrows():
                s_idx = np.searchsorted(self.full_time, row['Start_sec'])
                e_idx = np.searchsorted(self.full_time, row['End_sec'])
                if s_idx < e_idx:
                    for col in CHANNELS:
                        self.df_pm6[col] = fill_gap_with_red_noise(self.df_pm6[col].values, s_idx, e_idx)

            main_tab = self.tabs.widget(0)
            main_tab.update_raw(self.df_pm6)

            for item in main_tab.session_markers:
                if item in main_tab.p1.items:
                    main_tab.p1.removeItem(item)
                if item in main_tab.p2.items:
                    main_tab.p2.removeItem(item)
                if item in main_tab.p3.items:
                    main_tab.p3.removeItem(item)
            main_tab.session_markers.clear()
            
            if hasattr(main_tab, 'day_markers'):
                for item in main_tab.day_markers:
                    if item in main_tab.p1.items:
                        main_tab.p1.removeItem(item)
                    if item in main_tab.p2.items:
                        main_tab.p2.removeItem(item)
                    if item in main_tab.p3.items:
                        main_tab.p3.removeItem(item)
            main_tab.day_markers = []
            
            # Draw day markers
            current_day = self.pm6_start_dt.normalize() + pd.Timedelta(days=1)
            pm6_end_dt = self.pm6_start_dt + pd.to_timedelta(self.full_time[-1], unit='s')
            
            while current_day < pm6_end_dt:
                sec_offset = (current_day - self.pm6_start_dt).total_seconds()
                date_label = current_day.strftime('%Y-%m-%d')
                
                for p in [main_tab.p1, main_tab.p2, main_tab.p3]:
                    day_line = pg.InfiniteLine(pos=sec_offset, angle=90, pen=pg.mkPen(100, 200, 255, 150, style=Qt.DashLine), label=date_label, labelOpts={'position': 0.05, 'color': (150, 220, 255), 'movable': False, 'fill': (0, 0, 0, 100)})
                    day_line.setVisible(self.check_day_markers.isChecked())
                    p.addItem(day_line)
                    main_tab.day_markers.append(day_line)
                    
                current_day += pd.Timedelta(days=1)
            
            
            # Remove old day tabs on new log load (keep only 0-th index "Full overview")
            while self.tabs.count() > 1:
                self.tabs.removeTab(1)

            from collections import defaultdict
            day_tab_widgets = {}
            daily_target_counts = defaultdict(int)
            current_daily_counters = defaultdict(int)
            
            # Pre-count to identify duplicate sources on the same day
            for s in self.sessions:
                s_sec = s['start']
                session_dt = self.pm6_start_dt + pd.to_timedelta(s_sec, unit='s')
                date_str = session_dt.strftime('%d %b')
                daily_target_counts[(date_str, s['target'])] += 1

            created_tabs, skipped_tabs = [], []
            marker_index = 0  # Used to stagger label heights and avoid collision

            for s in self.sessions:
                target = s['target']
                s_sec = s['start']
                e_sec = s['end']
                
                s_idx = np.searchsorted(self.full_time, s_sec)
                e_idx = np.searchsorted(self.full_time, e_sec)
                
                if s_idx < e_idx and s_idx < len(self.full_time) and (e_sec - s_sec) > 300:
                    df_slice = self.df_pm6.iloc[s_idx:e_idx].copy()
                    
                    # Get exact start date and time
                    session_dt = self.pm6_start_dt + pd.to_timedelta(s_sec, unit='s')
                    date_str = session_dt.strftime('%d %b')  # Example: "04 Jan"
                    time_str = session_dt.strftime('%H:%M')  # Example: "17:59"
                    
                    # If the tab for this day does not exist, create it
                    if date_str not in day_tab_widgets:
                        day_tw = QTabWidget()
                        day_tw.currentChanged.connect(self.on_tab_changed) # Update analysis on click
                        self.tabs.addTab(day_tw, date_str)
                        day_tab_widgets[date_str] = day_tw
                        
                    current_daily_counters[(date_str, target)] += 1
                    
                    # If the source was observed more than once per day, add time clarification (HH:MM)
                    if daily_target_counts[(date_str, target)] > 1:
                        inner_tab_name = f"{target} ({time_str})"
                        marker_name = f"{target} ({date_str} {time_str})"
                    else:
                        inner_tab_name = target
                        marker_name = f"{target} ({date_str})"
                        
                    # Create the plot and hide it inside the current day's tab
                    target_tab = SignalTab(df_slice, self.pm6_start_dt, tab_name=marker_name, fs=self.fs)
                    day_tab_widgets[date_str].addTab(target_tab, inner_tab_name)
                    created_tabs.append(marker_name)

                    # Draw red markers on all 3 graphs of "Full overview"
                    marker_index += 1

                    brush = pg.mkBrush(255, 0, 0, 30) # semi-transparent red
                    line1 = pg.LinearRegionItem([s_sec, e_sec], movable=False, brush=brush, pen=None)
                    line2 = pg.LinearRegionItem([s_sec, e_sec], movable=False, brush=brush, pen=None)
                    line3 = pg.LinearRegionItem([s_sec, e_sec], movable=False, brush=brush, pen=None)

                    # Short source name only (no date) to keep label concise
                    short_name = inner_tab_name
                    
                    lbl_y_pos = 0.50
                    
                    # Add label as an InfiniteLine at start boundary (force angle=90 to be vertical)
                    lbl1 = pg.InfiniteLine(pos=s_sec, angle=90, pen=pg.mkPen(None), label=short_name, labelOpts={'position': lbl_y_pos, 'color': (255, 80, 80), 'movable': False, 'fill': (0, 0, 0, 100), 'angle': 90, 'anchor': (0.5, 0.5)})
                    lbl2 = pg.InfiniteLine(pos=s_sec, angle=90, pen=pg.mkPen(None), label=short_name, labelOpts={'position': lbl_y_pos, 'color': (255, 80, 80), 'movable': False, 'fill': (0, 0, 0, 100), 'angle': 90, 'anchor': (0.5, 0.5)})

                    is_visible = self.check_markers.isChecked()
                    for item in [line1, line2, line3, lbl1, lbl2]:
                        item.setVisible(is_visible)

                    main_tab.p1.addItem(line1)
                    main_tab.p1.addItem(lbl1)

                    main_tab.p2.addItem(line2)
                    main_tab.p2.addItem(lbl2)

                    main_tab.p3.addItem(line3)

                    main_tab.session_markers.extend([line1, line2, line3, lbl1, lbl2])
                else:
                    skipped_tabs.append(f"{target} ({s_sec:.0f}s - {e_sec:.0f}s)")

            self.change_active_channel(self.combo_channel.currentText())
            
            report_msg = MSG_CREATED_TABS_TEMPLATE.format(', '.join(created_tabs) if created_tabs else MSG_TAB_CREATION_NONE)
            if skipped_tabs:
                report_msg += "\n\n" + MSG_SKIPPED_TABS_TEMPLATE.format(', '.join(skipped_tabs))

            QMessageBox.information(self, MSG_LOG_PROCESSING_RESULT_TITLE, report_msg)

        except Exception as e:
            QMessageBox.critical(self, MSG_LOG_PROCESSING_ERROR_TITLE, f"Failed to process log: {str(e)}")


    def toggle_markers(self):
        """Toggle the visibility of session markers on the main plot."""
        if self.tabs.count() > 0:
            main_tab = self.tabs.widget(0)
            
            is_visible_sessions = self.check_markers.isChecked()
            for item in main_tab.session_markers:
                item.setVisible(is_visible_sessions)
                
            is_visible_days = self.check_day_markers.isChecked()
            if hasattr(main_tab, 'day_markers'):
                for item in main_tab.day_markers:
                    item.setVisible(is_visible_days)

    def open_batch_export(self):
        if self.df_pm6 is None or not hasattr(self, 'sessions') or not self.sessions:
            QMessageBox.warning(self, "Batch Export", "Load a PM6 file and REGI log first to extract sources.")
            return

        from gui.batch_export_dialog import BatchExportDialog
        dialog = BatchExportDialog(
            df_pm6=self.df_pm6,
            sessions=self.sessions,
            start_datetime=self.pm6_start_dt,
            fs=self.fs,
            window_size=self.spin_window.value(),
            n_sigmas=self.spin_sigmas.value(),
            apply_smoothing=self.check_smooth.isChecked(),
            parent=self
        )
        dialog.exec()

    
    
    def manual_clean_region(self):
        active_tab = self.get_active_tab()
        if not active_tab: return
        
        min_x, max_x = active_tab.region.getRegion()
        s_idx = np.searchsorted(active_tab.time_sec, min_x)
        e_idx = np.searchsorted(active_tab.time_sec, max_x)
        
        if s_idx < e_idx:
            new_df = active_tab.df_slice.copy()
            for col in CHANNELS:
                new_df[col] = fill_gap_with_red_noise(new_df[col].values, s_idx, e_idx)
                
            active_tab.update_raw(new_df)
            self.run_analysis()

    def run_analysis(self):
        """Run the signal processing pipeline."""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            return
            
        active_tab = self.get_active_tab()
        if not active_tab:
            return
        
        # Proceed with analysis even if on the main tab (global view).
        # Performance shortcuts will be handled inside the signal processing pipeline.
        
        idx = self.combo_band.currentIndex()
        if idx == 0:
            lowcut, highcut = 1.0 / 150.0, 1.0 / 5.0
        else:
            lowcut, highcut = 1.0 / 600.0, 1.0 / 150.0
        
        signal_duration_sec = len(active_tab.raw_signal) / self.fs
        
        if idx == 1 and signal_duration_sec < 600:
            QMessageBox.warning(
                self,
                MSG_ANALYSIS_DATA_TOO_SHORT_TITLE,
                MSG_ANALYSIS_DATA_TOO_SHORT_TEXT,
            )
            return
        
        try:
            window_size = self.spin_window.value()
            if window_size % 2 == 0:
                window_size += 1
                
            n_sigmas = self.spin_sigmas.value()
            apply_smoothing = self.check_smooth.isChecked()

            self.set_widgets_enabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            self.worker = SignalAnalysisWorker(
                active_tab.raw_signal,
                self.fs,
                lowcut,
                highcut,
                window_size,
                n_sigmas,
                apply_smoothing
            )
            
            def on_finished(result):
                self.set_widgets_enabled(True)
                self.progress_bar.setVisible(False)
                filtered_sig_downsampled, img_data_pooled = result
                try:
                    active_tab.update_filtered(filtered_sig_downsampled)
                    active_tab.update_spectrogram(img_data_pooled, lowcut, highcut)
                except RuntimeError:
                    pass
                
                # If the user changed tabs while the worker was busy, run analysis for the new tab now
                current_active_tab = self.get_active_tab()
                if current_active_tab and current_active_tab != active_tab:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, self.run_analysis)

            def on_error(err_str):
                self.set_widgets_enabled(True)
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, MSG_ANALYSIS_ERROR_TITLE, f"Failed to build plots:\n{err_str}")

            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(on_finished)
            self.worker.error.connect(on_error)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, MSG_ANALYSIS_ERROR_TITLE, f"Failed to start analysis:\n{str(e)}")

    def export_plots(self):
        active_tab = self.get_active_tab()
        if not active_tab:
            return

        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel, QDialogButtonBox, QGroupBox
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # --- Export options dialog ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Options")
        dlg.resize(340, 280)
        dlg_layout = QVBoxLayout(dlg)

        group_graphs = QGroupBox("Include sub-graphs")
        graph_layout = QVBoxLayout()
        chk_raw = QCheckBox("Raw Signal")
        chk_raw.setChecked(True)
        chk_filtered = QCheckBox("Filtered Signal (Scintillations)")
        chk_filtered.setChecked(True)
        chk_spec = QCheckBox("Spectrogram")
        chk_spec.setChecked(True)
        graph_layout.addWidget(chk_raw)
        graph_layout.addWidget(chk_filtered)
        graph_layout.addWidget(chk_spec)
        group_graphs.setLayout(graph_layout)
        dlg_layout.addWidget(group_graphs)

        group_fmt = QGroupBox("Format")
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("File format:"))
        combo_fmt = QComboBox()
        combo_fmt.addItems(["PNG (300 dpi)", "SVG (vector)"])
        fmt_layout.addWidget(combo_fmt)
        group_fmt.setLayout(fmt_layout)
        dlg_layout.addWidget(group_fmt)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return

        selected = []
        if chk_raw.isChecked():
            selected.append("raw")
        if chk_filtered.isChecked():
            selected.append("filtered")
        if chk_spec.isChecked():
            selected.append("spectrogram")

        if not selected:
            QMessageBox.warning(self, "Export", "No sub-graphs selected.")
            return

        # --- Build auto filename ---
        source = active_tab.tab_name.replace(' ', '_').replace('/', '-')
        channel = active_tab.current_channel
        date_str = active_tab.start_datetime.strftime('%Y%m%d') if active_tab.start_datetime else 'unknown'
        ext = ".svg" if "SVG" in combo_fmt.currentText() else ".png"
        default_name = f"{source}_{date_str}_{channel}{ext}"

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Export", default_name,
                                                   "PNG Images (*.png);;SVG Files (*.svg)")
        if not filepath:
            return

        try:
            n = len(selected)
            fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), constrained_layout=True)
            if n == 1:
                axes = [axes]

            ax_idx = 0
            time_h = active_tab.time_sec / 3600.0

            # Title with metadata
            fig.suptitle(
                f"{active_tab.tab_name}  |  {active_tab.start_datetime.strftime('%Y-%m-%d') if active_tab.start_datetime else ''}  |  {channel}",
                fontsize=13, fontweight='bold'
            )

            if "raw" in selected:
                ax = axes[ax_idx]; ax_idx += 1
                ax.plot(time_h, active_tab.raw_signal, color='steelblue', linewidth=0.6)
                ax.set_ylabel("Amplitude")
                ax.set_title("Raw Signal")
                ax.grid(True, alpha=0.3)

            if "filtered" in selected:
                ax = axes[ax_idx]; ax_idx += 1
                filtered = active_tab.curve_filtered.getData()[1]
                if filtered is not None and len(filtered) == len(time_h):
                    ax.plot(time_h, filtered, color='seagreen', linewidth=0.6)
                ax.set_ylabel("Amplitude")
                ax.set_title("Filtered Signal (Scintillations)")
                ax.grid(True, alpha=0.3)

            if "spectrogram" in selected:
                ax = axes[ax_idx]; ax_idx += 1
                img = active_tab.img_spec.image
                if img is not None:
                    t0 = active_tab.time_sec[0] / 3600.0
                    t1 = active_tab.time_sec[-1] / 3600.0
                    rect = active_tab.img_spec.boundingRect()
                    f0, f1 = rect.top(), rect.bottom()
                    im = ax.imshow(img.T, aspect='auto', origin='lower',
                                   extent=[t0, t1, f0, f1], cmap='viridis')
                    plt.colorbar(im, ax=ax, label='Power (dB)')
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title("CWT Spectrogram")

            axes[-1].set_xlabel("Time (hours from start)")

            dpi = 300 if ext == ".png" else 100
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, MSG_SAVE_ERROR_TITLE, str(e))

    def run_spectral_analysis(self):
        """Run the full spectral-correlation analysis on the active source transit."""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            return
            
        active_tab = self.get_active_tab()
        if not active_tab or active_tab.tab_name == MAIN_TAB_NAME:
            QMessageBox.warning(
                self, MSG_SPECTRAL_NO_SOURCE_TITLE, MSG_SPECTRAL_NO_SOURCE_TEXT,
            )
            return

        try:
            df = active_tab.df_slice

            # Compute P-M (interferometric difference) channels
            pm_signals = {
                '20 MHz Pol A': df['P1_20A'].values - df['M1_20A'].values,
                '20 MHz Pol B': df['P2_20B'].values - df['M2_20B'].values,
                '25 MHz Pol A': df['P3_25A'].values - df['M3_25A'].values,
                '25 MHz Pol B': df['P4_25B'].values - df['M4_25B'].values,
            }

            window_size = self.spin_window.value()
            if window_size % 2 == 0:
                window_size += 1
            n_sigmas = self.spin_sigmas.value()
            apply_smoothing = self.check_smooth.isChecked()

            signal_duration = len(df) / self.fs

            bands = [
                ('small', 1.0 / 150.0, 1.0 / 5.0),
                ('large', 1.0 / 600.0, 1.0 / 150.0),
            ]

            self.set_widgets_enabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            self.worker = SpectralAnalysisWorker(
                pm_signals, self.fs, signal_duration, bands,
                window_size, n_sigmas, apply_smoothing
            )
            
            # Find the day tab and source name
            day_tab = self.tabs.currentWidget()
            if not isinstance(day_tab, QTabWidget):
                self.set_widgets_enabled(True)
                self.progress_bar.setVisible(False)
                return

            inner_idx = day_tab.currentIndex()
            inner_name = day_tab.tabText(inner_idx)
            spectral_name = inner_name + SPECTRAL_TAB_SUFFIX

            def on_finished(band_results):
                self.set_widgets_enabled(True)
                self.progress_bar.setVisible(False)
                
                try:
                    # Replace existing SpectralTab if present
                    for i in range(day_tab.count()):
                        if day_tab.tabText(i) == spectral_name:
                            day_tab.removeTab(i)
                            break
    
                    spectral_tab = SpectralTab(inner_name, band_results)
                    day_tab.addTab(spectral_tab, spectral_name)
                    day_tab.setCurrentWidget(spectral_tab)
                except RuntimeError:
                    pass
                
            def on_error(err_str):
                self.set_widgets_enabled(True)
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, MSG_ANALYSIS_ERROR_TITLE, f"Failed to build plots:\n{err_str}")

            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(on_finished)
            self.worker.error.connect(on_error)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, MSG_ANALYSIS_ERROR_TITLE, f"Failed to build plots:\n{str(e)}")

    def run_global_spectral_analysis(self):
        """Run spectral analysis on the full PM6 dataset (matches professor's 'Global' approach)."""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            return
            
        if self.df_pm6 is None:
            QMessageBox.warning(self, MSG_SPECTRAL_NO_SOURCE_TITLE,
                                "Load a PM6 file first.")
            return

        try:
            df = self.df_pm6

            # Compute P-M interferometric channels over entire dataset
            pm_signals = {
                '20 MHz Pol A': df['P1_20A'].values - df['M1_20A'].values,
                '20 MHz Pol B': df['P2_20B'].values - df['M2_20B'].values,
                '25 MHz Pol A': df['P3_25A'].values - df['M3_25A'].values,
                '25 MHz Pol B': df['P4_25B'].values - df['M4_25B'].values,
            }

            window_size = self.spin_window.value()
            if window_size % 2 == 0:
                window_size += 1
            n_sigmas = self.spin_sigmas.value()
            apply_smoothing = self.check_smooth.isChecked()

            signal_duration = len(df) / self.fs

            bands = [
                ('small', 1.0 / 150.0, 1.0 / 5.0),
                ('large', 1.0 / 600.0, 1.0 / 150.0),
            ]

            self.set_widgets_enabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            self.worker = SpectralAnalysisWorker(
                pm_signals, self.fs, signal_duration, bands,
                window_size, n_sigmas, apply_smoothing
            )

            def on_finished(band_results):
                self.set_widgets_enabled(True)
                self.progress_bar.setVisible(False)
                
                # Place result as a top-level tab (not nested in a day group)
                # Replace existing global spectral tab if present
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) == GLOBAL_SPECTRAL_TAB_NAME:
                        self.tabs.removeTab(i)
                        break

                spectral_tab = SpectralTab("Global", band_results)
                idx = self.tabs.addTab(spectral_tab, GLOBAL_SPECTRAL_TAB_NAME)
                self.tabs.setCurrentIndex(idx)
                
            def on_error(err_str):
                self.set_widgets_enabled(True)
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, MSG_SPECTRAL_ERROR_TITLE, f"Global spectral analysis failed:\n{err_str}")

            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(on_finished)
            self.worker.error.connect(on_error)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(
                self, MSG_SPECTRAL_ERROR_TITLE,
                f"Global spectral analysis failed:\n{str(e)}",
            )

    def closeEvent(self, event):
        """Ensure any running background worker is safely terminated on app exit."""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        super().closeEvent(event)