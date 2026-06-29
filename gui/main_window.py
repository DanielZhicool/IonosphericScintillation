import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QComboBox, QSplitter, QMessageBox, QTabWidget,
                               QSpinBox, QDoubleSpinBox, QCheckBox)
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
    MSG_CREATED_TABS_TEMPLATE
)

# Core parsers and signal processing
from core.parsers import load_pm6_data, parse_regi_with_time, build_observation_sessions
from core.signal_processing import fill_gap_with_red_noise, process_signal_pipeline

CHANNELS = ['P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']

from gui.plotting import TimeAxisItem
from gui.tabs import SignalTab
from gui.spectral_tab import SpectralTab
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

        self.btn_spectral = QPushButton(BTN_SPECTRAL)
        self.btn_spectral.clicked.connect(self.run_spectral_analysis)
        self.btn_spectral.setEnabled(False)

        self.btn_global_spectral = QPushButton(BTN_GLOBAL_SPECTRAL)
        self.btn_global_spectral.clicked.connect(self.run_global_spectral_analysis)
        self.btn_global_spectral.setEnabled(False)

        # Left control panel layout
        control_layout.addWidget(self.btn_load_pm6)
        control_layout.addWidget(self.lbl_status)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel(LABEL_DISPLAY_CHANNEL))
        control_layout.addWidget(self.combo_channel)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_load_logs)
        control_layout.addWidget(self.check_markers)
        
        # Integrate cleaning controls
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel(LABEL_CLEANING_HEADER))
        control_layout.addWidget(QLabel(LABEL_WINDOW))
        control_layout.addWidget(self.spin_window)
        control_layout.addWidget(QLabel(LABEL_SIGMA))
        control_layout.addWidget(self.spin_sigmas)
        control_layout.addWidget(self.check_smooth)
        
        control_layout.addSpacing(15)
        control_layout.addWidget(self.btn_apply_noise)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel(LABEL_BAND))
        control_layout.addWidget(self.combo_band)
        control_layout.addWidget(self.btn_analyze)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_export)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_spectral)
        control_layout.addSpacing(4)
        control_layout.addWidget(self.btn_global_spectral)
        control_layout.addStretch()

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
            self.run_analysis()

    def change_active_channel(self, channel_name):
        active_tab = self.get_active_tab()
        if active_tab:
            active_tab.set_channel(channel_name)
            self.run_analysis()

    def set_widgets_enabled(self, enabled=True):
        """Enable or disable the cleaning and export controls."""
        self.combo_channel.setEnabled(enabled)
        self.btn_load_logs.setEnabled(enabled)
        self.check_markers.setEnabled(enabled)
        self.btn_apply_noise.setEnabled(enabled)
        self.btn_analyze.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
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
            pm6_start_dt = self.df_pm6['Datetime'].iloc[0]
            
            self.tabs.clear()
            # Pass pm6_start_dt into the tab
            main_tab = SignalTab(self.df_pm6, pm6_start_dt, tab_name=MAIN_TAB_NAME, fs=self.fs)
            self.tabs.addTab(main_tab, MAIN_TAB_NAME)
            
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
            df_logs, calibrations, sessions = build_observation_sessions(df_logs, pm6_max_sec)
            
            for _, row in calibrations.iterrows():
                s_idx = np.searchsorted(self.full_time, row['Start_sec'])
                e_idx = np.searchsorted(self.full_time, row['End_sec'])
                if s_idx < e_idx:
                    for col in CHANNELS:
                        self.df_pm6[col] = fill_gap_with_red_noise(self.df_pm6[col].values, s_idx, e_idx)

            main_tab = self.tabs.widget(0)
            main_tab.update_raw(self.df_pm6)

            for item in main_tab.session_markers:
                main_tab.p1.removeItem(item)
            main_tab.session_markers.clear()
            
            # Remove old day tabs on new log load (keep only 0-th index "Full overview")
            while self.tabs.count() > 1:
                self.tabs.removeTab(1)

            from collections import defaultdict
            day_tab_widgets = {}
            daily_target_counts = defaultdict(int)
            current_daily_counters = defaultdict(int)
            
            # Pre-count to identify duplicate sources on the same day
            for s in sessions:
                s_sec = s['start']
                session_dt = pm6_start_dt + pd.to_timedelta(s_sec, unit='s')
                date_str = session_dt.strftime('%d %b')
                daily_target_counts[(date_str, s['target'])] += 1

            created_tabs, skipped_tabs = [], []

            for s in sessions:
                target = s['target']
                s_sec = s['start']
                e_sec = s['end']
                
                s_idx = np.searchsorted(self.full_time, s_sec)
                e_idx = np.searchsorted(self.full_time, e_sec)
                
                if s_idx < e_idx and s_idx < len(self.full_time) and (e_sec - s_sec) > 300:
                    df_slice = self.df_pm6.iloc[s_idx:e_idx].copy()
                    
                    # Get exact start date and time
                    session_dt = pm6_start_dt + pd.to_timedelta(s_sec, unit='s')
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
                    target_tab = SignalTab(df_slice, pm6_start_dt, tab_name=marker_name, fs=self.fs)
                    day_tab_widgets[date_str].addTab(target_tab, inner_tab_name)
                    created_tabs.append(marker_name)

                    # Draw a green marker on "Full overview"
                    line = pg.PlotDataItem([s_sec, e_sec], [0, 0], pen=pg.mkPen((0, 255, 0), width=5))
                    text = pg.TextItem(marker_name, color=(0, 255, 0), anchor=(0.5, 0))
                    text.setPos((s_sec + e_sec) / 2, -20)
                    
                    is_visible = self.check_markers.isChecked()
                    line.setVisible(is_visible)
                    text.setVisible(is_visible)

                    main_tab.p1.addItem(line)
                    main_tab.p1.addItem(text)
                    main_tab.session_markers.extend([line, text])
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
            is_visible = self.check_markers.isChecked()
            for item in main_tab.session_markers:
                item.setVisible(is_visible)
    
    
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
        active_tab = self.get_active_tab()
        if not active_tab:
            return
        
        if active_tab.tab_name == MAIN_TAB_NAME:
            return
        
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

            filtered_sig_downsampled, img_data_pooled = process_signal_pipeline(
                active_tab.raw_signal,
                self.fs,
                lowcut,
                highcut,
                window_size,
                n_sigmas,
                apply_smoothing
            )
            
            active_tab.update_filtered(filtered_sig_downsampled)
            active_tab.update_spectrogram(img_data_pooled, lowcut, highcut)
            
        except Exception as e:
            QMessageBox.critical(self, MSG_ANALYSIS_ERROR_TITLE, f"Failed to build plots:\n{str(e)}")

    def export_plots(self):
        active_tab = self.get_active_tab()
        if not active_tab: return

        filepath, _ = QFileDialog.getSaveFileName(self, "Export Plot", SAVE_FILE_DEFAULT_NAME, SAVE_FILE_FILTER)
        if filepath:
            try:
                exporter = pg.exporters.ImageExporter(active_tab.graph_widget.scene())
                exporter.parameters()['width'] = 1920
                exporter.export(filepath)
            except Exception as e:
                QMessageBox.critical(self, MSG_SAVE_ERROR_TITLE, str(e))

    def run_spectral_analysis(self):
        """Run the full spectral-correlation analysis on the active source transit."""
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

            band_results = {}
            bands = [
                ('small', 1.0 / 150.0, 1.0 / 5.0),
                ('large', 1.0 / 600.0, 1.0 / 150.0),
            ]

            for band_key, lowcut, highcut in bands:
                min_period = 1.0 / lowcut
                if signal_duration < min_period:
                    band_results[band_key] = None
                    continue

                band_results[band_key] = run_spectral_pipeline(
                    pm_signals, self.fs, lowcut, highcut,
                    window_size, n_sigmas, apply_smoothing,
                )

            # Find the day tab and source name
            day_tab = self.tabs.currentWidget()
            if not isinstance(day_tab, QTabWidget):
                return

            inner_idx = day_tab.currentIndex()
            inner_name = day_tab.tabText(inner_idx)
            spectral_name = inner_name + SPECTRAL_TAB_SUFFIX

            # Replace existing SpectralTab if present
            for i in range(day_tab.count()):
                if day_tab.tabText(i) == spectral_name:
                    day_tab.removeTab(i)
                    break

            spectral_tab = SpectralTab(inner_name, band_results)
            day_tab.addTab(spectral_tab, spectral_name)
            day_tab.setCurrentWidget(spectral_tab)

        except Exception as e:
            QMessageBox.critical(
                self, MSG_SPECTRAL_ERROR_TITLE,
                f"Spectral analysis failed:\n{str(e)}",
            )

    def run_global_spectral_analysis(self):
        """Run spectral analysis on the full PM6 dataset (matches professor's 'Global' approach)."""
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

            band_results = {}
            bands = [
                ('small', 1.0 / 150.0, 1.0 / 5.0),
                ('large', 1.0 / 600.0, 1.0 / 150.0),
            ]
            for band_key, lowcut, highcut in bands:
                band_results[band_key] = run_spectral_pipeline(
                    pm_signals, self.fs, lowcut, highcut,
                    window_size, n_sigmas, apply_smoothing,
                )

            # Place result as a top-level tab (not nested in a day group)
            # Replace existing global spectral tab if present
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == GLOBAL_SPECTRAL_TAB_NAME:
                    self.tabs.removeTab(i)
                    break

            spectral_tab = SpectralTab("Global", band_results)
            idx = self.tabs.addTab(spectral_tab, GLOBAL_SPECTRAL_TAB_NAME)
            self.tabs.setCurrentIndex(idx)

        except Exception as e:
            QMessageBox.critical(
                self, MSG_SPECTRAL_ERROR_TITLE,
                f"Global spectral analysis failed:\n{str(e)}",
            )