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
    FILE_DIALOG_PM6_TITLE, FILE_DIALOG_REGI_TITLE,
    SAVE_FILE_DEFAULT_NAME, SAVE_FILE_FILTER,
    MSG_CREATED_TABS_TEMPLATE
)

# Core parsers and signal processing
from core.parsers import load_pm6_data, parse_regi_with_time
from core.signal_processing import (fill_gap_with_red_noise, bandpass_filter, 
                                    compute_cwt_spectrogram, clean_and_smooth_signal,
                                    upsample_pchip)

CHANNELS = ['P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']

from gui.plotting import TimeAxisItem
from gui.tabs import SignalTab


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
        control_layout.addStretch()

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        splitter.addWidget(control_panel)
        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 1000])

    def get_active_tab(self):
        current_widget = self.tabs.currentWidget()
        if isinstance(current_widget, SignalTab):
            # Это главная вкладка (Full overview)
            return current_widget
        elif isinstance(current_widget, QTabWidget):
            # Это вкладка конкретного дня. Берем активный график внутри неё.
            return current_widget.currentWidget()
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

            # --- 1. АСТРОФИЗИЧЕСКАЯ ПРОЕКЦИЯ (Звездные сутки) ---
            # Звездные сутки = 23 часа 56 минут 4 секунды (86164 секунды).
            # Радиоисточники смещаются ровно на это время каждый день относительно солнечного времени.
            SIDEREAL_DAY = 86164
            pm6_max_sec = self.full_time[-1]
            
            original_logs = df_logs.copy()
            max_log_sec = original_logs['End_sec'].max()
            
            # Если PM6 длиннее лога, размножаем расписание на все дни вперед
            if pm6_max_sec > max_log_sec:
                days_to_add = int(np.ceil((pm6_max_sec - max_log_sec) / SIDEREAL_DAY))
                projected_dfs = [original_logs]
                
                for day in range(1, days_to_add + 1):
                    df_shifted = original_logs.copy()
                    df_shifted['Start_sec'] += day * SIDEREAL_DAY
                    df_shifted['End_sec'] += day * SIDEREAL_DAY
                    projected_dfs.append(df_shifted)
                    
                df_logs = pd.concat(projected_dfs, ignore_index=True)
            # ----------------------------------------------------

            noise_targets = ['calibrovka', '3Czenit']
            calibrations = df_logs[df_logs['Target_Name'].isin(noise_targets)]
            
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
            
            # --- 2. СТРОГАЯ ХРОНОЛОГИЧЕСКАЯ ГРУППИРОВКА ---
            obs_logs = df_logs[~df_logs['Target_Name'].isin(noise_targets)].sort_values('Start_sec').reset_index(drop=True)
            
            sessions = []
            if not obs_logs.empty:
                current_target = obs_logs.iloc[0]['Target_Name']
                current_start = obs_logs.iloc[0]['Start_sec']
                current_end = obs_logs.iloc[0]['End_sec']
                
                for i in range(1, len(obs_logs)):
                    row = obs_logs.iloc[i]
                    is_same_session = (row['Target_Name'] == current_target) and (row['Start_sec'] - current_end < 3600)
                    
                    if is_same_session:
                        current_end = max(current_end, row['End_sec'])
                    else:
                        sessions.append({'target': current_target, 'start': current_start, 'end': current_end})
                        current_target = row['Target_Name']
                        current_start = row['Start_sec']
                        current_end = row['End_sec']
                        
                sessions.append({'target': current_target, 'start': current_start, 'end': current_end})

            # Удаляем старые вкладки дней при новой загрузке логов (оставляем только 0-й индекс "Full overview")
            while self.tabs.count() > 1:
                self.tabs.removeTab(1)

            from collections import defaultdict
            day_tab_widgets = {}
            daily_target_counts = defaultdict(int)
            current_daily_counters = defaultdict(int)
            
            # Предварительный подсчет для выявления дубликатов источников в один и тот же день
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
                    
                    # Получаем точную дату и время старта
                    session_dt = pm6_start_dt + pd.to_timedelta(s_sec, unit='s')
                    date_str = session_dt.strftime('%d %b')  # Пример: "04 Jan"
                    time_str = session_dt.strftime('%H:%M')  # Пример: "17:59"
                    
                    # Если вкладки для этого дня еще нет, создаем её
                    if date_str not in day_tab_widgets:
                        day_tw = QTabWidget()
                        day_tw.currentChanged.connect(self.on_tab_changed) # Чтобы анализ обновлялся при клике
                        self.tabs.addTab(day_tw, date_str)
                        day_tab_widgets[date_str] = day_tw
                        
                    current_daily_counters[(date_str, target)] += 1
                    
                    # Если источник наблюдался больше одного раза за сутки, добавляем уточнение (HH:MM)
                    if daily_target_counts[(date_str, target)] > 1:
                        inner_tab_name = f"{target} ({time_str})"
                        marker_name = f"{target} ({date_str} {time_str})"
                    else:
                        inner_tab_name = target
                        marker_name = f"{target} ({date_str})"
                        
                    # Создаем график и прячем его внутрь вкладки текущего дня
                    target_tab = SignalTab(df_slice, pm6_start_dt, tab_name=marker_name, fs=self.fs)
                    day_tab_widgets[date_str].addTab(target_tab, inner_tab_name)
                    created_tabs.append(marker_name)

                    # Рисуем зеленый маркер на "Full overview"
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
        """Run the signal processing pipeline: cleaning -> bandpass -> FSST."""
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

            # 1. Очистка от спайков (Хампель + Савицкий-Голей)
            cleaned_sig = clean_and_smooth_signal(
                active_tab.raw_signal,
                window_size=window_size,
                n_sigmas=n_sigmas,
                apply_smoothing=apply_smoothing,
            )
            
            # --- НОВЫЙ БЛОК: PCHIP Апсемплинг (x3) ---
            FACTOR = 3
            upsampled_sig, new_fs = upsample_pchip(cleaned_sig, self.fs, factor=FACTOR)
            
            # 2. Полосовая фильтрация на ВЫСОКОЙ частоте (new_fs)
            filtered_sig = bandpass_filter(upsampled_sig, lowcut, highcut, new_fs)
            
            # Отдаем в 1D график прореженный обратно сигнал, 
            # чтобы ось времени в интерфейсе не растянулась
            active_tab.update_filtered(filtered_sig[::FACTOR])
            
            # 3. Вычисление CWT на ВЫСОКОЙ частоте (идеальная детализация фазы)
            img_data = compute_cwt_spectrogram(filtered_sig, new_fs, lowcut, highcut)
            
            # --- Умное сжатие графики (Max Pooling) ---
            # Вместо грубого среза [::FACTOR], который создает "пиксельные лесенки",
            # мы берем максимум энергии в каждом окне из 3 точек. 
            N_orig = len(active_tab.raw_signal)
            
            # Защита от ошибок размерности: берем ровно N_orig * FACTOR точек
            img_data_exact = img_data[: N_orig * FACTOR, :]
            
            # Схлопываем (N_orig * 3) строк в N_orig строк, забирая максимальный пиксель
            img_data_pooled = img_data_exact.reshape(N_orig, FACTOR, img_data_exact.shape[1]).max(axis=1)
            
            # Отдаем сжатую картинку в интерфейс
            active_tab.update_spectrogram(img_data_pooled, lowcut, highcut)
            # -----------------------------------------
            
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