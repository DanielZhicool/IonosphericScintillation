import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QComboBox, QSplitter, QMessageBox, QTabWidget,
                               QSpinBox, QDoubleSpinBox, QCheckBox)
from PySide6.QtCore import Qt

# Core parsers and signal processing
from core.parsers import load_pm6_data, parse_regi_with_time
from core.signal_processing import (fill_gap_with_red_noise, bandpass_filter, 
                                    compute_fsst_spectrogram, clean_and_smooth_signal)

CHANNELS = ['P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']




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
                # Add current seconds to start timestamp
                dt = pd.to_datetime(self.start_timestamp + v, unit='s')
                strings.append(dt.strftime('%H:%M:%S'))
            except Exception:
                strings.append("")  # Protection against out-of-range zoom errors
        return strings




class SignalTab(QWidget):
    """Widget for a signal tab; holds a DataFrame for a target."""
    def __init__(self, df_slice, start_datetime, tab_name="Полный обзор", fs=1.0):
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
        self.p1 = self.graph_widget.addPlot(title=f"[{self.tab_name}] Сырые данные ({self.current_channel})", axisItems={'bottom': axis_p1})
        self.p1.showGrid(x=True, y=True)
        self.curve_raw = self.p1.plot(self.time_sec, self.raw_signal, pen='b')
        
        self.region = pg.LinearRegionItem()
        region_end = min(self.time_sec[0] + 100, self.time_sec[-1])
        self.region.setRegion([self.time_sec[0], region_end])
        self.region.setZValue(10)
        self.p1.addItem(self.region, ignoreBounds=True)

        self.graph_widget.nextRow()
        
        # Ionospheric scintillations
        axis_p2 = TimeAxisItem(self.start_datetime, orientation='bottom')
        self.p2 = self.graph_widget.addPlot(title="Ионосферные мерцания", axisItems={'bottom': axis_p2})
        self.p2.showGrid(x=True, y=True)
        self.p2.setXLink(self.p1)
        self.curve_filtered = self.p2.plot(pen='g')

        self.graph_widget.nextRow()

        # Spectrogram
        axis_p3 = TimeAxisItem(self.start_datetime, orientation='bottom')
        self.p3 = self.graph_widget.addPlot(title="FSST Спектрограмма", axisItems={'bottom': axis_p3})
        self.p3.setXLink(self.p1)
        self.img_spec = pg.ImageItem()
        self.p3.addItem(self.img_spec)
        self.img_spec.setColorMap(pg.colormap.get('viridis'))  # Use 'viridis' colormap

    def set_channel(self, channel_name):
        self.current_channel = channel_name
        self.raw_signal = self.df_slice[channel_name].values
        # Update title keeping tab name
        self.p1.setTitle(f"[{self.tab_name}] Сырые данные ({channel_name})")
        self.curve_raw.setData(self.time_sec, self.raw_signal)

    def update_raw(self, df_updated):
        self.df_slice = df_updated.copy()
        self.set_channel(self.current_channel)

    def update_filtered(self, filtered_signal):
        self.curve_filtered.setData(self.time_sec, filtered_signal)

    def update_spectrogram(self, img_data, lowcut, highcut):
        self.img_spec.setImage(img_data, autoLevels=True)
        # Map image to its real frequency range
        freq_height = highcut - lowcut
        self.img_spec.setRect(pg.QtCore.QRectF(self.time_sec[0], lowcut, self.time_sec[-1]-self.time_sec[0], freq_height))
        self.p3.setYRange(lowcut, highcut)


class Uran4App(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("URAN-4 Ionospheric Scintillation Analyzer")
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
        self.btn_load_pm6 = QPushButton("1. Загрузить данные (PM6)")
        self.btn_load_pm6.clicked.connect(self.load_pm6)
        self.lbl_status = QLabel("Файл PM6 не загружен")

        self.combo_channel = QComboBox()
        self.combo_channel.addItems(CHANNELS)
        self.combo_channel.currentTextChanged.connect(self.change_active_channel)
        self.combo_channel.setEnabled(False)

        self.btn_load_logs = QPushButton("2. Загрузить regi и создать вкладки")
        self.btn_load_logs.clicked.connect(self.auto_clean_and_split)
        self.btn_load_logs.setEnabled(False)

        self.check_markers = QCheckBox("Показывать разметку источников на главном графике")
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

        self.check_smooth = QCheckBox("Включить сглаживание")
        self.check_smooth.setChecked(True)
        self.check_smooth.stateChanged.connect(self.run_analysis)
        self.check_smooth.setEnabled(False)

        self.btn_apply_noise = QPushButton("3. Вырезать зону вручную")
        self.btn_apply_noise.clicked.connect(self.manual_clean_region)
        self.btn_apply_noise.setEnabled(False)

        self.combo_band = QComboBox()
        self.combo_band.addItems(["Малые пузырьки (5 - 150 сек)", "Большие облака (150 - 600 сек)"])
        self.combo_band.currentIndexChanged.connect(self.run_analysis)

        self.btn_analyze = QPushButton("4. Принудительно обновить спектр")
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setEnabled(False)
        
        self.btn_export = QPushButton("5. Экспорт графиков")
        self.btn_export.clicked.connect(self.export_plots)
        self.btn_export.setEnabled(False)

        # Left control panel layout
        control_layout.addWidget(self.btn_load_pm6)
        control_layout.addWidget(self.lbl_status)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Отображаемый канал:"))
        control_layout.addWidget(self.combo_channel)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_load_logs)
        control_layout.addWidget(self.check_markers)
        
        # Integrate cleaning controls
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("<b>Параметры очистки помех:</b>"))
        control_layout.addWidget(QLabel("Размер окна фильтра (отсчеты):"))
        control_layout.addWidget(self.spin_window)
        control_layout.addWidget(QLabel("Порог отклонения (Сигма):"))
        control_layout.addWidget(self.spin_sigmas)
        control_layout.addWidget(self.check_smooth)
        
        control_layout.addSpacing(15)
        control_layout.addWidget(self.btn_apply_noise)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Диапазон фильтрации:"))
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
        return self.tabs.currentWidget()

    def on_tab_changed(self, index):
        if index >= 0 and self.df_pm6 is not None:
            self.run_analysis()

    def change_active_channel(self, channel_name):
        active_tab = self.get_active_tab()
        if active_tab:
            active_tab.set_channel(channel_name)
            self.run_analysis()

    def set_widgets_enabled(self, enabled=True):
        """Включение/выключение блоков настройки параметров."""
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
        filepath, _ = QFileDialog.getOpenFileName(self, "Открыть PM6", "", "PM6 Files (*.PM6);;All Files (*)")
        if not filepath: return
            
        try:
            self.df_pm6 = load_pm6_data(filepath)
            self.full_time = self.df_pm6['Time_sec'].values
            
            # Extract exact pm6 start datetime
            pm6_start_dt = self.df_pm6['Datetime'].iloc[0]
            
            self.tabs.clear()
            # Pass pm6_start_dt into the tab
            main_tab = SignalTab(self.df_pm6, pm6_start_dt, tab_name="Полный обзор", fs=self.fs)
            self.tabs.addTab(main_tab, "Полный обзор")
            
            self.lbl_status.setText(f"Загружено отсчетов: {len(self.df_pm6)}")
            self.set_widgets_enabled(True)
            self.run_analysis()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def auto_clean_and_split(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Открыть regi лог", "", "Text Files (*.txt);;All Files (*)")
        if not filepath: return

        try:
            pm6_start_dt = self.df_pm6['Datetime'].iloc[0]
            
            df_logs = parse_regi_with_time(filepath, self.df_pm6['Datetime'].iloc[0])
            if df_logs.empty:
                QMessageBox.warning(self, "Пустой файл", "Не удалось найти события в файле.")
                return

            noise_targets = ['calibrovka', '3Czenit']
            calibrations = df_logs[df_logs['Target_Name'].isin(noise_targets)]
            
            for _, row in calibrations.iterrows():
                s_idx = np.searchsorted(self.full_time, row['Start_sec'])
                e_idx = np.searchsorted(self.full_time, row['End_sec'])
                if s_idx < e_idx:
                    for col in CHANNELS:
                        self.df_pm6[col] = fill_gap_with_red_noise(self.df_pm6[col].values, s_idx, e_idx)

            # Update main tab with cleaned data
            main_tab = self.tabs.widget(0)
            main_tab.update_raw(self.df_pm6)

            for item in main_tab.session_markers:
                main_tab.p1.removeItem(item)
            main_tab.session_markers.clear()
            
            unique_targets = df_logs[~df_logs['Target_Name'].isin(noise_targets)]['Target_Name'].unique()
            created_tabs, skipped_tabs = [], []

            # Session gap threshold in seconds (3600 = 1 hour).
            GAP_THRESHOLD = 3600 



            for target in unique_targets:
                target_events = df_logs[df_logs['Target_Name'] == target].sort_values('Start_sec')
                if target_events.empty: continue
                
                sessions = []
                current_start = target_events.iloc[0]['Start_sec']
                current_end = target_events.iloc[0]['End_sec']

                for i in range(1, len(target_events)):
                    row = target_events.iloc[i]
                    if row['Start_sec'] - current_end > GAP_THRESHOLD:
                        sessions.append((current_start, current_end))
                        current_start = row['Start_sec']
                    current_end = row['End_sec']
                sessions.append((current_start, current_end))

                for idx, (s_sec, e_sec) in enumerate(sessions):
                    s_idx = np.searchsorted(self.full_time, s_sec)
                    e_idx = np.searchsorted(self.full_time, e_sec)
                    
                    if s_idx < e_idx and s_idx < len(self.full_time) and e_idx > 0:
                        df_slice = self.df_pm6.iloc[s_idx:e_idx].copy()
                        tab_name = f"{target} ({idx+1})" if len(sessions) > 1 else target
                        
                        # Передаем tab_name во вкладку
                        target_tab = SignalTab(df_slice, pm6_start_dt, tab_name=tab_name, fs=self.fs)
                        self.tabs.addTab(target_tab, tab_name)
                        created_tabs.append(tab_name)

                        # Рисуем разметку на главном графике
                        line = pg.PlotDataItem([s_sec, e_sec], [0, 0], pen=pg.mkPen((0, 255, 0), width=5))
                        text = pg.TextItem(tab_name, color=(0, 255, 0), anchor=(0.5, 0))
                        text.setPos((s_sec + e_sec) / 2, -20) 
                        
                        # Учитываем текущее состояние чекбокса
                        is_visible = self.check_markers.isChecked()
                        line.setVisible(is_visible)
                        text.setVisible(is_visible)

                        main_tab.p1.addItem(line)
                        main_tab.p1.addItem(text)
                        
                        # Сохраняем в список вкладки для управления видимостью
                        main_tab.session_markers.extend([line, text])
                    else:
                        skipped_tabs.append(f"{target} (Сессия {idx+1})")

            self.change_active_channel(self.combo_channel.currentText())
            
            report_msg = f"✅ Созданы вкладки: {', '.join(created_tabs) if created_tabs else 'Нет'}\n\n"
            if skipped_tabs:
                report_msg += f"⚠️ Пропущены (нет данных в PM6):\n{', '.join(skipped_tabs)}"
            QMessageBox.information(self, "Результат", report_msg)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось обработать лог: {str(e)}")


    def toggle_markers(self):
        """Включает и выключает видимость зеленых линий на главной вкладке."""
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
        """Запуск сквозного конвейера: Очистка параметров -> Полосовой фильтр -> FSST."""
        active_tab = self.get_active_tab()
        if not active_tab: return
        
        if active_tab.tab_name == "Полный обзор":
            return
        
        idx = self.combo_band.currentIndex()
        if idx == 0:
            lowcut, highcut = 1.0/150.0, 1.0/5.0
        else:
            lowcut, highcut = 1.0/600.0, 1.0/150.0
        
        signal_duration_sec = len(active_tab.raw_signal) / self.fs
        
        # If 'Large clouds' selected but signal < 600s (10 min)
        if idx == 1 and signal_duration_sec < 600:
            QMessageBox.warning(self, "Недостаточно данных", 
                              f"Длина сигнала ({int(signal_duration_sec)} сек) слишком мала "
                              "для поиска Больших облаков (волн до 600 сек).\n\n"
                              "Результаты анализа будут содержать сильные краевые артефакты. "
                              "Для коротких наблюдений используйте режим 'Малые пузырьки'.")
            return # Прерываем расчет, чтобы не рисовать пиксельный мусор
        
        try:
            # Read dynamic parameters from GUI
            window_size = self.spin_window.value()
            # Гарантируем нечетное число для корректной работы фильтра Савицкого-Голея
            if window_size % 2 == 0:
                window_size += 1
                
            n_sigmas = self.spin_sigmas.value()
            apply_smoothing = self.check_smooth.isChecked()

            # Step 1: Clean spikes and clusters
            cleaned_sig = clean_and_smooth_signal(
                active_tab.raw_signal, 
                window_size=window_size, 
                n_sigmas=n_sigmas, 
                apply_smoothing=apply_smoothing
            )
            
            # Step 2: Bandpass filter the cleaned signal
            filtered_sig = bandpass_filter(cleaned_sig, lowcut, highcut, self.fs)
            active_tab.update_filtered(filtered_sig)
            
            # Step 3: Compute synchrosqueezed spectrogram (lowcut/highcut added)
            img_data = compute_fsst_spectrogram(filtered_sig, self.fs, lowcut, highcut)
            active_tab.update_spectrogram(img_data, lowcut, highcut)
            
        except Exception as e:
            # Note: now shows an error dialog instead of a blank screen on failure
            QMessageBox.critical(self, "Ошибка анализа", f"Сбой при построении графиков:\n{str(e)}")

    def export_plots(self):
        active_tab = self.get_active_tab()
        if not active_tab: return

        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить", "graph.png", "PNG (*.png)")
        if filepath:
            try:
                exporter = pg.exporters.ImageExporter(active_tab.graph_widget.scene())
                exporter.parameters()['width'] = 1920
                exporter.export(filepath)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))