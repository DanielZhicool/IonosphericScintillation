import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QComboBox, QSplitter, QMessageBox, QTabWidget)
from PySide6.QtCore import Qt

# Импортируем нашу математику из папки core
from core.parsers import load_pm6_data, parse_regi_with_time
from core.signal_processing import fill_gap_with_red_noise, bandpass_filter, compute_fsst_spectrogram

CHANNELS = ['P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']

class SignalTab(QWidget):
    """Виджет отдельной вкладки. Хранит полный DataFrame для конкретного объекта."""
    def __init__(self, df_slice, fs=1.0):
        super().__init__()
        self.df_slice = df_slice.copy()
        self.time_sec = self.df_slice['Time_sec'].values
        self.fs = fs
        
        self.current_channel = 'P1_20A'
        self.raw_signal = self.df_slice[self.current_channel].values

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.p1 = self.graph_widget.addPlot(title=f"Сырые данные ({self.current_channel})")
        self.p1.showGrid(x=True, y=True)
        self.curve_raw = self.p1.plot(self.time_sec, self.raw_signal, pen='b')
        
        self.region = pg.LinearRegionItem()
        region_end = min(self.time_sec[0] + 100, self.time_sec[-1])
        self.region.setRegion([self.time_sec[0], region_end])
        self.region.setZValue(10)
        self.p1.addItem(self.region, ignoreBounds=True)

        self.graph_widget.nextRow()
        
        self.p2 = self.graph_widget.addPlot(title="Ионосферные мерцания")
        self.p2.showGrid(x=True, y=True)
        self.p2.setXLink(self.p1)
        self.curve_filtered = self.p2.plot(pen='g')

        self.graph_widget.nextRow()

        self.p3 = self.graph_widget.addPlot(title="FSST Спектрограмма")
        self.p3.setXLink(self.p1)
        self.img_spec = pg.ImageItem()
        self.p3.addItem(self.img_spec)
        self.img_spec.setColorMap(pg.colormap.get('viridis'))

    def set_channel(self, channel_name):
        self.current_channel = channel_name
        self.raw_signal = self.df_slice[channel_name].values
        self.p1.setTitle(f"Сырые данные ({channel_name})")
        self.curve_raw.setData(self.time_sec, self.raw_signal)
        # Мы больше не очищаем нижние графики, так как авто-расчет сразу же их перезапишет

    def update_raw(self, df_updated):
        self.df_slice = df_updated.copy()
        self.set_channel(self.current_channel)

    def update_filtered(self, filtered_signal):
        self.curve_filtered.setData(self.time_sec, filtered_signal)

    def update_spectrogram(self, img_data, lowcut, highcut):
        self.img_spec.setImage(img_data, autoLevels=True)
        freq_range = self.fs / 2
        self.img_spec.setRect(pg.QtCore.QRectF(self.time_sec[0], 0, self.time_sec[-1]-self.time_sec[0], freq_range))
        self.p3.setYRange(lowcut, highcut)


class Uran4App(QMainWindow):
    """Главное окно приложения."""
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
        
        self.btn_load_pm6 = QPushButton("1. Загрузить данные (PM6)")
        self.btn_load_pm6.clicked.connect(self.load_pm6)
        self.lbl_status = QLabel("Файл PM6 не загружен")

        self.combo_channel = QComboBox()
        self.combo_channel.addItems(CHANNELS)
        # Автоматический пересчет при смене канала
        self.combo_channel.currentTextChanged.connect(self.change_active_channel)
        self.combo_channel.setEnabled(False)

        self.btn_load_logs = QPushButton("2. Загрузить regi и создать вкладки")
        self.btn_load_logs.clicked.connect(self.auto_clean_and_split)
        self.btn_load_logs.setEnabled(False)

        self.btn_apply_noise = QPushButton("3. Вырезать зону вручную (активная вкладка)")
        self.btn_apply_noise.clicked.connect(self.manual_clean_region)
        self.btn_apply_noise.setEnabled(False)

        self.combo_band = QComboBox()
        self.combo_band.addItems(["Малые пузырьки (5 - 150 сек)", "Большие облака (150 - 600 сек)"])
        # Автоматический пересчет при смене диапазона (малые/большие пузырьки)
        self.combo_band.currentIndexChanged.connect(self.run_analysis)

        # Кнопку переименовали для ясности (она теперь нужна только для принудительного обновления)
        self.btn_analyze = QPushButton("4. Принудительно обновить спектр")
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setEnabled(False)
        
        self.btn_export = QPushButton("5. Экспорт графиков")
        self.btn_export.clicked.connect(self.export_plots)
        self.btn_export.setEnabled(False)

        control_layout.addWidget(self.btn_load_pm6)
        control_layout.addWidget(self.lbl_status)
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("Отображаемый канал:"))
        control_layout.addWidget(self.combo_channel)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.btn_load_logs)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.btn_apply_noise)
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("Диапазон фильтрации:"))
        control_layout.addWidget(self.combo_band)
        control_layout.addWidget(self.btn_analyze)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.btn_export)
        control_layout.addStretch()

        self.tabs = QTabWidget()
        # Автоматический пересчет при переключении вкладок
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        splitter.addWidget(control_panel)
        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 1000])

    def get_active_tab(self):
        return self.tabs.currentWidget()

    def on_tab_changed(self, index):
        """Срабатывает при клике на новую вкладку (например, переход от 3C405 к 3C144)."""
        if index >= 0 and self.df_pm6 is not None:
            self.run_analysis()

    def change_active_channel(self, channel_name):
        """Срабатывает при выборе нового канала в выпадающем списке."""
        active_tab = self.get_active_tab()
        if active_tab:
            active_tab.set_channel(channel_name)
            # Запускаем автоматический расчет спектра для нового канала
            self.run_analysis()

    def load_pm6(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Открыть PM6", "", "PM6 Files (*.PM6);;All Files (*)")
        if not filepath: return
            
        try:
            self.df_pm6 = load_pm6_data(filepath)
            self.full_time = self.df_pm6['Time_sec'].values
            
            self.tabs.clear()
            main_tab = SignalTab(self.df_pm6, self.fs)
            self.tabs.addTab(main_tab, "Полный обзор")
            
            self.lbl_status.setText(f"Загружено отсчетов: {len(self.df_pm6)}")
            self.combo_channel.setEnabled(True)
            self.btn_load_logs.setEnabled(True)
            self.btn_apply_noise.setEnabled(True)
            self.btn_analyze.setEnabled(True)
            self.btn_export.setEnabled(True)
            
            # Строим первый спектр автоматически при загрузке файла
            self.run_analysis()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def auto_clean_and_split(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Открыть regi лог", "", "Text Files (*.txt);;All Files (*)")
        if not filepath: return

        try:
            pm6_start_dt = self.df_pm6['Datetime'].iloc[0]
            
            # 1. ВЫЗОВ ИЗ ЯДРА (Вся сложная магия с датами происходит там)
            df_logs = parse_regi_with_time(filepath, pm6_start_dt)

            if df_logs.empty:
                QMessageBox.warning(self, "Пустой файл", "Не удалось найти события в файле.")
                return

            # 2. Вырезаем калибровки и зениты
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

            # 3. Создаем вкладки и собираем отчет
            unique_targets = df_logs[~df_logs['Target_Name'].isin(noise_targets)]['Target_Name'].unique()
            created_tabs, skipped_tabs = [], []

            for target in unique_targets:
                target_events = df_logs[df_logs['Target_Name'] == target]
                if target_events.empty: continue
                
                s_idx = np.searchsorted(self.full_time, target_events['Start_sec'].min())
                e_idx = np.searchsorted(self.full_time, target_events['End_sec'].max())
                
                if s_idx < e_idx and s_idx < len(self.full_time) and e_idx > 0:
                    df_slice = self.df_pm6.iloc[s_idx:e_idx].copy()
                    target_tab = SignalTab(df_slice, self.fs)
                    self.tabs.addTab(target_tab, target)
                    created_tabs.append(target)
                else:
                    skipped_tabs.append(target)

            self.change_active_channel(self.combo_channel.currentText())
            
            # Вывод отчета
            report_msg = "Обработка логов завершена.\n\n"
            report_msg += f"Созданы вкладки: {', '.join(created_tabs) if created_tabs else 'Нет'}\n\n"
            if skipped_tabs:
                report_msg += f"Пропущены (нет данных в PM6):\n{', '.join(skipped_tabs)}"
            QMessageBox.information(self, "Результат", report_msg)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось обработать лог: {str(e)}")

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
            # Авто-пересчет спектра после ручного вырезания помех
            self.run_analysis()

    def run_analysis(self):
        active_tab = self.get_active_tab()
        if not active_tab: return
        
        idx = self.combo_band.currentIndex()
        if idx == 0:
            lowcut, highcut = 1.0/150.0, 1.0/5.0
        else:
            lowcut, highcut = 1.0/600.0, 1.0/150.0
            
        try:
            filtered_sig = bandpass_filter(active_tab.raw_signal, lowcut, highcut, self.fs)
            active_tab.update_filtered(filtered_sig)
            
            img_data = compute_fsst_spectrogram(filtered_sig, self.fs)
            active_tab.update_spectrogram(img_data, lowcut, highcut)
        except Exception as e:
            # Игнорируем ошибки при пустых данных (например, когда вкладка еще только создается)
            pass

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