import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QComboBox, QSplitter, QMessageBox)
from PySide6.QtCore import Qt
from scipy.signal import butter, filtfilt
from ssqueezepy import ssq_stft

# Utility functions for data loading, filtering, and noise generation

def load_pm6_data(filepath):
    """Loading PM6 data and converting time (Excel Date)."""
    columns = ['MJD', 'P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 
               'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']
    df = pd.read_csv(filepath, sep=r'\s+', names=columns, header=30, encoding_errors='ignore')
    df['Datetime'] = pd.to_datetime(df['MJD'], unit='D', origin='1899-12-30')
    df['Time_sec'] = (df['Datetime'] - df['Datetime'].iloc[0]).dt.total_seconds()
    return df

def parse_regi_file(filepath):
    """Parsing log file (regi). Returns DataFrame of events."""
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[1] == '-':
                try:
                    start_time = parts[0]
                    end_time = parts[2].replace('(', '').replace(')', '')
                    name = parts[-1]
                    events.append({
                        'Start_Time': start_time,
                        'End_Time': end_time,
                        'Target_Name': name
                    })
                except IndexError:
                    continue
    return pd.DataFrame(events)

def fill_gap_with_red_noise(signal, start_idx, end_idx, context_window=50):
    """Replaces the segment with red noise based on surrounding context."""
    left_context = signal[max(0, start_idx - context_window) : start_idx]
    right_context = signal[end_idx : min(len(signal), end_idx + context_window)]
    
    valid_context = np.concatenate([left_context, right_context])
    if len(valid_context) == 0:
        return signal 
        
    target_mean = np.mean(valid_context)
    target_std = np.std(valid_context)
    
    length = end_idx - start_idx
    r = 0.95
    white_noise = np.random.normal(0, 1, length)
    red_noise = np.zeros(length)
    
    red_noise[0] = white_noise[0]
    for i in range(1, length):
        red_noise[i] = r * red_noise[i-1] + np.sqrt(1 - r**2) * white_noise[i]
        
    red_noise = red_noise - np.mean(red_noise)
    red_noise = (red_noise / (np.std(red_noise) + 1e-9)) * target_std
    red_noise = red_noise + target_mean
    
    signal_cleaned = signal.copy()
    signal_cleaned[start_idx:end_idx] = red_noise
    return signal_cleaned

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Двунаправленный фильтр Баттерворта."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ==========================================
# ПРЕДСТАВЛЕНИЕ И КОНТРОЛЛЕР (GUI)
# ==========================================

class Uran4App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("URAN-4 Ionospheric Scintillation Analyzer")
        self.resize(1200, 800)
        
        self.df = None
        self.raw_signal = None
        self.time_sec = None
        self.fs = 1.0 # 1 Гц (шаг 1 секунда)
        
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ЛЕВАЯ ПАНЕЛЬ (Управление)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 1. Загрузка PM6
        self.btn_load_pm6 = QPushButton("1. Загрузить данные (PM6)")
        self.btn_load_pm6.clicked.connect(self.load_pm6)
        
        self.lbl_status_pm6 = QLabel("Файл PM6 не загружен")
        self.lbl_status_pm6.setWordWrap(True)

        # 2. Загрузка логов и авто-очистка
        self.btn_load_logs = QPushButton("2. Загрузить логи (regi) и авто-очистить")
        self.btn_load_logs.clicked.connect(self.auto_clean_from_logs)
        self.btn_load_logs.setEnabled(False)

        # 3. Ручное удаление (на случай если логи пропустили помеху)
        self.btn_apply_noise = QPushButton("3. Вырезать зону вручную (по графику)")
        self.btn_apply_noise.clicked.connect(self.manual_clean_region)
        self.btn_apply_noise.setEnabled(False)

        # 4. Фильтрация и спектр
        self.combo_band = QComboBox()
        self.combo_band.addItems([
            "Малые пузырьки (5 - 150 сек)", 
            "Большие облака (150 - 600 сек)"
        ])
        
        self.btn_analyze = QPushButton("4. Построить спектр мерцаний (FSST)")
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setEnabled(False)
        
        self.lbl_status_analysis = QLabel("")
        self.lbl_status_analysis.setWordWrap(True)

        # Сборка панели управления
        control_layout.addWidget(self.btn_load_pm6)
        control_layout.addWidget(self.lbl_status_pm6)
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("Автоматическая очистка калибровок:"))
        control_layout.addWidget(self.btn_load_logs)
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("Дополнительная ручная зачистка:"))
        control_layout.addWidget(self.btn_apply_noise)
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("Диапазон фильтрации:"))
        control_layout.addWidget(self.combo_band)
        control_layout.addWidget(self.btn_analyze)
        control_layout.addWidget(self.lbl_status_analysis)
        control_layout.addStretch()

        # ПРАВАЯ ПАНЕЛЬ (Графики PyQtGraph)
        self.graph_widget = pg.GraphicsLayoutWidget()
        
        # Сырые данные
        self.p1 = self.graph_widget.addPlot(title="Сырые данные (Полная мощность 20 МГц, Пол. А)")
        self.p1.showGrid(x=True, y=True)
        self.curve_raw = self.p1.plot(pen='b')
        self.p1.addItem(self.region, ignoreBounds=True)
        self.graph_widget.nextRow()
        
        # Отфильтрованные данные
        self.p2 = self.graph_widget.addPlot(title="Выделенные ионосферные мерцания")
        self.p2.showGrid(x=True, y=True)
        self.p2.setXLink(self.p1)
        self.curve_filtered = self.p2.plot(pen='g')
        self.graph_widget.nextRow()

        # Спектрограмма
        self.p3 = self.graph_widget.addPlot(title="FSST Спектрограмма")
        self.p3.setXLink(self.p1)
        self.img_spec = pg.ImageItem()
        self.p3.addItem(self.img_spec)
        self.img_spec.setColorMap(pg.colormap.get('viridis'))

        splitter.addWidget(control_panel)
        splitter.addWidget(self.graph_widget)
        splitter.setSizes([300, 900])

    # ==========================================
    # СЛОТЫ ИНТЕРФЕЙСА
    # ==========================================

    def load_pm6(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Открыть PM6", "", "PM6 Files (*.PM6);;All Files (*)")
        if not filepath: return
            
        try:
            self.df = load_pm6_data(filepath)
            self.time_sec = self.df['Time_sec'].values
            self.raw_signal = self.df['P1_20A'].values
            
            self.curve_raw.setData(self.time_sec, self.raw_signal)
            self.region.setRegion([self.time_sec[0], self.time_sec[min(100, len(self.time_sec)-1)]])
            
            start_dt = self.df['Datetime'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
            self.lbl_status_pm6.setText(f"Загружено отсчетов: {len(self.df)}\nНачало: {start_dt}")
            
            self.btn_load_logs.setEnabled(True)
            self.btn_apply_noise.setEnabled(True)
            self.btn_analyze.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки PM6", str(e))

    def auto_clean_from_logs(self):
        """Парсинг логов и автоматическое вырезание калибровок."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Открыть regi лог", "", "Text Files (*.txt);;All Files (*)")
        if not filepath: return

        try:
            df_logs = parse_regi_file(filepath)
            calibrations = df_logs[df_logs['Target_Name'] == 'calibrovka']
            
            if calibrations.empty:
                QMessageBox.information(self, "Информация", "В файле логов не найдено интервалов 'calibrovka'.")
                return

            pm6_start_dt = self.df['Datetime'].iloc[0]
            base_date = pm6_start_dt.date()
            cut_count = 0

            # Синхронизация времени
            for _, row in calibrations.iterrows():
                try:
                    start_dt = pd.to_datetime(f"{base_date} {row['Start_Time']}:00")
                    end_dt = pd.to_datetime(f"{base_date} {row['End_Time']}:00")
                    
                    # Обработка смены суток (если время в логе ушло за полночь по отношению к старту PM6)
                    if start_dt < pm6_start_dt - pd.Timedelta(hours=12):
                        start_dt += pd.Timedelta(days=1)
                        end_dt += pd.Timedelta(days=1)
                    if end_dt < start_dt:
                        end_dt += pd.Timedelta(days=1)

                    # Перевод в относительные секунды
                    start_sec = (start_dt - pm6_start_dt).total_seconds()
                    end_sec = (end_dt - pm6_start_dt).total_seconds()

                    # Перевод в индексы массива
                    start_idx = np.searchsorted(self.time_sec, start_sec)
                    end_idx = np.searchsorted(self.time_sec, end_sec)

                    # Проверка границ массива
                    if 0 <= start_idx < len(self.raw_signal) and start_idx < end_idx:
                        # Вшиваем красный шум
                        self.raw_signal = fill_gap_with_red_noise(
                            self.raw_signal, start_idx, min(end_idx, len(self.raw_signal))
                        )
                        cut_count += 1
                except Exception as iter_e:
                    print(f"Ошибка парсинга строки калибровки: {iter_e}")
                    continue
            
            # Обновление графика
            self.curve_raw.setData(self.time_sec, self.raw_signal)
            QMessageBox.information(self, "Успех", f"Успешно вырезано {cut_count} калибровочных ступеней. \nПустоты заполнены красным шумом.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка обработки логов", str(e))

    def manual_clean_region(self):
        """Ручное вырезание по синему графику (LinearRegionItem)."""
        if self.raw_signal is None: return
        min_x, max_x = self.region.getRegion()
        
        start_idx = np.searchsorted(self.time_sec, min_x)
        end_idx = np.searchsorted(self.time_sec, max_x)
        
        if start_idx < end_idx:
            self.raw_signal = fill_gap_with_red_noise(self.raw_signal, start_idx, end_idx)
            self.curve_raw.setData(self.time_sec, self.raw_signal)

    def run_analysis(self):
        """Запуск полосовой фильтрации и FSST."""
        if self.raw_signal is None: return
        
        idx = self.combo_band.currentIndex()
        if idx == 0:
            lowcut, highcut = 1.0/150.0, 1.0/5.0
        else:
            lowcut, highcut = 1.0/600.0, 1.0/150.0
            
        filtered_sig = bandpass_filter(self.raw_signal, lowcut, highcut, self.fs)
        self.curve_filtered.setData(self.time_sec, filtered_sig)
        
        try:
            self.lbl_status_analysis.setText("Расчет FSST... Пожалуйста, подождите.")
            QApplication.processEvents() 
            
            Tx, _, freqs, _ = ssq_stft(filtered_sig, fs=self.fs)
            Tx_abs = np.abs(Tx)
            img_data = Tx_abs.T 
            
            self.img_spec.setImage(img_data, autoLevels=True)
            
            freq_range = self.fs / 2
            self.img_spec.setRect(pg.QtCore.QRectF(self.time_sec[0], 0, self.time_sec[-1]-self.time_sec[0], freq_range))
            self.p3.setYRange(lowcut, highcut) 
            
            self.lbl_status_analysis.setText("Анализ (Спектрограмма) завершен успешно.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", str(e))
            self.lbl_status_analysis.setText("Ошибка при расчете.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Uran4App()
    window.show()
    sys.exit(app.exec())