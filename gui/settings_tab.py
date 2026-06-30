import core.config as cfg
from PySide6.QtWidgets import QDialog, QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton, QVBoxLayout, QScrollArea, QMessageBox

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(400, 600)
        layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.form_layout = QFormLayout(content)
        
        self.inputs = {}
        
        # CWT Params
        self.add_spinbox("CWT_NV (Voices per octave)", "CWT_NV", cfg.CWT_NV, 16, 512, step=16)
        self.add_double_spinbox("MORSE_GAMMA (Wavelet symmetry)", "MORSE_GAMMA", cfg.MORSE_GAMMA, 1.0, 10.0, step=1.0)
        self.add_double_spinbox("MORSE_BETA (Wavelet time-bandwidth)", "MORSE_BETA", cfg.MORSE_BETA, 1.0, 120.0, step=1.0)
        self.add_double_spinbox("GAUSSIAN_SIGMA_FREQ (Blur frequency)", "GAUSSIAN_SIGMA_FREQ", cfg.GAUSSIAN_SIGMA_FREQ, 0.0, 10.0, step=0.1)
        self.add_double_spinbox("GAUSSIAN_SIGMA_TIME (Blur time)", "GAUSSIAN_SIGMA_TIME", cfg.GAUSSIAN_SIGMA_TIME, 0.0, 10.0, step=0.1)
        self.add_double_spinbox("CWT_DYNAMIC_RANGE_DB", "CWT_DYNAMIC_RANGE_DB", cfg.CWT_DYNAMIC_RANGE_DB, 10.0, 100.0, step=5.0)
        
        # Spectral Params
        self.add_spinbox("MTM_N_TAPERS (DPSS tapers)", "MTM_N_TAPERS", cfg.MTM_N_TAPERS, 1, 20, step=1)
        self.add_double_spinbox("MTM_NW (Time-bandwidth product)", "MTM_NW", cfg.MTM_NW, 1.0, 10.0, step=0.5)
        self.add_double_spinbox("FTEST_CONFIDENCE", "FTEST_CONFIDENCE", cfg.FTEST_CONFIDENCE, 0.5, 0.999, step=0.01)
        
        self.add_double_spinbox("CROSS_SPECTRUM_DX (Baseline m)", "CROSS_SPECTRUM_DX", cfg.CROSS_SPECTRUM_DX, 100.0, 10000.0, step=100.0)
        self.add_spinbox("VELOCITY_N_PEAKS", "VELOCITY_N_PEAKS", cfg.VELOCITY_N_PEAKS, 1, 10, step=1)
        self.add_spinbox("PCHIP_FACTOR (Upsampling)", "PCHIP_FACTOR", cfg.PCHIP_FACTOR, 1, 10, step=1)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        btn_apply = QPushButton("Apply Settings")
        btn_apply.clicked.connect(self.apply_settings)
        layout.addWidget(btn_apply)
        
    def add_spinbox(self, label, name, current_val, min_val, max_val, step):
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(int(current_val))
        sb.setSingleStep(step)
        self.form_layout.addRow(label, sb)
        self.inputs[name] = sb

    def add_double_spinbox(self, label, name, current_val, min_val, max_val, step):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(float(current_val))
        sb.setSingleStep(step)
        self.form_layout.addRow(label, sb)
        self.inputs[name] = sb

    def apply_settings(self):
        cfg.CWT_NV = self.inputs["CWT_NV"].value()
        cfg.MORSE_GAMMA = self.inputs["MORSE_GAMMA"].value()
        cfg.MORSE_BETA = self.inputs["MORSE_BETA"].value()
        cfg.GAUSSIAN_SIGMA_FREQ = self.inputs["GAUSSIAN_SIGMA_FREQ"].value()
        cfg.GAUSSIAN_SIGMA_TIME = self.inputs["GAUSSIAN_SIGMA_TIME"].value()
        cfg.CWT_DYNAMIC_RANGE_DB = self.inputs["CWT_DYNAMIC_RANGE_DB"].value()
        
        cfg.MTM_N_TAPERS = self.inputs["MTM_N_TAPERS"].value()
        cfg.MTM_NW = self.inputs["MTM_NW"].value()
        cfg.FTEST_CONFIDENCE = self.inputs["FTEST_CONFIDENCE"].value()
        
        cfg.CROSS_SPECTRUM_DX = self.inputs["CROSS_SPECTRUM_DX"].value()
        cfg.VELOCITY_N_PEAKS = self.inputs["VELOCITY_N_PEAKS"].value()
        cfg.PCHIP_FACTOR = self.inputs["PCHIP_FACTOR"].value()
        
        self.accept()
        if self.parent() and hasattr(self.parent(), "run_analysis"):
            self.parent().run_analysis()
