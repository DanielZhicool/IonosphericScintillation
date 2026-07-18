import json
import os

from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import core.config as cfg

# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------
_BUILTIN_PRESETS = {
    "Default": {
        "CWT_NV_BUBBLES": 32,
        "CWT_NV_CLOUDS": 64,
        "MORSE_GAMMA": 3.0,
        "MORSE_BETA": 30.0,
        "GAUSSIAN_SIGMA_FREQ": 1.0,
        "GAUSSIAN_SIGMA_TIME": 1.0,
        "CWT_DYNAMIC_RANGE_DB": 40.0,
        "MTM_N_TAPERS": 7,
        "MTM_NW": 4.0,
        "FTEST_CONFIDENCE": 0.95,
        "CROSS_SPECTRUM_DX": 2500.0,
        "VELOCITY_N_PEAKS": 3,
        "PCHIP_FACTOR": 3,
    },
    "High Resolution": {
        "CWT_NV_BUBBLES": 64,
        "CWT_NV_CLOUDS": 128,
        "MORSE_GAMMA": 3.0,
        "MORSE_BETA": 50.0,
        "GAUSSIAN_SIGMA_FREQ": 0.5,
        "GAUSSIAN_SIGMA_TIME": 0.5,
        "CWT_DYNAMIC_RANGE_DB": 60.0,
        "MTM_N_TAPERS": 10,
        "MTM_NW": 4.0,
        "FTEST_CONFIDENCE": 0.999,
        "CROSS_SPECTRUM_DX": 2500.0,
        "VELOCITY_N_PEAKS": 5,
        "PCHIP_FACTOR": 4,
    },
    "Fast Preview": {
        "CWT_NV_BUBBLES": 16,
        "CWT_NV_CLOUDS": 32,
        "MORSE_GAMMA": 3.0,
        "MORSE_BETA": 20.0,
        "GAUSSIAN_SIGMA_FREQ": 1.5,
        "GAUSSIAN_SIGMA_TIME": 1.5,
        "CWT_DYNAMIC_RANGE_DB": 40.0,
        "MTM_N_TAPERS": 5,
        "MTM_NW": 4.0,
        "FTEST_CONFIDENCE": 0.95,
        "CROSS_SPECTRUM_DX": 2500.0,
        "VELOCITY_N_PEAKS": 3,
        "PCHIP_FACTOR": 2,
    },
}


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(430, 680)
        layout = QVBoxLayout(self)

        # --- Preset bar ---
        group_preset = QGroupBox("Presets")
        preset_layout = QVBoxLayout()

        # 1. Built-in presets
        row_builtin = QHBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["Custom"] + list(_BUILTIN_PRESETS.keys()))
        model = self.combo_preset.model()
        if isinstance(model, QStandardItemModel):
            item = model.item(0)
            if item is not None:
                item.setEnabled(False)
        self.combo_preset.setToolTip("Select a built-in preset to instantly populate the fields below")
        self.combo_preset.currentIndexChanged.connect(self._load_preset)

        row_builtin.addWidget(QLabel("Preset:"))
        row_builtin.addWidget(self.combo_preset, stretch=1)
        preset_layout.addLayout(row_builtin)

        # 2. File operations
        row_file = QHBoxLayout()
        btn_save_file = QPushButton("Save to file…")
        btn_save_file.setToolTip("Save current form values as a JSON preset file")
        btn_save_file.clicked.connect(self._save_to_file)
        btn_load_file = QPushButton("Load from file…")
        btn_load_file.setToolTip("Load a previously saved JSON preset file")
        btn_load_file.clicked.connect(self._load_from_file)

        row_file.addWidget(btn_save_file)
        row_file.addWidget(btn_load_file)
        preset_layout.addLayout(row_file)

        group_preset.setLayout(preset_layout)
        layout.addWidget(group_preset)

        # --- Parameter form ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.form_layout = QFormLayout(content)

        self.inputs = {}

        # CWT Params
        self.add_spinbox("CWT_NV Bubbles (Voices per octave)", "CWT_NV_BUBBLES", cfg.CWT_NV_BUBBLES, 16, 512, step=16)
        self.add_spinbox("CWT_NV Clouds (Voices per octave)", "CWT_NV_CLOUDS", cfg.CWT_NV_CLOUDS, 16, 512, step=16)
        self.add_double_spinbox("MORSE_GAMMA (Wavelet symmetry)", "MORSE_GAMMA", cfg.MORSE_GAMMA, 1.0, 10.0, step=1.0)
        self.add_double_spinbox(
            "MORSE_BETA (Wavelet time-bandwidth)", "MORSE_BETA", cfg.MORSE_BETA, 1.0, 120.0, step=1.0
        )
        self.add_double_spinbox(
            "GAUSSIAN_SIGMA_FREQ (Blur frequency)", "GAUSSIAN_SIGMA_FREQ", cfg.GAUSSIAN_SIGMA_FREQ, 0.0, 10.0, step=0.1
        )
        self.add_double_spinbox(
            "GAUSSIAN_SIGMA_TIME (Blur time)", "GAUSSIAN_SIGMA_TIME", cfg.GAUSSIAN_SIGMA_TIME, 0.0, 10.0, step=0.1
        )
        self.add_double_spinbox(
            "CWT_DYNAMIC_RANGE_DB", "CWT_DYNAMIC_RANGE_DB", cfg.CWT_DYNAMIC_RANGE_DB, 10.0, 100.0, step=5.0
        )

        # Spectral Params
        self.add_spinbox("MTM_N_TAPERS (DPSS tapers)", "MTM_N_TAPERS", cfg.MTM_N_TAPERS, 1, 20, step=1)
        self.add_double_spinbox("MTM_NW (Time-bandwidth product)", "MTM_NW", cfg.MTM_NW, 1.0, 10.0, step=0.5)
        self.add_double_spinbox("FTEST_CONFIDENCE", "FTEST_CONFIDENCE", cfg.FTEST_CONFIDENCE, 0.5, 0.999, step=0.01)

        self.add_double_spinbox(
            "CROSS_SPECTRUM_DX (Baseline m)", "CROSS_SPECTRUM_DX", cfg.CROSS_SPECTRUM_DX, 100.0, 10000.0, step=100.0
        )
        self.add_spinbox("VELOCITY_N_PEAKS", "VELOCITY_N_PEAKS", cfg.VELOCITY_N_PEAKS, 1, 10, step=1)
        self.add_spinbox("PCHIP_FACTOR (Upsampling)", "PCHIP_FACTOR", cfg.PCHIP_FACTOR, 1, 10, step=1)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.apply_settings)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._match_current_config_to_preset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def add_spinbox(self, label, name, current_val, min_val, max_val, step):
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(int(current_val))
        sb.setSingleStep(step)
        sb.valueChanged.connect(self._on_field_edited)
        self.form_layout.addRow(label, sb)
        self.inputs[name] = sb

    def add_double_spinbox(self, label, name, current_val, min_val, max_val, step):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(float(current_val))
        sb.setSingleStep(step)
        sb.setDecimals(3)
        sb.valueChanged.connect(self._on_field_edited)
        self.form_layout.addRow(label, sb)
        self.inputs[name] = sb

    def _current_values(self) -> dict:
        """Return current form values as a plain dict."""
        return {name: widget.value() for name, widget in self.inputs.items()}

    def _apply_values(self, values: dict):
        """Populate form widgets from a dict (ignores unknown keys)."""
        # Block signals globally so setting values doesn't trigger _on_field_edited
        for widget in self.inputs.values():
            widget.blockSignals(True)

        for name, val in values.items():
            if name in self.inputs:
                self.inputs[name].setValue(val)

        for widget in self.inputs.values():
            widget.blockSignals(False)

    def _on_field_edited(self):
        """When a user manually edits a field, switch dropdown to Custom."""
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentText("Custom")
        self.combo_preset.blockSignals(False)

    def _match_current_config_to_preset(self):
        """Check if current form values exactly match any preset, and update the dropdown."""
        current_vals = self._current_values()
        for preset_name, preset_vals in _BUILTIN_PRESETS.items():
            match = True
            for k, v in preset_vals.items():
                if k not in current_vals or abs(current_vals[k] - v) > 1e-5:
                    match = False
                    break
            if match:
                self.combo_preset.blockSignals(True)
                self.combo_preset.setCurrentText(preset_name)
                self.combo_preset.blockSignals(False)
                return

        # No match found, set to Custom
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentText("Custom")
        self.combo_preset.blockSignals(False)

    # ------------------------------------------------------------------
    # Preset actions
    # ------------------------------------------------------------------
    def _load_preset(self):
        preset_name = self.combo_preset.currentText()
        if preset_name in _BUILTIN_PRESETS:
            self._apply_values(_BUILTIN_PRESETS[preset_name])

    def _save_to_file(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", os.path.expanduser("~/preset.json"), "JSON Files (*.json)"
        )
        if filepath:
            try:
                with open(filepath, "w") as f:
                    json.dump(self._current_values(), f, indent=2)
                QMessageBox.information(self, "Preset Saved", f"Preset saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _load_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Preset", os.path.expanduser("~"), "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath) as f:
                    values = json.load(f)
                self._apply_values(values)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    # ------------------------------------------------------------------
    # Apply to live config
    # ------------------------------------------------------------------
    def apply_settings(self):
        cfg.CWT_NV_BUBBLES = self.inputs["CWT_NV_BUBBLES"].value()
        cfg.CWT_NV_CLOUDS = self.inputs["CWT_NV_CLOUDS"].value()
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
        parent = self.parent()
        if parent is not None:
            run_analysis = getattr(parent, "run_analysis", None)
            if run_analysis is not None:
                run_analysis()
