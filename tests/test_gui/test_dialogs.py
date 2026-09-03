"""Automated GUI integration tests for SettingsDialog and BatchExportDialog."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pytestqt.qtbot import QtBot

from core.config import ProcessingConfig
from gui.batch_export_dialog import BatchExportDialog
from gui.settings_tab import SettingsDialog


def test_settings_dialog_presets(qtbot: QtBot) -> None:
    initial_cfg = ProcessingConfig()
    dialog = SettingsDialog(current_config=initial_cfg)
    qtbot.addWidget(dialog)

    assert dialog.combo_preset.currentText() == "Default"

    # Switch to High Resolution preset
    dialog.combo_preset.setCurrentText("High Resolution")
    cfg_high = dialog.get_config()
    assert cfg_high.cwt_nv_bubbles == 64
    assert cfg_high.mtm_n_tapers == 10

    # Switch to Fast Preview preset
    dialog.combo_preset.setCurrentText("Fast Preview")
    cfg_fast = dialog.get_config()
    assert cfg_fast.cwt_nv_bubbles == 16
    assert cfg_fast.mtm_n_tapers == 5


def test_batch_export_dialog_init(
    qtbot: QtBot, mock_dataset: pd.DataFrame, mock_sessions: list[dict[str, Any]]
) -> None:
    start_dt = mock_dataset["Datetime"].iloc[0]
    dialog = BatchExportDialog(
        df_pm6=mock_dataset,
        df_pm6_original=mock_dataset.copy(),
        sessions=mock_sessions,
        start_datetime=start_dt,
        fs=1.0,
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=True,
    )
    qtbot.addWidget(dialog)

    # Verify sessions list includes Full Overview + mock sessions
    assert dialog.list_sessions.count() == 3
    assert dialog.list_channels.count() == 12


def test_batch_export_dialog_cancel(
    qtbot: QtBot, mock_dataset: pd.DataFrame, mock_sessions: list[dict[str, Any]]
) -> None:
    start_dt = mock_dataset["Datetime"].iloc[0]
    dialog = BatchExportDialog(
        df_pm6=mock_dataset,
        df_pm6_original=mock_dataset.copy(),
        sessions=mock_sessions,
        start_datetime=start_dt,
        fs=1.0,
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=True,
    )
    qtbot.addWidget(dialog)
    dialog.close_or_cancel()


def test_settings_dialog_load_config_and_provenance(qtbot: QtBot) -> None:
    dialog = SettingsDialog(current_config=ProcessingConfig())
    qtbot.addWidget(dialog)

    # 1. Apply lowercase dataclass keys
    dialog._apply_values({"cwt_nv_bubbles": 64, "mtm_n_tapers": 10})
    conf = dialog.get_config()
    assert conf.cwt_nv_bubbles == 64
    assert conf.mtm_n_tapers == 10

    # 2. Apply nested provenance structure
    dialog._apply_values({"config": {"cwt_nv_bubbles": 16, "mtm_n_tapers": 5}})
    conf2 = dialog.get_config()
    assert conf2.cwt_nv_bubbles == 16
    assert conf2.mtm_n_tapers == 5
