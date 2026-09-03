"""Automated GUI integration tests for Uran4App main window."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytestqt.qtbot import QtBot

from gui.main_window import Uran4App
from gui.tabs import SignalTab


def test_main_window_init(qtbot: QtBot) -> None:
    app = Uran4App()
    qtbot.addWidget(app)

    assert app.windowTitle() == "URAN-4 Ionospheric Scintillation Analyzer"
    assert app.df_pm6 is None
    assert not app.combo_channel.isEnabled()
    assert not app.btn_analyze.isEnabled()


def test_main_window_data_loaded_state(qtbot: QtBot, mock_dataset: pd.DataFrame) -> None:
    app = Uran4App()
    qtbot.addWidget(app)

    app.df_pm6 = mock_dataset
    app.df_pm6_original = mock_dataset.copy()
    app.full_time = np.asarray(mock_dataset["Time_sec"].to_numpy(), dtype=float)
    app.pm6_start_dt = mock_dataset["Datetime"].iloc[0]

    main_tab = SignalTab(
        app.df_pm6,
        app.pm6_start_dt,
        tab_name="Full Overview",
        fs=app.fs,
        full_datetime_series=np.asarray(app.df_pm6["Datetime"].to_numpy()),
    )
    app.tabs.addTab(main_tab, "Full Overview")
    app.set_widgets_enabled(True)

    assert app.combo_channel.isEnabled()
    assert app.btn_analyze.isEnabled()
    assert app.get_active_tab() is main_tab
    assert len(app.get_all_signal_tabs()) == 1


def test_main_window_channel_switching(qtbot: QtBot, mock_dataset: pd.DataFrame) -> None:
    app = Uran4App()
    qtbot.addWidget(app)

    app.df_pm6 = mock_dataset
    app.df_pm6_original = mock_dataset.copy()
    app.full_time = np.asarray(mock_dataset["Time_sec"].to_numpy(), dtype=float)
    app.pm6_start_dt = mock_dataset["Datetime"].iloc[0]

    main_tab = SignalTab(app.df_pm6, app.pm6_start_dt, tab_name="Full Overview", fs=app.fs)
    app.tabs.addTab(main_tab, "Full Overview")

    # Switch channel to P3_25A
    app.combo_channel.setCurrentText("P3_25A")
    assert main_tab.current_channel == "P3_25A"


def test_main_window_marker_toggle(qtbot: QtBot, mock_dataset: pd.DataFrame) -> None:
    app = Uran4App()
    qtbot.addWidget(app)

    app.df_pm6 = mock_dataset
    app.pm6_start_dt = mock_dataset["Datetime"].iloc[0]
    main_tab = SignalTab(app.df_pm6, app.pm6_start_dt, tab_name="Full Overview", fs=app.fs)
    app.tabs.addTab(main_tab, "Full Overview")

    app.check_markers.setChecked(False)
    app.toggle_markers()
    assert not app.check_markers.isChecked()

    app.check_markers.setChecked(True)
    app.toggle_markers()
    assert app.check_markers.isChecked()


def test_main_window_close_event_safety(qtbot: QtBot) -> None:
    app = Uran4App()
    qtbot.addWidget(app)
    app.close()
