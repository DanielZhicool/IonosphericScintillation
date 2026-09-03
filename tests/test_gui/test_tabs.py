"""Automated GUI integration tests for SignalTab and SpectralTab widgets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytestqt.qtbot import QtBot

from core.parsers import extract_pm_signals
from core.spectral_analysis import run_spectral_pipeline
from gui.spectral_tab import SpectralTab
from gui.tabs import SignalTab


def test_signal_tab_lifecycle(qtbot: QtBot, mock_dataset: pd.DataFrame) -> None:
    start_dt = mock_dataset["Datetime"].iloc[0]
    tab = SignalTab(mock_dataset, start_dt, tab_name="3C144", fs=1.0)
    qtbot.addWidget(tab)

    assert tab.tab_name == "3C144"
    assert tab.current_channel == "P1_20A"
    assert len(tab.raw_signal) == 600

    # Test channel switching
    tab.set_channel("P2_20B")
    assert tab.current_channel == "P2_20B"

    # Test filtered curve update
    filtered = np.random.randn(600)
    tab.update_filtered(filtered)
    assert tab.curve_filtered.xData is not None
    assert len(tab.curve_filtered.xData) == 600

    # Test spectrogram update
    spec_img = np.random.rand(64, 200)
    tab.update_spectrogram(spec_img, lowcut=1.0 / 150.0, highcut=1.0 / 5.0)
    assert tab.img_spec.image is not None


def test_spectral_tab_rendering(qtbot: QtBot, mock_dataset: pd.DataFrame) -> None:
    pm_signals = extract_pm_signals(mock_dataset)
    res_small = run_spectral_pipeline(
        pm_signals,
        fs=1.0,
        lowcut=1.0 / 150.0,
        highcut=1.0 / 5.0,
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=True,
    )
    res_large = run_spectral_pipeline(
        pm_signals,
        fs=1.0,
        lowcut=1.0 / 600.0,
        highcut=1.0 / 150.0,
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=True,
    )
    band_results = {"small": res_small, "large": res_large}

    spectral_tab = SpectralTab(source_name="3C144", band_results=band_results)
    qtbot.addWidget(spectral_tab)

    assert spectral_tab.source_name == "3C144"
    assert spectral_tab.band_tabs.count() == 2
    assert "Small bubbles" in spectral_tab.band_tabs.tabText(0)
    assert "Large clouds" in spectral_tab.band_tabs.tabText(1)
