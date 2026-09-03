"""Automated GUI integration tests for Qt background worker threads."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytestqt.qtbot import QtBot

from core.parsers import extract_pm_signals
from gui.workers import SignalAnalysisWorker, SpectralAnalysisWorker


def test_signal_analysis_worker_execution(qtbot: QtBot, mock_dataset: pd.DataFrame) -> None:
    raw_sig = np.asarray(mock_dataset["P1_20A"].to_numpy(), dtype=float)
    worker = SignalAnalysisWorker(
        raw_signal=raw_sig,
        fs=1.0,
        lowcut=1.0 / 150.0,
        highcut=1.0 / 5.0,
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=True,
    )

    with qtbot.waitSignal(worker.finished, timeout=10000) as blocker:
        worker.start()

    assert blocker.args is not None
    filtered_sig, img_data = blocker.args[0]
    assert len(filtered_sig) == len(raw_sig)
    assert isinstance(img_data, np.ndarray)


def test_spectral_analysis_worker_execution(qtbot: QtBot, mock_dataset: pd.DataFrame) -> None:
    pm_signals = extract_pm_signals(mock_dataset)
    worker = SpectralAnalysisWorker(
        pm_signals=pm_signals,
        fs=1.0,
        signal_duration=600.0,
        bands=[("small", 1.0 / 150.0, 1.0 / 5.0)],
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=True,
    )

    progress_emitted: list[int] = []
    worker.progress.connect(progress_emitted.append)

    with qtbot.waitSignal(worker.finished, timeout=15000) as blocker:
        worker.start()

    assert blocker.args is not None
    band_results = blocker.args[0]
    assert "small" in band_results
    assert band_results["small"] is not None
    assert len(progress_emitted) > 0
