"""
Dataclasses and result containers for spectral analysis and signal processing pipelines.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultitaperPSDResult:
    """
    Result container for Thomson Multitaper Power Spectral Density with optional Jackknife 95% CIs.
    """

    freqs: np.ndarray
    psd: np.ndarray
    log_psd_se: np.ndarray | None = None
    ci95_low: np.ndarray | None = None
    ci95_high: np.ndarray | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        """Enable tuple unpacking: freqs, psd = result."""
        return iter((self.freqs, self.psd))


@dataclass(frozen=True)
class VelocityEstimate:
    """
    Structured result for an ionospheric drift velocity estimate derived from cross-phase regression.
    """

    peak_freq: float
    period: float
    mean_coherence: float
    phase_slope: float  # dphi/df in rad/Hz
    dt: float  # time delay in seconds
    dt_ci95: tuple[float, float]  # 95% confidence interval for dt
    velocity: float  # horizontal velocity in m/s (signed)
    velocity_ci95: tuple[float, float]  # 95% confidence interval for velocity magnitude
    is_valid: bool  # True if coherence >= threshold and fit succeeded
    gating_reason: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Enable dict-like indexing v['velocity'] for backward compatibility."""
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for backward compatibility with GUI/export components."""
        return {
            "period": self.period,
            "phase_deg": np.degrees(self.phase_slope * self.peak_freq),
            "dt": self.dt,
            "velocity": self.velocity,
            "mean_coherence": self.mean_coherence,
            "is_valid": self.is_valid,
            "gating_reason": self.gating_reason,
            "dt_ci95": self.dt_ci95,
            "velocity_ci95": self.velocity_ci95,
        }


@dataclass(frozen=True)
class TimeFrequencyResult:
    """
    Result container for CWT / Synchrosqueezed time-frequency spectrogram analysis.
    """

    time_sec: np.ndarray
    freqs: np.ndarray
    spectrogram: np.ndarray
    raw_coefficients: np.ndarray | None = None
    decimation_factor: int = 1
