import numpy as np
import pytest

from core.config import ProcessingConfig
from core.signal_processing import bandpass_filter, clean_and_smooth_signal


def test_bandpass_filter_validation() -> None:
    """Verify boundary checks and parameter validation in bandpass_filter."""
    fs = 100.0  # Nyquist is 50.0 Hz
    data = np.random.randn(500)

    # Valid run
    filtered = bandpass_filter(data, lowcut=1.0, highcut=20.0, fs=fs)
    assert len(filtered) == len(data)

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        bandpass_filter(np.array([]), lowcut=1.0, highcut=20.0, fs=fs)

    # Invalid fs
    with pytest.raises(ValueError, match="fs must be positive"):
        bandpass_filter(data, lowcut=1.0, highcut=20.0, fs=0.0)

    # lowcut <= 0
    with pytest.raises(ValueError, match="lowcut must be > 0"):
        bandpass_filter(data, lowcut=0.0, highcut=20.0, fs=fs)

    # highcut <= lowcut
    with pytest.raises(ValueError, match="greater than lowcut"):
        bandpass_filter(data, lowcut=20.0, highcut=10.0, fs=fs)

    # highcut >= Nyquist (50.0 Hz)
    with pytest.raises(ValueError, match="strictly less than Nyquist"):
        bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=fs)


def test_clean_and_smooth_signal_validation() -> None:
    """Verify boundary checks in clean_and_smooth_signal."""
    data = np.random.randn(100)

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        clean_and_smooth_signal(np.array([]))

    # Invalid window_size
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        clean_and_smooth_signal(data, window_size=0)

    # Invalid n_sigmas
    with pytest.raises(ValueError, match="n_sigmas must be > 0"):
        clean_and_smooth_signal(data, n_sigmas=-1.0)


def test_processing_config_immutability() -> None:
    """Verify that ProcessingConfig is immutable and holds expected defaults."""
    cfg_inst = ProcessingConfig(sampling_rate=10.0, tukey_alpha=0.1)
    assert cfg_inst.sampling_rate == 10.0
    assert cfg_inst.tukey_alpha == 0.1

    with pytest.raises(AttributeError):
        cfg_inst.sampling_rate = 20.0  # type: ignore[misc] # Frozen dataclass prevents mutation
