import numpy as np
import pytest

from core.signal_processing import upsample_pchip


@pytest.mark.parametrize("factor", [2, 3, 5])
def test_upsample_pchip_time_grid_math(factor: int) -> None:
    """Verify that upsample_pchip maintains exact time grid spacing and endpoint alignment across factors."""
    fs = 10.0  # 10 Hz
    N = 101  # 10 seconds of data (100 intervals)
    t = np.arange(N) / fs
    signal = np.sin(2 * np.pi * 1.0 * t)

    upsampled_signal, new_fs = upsample_pchip(signal, fs=fs, factor=factor)

    expected_new_fs = fs * factor
    expected_length = (N - 1) * factor + 1
    t_new = np.arange(len(upsampled_signal)) / new_fs

    assert new_fs == pytest.approx(expected_new_fs)
    assert len(upsampled_signal) == expected_length
    assert t_new[-1] == pytest.approx(t[-1])
    assert np.diff(t_new).mean() == pytest.approx(1.0 / expected_new_fs)


def test_upsample_pchip_edge_cases() -> None:
    """Test PCHIP upsampling with short signals, factor=1, and invalid inputs."""
    signal = np.array([1.0, 2.0, 3.0])

    # Factor = 1 returns copy
    up_sig, new_fs = upsample_pchip(signal, fs=1.0, factor=1)
    np.testing.assert_array_equal(up_sig, signal)
    assert new_fs == 1.0

    # Short signal (< 2 elements) returns copy
    short_sig = np.array([5.0])
    up_short, new_fs_short = upsample_pchip(short_sig, fs=2.0, factor=4)
    np.testing.assert_array_equal(up_short, short_sig)
    assert new_fs_short == 2.0

    # Invalid parameters
    with pytest.raises(ValueError, match="fs must be positive"):
        upsample_pchip(signal, fs=0.0, factor=2)

    with pytest.raises(ValueError, match="Upsampling factor must be >= 1"):
        upsample_pchip(signal, fs=1.0, factor=0)

    with pytest.raises(ValueError, match="empty"):
        upsample_pchip(np.array([]), fs=1.0, factor=2)
