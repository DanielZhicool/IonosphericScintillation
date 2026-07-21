from pathlib import Path

import pandas as pd
import pytest

from core.parsers import ParseWarning, load_pm6_data, parse_regi_with_time


def test_load_pm6_data_timestamps(tmp_path: Path) -> None:
    """Test load_pm6_data timestamp calculation and sampling rate extraction."""
    pm6_file = tmp_path / "test_data.PM6"

    # Create synthetic PM6 lines (header of 30 lines + data)
    header_lines = [f"Header line {i}\n" for i in range(30)]
    # MJD starting at 50000.0 (incrementing by 1 second = 1/86400 days)
    data_lines = []
    base_mjd = 50000.0
    sec_in_day = 86400.0
    for i in range(10):
        mjd = base_mjd + (i * 2.0) / sec_in_day  # 2-second sampling interval (0.5 Hz)
        data_lines.append(f"{mjd:.12f} 100 50 100 50 100 50 100 50\n")

    pm6_file.write_text("".join(header_lines + data_lines), encoding="utf-8")

    df = load_pm6_data(str(pm6_file))

    assert "Time_sec" in df.columns
    assert len(df) == 10
    assert df["Time_sec"].iloc[0] == 0.0
    assert df["Time_sec"].iloc[1] == pytest.approx(2.0, abs=1e-3)
    assert df["Time_sec"].iloc[-1] == pytest.approx(18.0)
    assert df.attrs["fs"] == pytest.approx(0.5)  # 1 / 2 seconds = 0.5 Hz


def test_parse_regi_with_time_modes(tmp_path: Path) -> None:
    """Test REGI log parsing diagnostic modes: strict, warn, and repair."""
    regi_file = tmp_path / "test_log.txt"
    # Typo line where start time (21:06) is after end time (20:30) within 2 hours
    log_content = (
        "20:00 - (20:25) target1\n"
        "21:06 - (20:30) target2\n"  # 1-hour offset typo (21:06 -> should be 20:06)
    )
    regi_file.write_text(log_content, encoding="utf-8")
    pm6_start_dt = pd.Timestamp("2026-07-21 20:00:00")

    # Mode 1: 'strict' raises ValueError
    with pytest.raises(ValueError, match="End time precedes start time"):
        parse_regi_with_time(str(regi_file), pm6_start_dt, mode="strict")

    # Mode 2: 'warn' retains raw timestamp without silent modification and returns ParseWarning
    df_warn, warnings_list = parse_regi_with_time(str(regi_file), pm6_start_dt, mode="warn", return_warnings=True)
    assert len(warnings_list) == 1
    assert isinstance(warnings_list[0], ParseWarning)
    assert warnings_list[0].line_number == 2
    # Start_sec for line 2 should correspond to 21:06 (3960 seconds relative to 20:00)
    assert df_warn["Start_sec"].iloc[1] == pytest.approx(3960.0)

    # Mode 3: 'repair' applies proposed 1-hour correction (20:06 -> 360 seconds relative to 20:00)
    df_repair, warnings_repair = parse_regi_with_time(str(regi_file), pm6_start_dt, mode="repair", return_warnings=True)
    assert len(warnings_repair) == 1
    assert df_repair["Start_sec"].iloc[1] == pytest.approx(360.0)
