import re
import warnings
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd

import core.config as cfg


@dataclass(frozen=True)
class ParseWarning:
    """Diagnostic record generated when log or data parsing detects anomalies."""

    line_number: int | None
    raw_line: str
    original_value: str
    proposed_value: str
    reason: str


def load_pm6_data(filepath: str) -> pd.DataFrame:
    """
    Load PM6 data and convert the MJD date format to Datetime and physical Time_sec.

    Args:
        filepath: Path to the PM6 text file.

    Returns:
        DataFrame containing parsed PM6 data with computed P-M interferometric differences
        and derived physical sampling frequency metadata attached to `df.attrs["fs"]`.

    Raises:
        ValueError: If file is empty or cannot be parsed.
    """
    columns = ["MJD", "P1_20A", "M1_20A", "P2_20B", "M2_20B", "P3_25A", "M3_25A", "P4_25B", "M4_25B"]
    try:
        df = pd.read_csv(filepath, sep=r"\s+", names=columns, skiprows=30, header=None, encoding_errors="ignore")
    except Exception as err:
        raise ValueError(f"Failed to read PM6 file '{filepath}': {err}") from err

    if df.empty:
        raise ValueError(f"PM6 data file '{filepath}' is empty.")

    # Convert signal columns to float
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # MJD to Datetime conversion (Excel 1899-12-30 origin)
    df["Datetime"] = pd.to_datetime(df["MJD"], unit="D", origin="1899-12-30")

    # Derive true physical Time_sec from timestamps
    start_dt = df["Datetime"].iloc[0]
    df["Time_sec"] = (df["Datetime"] - start_dt).dt.total_seconds()

    # Integrity & Sampling Frequency verification
    dt = np.diff(np.asarray(df["Time_sec"], dtype=np.float64))
    if len(dt) > 0:
        median_dt = float(np.median(dt))
        if median_dt <= 0:
            warnings.warn(
                f"Non-positive median time step in '{filepath}'. Defaulting fs=1.0.",
                category=UserWarning,
                stacklevel=2,
            )
            estimated_fs = 1.0
        else:
            estimated_fs = 1.0 / median_dt
    else:
        estimated_fs = 1.0

    df.attrs["fs"] = estimated_fs
    df.attrs["filepath"] = filepath

    # Compute P-M interferometric differences
    df["20 MHz Pol A (P-M)"] = df["P1_20A"] - df["M1_20A"]
    df["20 MHz Pol B (P-M)"] = df["P2_20B"] - df["M2_20B"]
    df["25 MHz Pol A (P-M)"] = df["P3_25A"] - df["M3_25A"]
    df["25 MHz Pol B (P-M)"] = df["P4_25B"] - df["M4_25B"]

    return df


@overload
def parse_regi_with_time(
    filepath: str,
    pm6_start_dt: pd.Timestamp,
    mode: str = ...,
    return_warnings: Literal[False] = ...,
) -> pd.DataFrame: ...


@overload
def parse_regi_with_time(
    filepath: str,
    pm6_start_dt: pd.Timestamp,
    mode: str = ...,
    return_warnings: Literal[True] = ...,
) -> tuple[pd.DataFrame, list[ParseWarning]]: ...


def parse_regi_with_time(
    filepath: str,
    pm6_start_dt: pd.Timestamp,
    mode: str = "warn",
    return_warnings: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[ParseWarning]]:
    """
    Reads a log file with regex, handles midnight rollovers, and returns a DataFrame with times in seconds.

    Args:
        filepath: Path to the registration log file.
        pm6_start_dt: The starting timestamp of the corresponding PM6 observation.
        mode: Data repair behavior: 'strict' (raise exception), 'warn' (keep original without silent fix),
              or 'repair' (apply correction and log warning).
        return_warnings: If True, returns a tuple of (df_logs, warnings_list).

    Returns:
        DataFrame containing parsed log events (or tuple of (DataFrame, warnings) if return_warnings=True).

    Raises:
        ValueError: If mode is 'strict' and an invalid time anomaly is encountered.
    """
    if mode not in {"strict", "warn", "repair"}:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of 'strict', 'warn', 'repair'.")

    events = []
    log_pattern = re.compile(r"^(\d{1,2}:\d{1,2})\s*-?\s*\((\d{1,2}:\d{1,2})\).*?(\S+)$")
    warnings_list: list[ParseWarning] = []

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines, start=1):
        line_str = line.strip()
        if not line_str:
            continue

        match = log_pattern.match(line_str)
        if match:
            events.append(
                {
                    "line_number": line_idx,
                    "raw_line": line_str,
                    "Start_Time": match.group(1),
                    "End_Time": match.group(2),
                    "Target_Name": match.group(3),
                }
            )

    df_logs = pd.DataFrame(events)
    if df_logs.empty:
        df_logs.attrs["warnings"] = warnings_list
        return (df_logs, warnings_list) if return_warnings else df_logs

    current_date = pm6_start_dt.normalize()
    last_dt = pm6_start_dt

    start_secs, end_secs = [], []
    for _, row in df_logs.iterrows():
        line_idx = row["line_number"]
        raw_line = row["raw_line"]

        try:
            s_dt = pd.to_datetime(f"{current_date.date()} {row['Start_Time']}:00")
            e_dt = pd.to_datetime(f"{current_date.date()} {row['End_Time']}:00")
        except (ValueError, TypeError):
            start_secs.append(np.nan)
            end_secs.append(np.nan)
            continue

        if s_dt.hour < last_dt.hour and (last_dt.hour - s_dt.hour) > 6:
            current_date += pd.Timedelta(days=1)
            s_dt += pd.Timedelta(days=1)
            e_dt += pd.Timedelta(days=1)

        if e_dt < s_dt:
            e_dt_rolled = e_dt + pd.Timedelta(days=1)
            duration_rolled = e_dt_rolled - s_dt
            if duration_rolled <= pd.Timedelta(hours=2):
                e_dt = e_dt_rolled
            else:
                time_diff = s_dt - e_dt
                if time_diff < pd.Timedelta(hours=2):
                    proposed_s_dt = s_dt - pd.Timedelta(hours=1)
                    parse_warn = ParseWarning(
                        line_number=line_idx,
                        raw_line=raw_line,
                        original_value=s_dt.strftime("%H:%M"),
                        proposed_value=proposed_s_dt.strftime("%H:%M"),
                        reason="End time precedes start time; potential 1-hour offset typo.",
                    )
                    warnings_list.append(parse_warn)
                    if mode == "strict":
                        raise ValueError(
                            f"End time precedes start time for line {line_idx}: '{raw_line}' ({parse_warn.reason})"
                        )
                    elif mode == "repair":
                        s_dt = proposed_s_dt
                    # If mode == 'warn', retain original s_dt without modification
                else:
                    e_dt = e_dt_rolled

        start_secs.append((s_dt - pm6_start_dt).total_seconds())
        end_secs.append((e_dt - pm6_start_dt).total_seconds())
        last_dt = s_dt

    df_logs["Start_sec"] = start_secs
    df_logs["End_sec"] = end_secs
    result_df = df_logs.dropna().drop(columns=["line_number", "raw_line"])
    result_df.attrs["warnings"] = warnings_list

    return (result_df, warnings_list) if return_warnings else result_df


def build_observation_sessions(df_logs: pd.DataFrame, pm6_max_sec: float) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Projects logs across sidereal days and groups them into contiguous sessions.

    Astrophysical Projection:
    A sidereal day is 23 hours 56 minutes 4 seconds (86164 seconds).
    Radio sources shift by exactly this amount every day relative to solar time.
    If the PM6 file is longer than the log, we multiply the schedule forward.

    Args:
        df_logs: DataFrame containing parsed log events.
        pm6_max_sec: Maximum time length in seconds of the PM6 file.

    Returns:
        Tuple containing:
        - df_logs: Projected logs.
        - calibrations: Calibration blocks.
        - sessions: Chronological observation sessions.
    """
    original_logs = df_logs.copy()
    max_log_sec = original_logs["End_sec"].max()

    if pm6_max_sec > max_log_sec:
        days_to_add = int(np.ceil((pm6_max_sec - max_log_sec) / cfg.SIDEREAL_DAY))
        projected_dfs = [original_logs]

        for day in range(1, days_to_add + 1):
            df_shifted = original_logs.copy()
            df_shifted["Start_sec"] += day * cfg.SIDEREAL_DAY
            df_shifted["End_sec"] += day * cfg.SIDEREAL_DAY
            projected_dfs.append(df_shifted)

        df_logs = pd.concat(projected_dfs, ignore_index=True)

    noise_targets = ["calibrovka", "3Czenit"]
    calibrations = df_logs[df_logs["Target_Name"].isin(noise_targets)]

    # Strict chronological grouping
    obs_logs = df_logs[~df_logs["Target_Name"].isin(noise_targets)].sort_values("Start_sec").reset_index(drop=True)

    sessions = []
    if not obs_logs.empty:
        current_target = obs_logs.iloc[0]["Target_Name"]
        current_start = obs_logs.iloc[0]["Start_sec"]
        current_end = obs_logs.iloc[0]["End_sec"]

        for i in range(1, len(obs_logs)):
            row = obs_logs.iloc[i]
            is_same_session = (row["Target_Name"] == current_target) and (
                row["Start_sec"] - current_end < cfg.SESSION_MERGE_GAP
            )

            if is_same_session:
                current_end = max(current_end, row["End_sec"])
            else:
                sessions.append({"target": current_target, "start": current_start, "end": current_end})
                current_target = row["Target_Name"]
                current_start = row["Start_sec"]
                current_end = row["End_sec"]

        sessions.append({"target": current_target, "start": current_start, "end": current_end})

    return df_logs, calibrations, sessions
