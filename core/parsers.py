import pandas as pd
import numpy as np
import re

import core.config as cfg

def load_pm6_data(filepath: str) -> pd.DataFrame:
    """
    Load PM6 data and convert the Excel date format to datetime.

    Args:
        filepath: Path to the PM6 text file.

    Returns:
        DataFrame containing parsed PM6 data with computed P-M interferometric differences.
    """
    columns = ['MJD', 'P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 
               'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']
    df = pd.read_csv(filepath, sep=r'\s+', names=columns, header=30, encoding_errors='ignore')
    df['Datetime'] = pd.to_datetime(df['MJD'], unit='D', origin='1899-12-30')
    df['Time_sec'] = (df['Datetime'] - df['Datetime'].iloc[0]).dt.total_seconds()
    
    # Compute P-M interferometric differences
    df['20 MHz Pol A (P-M)'] = df['P1_20A'] - df['M1_20A']
    df['20 MHz Pol B (P-M)'] = df['P2_20B'] - df['M2_20B']
    df['25 MHz Pol A (P-M)'] = df['P3_25A'] - df['M3_25A']
    df['25 MHz Pol B (P-M)'] = df['P4_25B'] - df['M4_25B']
    
    return df

def parse_regi_with_time(filepath: str, pm6_start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Reads a log file with regex, handles midnight rollovers, and returns a DataFrame with times in seconds.

    Args:
        filepath: Path to the registration log file.
        pm6_start_dt: The starting timestamp of the corresponding PM6 observation.

    Returns:
        DataFrame containing chronologically sorted log events.
    """
    events = []
    
    # Pattern: captures start (HH:MM), end in parentheses, and target name
    log_pattern = re.compile(r'^(\d{1,2}:\d{1,2})\s*-?\s*\((\d{1,2}:\d{1,2})\).*?(\S+)$')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            match = log_pattern.match(line)
            if match:
                events.append({
                    'Start_Time': match.group(1),
                    'End_Time': match.group(2),
                    'Target_Name': match.group(3)
                })

    df_logs = pd.DataFrame(events)
    if df_logs.empty:
        return df_logs

    # Chronological time parsing
    current_date = pm6_start_dt.normalize()
    last_dt = pm6_start_dt

    start_secs, end_secs = [], []
    for _, row in df_logs.iterrows():
        try:
            s_dt = pd.to_datetime(f"{current_date.date()} {row['Start_Time']}:00")
            
            # Adjust for midnight rollover
            if s_dt.hour < last_dt.hour and (last_dt.hour - s_dt.hour) > 6:
                current_date += pd.Timedelta(days=1)
                s_dt += pd.Timedelta(days=1)

            e_dt = pd.to_datetime(f"{current_date.date()} {row['End_Time']}:00")
            
            if e_dt < s_dt:
                e_dt += pd.Timedelta(days=1)

            start_secs.append((s_dt - pm6_start_dt).total_seconds())
            end_secs.append((e_dt - pm6_start_dt).total_seconds())
            last_dt = s_dt 
        except (ValueError, TypeError):
            start_secs.append(np.nan)
            end_secs.append(np.nan)
            
    df_logs['Start_sec'] = start_secs
    df_logs['End_sec'] = end_secs
    
    return df_logs.dropna()


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
    max_log_sec = original_logs['End_sec'].max()
    
    if pm6_max_sec > max_log_sec:
        days_to_add = int(np.ceil((pm6_max_sec - max_log_sec) / cfg.SIDEREAL_DAY))
        projected_dfs = [original_logs]
        
        for day in range(1, days_to_add + 1):
            df_shifted = original_logs.copy()
            df_shifted['Start_sec'] += day * cfg.SIDEREAL_DAY
            df_shifted['End_sec'] += day * cfg.SIDEREAL_DAY
            projected_dfs.append(df_shifted)
            
        df_logs = pd.concat(projected_dfs, ignore_index=True)

    noise_targets = ['calibrovka', '3Czenit']
    calibrations = df_logs[df_logs['Target_Name'].isin(noise_targets)]
    
    # Strict chronological grouping
    obs_logs = df_logs[~df_logs['Target_Name'].isin(noise_targets)].sort_values('Start_sec').reset_index(drop=True)
    
    sessions = []
    if not obs_logs.empty:
        current_target = obs_logs.iloc[0]['Target_Name']
        current_start = obs_logs.iloc[0]['Start_sec']
        current_end = obs_logs.iloc[0]['End_sec']
        
        for i in range(1, len(obs_logs)):
            row = obs_logs.iloc[i]
            is_same_session = (row['Target_Name'] == current_target) and (row['Start_sec'] - current_end < cfg.SESSION_MERGE_GAP)
            
            if is_same_session:
                current_end = max(current_end, row['End_sec'])
            else:
                sessions.append({'target': current_target, 'start': current_start, 'end': current_end})
                current_target = row['Target_Name']
                current_start = row['Start_sec']
                current_end = row['End_sec']
                
        sessions.append({'target': current_target, 'start': current_start, 'end': current_end})
        
    return df_logs, calibrations, sessions