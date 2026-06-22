import pandas as pd
import numpy as np

def load_pm6_data(filepath):
    """Загрузка данных PM6 и конвертация времени (Excel Date)."""
    columns = ['MJD', 'P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 
               'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']
    df = pd.read_csv(filepath, sep=r'\s+', names=columns, header=30, encoding_errors='ignore')
    df['Datetime'] = pd.to_datetime(df['MJD'], unit='D', origin='1899-12-30')
    df['Time_sec'] = (df['Datetime'] - df['Datetime'].iloc[0]).dt.total_seconds()
    return df

def parse_regi_with_time(filepath, pm6_start_dt):
    """
    Умный парсер: читает лог-файл, учитывает переходы через полночь 
    и сразу возвращает DataFrame с готовыми секундами (Start_sec, End_sec).
    """
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[1] == '-':
                try:
                    events.append({
                        'Start_Time': parts[0],
                        'End_Time': parts[2].replace('(', '').replace(')', ''),
                        'Target_Name': parts[-1]
                    })
                except IndexError:
                    continue
                    
    df_logs = pd.DataFrame(events)
    if df_logs.empty:
        return df_logs

    # Хронологический разбор времени
    current_date = pm6_start_dt.normalize()
    last_dt = pm6_start_dt

    start_secs, end_secs = [], []
    for _, row in df_logs.iterrows():
        try:
            s_dt = pd.to_datetime(f"{current_date.date()} {row['Start_Time']}:00")
            
            if s_dt.hour < last_dt.hour and (last_dt.hour - s_dt.hour) > 6:
                current_date += pd.Timedelta(days=1)
                s_dt += pd.Timedelta(days=1)

            e_dt = pd.to_datetime(f"{current_date.date()} {row['End_Time']}:00")
            
            if e_dt < s_dt:
                e_dt += pd.Timedelta(days=1)

            start_secs.append((s_dt - pm6_start_dt).total_seconds())
            end_secs.append((e_dt - pm6_start_dt).total_seconds())
            last_dt = s_dt 
        except:
            start_secs.append(np.nan)
            end_secs.append(np.nan)
            
    df_logs['Start_sec'] = start_secs
    df_logs['End_sec'] = end_secs
    
    return df_logs.dropna()