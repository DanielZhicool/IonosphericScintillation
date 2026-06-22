import pandas as pd

def load_pm6_data(filepath):
    """Loading PM6 data and converting time (Excel Date)."""
    columns = ['MJD', 'P1_20A', 'M1_20A', 'P2_20B', 'M2_20B', 
               'P3_25A', 'M3_25A', 'P4_25B', 'M4_25B']
    df = pd.read_csv(filepath, sep=r'\s+', names=columns, header=30, encoding_errors='ignore')
    df['Datetime'] = pd.to_datetime(df['MJD'], unit='D', origin='1899-12-30')
    df['Time_sec'] = (df['Datetime'] - df['Datetime'].iloc[0]).dt.total_seconds()
    return df

def parse_regi_file(filepath):
    """Parsing log file (regi). Returns DataFrame of events."""
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[1] == '-':
                try:
                    start_time = parts[0]
                    end_time = parts[2].replace('(', '').replace(')', '')
                    name = parts[-1]
                    events.append({
                        'Start_Time': start_time,
                        'End_Time': end_time,
                        'Target_Name': name
                    })
                except IndexError:
                    continue
    return pd.DataFrame(events)