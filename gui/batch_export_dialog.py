import os
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, 
    QProgressBar, QLabel, QFileDialog, QMessageBox, QGroupBox, 
    QCheckBox, QAbstractItemView, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal

from core.signal_processing import process_signal_pipeline
from core.spectral_analysis import run_spectral_pipeline
from gui.constants import CHANNELS
from scipy.signal import find_peaks
from gui.spectral_tab import SpectralTab

class BatchExportWorker(QThread):
    """Background thread to process and export plots to avoid freezing UI."""
    progress = Signal(int, str)  # percentage, status_text
    finished_ok = Signal(int)    # number of files saved
    error = Signal(str)

    def __init__(self, df_pm6, start_datetime, fs, window_size, n_sigmas, apply_smoothing,
                 selected_sessions, all_sessions, selected_channels, graphs_config, output_dir):
        super().__init__()
        self.df_pm6 = df_pm6
        self.start_datetime = start_datetime
        self.fs = fs
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.apply_smoothing = apply_smoothing
        self.selected_sessions = selected_sessions
        self.all_sessions = all_sessions
        self.selected_channels = selected_channels
        self.graphs_config = graphs_config
        self.output_dir = output_dir
        self._is_cancelled = False
        
        # Determine bands based on selection
        band_sel = graphs_config.get('band_selection', 'Both')
        all_bands = [
            ('5-150s', 1.0 / 150.0, 1.0 / 5.0),
            ('150-600s', 1.0 / 600.0, 1.0 / 150.0),
        ]
        if 'Small' in band_sel:
            self.bands = [all_bands[0]]
        elif 'Large' in band_sel:
            self.bands = [all_bands[1]]
        else:
            self.bands = all_bands

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            # Save batch configuration settings
            if self.selected_sessions:
                import json
                import core.config as cfg
                settings = {
                    'window_size': self.window_size,
                    'n_sigmas': self.n_sigmas,
                    'apply_smoothing': self.apply_smoothing,
                    'graphs_config': self.graphs_config,
                    'selected_channels': self.selected_channels,
                    'bands': self.bands,
                    'cwt_config': {
                        'nv': cfg.CWT_NV,
                        'gamma': cfg.MORSE_GAMMA,
                        'beta': cfg.MORSE_BETA,
                        'sigma_freq': cfg.GAUSSIAN_SIGMA_FREQ,
                        'sigma_time': cfg.GAUSSIAN_SIGMA_TIME,
                        'dynamic_range_db': cfg.CWT_DYNAMIC_RANGE_DB,
                        'pchip_factor': cfg.PCHIP_FACTOR
                    }
                }
                with open(os.path.join(self.output_dir, "BatchExportSettings.txt"), "w") as f:
                    f.write("Batch Export Settings\n")
                    f.write("=====================\n")
                    f.write(json.dumps(settings, indent=4))
                    
            # Count spectral tasks (one per session per band if any spectral plot is requested)
            needs_spectral = any(self.graphs_config.get(k, False) for k in ('psd', 'ftest', 'cross', 'idve'))
            spectral_tasks = len(self.selected_sessions) * len(self.bands) if needs_spectral else 0
            total_tasks = len(self.selected_sessions) * len(self.selected_channels) * len(self.bands) + spectral_tasks
            tasks_done = 0
            saved_count = 0

            time_full = self.df_pm6['Time_sec'].values

            for session in self.selected_sessions:
                if self._is_cancelled:
                    break

                s_sec, e_sec, target = session['start'], session['end'], session['target']
                s_idx = np.searchsorted(time_full, s_sec)
                e_idx = np.searchsorted(time_full, e_sec)
                
                if s_idx >= e_idx:
                    continue
                
                df_slice = self.df_pm6.iloc[s_idx:e_idx]
                time_sec = df_slice['Time_sec'].values
                time_h = time_sec / 3600.0
                signal_duration = len(df_slice) / self.fs
                
                session_dt = self.start_datetime + pd.to_timedelta(s_sec, unit='s')
                date_str = session_dt.strftime('%Y%m%d')
                
                # Check if we need spectral analysis
                needs_spectral = (self.graphs_config.get('psd', False) or 
                                  self.graphs_config.get('ftest', False) or 
                                  self.graphs_config.get('cross', False) or 
                                  self.graphs_config.get('idve', False))
                spectral_results = None
                
                if needs_spectral:
                    pm_signals = {
                        '20 MHz Pol A': df_slice['P1_20A'].values - df_slice['M1_20A'].values,
                        '20 MHz Pol B': df_slice['P2_20B'].values - df_slice['M2_20B'].values,
                        '25 MHz Pol A': df_slice['P3_25A'].values - df_slice['M3_25A'].values,
                        '25 MHz Pol B': df_slice['P4_25B'].values - df_slice['M4_25B'].values,
                    }
                    
                    self.progress.emit(int((tasks_done / max(1, total_tasks)) * 100), 
                                       f"Running spectral analysis for {target} ({date_str})...")
                    
                    band_results = {}
                    for band_key, lowcut, highcut in self.bands:
                        if self._is_cancelled: break
                        min_period = 1.0 / lowcut
                        if signal_duration >= min_period:
                            res = run_spectral_pipeline(
                                pm_signals, self.fs, lowcut, highcut,
                                self.window_size, self.n_sigmas, self.apply_smoothing,
                                progress_callback=lambda x: None
                            )
                            band_results[band_key] = res
                    spectral_results = band_results
                
                # Process Spectral Exports per session (cross-channel)
                if needs_spectral and spectral_results:
                    for band_key, res in spectral_results.items():
                        if res is None: continue
                        
                        freqs = res['freqs']
                        mask = (freqs >= res['lowcut']) & (freqs <= res['highcut']) & (freqs > 0)
                        periods = (1.0 / freqs[mask])[::-1]
                        chs = [('20 MHz Pol A', 0, 0), ('20 MHz Pol B', 0, 1),
                               ('25 MHz Pol A', 1, 0), ('25 MHz Pol B', 1, 1)]
                        safe_target = target.replace('/', '-').replace(' ', '_')
                        
                        # 1. Multitaper PSD
                        if self.graphs_config.get('psd', False):
                            fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
                            fig.suptitle(f"{target}  |  {session_dt.strftime('%Y-%m-%d')}  |  PSD ({band_key})", 
                                         fontsize=14, fontweight='bold')
                            for ch_name, r, c in chs:
                                ax = axes[r, c]
                                if ch_name in res['psd']:
                                    vals = res['psd'][ch_name][mask][::-1]
                                    ax.plot(periods, vals, color='#42A5F5', linewidth=1.0)
                                    ax.set_yscale('log')
                                    ax.set_xlabel("Period (Sec)")
                                    ax.set_ylabel("Spectral Power (dB)")
                                    ax.grid(True, alpha=0.3)
                                    
                                    peaks, _ = find_peaks(vals, distance=max(1, len(vals)//50))
                                    if len(peaks) > 0:
                                        top_idx = sorted(peaks, key=lambda i: vals[i], reverse=True)[:5]
                                        top_periods = sorted([periods[i] for i in top_idx], reverse=True)
                                        peaks_str = ", ".join(f"{tp:.1f}" for tp in top_periods)
                                        ax.set_title(f"{ch_name}\nTop 5 periods (s): {peaks_str}", color='darkred', fontweight='bold')
                                        ax.plot([periods[i] for i in top_idx], [vals[i] for i in top_idx], 'rv', markersize=8)
                                    else:
                                        ax.set_title(ch_name)
                            plt.savefig(os.path.join(self.output_dir, f"{date_str}_{safe_target}_PSD_{band_key}.png"), dpi=200)
                            plt.close(fig)
                            saved_count += 1

                        # 2. Thomson F-Test
                        if self.graphs_config.get('ftest', False):
                            fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
                            fig.suptitle(f"{target}  |  {session_dt.strftime('%Y-%m-%d')}  |  F-Test ({band_key})", 
                                         fontsize=14, fontweight='bold')
                            threshold = res['ftest'].get('threshold', 1.0)
                            for ch_name, r, c in chs:
                                ax = axes[r, c]
                                if ch_name in res['ftest']:
                                    vals = res['ftest'][ch_name][mask][::-1]
                                    ax.plot(periods, vals, color='#42A5F5', linewidth=0.8)
                                    ax.axhline(threshold, color='r', linestyle='--', alpha=0.7)
                                    conf_pct = res['ftest'].get('confidence', 0.99) * 100
                                    p_min, p_max = min(periods), max(periods)
                                    ax.text(p_min + (p_max - p_min)*0.02, threshold, 
                                            f"{conf_pct:.0f}% Confidence Threshold (F={threshold:.2f})",
                                            color='r', fontweight='bold', va='bottom', ha='left', fontsize=10)
                                    
                                    T0 = res['ftest'].get(ch_name + '_T0', None)
                                    if T0:
                                        t2, t3 = T0/2.0, T0/3.0
                                        title_str = f"{ch_name}\nT0 = {T0:.1f} s"
                                        y_max = max(vals) if len(vals) > 0 else threshold
                                        
                                        if min(periods) <= t2 <= max(periods):
                                            title_str += f" | 2T: {t2:.1f} s"
                                            ax.axvline(t2, color='m', linestyle='--', alpha=0.5)
                                            ax.text(t2, threshold + (y_max - threshold)*0.1, '2T', color='m', ha='center', va='bottom', fontsize=10, fontweight='bold')
                                            
                                        if min(periods) <= t3 <= max(periods):
                                            title_str += f" | 3T: {t3:.1f} s"
                                            ax.axvline(t3, color='m', linestyle='--', alpha=0.5)
                                            ax.text(t3, threshold + (y_max - threshold)*0.1, '3T', color='m', ha='center', va='bottom', fontsize=10, fontweight='bold')

                                        ax.set_title(title_str, color='darkblue')
                                        ax.axvline(T0, color='k', linewidth=1.5)
                                        
                                        # Add label for T0
                                        ax.text(T0, y_max, 'T0', color='k', ha='center', va='bottom', fontsize=10, fontweight='bold')
                                        
                                        t0_idx = int(np.argmin(np.abs(periods - T0)))
                                        if 0 <= t0_idx < len(vals):
                                            ax.plot([periods[t0_idx]], [vals[t0_idx]], 'r*', markersize=12)
                                    else:
                                        ax.set_title(ch_name)
                                        
                                    ax.set_xlabel("Period (Sec)")
                                    ax.set_ylabel("F-Statistic")
                                    ax.grid(True, alpha=0.3)
                            plt.savefig(os.path.join(self.output_dir, f"{date_str}_{safe_target}_FTest_{band_key}.png"), dpi=200)
                            plt.close(fig)
                            saved_count += 1

                        # 3. Cross-Spectrum
                        if self.graphs_config.get('cross', False):
                            fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
                            fig.suptitle(f"{target}  |  {session_dt.strftime('%Y-%m-%d')}  |  Cross-Spectrum ({band_key})", 
                                         fontsize=14, fontweight='bold')
                            cross_data = res['cross']
                            pols = [('Pol A', 0), ('Pol B', 1)]
                            for pol_name, r in pols:
                                if pol_name in cross_data:
                                    vals_pow = cross_data[pol_name]['power'][mask][::-1]
                                    vals_re = cross_data[pol_name]['real'][mask][::-1]
                                    vals_im = cross_data[pol_name]['imag'][mask][::-1]
                                    
                                    ax_pow, ax_re, ax_im = axes[r, 0], axes[r, 1], axes[r, 2]
                                    
                                    ax_pow.plot(periods, vals_pow, color='k', linewidth=1.0)
                                    ax_pow.set_ylabel("Power")
                                    
                                    ax_re.plot(periods, vals_re, color='#42A5F5', linewidth=1.0)
                                    ax_re.axhline(0, color='gray', linestyle='--')
                                    ax_re.set_ylabel("Re(Pxy)")
                                    if r == 0: ax_re.set_title("Co-spectrum (In-phase)")
                                    
                                    ax_im.plot(periods, vals_im, color='#EF5350', linewidth=1.0)
                                    ax_im.axhline(0, color='gray', linestyle='--')
                                    ax_im.set_ylabel("Im(Pxy)")
                                    if r == 0: ax_im.set_title("Quadrature spectrum (Phase shift)")
                                    
                                    peaks, _ = find_peaks(vals_pow, distance=max(1, len(vals_pow)//50))
                                    if len(peaks) > 0:
                                        top_idx = sorted(peaks, key=lambda i: vals_pow[i], reverse=True)[:3]
                                        top_periods = [periods[i] for i in top_idx]
                                        peaks_str = ", ".join(f"{tp:.1f}" for tp in sorted(top_periods, reverse=True))
                                        ax_pow.set_title(f"{pol_name} (20 vs 25 MHz)\nTop 3 periods (s): {peaks_str}", color='darkred')
                                        ax_pow.plot([periods[i] for i in top_idx], [vals_pow[i] for i in top_idx], 'rv', markersize=8)
                                        for px in top_periods:
                                            ax_re.axvline(px, color='r', linestyle=':', alpha=0.7)
                                            ax_im.axvline(px, color='r', linestyle=':', alpha=0.7)
                                    else:
                                        ax_pow.set_title(f"{pol_name} (20 vs 25 MHz)")
                                        
                                    for ax in [ax_pow, ax_re, ax_im]:
                                        ax.set_xlabel("Period (Sec)")
                                        ax.grid(True, alpha=0.3)
                            plt.savefig(os.path.join(self.output_dir, f"{date_str}_{safe_target}_Cross_{band_key}.png"), dpi=200)
                            plt.close(fig)
                            saved_count += 1
                            
                        # 4. IDVE txt log
                        if self.graphs_config.get('idve', False):
                            txt = SpectralTab._format_velocity_table(res.get('velocities', {}), band_key)
                            with open(os.path.join(self.output_dir, f"{date_str}_{safe_target}_IDVE_{band_key}.txt"), "w") as f:
                                f.write(f"Target: {target} | Date: {session_dt.strftime('%Y-%m-%d')}\n\n")
                                f.write(txt)
                            saved_count += 1
                        tasks_done += 1
                
                # Process Channel Time Domain Exports
                for channel in self.selected_channels:
                    if self._is_cancelled:
                        break
                        
                    raw_signal = df_slice[channel].values
                    
                    for band_key, cwt_low, cwt_high in self.bands:
                        self.progress.emit(int((tasks_done / max(1, total_tasks)) * 100), 
                                           f"Processing {target} | {channel} | {date_str} | {band_key}")
                        
                        # Need CWT/Filtering?
                        needs_filtered = self.graphs_config.get('filtered', False)
                        needs_spec = self.graphs_config.get('spectrogram', False)
                        
                        filtered_sig = None
                        img_data = None
                        
                        if needs_filtered or needs_spec:
                            f_sig, i_data = process_signal_pipeline(
                                raw_signal, self.fs, cwt_low, cwt_high, 
                                self.window_size, self.n_sigmas, self.apply_smoothing,
                                progress_callback=lambda x: None
                            )
                            filtered_sig = f_sig
                            img_data = i_data

                        # Plotting Time Domain
                        time_plots = []
                        if self.graphs_config.get('raw', False): time_plots.append('raw')
                        if needs_filtered: time_plots.append('filtered')
                        if needs_spec: time_plots.append('spectrogram')
                        
                        if time_plots:
                            n = len(time_plots)
                            fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), constrained_layout=True)
                            if n == 1: axes = [axes]
                            
                            fig.suptitle(f"{target}  |  {session_dt.strftime('%Y-%m-%d')}  |  {channel}  |  {band_key}", 
                                         fontsize=14, fontweight='bold')
                            
                            ax_idx = 0
                            if 'raw' in time_plots:
                                ax = axes[ax_idx]; ax_idx += 1
                                ax.plot(time_h, raw_signal, color='steelblue', linewidth=0.6)
                                ax.set_ylabel("Amplitude")
                                ax.set_title("Raw Signal")
                                ax.grid(True, alpha=0.3)
                                
                            if 'filtered' in time_plots:
                                ax = axes[ax_idx]; ax_idx += 1
                                ax.plot(time_h, filtered_sig, color='seagreen', linewidth=0.6)
                                ax.set_ylabel("Amplitude")
                                ax.set_title(f"Filtered Signal (Scintillations, {band_key})")
                                ax.grid(True, alpha=0.3)
                                
                            if 'spectrogram' in time_plots:
                                ax = axes[ax_idx]; ax_idx += 1
                                t0, t1 = time_h[0], time_h[-1]
                                im = ax.imshow(img_data.T, aspect='auto', origin='lower',
                                               extent=[t0, t1, cwt_low, cwt_high], cmap='viridis')
                                plt.colorbar(im, ax=ax, label='Power (dB)')
                                ax.set_ylabel("Frequency (Hz)")
                                ax.set_title(f"CWT Spectrogram ({band_key})")
                                
                            for ax in axes:
                                if target == 'Full Overview':
                                    if self.graphs_config.get('markers', False):
                                        for ms in self.all_sessions:
                                            if ms['target'] != 'Full Overview':
                                                s_h = ms['start'] / 3600.0
                                                e_h = ms['end'] / 3600.0
                                                mid_h = (s_h + e_h) / 2.0
                                                y_min, y_max = ax.get_ylim()
                                                # Draw red shaded region with strong borders
                                                ax.axvspan(s_h, e_h, facecolor='red', edgecolor='red', linewidth=1.5, linestyle='--', alpha=0.15)
                                                
                                                # Draw text vertically in the middle, using a white background box so it doesn't blend with spectrogram
                                                if ax == axes[0]:
                                                    ax.text(mid_h, y_min + (y_max - y_min) * 0.5, ms['target'], 
                                                            rotation=90, va='center', ha='center', 
                                                            color='#FF4444', alpha=1.0, fontweight='bold',
                                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                                    
                                    if self.graphs_config.get('day_markers', False):
                                        current_day = self.start_datetime.normalize() + pd.Timedelta(days=1)
                                        end_dt = self.start_datetime + pd.to_timedelta(time_sec[-1], unit='s')
                                        while current_day < end_dt:
                                            sec_offset = (current_day - self.start_datetime).total_seconds()
                                            h_offset = sec_offset / 3600.0
                                            y_max = ax.get_ylim()[1]
                                            ax.axvline(h_offset, color='black', linestyle=':', alpha=0.6)
                                            if ax == axes[0]:
                                                ax.text(h_offset, y_max, current_day.strftime('%Y-%m-%d'), rotation=90, va='top', ha='right', 
                                                        color='black', alpha=0.9, fontweight='bold',
                                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
                                            current_day += pd.Timedelta(days=1)
                            
                            axes[-1].set_xlabel("Time (hours from start)")
                            
                            safe_target = target.replace('/', '-').replace(' ', '_')
                            fname = f"{date_str}_{safe_target}_{channel}_TimeDomain_{band_key}.png"
                            out_path = os.path.join(self.output_dir, fname)
                            plt.savefig(out_path, dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            saved_count += 1
                        
                        tasks_done += 1

            if not self._is_cancelled:
                self.progress.emit(100, "Done!")
                self.finished_ok.emit(saved_count)
            else:
                self.progress.emit(100, "Cancelled.")
                self.error.emit("Export was cancelled.")
                
        except Exception:
            err = traceback.format_exc()
            self.error.emit(err)


class BatchExportDialog(QDialog):
    """Dialog for configuring and running a batch export of plots."""
    def __init__(self, df_pm6, df_pm6_original, sessions, start_datetime, fs, window_size, n_sigmas, apply_smoothing, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Export")
        self.resize(700, 750)
        
        self.df_pm6 = df_pm6
        self.df_pm6_original = df_pm6_original
        self.sessions = sessions
        self.start_datetime = start_datetime
        self.fs = fs
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.apply_smoothing = apply_smoothing
        
        # Add Full Overview pseudo-session
        max_end = max(s['end'] for s in sessions) if sessions else 0
        full_session = {'start': 0, 'end': max_end, 'target': 'Full Overview'}
        self.sessions = [full_session] + sessions
        
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        lists_layout = QHBoxLayout()
        
        # Sessions list
        grp_sessions = QGroupBox("1. Select Sessions")
        v_sess = QVBoxLayout()
        self.list_sessions = QListWidget()
        self.list_sessions.setSelectionMode(QAbstractItemView.MultiSelection)
        
        self.session_map = {}
        for idx, s in enumerate(self.sessions):
            session_dt = self.start_datetime + pd.to_timedelta(s['start'], unit='s')
            date_str = session_dt.strftime('%d %b %H:%M')
            label = f"{s['target']} ({date_str})"
            self.list_sessions.addItem(label)
            self.session_map[label] = s
            
        # Select all by default
        self.list_sessions.selectAll()
        v_sess.addWidget(self.list_sessions)
        grp_sessions.setLayout(v_sess)
        lists_layout.addWidget(grp_sessions)
        
        # Channels list
        grp_channels = QGroupBox("2. Select Channels")
        v_chan = QVBoxLayout()
        self.list_channels = QListWidget()
        self.list_channels.setSelectionMode(QAbstractItemView.MultiSelection)
        for ch in CHANNELS:
            self.list_channels.addItem(ch)
        self.list_channels.item(0).setSelected(True) # Select first by default
        v_chan.addWidget(self.list_channels)
        grp_channels.setLayout(v_chan)
        lists_layout.addWidget(grp_channels)
        
        layout.addLayout(lists_layout)
        
        # Spectral Band Selector
        h_band = QHBoxLayout()
        self.lbl_band = QLabel("Spectral Band:")
        self.combo_band = QComboBox()
        self.combo_band.addItems(["Small bubbles (5 - 150 s)", "Large clouds (150 - 600 s)", "Both"])
        h_band.addWidget(self.lbl_band)
        h_band.addWidget(self.combo_band)
        layout.addLayout(h_band)
        
        # Plots to include
        grp_plots = QGroupBox("3. Plots to Include")
        v_plots = QVBoxLayout()
        h_plots1 = QHBoxLayout()
        self.chk_raw = QCheckBox("Raw Signal")
        self.chk_raw.setChecked(True)
        self.chk_filtered = QCheckBox("Filtered (Scintillations)")
        self.chk_filtered.setChecked(True)
        self.chk_spec = QCheckBox("CWT Spectrogram")
        self.chk_spec.setChecked(True)
        h_plots1.addWidget(self.chk_raw)
        h_plots1.addWidget(self.chk_filtered)
        h_plots1.addWidget(self.chk_spec)
        
        h_plots2 = QHBoxLayout()
        self.chk_spectral_psd = QCheckBox("Multitaper PSD")
        self.chk_spectral_ftest = QCheckBox("Thomson F-Test")
        self.chk_spectral_cross = QCheckBox("Cross-Spectrum")
        self.chk_spectral_idve = QCheckBox("IDVE (txt log)")
        h_plots2.addWidget(self.chk_spectral_psd)
        h_plots2.addWidget(self.chk_spectral_ftest)
        h_plots2.addWidget(self.chk_spectral_cross)
        h_plots2.addWidget(self.chk_spectral_idve)
        
        v_plots.addLayout(h_plots1)
        v_plots.addLayout(h_plots2)
        
        self.chk_markers_export = QCheckBox("Draw Session Markers (Full Overview)")
        self.chk_day_markers_export = QCheckBox("Draw Day Markers (Full Overview)")
        v_plots.addWidget(self.chk_markers_export)
        v_plots.addWidget(self.chk_day_markers_export)
        
        self.chk_use_cleaned = QCheckBox("Use manually cleaned data (if any)")
        self.chk_use_cleaned.setChecked(True)
        self.chk_use_cleaned.setToolTip("Uncheck to export using the original unmodified PM6 data")
        v_plots.addWidget(self.chk_use_cleaned)
        
        grp_plots.setLayout(v_plots)
        layout.addWidget(grp_plots)
        
        # Output directory
        grp_out = QGroupBox("4. Output Directory")
        h_out = QHBoxLayout()
        self.lbl_outdir = QLabel("No directory selected")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_dir)
        h_out.addWidget(self.lbl_outdir)
        h_out.addWidget(btn_browse)
        grp_out.setLayout(h_out)
        layout.addWidget(grp_out)
        
        # Progress and Actions
        self.lbl_status = QLabel("Ready.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        
        h_actions = QHBoxLayout()
        self.btn_export = QPushButton("Start Export")
        self.btn_export.clicked.connect(self.start_export)
        self.btn_cancel = QPushButton("Close")
        self.btn_cancel.clicked.connect(self.close_or_cancel)
        h_actions.addWidget(self.btn_export)
        h_actions.addWidget(self.btn_cancel)
        
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress)
        layout.addLayout(h_actions)
        
        self.output_dir = ""

    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.output_dir = d
            self.lbl_outdir.setText(d)
            
    def close_or_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.btn_cancel.setText("Cancelling...")
            self.btn_cancel.setEnabled(False)
        else:
            self.reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().closeEvent(event)

    def start_export(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return
            
        selected_sess = [self.session_map[item.text()] for item in self.list_sessions.selectedItems()]
        selected_chans = [item.text() for item in self.list_channels.selectedItems()]
        
        if not selected_sess or not selected_chans:
            QMessageBox.warning(self, "Error", "Please select at least one session and channel.")
            return
            
        graphs_config = {
            'raw': self.chk_raw.isChecked(),
            'filtered': self.chk_filtered.isChecked(),
            'spectrogram': self.chk_spec.isChecked(),
            'psd': self.chk_spectral_psd.isChecked(),
            'ftest': self.chk_spectral_ftest.isChecked(),
            'cross': self.chk_spectral_cross.isChecked(),
            'idve': self.chk_spectral_idve.isChecked(),
            'markers': self.chk_markers_export.isChecked(),
            'day_markers': self.chk_day_markers_export.isChecked(),
            'band_selection': self.combo_band.currentText()
        }
        
        if not any(graphs_config.values()) and not any(v for k, v in graphs_config.items() if isinstance(v, bool)):
            QMessageBox.warning(self, "Error", "Please select at least one plot type.")
            return

        self.btn_export.setEnabled(False)
        self.btn_cancel.setText("Cancel")
        self.progress.setValue(0)
        
        df_to_use = self.df_pm6 if self.chk_use_cleaned.isChecked() else self.df_pm6_original

        self.worker = BatchExportWorker(
            df_pm6=df_to_use,
            start_datetime=self.start_datetime,
            fs=self.fs,
            window_size=self.window_size,
            n_sigmas=self.n_sigmas,
            apply_smoothing=self.apply_smoothing,
            selected_sessions=selected_sess,
            all_sessions=self.sessions,
            selected_channels=selected_chans,
            graphs_config=graphs_config,
            output_dir=self.output_dir
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def update_progress(self, val, text):
        self.progress.setValue(val)
        self.lbl_status.setText(text)
        
    def on_finished(self, saved_count):
        self.progress.setValue(100)
        self.lbl_status.setText("Export complete.")
        QMessageBox.information(self, "Success", f"Batch export finished.\nSaved {saved_count} files.")
        self.btn_export.setEnabled(True)
        self.btn_cancel.setText("Close")
        self.btn_cancel.setEnabled(True)
        
    def on_error(self, err_msg):
        self.lbl_status.setText("Error/Cancelled.")
        QMessageBox.warning(self, "Export Halted", err_msg)
        self.btn_export.setEnabled(True)
        self.btn_cancel.setText("Close")
        self.btn_cancel.setEnabled(True)
