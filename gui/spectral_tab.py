"""
SpectralTab widget for displaying Multitaper PSD, F-Test,
Cross-Spectrum, and ionospheric drift velocity results.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QTextEdit,
)
from PySide6.QtCore import Qt
from scipy.signal import find_peaks

from gui.constants import (
    SPECTRAL_TAB_PSD_TITLE,
    SPECTRAL_TAB_FTEST_TITLE,
    SPECTRAL_TAB_VELOCITY_HEADER,
    SPECTRAL_BAND_SMALL,
    SPECTRAL_BAND_LARGE,
    SPECTRAL_INSUFFICIENT_DATA,
)


class SpectralTab(QWidget):
    """
    Displays spectral-correlation analysis results for a radio source transit.

    Layout per frequency band (sub-tab):
        Nested QTabWidget with:
        - Multitaper PSD (2x2 grid)
        - Thomson F-Test (2x2 grid)
        - Cross-Spectrum (2x2 grid: Coherence & Phase)
        - Velocity Summary (Text Table)
    """

    def __init__(self, source_name, band_results, parent=None):
        """
        Args:
            source_name: e.g. "3C144".
            band_results: dict with keys ``'small'`` and ``'large'``.
                Each value is either a results dict produced by
                ``run_spectral_pipeline`` or ``None`` (signal too short).
        """
        super().__init__(parent)
        self.source_name = source_name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.band_tabs = QTabWidget()
        layout.addWidget(self.band_tabs)

        self.band_tabs.addTab(
            self._create_band_widget(band_results.get('small'),
                                     SPECTRAL_BAND_SMALL),
            SPECTRAL_BAND_SMALL,
        )
        self.band_tabs.addTab(
            self._create_band_widget(band_results.get('large'),
                                     SPECTRAL_BAND_LARGE),
            SPECTRAL_BAND_LARGE,
        )

    # ------------------------------------------------------------------
    # Band sub-tab builder
    # ------------------------------------------------------------------

    def _create_band_widget(self, results, band_label):
        """Build the widget for one frequency band's analysis results."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        if results is None:
            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText(SPECTRAL_INSUFFICIENT_DATA)
            layout.addWidget(text)
            return widget

        # Create inner tabs for analysis types to prevent clutter
        analysis_tabs = QTabWidget()
        layout.addWidget(analysis_tabs)

        freqs = results['freqs']
        lowcut = results['lowcut']
        highcut = results['highcut']

        # Mask to the band of interest, skip DC
        mask = (freqs >= lowcut) & (freqs <= highcut) & (freqs > 0)
        masked_freqs = freqs[mask]
        # Invert freq → period, reverse so periods are ascending
        periods = (1.0 / masked_freqs)[::-1]

        period_min = 1.0 / highcut
        period_max = 1.0 / lowcut

        # ── 1. PSD Tab (2x2 Grid) ────────────────────────────────────
        psd_widget = pg.GraphicsLayoutWidget()
        analysis_tabs.addTab(psd_widget, SPECTRAL_TAB_PSD_TITLE)
        self._build_psd_grid(
            psd_widget, results['psd'], periods, mask, period_min, period_max,
        )

        # ── 2. F-Test Tab (2x2 Grid) ─────────────────────────────────
        ftest_widget = pg.GraphicsLayoutWidget()
        analysis_tabs.addTab(ftest_widget, SPECTRAL_TAB_FTEST_TITLE)
        self._build_ftest_grid(
            ftest_widget, results['ftest'], periods, mask,
            freqs, period_min, period_max,
        )

        # ── 3. Cross-Spectrum Tab (2x2 Grid) ─────────────────────────
        cross_widget = pg.GraphicsLayoutWidget()
        analysis_tabs.addTab(cross_widget, "Cross-Spectrum")
        self._build_cross_spectrum(
            cross_widget, results['cross'], periods, mask, period_min, period_max
        )

        # ── 4. Velocity summary ──────────────────────────────────────
        velocity_text = QTextEdit()
        velocity_text.setReadOnly(True)
        velocity_text.setFontFamily('Consolas')
        velocity_text.setPlainText(
            self._format_velocity_table(
                results.get('velocities', {}), band_label,
            )
        )
        analysis_tabs.addTab(velocity_text, SPECTRAL_TAB_VELOCITY_HEADER)

        return widget

    def _build_psd_grid(self, graph, data_dict, periods, mask, p_min, p_max):
        """Build 2x2 PSD grid — all channels in blue with Top-5 peak markers."""
        channels = [
            ('20 MHz Pol A', 0, 0),
            ('20 MHz Pol B', 0, 1),
            ('25 MHz Pol A', 1, 0),
            ('25 MHz Pol B', 1, 1),
        ]
        link_plot = None
        for ch_name, row, col in channels:
            if ch_name not in data_dict:
                continue
            label = ch_name
            p = graph.addPlot(row=row, col=col, title=label)
            p.setLabel('bottom', 'Period (Sec)')
            p.setLabel('left', 'Spectral Power (dB)')
            p.showGrid(x=True, y=True)
            p.setXRange(p_min, p_max)
            p.setLogMode(x=False, y=True)
            if link_plot is None:
                link_plot = p
            else:
                p.setXLink(link_plot)

            vals = data_dict[ch_name][mask][::-1]
            p.plot(periods, vals, pen=pg.mkPen('#42A5F5', width=1.0))  # uniform blue

            peaks, _ = find_peaks(vals, distance=max(1, len(vals) // 50))
            if len(peaks) > 0:
                top_idx = sorted(peaks, key=lambda i: vals[i], reverse=True)[:5]
                top_periods = sorted([periods[i] for i in top_idx], reverse=True)
                peaks_str = ", ".join(f"{tp:.1f}" for tp in top_periods)
                title_html = (f"<span style='font-size:12pt;'>{label}</span><br>"
                              f"<span style='color:#FF5555; font-size:11pt; font-weight:bold;'>"
                              f"Top 5 periods (s): {peaks_str}</span>")
                p.setTitle(title_html)
                peak_x = [periods[i] for i in top_idx]
                peak_y = [vals[i] for i in top_idx]
                p.plot(peak_x, peak_y, pen=None, symbol='t',
                       symbolPen='r', symbolBrush='r', symbolSize=10)

    def _build_ftest_grid(self, graph, data_dict, periods, mask,
                          freqs_full, p_min, p_max):
        """Build 2x2 F-Test grid matching the MATLAB reference exactly."""
        channels = [
            ('20 MHz Pol A', 0, 0),
            ('20 MHz Pol B', 0, 1),
            ('25 MHz Pol A', 1, 0),
            ('25 MHz Pol B', 1, 1),
        ]
        threshold = data_dict.get('threshold', 1.0)
        confidence_pct = data_dict.get('confidence', 0.95) * 100
        link_plot = None

        for ch_name, row, col in channels:
            if ch_name not in data_dict:
                continue

            label = ch_name
            T0 = data_dict.get(ch_name + '_T0', None)

            if T0 is not None:
                t2, t3 = T0 / 2.0, T0 / 3.0
                title_html = f"<span style='font-size:12pt;'>{label}</span><br><span style='color:#42A5F5; font-size:11pt; font-weight:bold;'>T₀ = {T0:.1f} s"
                if p_min <= t2 <= p_max:
                    title_html += f" | 2T: {t2:.1f} s"
                if p_min <= t3 <= p_max:
                    title_html += f" | 3T: {t3:.1f} s"
                title_html += "</span>"
            else:
                title_html = f"<span style='font-size:12pt;'>{label}</span>"

            p = graph.addPlot(row=row, col=col, title=title_html)
            p.setLabel('bottom', 'Period (Sec)')
            p.setLabel('left', 'F-Statistic')
            p.showGrid(x=True, y=True)
            p.setXRange(p_min, p_max)
            if link_plot is None:
                link_plot = p
            else:
                p.setXLink(link_plot)

            vals = data_dict[ch_name][mask][::-1]
            p.plot(periods, vals, pen=pg.mkPen('#42A5F5', width=0.8))  # uniform blue

            # Confidence threshold dashed red line
            thresh_line = pg.InfiniteLine(
                pos=threshold, angle=0,
                pen=pg.mkPen('#FF4444', width=2.5, style=Qt.DashLine),
            )
            p.addItem(thresh_line)
            thresh_html = f"<div style='font-size: 11pt; font-weight: bold; color: #FF4444;'>{confidence_pct:.0f}% Confidence Threshold (F={threshold:.2f})</div>"
            thresh_label = pg.TextItem(
                html=thresh_html, anchor=(0, 1)
            )
            thresh_label.setPos(p_min + (p_max - p_min) * 0.02, threshold)
            p.addItem(thresh_label)

            if T0 is not None:
                # Black vertical stem at T0
                vl_t0 = pg.InfiniteLine(
                    pos=T0, angle=90,
                    pen=pg.mkPen('w', width=2.0),
                )
                p.addItem(vl_t0)
                t0_html = "<div style='font-size: 12pt; font-weight: bold; color: white;'>T₀</div>"
                t0_label = pg.TextItem(html=t0_html, anchor=(0.5, 1))
                t0_label.setPos(T0, p.viewRange()[1][1] if p.viewRange()[1][1] > 0 else threshold * 5)
                p.addItem(t0_label)

                # Red star marker at T0
                # Find the index in the masked/reversed array closest to T0
                t0_idx = int(np.argmin(np.abs(periods - T0)))
                if 0 <= t0_idx < len(vals):
                    p.plot([periods[t0_idx]], [vals[t0_idx]],
                           pen=None, symbol='star',
                           symbolPen='r', symbolBrush='r', symbolSize=14)

                # Magenta dashed lines at 3T and 2T (harmonics / sub-harmonics)
                t2, t3 = T0 / 2.0, T0 / 3.0
                for harm_period, harm_label in [(t3, '3T'), (t2, '2T')]:
                    if p_min <= harm_period <= p_max:
                        vl = pg.InfiniteLine(
                            pos=harm_period, angle=90,
                            pen=pg.mkPen('#FF00FF', width=1.5, style=Qt.DashLine),
                        )
                        p.addItem(vl)
                        hl_html = f"<div style='font-size: 11pt; font-weight: bold; color: #FF00FF;'>{harm_label}</div>"
                        hl = pg.TextItem(html=hl_html, anchor=(0.5, 1))
                        hl.setPos(harm_period, threshold * 2)
                        p.addItem(hl)


    def _build_cross_spectrum(self, graph, cross_data, periods, mask, p_min, p_max):
        """Helper to build a 2x3 grid for Cross-Spectrum matching MATLAB reference."""
        p_pwa = graph.addPlot(row=0, col=0)
        p_rea = graph.addPlot(row=0, col=1, title="Co-spectrum (In-phase)")
        p_ima = graph.addPlot(row=0, col=2, title="Quadrature spectrum (Phase shift)")
        
        p_pwb = graph.addPlot(row=1, col=0)
        p_reb = graph.addPlot(row=1, col=1, title="Co-spectrum (In-phase)")
        p_imb = graph.addPlot(row=1, col=2, title="Quadrature spectrum (Phase shift)")
        
        all_plots = [p_pwa, p_rea, p_ima, p_pwb, p_reb, p_imb]
        for p in all_plots:
            p.setLabel('bottom', 'Period (s)')
            p.showGrid(x=True, y=True)
            p.setXRange(p_min, p_max)
            
            # Horizontal line at 0 for real/imag
            if p in (p_rea, p_ima, p_reb, p_imb):
                zero_line = pg.InfiniteLine(
                    pos=0, angle=0, pen=pg.mkPen('gray', width=1.5, style=Qt.DashLine)
                )
                p.addItem(zero_line)
                
        # Link X axes
        for p in all_plots[1:]:
            p.setXLink(all_plots[0])
            
        p_pwa.setLabel('left', 'Power')
        p_pwb.setLabel('left', 'Power')
        p_rea.setLabel('left', 'Re(Pxy)')
        p_reb.setLabel('left', 'Re(Pxy)')
        p_ima.setLabel('left', 'Im(Pxy)')
        p_imb.setLabel('left', 'Im(Pxy)')
        
        for pol_name, row_plots in [('Pol A', (p_pwa, p_rea, p_ima)), ('Pol B', (p_pwb, p_reb, p_imb))]:
            if pol_name not in cross_data:
                continue
                
            p_pow, p_re, p_im = row_plots
            
            vals_pow = cross_data[pol_name]['power'][mask][::-1]
            vals_re = cross_data[pol_name]['real'][mask][::-1]
            vals_im = cross_data[pol_name]['imag'][mask][::-1]
            
            # Using white instead of black for the power plot since the app is dark mode
            p_pow.plot(periods, vals_pow, pen=pg.mkPen('w', width=1.5))
            p_re.plot(periods, vals_re, pen=pg.mkPen('#42A5F5', width=1.5)) # Blue
            p_im.plot(periods, vals_im, pen=pg.mkPen('#EF5350', width=1.5)) # Red
            
            # Peak detection (Top-3 for Cross Spectrum)
            peaks, _ = find_peaks(vals_pow, distance=max(1, len(vals_pow)//50))
            if len(peaks) > 0:
                top_indices = sorted(peaks, key=lambda i: vals_pow[i], reverse=True)[:3]
                top_periods = [periods[i] for i in top_indices]
                
                peaks_str = ", ".join(f"{tp:.1f}" for tp in sorted(top_periods, reverse=True))
                title_html = (
                    f"Pol {pol_name[-1]} (20 MHz vs 25 MHz)<br>"
                    f"<span style='color:red; font-size: 10pt;'>Amplitude |Pxy| (Top 3: {peaks_str} s)</span>"
                )
                p_pow.setTitle(title_html)
                
                peak_x = [periods[i] for i in top_indices]
                peak_y = [vals_pow[i] for i in top_indices]
                p_pow.plot(peak_x, peak_y, pen=None, symbol='t', symbolPen='r', symbolBrush='r', symbolSize=10)
                
                # Add vertical lines to real and imag plots at the peak periods
                for px in peak_x:
                    vl_re = pg.InfiniteLine(pos=px, angle=90, pen=pg.mkPen('r', width=1.5, style=Qt.DotLine))
                    p_re.addItem(vl_re)
                    vl_im = pg.InfiniteLine(pos=px, angle=90, pen=pg.mkPen('r', width=1.5, style=Qt.DotLine))
                    p_im.addItem(vl_im)
            else:
                p_pow.setTitle(f"Pol {pol_name[-1]} (20 MHz vs 25 MHz)<br>Amplitude |Pxy|")

    # ------------------------------------------------------------------
    # Velocity table formatter
    # ------------------------------------------------------------------

    @staticmethod
    def _format_velocity_table(velocities, band_label):
        """Render velocity estimates as a plain-text table."""
        lines = [
            SPECTRAL_TAB_VELOCITY_HEADER,
            "Model: Line-of-sight transmission.  "
            "Beam separation (dx) = 2500 m",
            f"Band: {band_label}",
            "-" * 70,
        ]

        for pol_name in ('Pol A', 'Pol B'):
            pol_data = velocities.get(pol_name, [])
            lines.append(f"\n[{pol_name} (20 MHz vs 25 MHz)]")

            if not pol_data:
                lines.append("  No significant peaks found.")
                continue

            for i, v in enumerate(pol_data):
                lines.append(
                    f"  Peak {i + 1}: Period = {v['period']:7.1f} s | "
                    f"Phase = {v['phase_deg']:7.1f}\u00b0 | "
                    f"dt = {v['dt']:7.1f} s | "
                    f"Velocity = {v['velocity']:7.1f} m/s"
                )

        return '\n'.join(lines)
