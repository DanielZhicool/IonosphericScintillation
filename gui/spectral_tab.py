"""
SpectralTab widget for displaying Multitaper PSD, F-Test,
Cross-Spectrum, and ionospheric drift velocity results.
"""

import numpy as np
import pyqtgraph as pg
from scipy.signal import find_peaks
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QTextEdit, QSplitter,
)
from PySide6.QtCore import Qt

from gui.constants import (
    SPECTRAL_TAB_PSD_TITLE,
    SPECTRAL_TAB_FTEST_TITLE,
    SPECTRAL_TAB_CROSS_POWER_TITLE,
    SPECTRAL_TAB_CROSS_PHASE_TITLE,
    SPECTRAL_TAB_VELOCITY_HEADER,
    SPECTRAL_BAND_SMALL,
    SPECTRAL_BAND_LARGE,
    SPECTRAL_CHANNEL_COLORS,
    SPECTRAL_CROSS_COLORS,
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
        self._build_2x2_grid(
            psd_widget, results['psd'], periods, mask, period_min, period_max,
            ylabel='PSD', log_y=True
        )

        # ── 2. F-Test Tab (2x2 Grid) ─────────────────────────────────
        ftest_widget = pg.GraphicsLayoutWidget()
        analysis_tabs.addTab(ftest_widget, SPECTRAL_TAB_FTEST_TITLE)
        threshold = results['ftest'].get('threshold', 1.0)
        self._build_2x2_grid(
            ftest_widget, results['ftest'], periods, mask, period_min, period_max,
            ylabel='F-statistic', threshold=threshold
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

    def _build_2x2_grid(self, graph, data_dict, periods, mask, p_min, p_max, ylabel, log_y=False, threshold=None):
        """Helper to build a 2x2 grid of plots for the 4 P-M channels."""
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
                
            p = graph.addPlot(row=row, col=col, title=f"{ch_name}")
            p.setLabel('bottom', 'Period (s)')
            p.setLabel('left', ylabel)
            p.showGrid(x=True, y=True)
            p.setXRange(p_min, p_max)
            if log_y:
                p.setLogMode(x=False, y=True)
                
            if link_plot is None:
                link_plot = p
            else:
                p.setXLink(link_plot)
                
            if threshold is not None:
                thresh_line = pg.InfiniteLine(
                    pos=threshold, angle=0,
                    pen=pg.mkPen('r', width=1.5, style=Qt.DashLine),
                )
                p.addItem(thresh_line)
                thresh_label = pg.TextItem(f'95 % ({threshold:.1f})', color='r', anchor=(1, 1))
                thresh_label.setPos(p_max, threshold)
                p.addItem(thresh_label)
                
            vals = data_dict[ch_name][mask][::-1]
            color = SPECTRAL_CHANNEL_COLORS.get(ch_name, '#FFFFFF')
            p.plot(periods, vals, pen=pg.mkPen(color, width=1.5))

            # Peak finding for annotations (matching MATLAB reference)
            peaks, _ = find_peaks(vals, distance=max(1, len(vals)//50))
            if len(peaks) > 0:
                # Top 5 by amplitude
                top_indices = sorted(peaks, key=lambda i: vals[i], reverse=True)[:5]
                top_periods = [periods[i] for i in top_indices]
                
                # HTML Title with red subtitle for peaks
                peaks_str = ", ".join(f"{tp:.1f}" for tp in sorted(top_periods, reverse=True))
                title_html = f"{ch_name}<br><span style='color:red; font-size: 10pt;'>Top-5 periods (s): {peaks_str}</span>"
                p.setTitle(title_html)
                
                # Draw red markers on the peaks
                peak_x = [periods[i] for i in top_indices]
                peak_y = [vals[i] for i in top_indices]
                p.plot(peak_x, peak_y, pen=None, symbol='t', symbolPen='r', symbolBrush='r', symbolSize=10)

    def _build_cross_spectrum(self, graph, cross_data, periods, mask, p_min, p_max):
        """Helper to build a 2x2 grid for Cross-Spectrum (Row 1: Coherence, Row 2: Phase)."""
        p_coh_a = graph.addPlot(row=0, col=0, title="Coherence (Pol A: 20 vs 25 MHz)")
        p_coh_b = graph.addPlot(row=0, col=1, title="Coherence (Pol B: 20 vs 25 MHz)")
        
        p_phs_a = graph.addPlot(row=1, col=0, title="Phase (Pol A: 20 vs 25 MHz)")
        p_phs_b = graph.addPlot(row=1, col=1, title="Phase (Pol B: 20 vs 25 MHz)")
        
        for p in [p_coh_a, p_coh_b, p_phs_a, p_phs_b]:
            p.setLabel('bottom', 'Period (s)')
            p.showGrid(x=True, y=True)
            p.setXRange(p_min, p_max)
            
        p_coh_b.setXLink(p_coh_a)
        p_phs_a.setXLink(p_coh_a)
        p_phs_b.setXLink(p_coh_a)
        
        p_coh_a.setLabel('left', 'Coherence')
        p_coh_a.setYRange(0, 1)
        p_coh_b.setLabel('left', 'Coherence')
        p_coh_b.setYRange(0, 1)
        
        p_phs_a.setLabel('left', 'Phase (°)')
        p_phs_a.setYRange(-180, 180)
        p_phs_b.setLabel('left', 'Phase (°)')
        p_phs_b.setYRange(-180, 180)
        
        if 'Pol A' in cross_data:
            vals_coh = cross_data['Pol A']['coherence'][mask][::-1]
            vals_phs = cross_data['Pol A']['phase'][mask][::-1]
            p_coh_a.plot(periods, vals_coh, pen=pg.mkPen(SPECTRAL_CROSS_COLORS.get('Pol A', 'w'), width=1.5))
            p_phs_a.plot(periods, vals_phs, pen=pg.mkPen(SPECTRAL_CROSS_COLORS.get('Pol A', 'w'), width=1.5))
            
        if 'Pol B' in cross_data:
            vals_coh = cross_data['Pol B']['coherence'][mask][::-1]
            vals_phs = cross_data['Pol B']['phase'][mask][::-1]
            p_coh_b.plot(periods, vals_coh, pen=pg.mkPen(SPECTRAL_CROSS_COLORS.get('Pol B', 'w'), width=1.5))
            p_phs_b.plot(periods, vals_phs, pen=pg.mkPen(SPECTRAL_CROSS_COLORS.get('Pol B', 'w'), width=1.5))

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
