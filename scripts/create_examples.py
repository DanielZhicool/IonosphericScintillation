"""Script to generate 7 educational Jupyter notebooks in examples/."""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {"language_info": {"name": "python", "version": "3.11"}, "orig_nbformat": 4},
        "nbformat": 4,
        "nbformat_minor": 2,
    }


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_all_notebooks():
    out_dir = Path("examples")
    out_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Notebook 1: 01_signal_generation.ipynb
    # -------------------------------------------------------------------------
    nb1 = make_notebook(
        [
            md_cell("""# 01: Progressive Signal Synthesis & Scintillation Models

This notebook demonstrates the physical and mathematical signal models used to simulate **URAN-4 ionospheric scintillation** observations.

### Mathematical Formulation
1. **Gaussian White Noise**: $w(t) \\sim \\mathcal{N}(0, \\sigma^2)$
2. **Kolmogorov Power-Law Colored Noise**: $PSD(f) \\propto f^{-8/3}$, representing turbulent ionospheric density fluctuations.
3. **Deterministic Harmonics**: Scintillation wave modes embedded at peak frequencies $f_0$.
4. **Dual-Channel Phase Delay**: Delay $\\tau = dx / v$ between dual-beam observations ($dx = 2500\\text{ m}$).
"""),
            code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from core.synthetic_generator import generate_synthetic_scintillation, generate_power_law_noise

# Set random seed for reproducibility
np.random.seed(42)
fs = 1.0  # Hz
duration = 300  # seconds
t = np.arange(0, duration, 1.0 / fs)

# 1. Pure Sine Wave vs White Noise vs Power-law Colored Noise
sine_wave = 0.5 * np.sin(2 * np.pi * 0.05 * t)
white_noise = 0.2 * np.random.randn(len(t))
colored_noise = 0.2 * generate_power_law_noise(len(t), fs=fs, spectral_index=8/3)


# Plot generated time series
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, sine_wave + white_noise, label="Sine + White Noise", alpha=0.8)
plt.plot(t, colored_noise, label="f^-8/3 Colored Noise", alpha=0.8)
plt.title("Synthetic Ionospheric Signal Components")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (Volt)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# 2. Dual-Channel Delay Simulation (dx = 2500m, dt = 2.0s -> v = 1250 m/s)
ch1, ch2 = generate_synthetic_scintillation(
    length=300,
    fs=fs,
    delay_sec=2.0,
)
t_axis = t

plt.subplot(2, 1, 2)
plt.plot(t_axis[:100], ch1[:100], label="Channel 1 (20 MHz)", color="navy")
plt.plot(t_axis[:100], ch2[:100], label="Channel 2 (25 MHz, dt=2.0s)", color="crimson", linestyle="--")

plt.title("Dual-Channel Scintillation Signal with Injected Delay (dt = 2.0s)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
"""),
        ]
    )
    (out_dir / "01_signal_generation.ipynb").write_text(json.dumps(nb1, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Notebook 2: 02_preprocessing.ipynb
    # -------------------------------------------------------------------------
    nb2 = make_notebook(
        [
            md_cell("""# 02: Step-by-Step Signal Preprocessing & Scale Separation

This notebook demonstrates the digital signal processing (DSP) cleaning pipeline applied to raw URAN-4 time series.

### Pipeline Stages
1. **Hampel Filter**: Rolling median & MAD outlier detection to remove industrial impulse spikes without distorting surrounding clean data.
2. **Savitzky-Golay Filter**: Polynomial smoothing ($k=2$) to preserve peak amplitudes while suppressing high-frequency measurement noise.
3. **Butterworth Bandpass Filtering**: 4th-order SOS filter isolating **Small Scale Bubbles** ($5\\text{--}150\\text{ s}$) and **Large Scale Clouds** ($150\\text{--}600\\text{ s}$).
"""),
            code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from core.signal_processing import bandpass_filter, clean_and_smooth_signal

np.random.seed(123)
fs = 1.0
t = np.arange(0, 500, 1.0 / fs)

# Generate clean signal + industrial impulse spikes
clean_signal = 1.0 + 0.3 * np.sin(2 * np.pi * 0.02 * t) + 0.2 * np.sin(2 * np.pi * 0.005 * t)
noisy_signal = clean_signal + 0.05 * np.random.randn(len(t))

# Inject spikes (>10 sigma)
noisy_signal[100] += 5.0
noisy_signal[250] -= 4.5
noisy_signal[380] += 6.0

# 1. Hampel Outlier Removal
cleaned_hampel = clean_and_smooth_signal(noisy_signal, window_size=15, n_sigmas=3.0, apply_smoothing=False)

# 2. Savitzky-Golay Polynomial Smoothing
smoothed_signal = clean_and_smooth_signal(noisy_signal, window_size=15, n_sigmas=3.0, apply_smoothing=True)

# 3. Bandpass Scale Separation
small_bubbles = bandpass_filter(smoothed_signal, lowcut=1/150, highcut=1/5, fs=fs)
large_clouds = bandpass_filter(smoothed_signal, lowcut=1/600, highcut=1/150, fs=fs)

# Plot Preprocessing Steps
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(t, noisy_signal, color="darkred", alpha=0.7, label="Raw Signal with Spikes")
axes[0].plot(t, cleaned_hampel, color="blue", alpha=0.8, label="Hampel Filtered")
axes[0].set_title("1. Hampel Filter (Outlier Rejection)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, cleaned_hampel, color="gray", alpha=0.5, label="Hampel Output")
axes[1].plot(t, smoothed_signal, color="green", linewidth=1.5, label="Savitzky-Golay Smoothed")
axes[1].set_title("2. Savitzky-Golay Polynomial Smoothing")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, small_bubbles, color="purple", label="Small Bubbles (5 - 150s)")
axes[2].set_title("3a. Bandpass Scale: Small Bubbles (0.0067 Hz - 0.20 Hz)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, large_clouds, color="teal", label="Large Clouds (150 - 600s)")
axes[3].set_title("3b. Bandpass Scale: Large Clouds (0.00167 Hz - 0.0067 Hz)")
axes[3].set_xlabel("Time (s)")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""),
        ]
    )
    (out_dir / "02_preprocessing.ipynb").write_text(json.dumps(nb2, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Notebook 3: 03_wavelet_analysis.ipynb
    # -------------------------------------------------------------------------
    nb3 = make_notebook(
        [
            md_cell("""# 03: Time-Frequency Spectroscopy (CWT vs. Synchrosqueezing SST)

This notebook compares **Continuous Wavelet Transform (CWT)** and **Synchrosqueezed Wavelet Transform (SST)** using Generalized Morse Wavelets (GMW, $\\gamma=3, \\beta=30$).

### Key Advantages of Synchrosqueezing (SST)
- Reassigns CWT energy coefficients along the frequency axis based on instantaneous frequency $\\omega(a, b) = -i \\frac{\\partial_b W(a, b)}{W(a, b)}$.
- Sharpen ridges and eliminates vertical smearing ("smearing fans") in time-frequency space.
"""),
            code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from core.signal_processing import compute_cwt_spectrogram

fs = 1.0  # Hz
t = np.arange(0, 300, 1.0 / fs)

# Generate non-stationary signal with two transient bursts
signal = (
    np.sin(2 * np.pi * 0.05 * t) * (t < 150) +
    np.sin(2 * np.pi * 0.12 * t) * (t >= 150) +
    0.1 * np.random.randn(len(t))
)

# Compute CWT Spectrogram (Standard Morse CWT)
cwt_img = compute_cwt_spectrogram(signal, fs=fs, lowcut=0.01, highcut=0.4, use_ssq=False)

# Compute SST Spectrogram (Synchrosqueezed CWT)
sst_img = compute_cwt_spectrogram(signal, fs=fs, lowcut=0.01, highcut=0.4, use_ssq=True)

# Plot CWT vs SST Side-by-Side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

im1 = ax1.imshow(cwt_img, aspect="auto", origin="lower", cmap="viridis", extent=[0, 300, 0.01, 0.4])
ax1.set_title("Standard CWT Spectrogram (Morse Wavelet)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")
fig.colorbar(im1, ax=ax1, label="Log Power (dB)")

im2 = ax2.imshow(sst_img, aspect="auto", origin="lower", cmap="magma", extent=[0, 300, 0.01, 0.4])
ax2.set_title("Synchrosqueezed SST Spectrogram (Ridge Sharpened)")
ax2.set_xlabel("Time (s)")
fig.colorbar(im2, ax=ax2, label="Log Power (dB)")

plt.tight_layout()
plt.show()
"""),
        ]
    )
    (out_dir / "03_wavelet_analysis.ipynb").write_text(json.dumps(nb3, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Notebook 4: 04_multitaper_psd.ipynb
    # -------------------------------------------------------------------------
    nb4 = make_notebook(
        [
            md_cell("""# 04: Thomson Multitaper PSD & Harmonic Detection (F-Test)

This notebook demonstrates **Thomson Multitaper Power Spectral Density (PSD)** estimation using Discrete Prolate Spheroidal Sequences (DPSS/Slepian tapers, $K=7, NW=4.0$), non-parametric **Jackknife 95% Confidence Intervals**, and **Thomson F-Test** for deterministic harmonic line detection with **Benjamini-Hochberg FDR control** ($\alpha=0.05$).
"""),
            code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from core.spectral_analysis import compute_multitaper_psd, compute_ftest

fs = 1.0  # Hz
N = 1000
t = np.arange(N) / fs

# Background red noise + strong embedded harmonic at f0 = 0.15 Hz
signal = np.sin(2 * np.pi * 0.15 * t) + 0.5 * np.random.randn(N)

# 1. Thomson Multitaper PSD with Jackknife 95% CI
psd_res = compute_multitaper_psd(signal, fs=fs, nw=4.0, n_tapers=7, compute_ci=True)

# 2. Thomson F-Test for Harmonic Detection
freqs, f_stat, f_threshold, T0 = compute_ftest(signal, fs=fs, nw=4.0, n_tapers=7, fdr_alpha=0.05)

# Plot PSD with Jackknife Bounds & F-Test Threshold
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# PSD Plot
ax1.semilogy(psd_res.freqs, psd_res.psd, color="navy", label="Multitaper PSD")
if psd_res.ci95_low is not None and psd_res.ci95_high is not None:
    ax1.fill_between(psd_res.freqs, psd_res.ci95_low, psd_res.ci95_high, color="blue", alpha=0.2, label="Jackknife 95% CI")
ax1.set_title("Thomson Multitaper PSD (K=7, NW=4.0)")
ax1.set_ylabel("Power Spectral Density (V²/Hz)")
ax1.legend()
ax1.grid(True, which="both", linestyle="--", alpha=0.5)

# F-Test Plot
ax2.plot(freqs, f_stat, color="crimson", label="F-Statistic")
ax2.axhline(f_threshold, color="black", linestyle=":", label="F-Threshold (FDR alpha=0.05)")
ax2.set_title(f"Thomson F-Test for Deterministic Harmonics (Dominant T0: {T0} s)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("F-Statistic")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print(f"Dominant Oscillation Period T0: {T0}")
"""),
        ]
    )
    (out_dir / "04_multitaper_psd.ipynb").write_text(json.dumps(nb4, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Notebook 5: 05_cross_spectrum_velocity.ipynb
    # -------------------------------------------------------------------------
    nb5 = make_notebook(
        [
            md_cell("""# 05: Dual-Channel Cross-Spectral Drift Velocity Estimation

This notebook demonstrates **Ionospheric Drift Velocity Estimation (IDVE)** via cross-spectral analysis between dual-channel URAN-4 signals.

### Estimation Steps
1. **Cross-Spectral Density**: $S_{xy}(f) = C_{xy}(f) - i Q_{xy}(f)$
2. **Magnitude-Squared Coherence**: $C_{xy}(f) = \\frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)}$
3. **Phase Spectrum Unwrapping**: $\\phi(f) = \\arg(S_{xy}(f))$
4. **Weighted Linear Regression**: $\\phi(f) = \\phi_0 - 2\\pi f \\tau$ over high-coherence bands ($C_{xy} \\ge 0.7$).
5. **Velocity Recovery**: $v = \\text{sign}(\\tau) \\frac{dx}{|\\tau|}$ ($dx = 2500\\text{ m}$).
"""),
            code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from core.synthetic_generator import generate_synthetic_scintillation
from core.spectral_analysis import compute_cross_spectrum, estimate_velocities, find_spectral_peaks

fs = 1.0
dx = 2500.0  # meters baseline
dt_true = 2.0  # seconds delay -> v_true = 1250 m/s

# Generate dual-channel scintillation data
ch1, ch2 = generate_synthetic_scintillation(length=400, fs=fs, delay_sec=dt_true)

# Compute Cross-Spectrum & Coherence
freqs, cross_power, phase_deg, coherence, _, _ = compute_cross_spectrum(ch1, ch2, fs=fs)

# Detect Spectral Peaks
peak_indices = find_spectral_peaks(cross_power, freqs, lowcut_hz=0.01, highcut_hz=0.4, n_peaks=3)

# Estimate Velocity & Confidence Intervals
estimates = estimate_velocities(
    phase_deg, freqs, peak_indices, coherence=coherence, dx=dx, min_coherence=0.7
)

# Plot Coherence & Phase Slope Regression
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(freqs, coherence, color="teal", label="Coherence Cxy(f)")
ax1.axhline(0.7, color="red", linestyle="--", label="Coherence Threshold (0.7)")
ax1.set_title("Magnitude-Squared Coherence")
ax1.set_ylabel("Coherence")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(freqs, phase_deg, "o", color="gray", alpha=0.5, label="Raw Phase Deg")
for v_est in estimates:
    if v_est.is_valid:
        ax2.plot(v_est.peak_freq, np.degrees(2*np.pi*v_est.peak_freq*v_est.dt), "r*", markersize=12, label=f"Valid Peak v={v_est.velocity:.1f} m/s")

ax2.set_title(f"Cross-Spectral Phase Difference (True v = {dx/dt_true:.1f} m/s)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Phase (deg)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Print Estimated Velocities
for i, est in enumerate(estimates):
    print(f"Peak {i+1} at f={est.peak_freq:.3f} Hz:")
    print(f"  Delay dt = {est.dt:.3f} s (95% CI: [{est.dt_ci95[0]:.3f}, {est.dt_ci95[1]:.3f}])")
    print(f"  Velocity v = {est.velocity:.1f} m/s (Valid: {est.is_valid}, Gating: {est.gating_reason})")
"""),
        ]
    )
    (out_dir / "05_cross_spectrum_velocity.ipynb").write_text(json.dumps(nb5, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Notebook 6: 06_validation_pipeline.ipynb
    # -------------------------------------------------------------------------
    nb6 = make_notebook(
        [
            md_cell("""# 06: End-to-End Synthetic Pipeline Validation

This notebook performs a quantitative validation of the complete processing pipeline by comparing extracted parameters against injected ground truth values.
"""),
            code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd

from core.synthetic_generator import generate_synthetic_scintillation
from core.signal_processing import clean_and_smooth_signal
from core.spectral_analysis import compute_cross_spectrum, estimate_velocities, find_spectral_peaks

# Ground Truth Injected Parameters
TRUE_DT = 1.5  # seconds
TRUE_DX = 2500.0  # meters
TRUE_VELOCITY = TRUE_DX / TRUE_DT  # 1666.7 m/s

# 1. Generate Synthetic Data
ch1, ch2 = generate_synthetic_scintillation(length=300, fs=1.0, delay_sec=TRUE_DT)

# 2. Preprocess Signals
ch1_clean = clean_and_smooth_signal(ch1, window_size=15, n_sigmas=3.0, apply_smoothing=True)
ch2_clean = clean_and_smooth_signal(ch2, window_size=15, n_sigmas=3.0, apply_smoothing=True)

# 3. Spectral Analysis & Drift Velocity Estimation
freqs, cross_power, phase_deg, coherence, _, _ = compute_cross_spectrum(ch1_clean, ch2_clean, fs=1.0)
peaks = find_spectral_peaks(cross_power, freqs, lowcut_hz=0.01, highcut_hz=0.4, n_peaks=1)
v_est = estimate_velocities(phase_deg, freqs, peaks, coherence=coherence, dx=TRUE_DX)[0]

# 4. Display Validation Table
results_df = pd.DataFrame([{
    "Parameter": "Time Delay dt (s)",
    "Injected Ground Truth": TRUE_DT,
    "Pipeline Recovered": round(v_est.dt, 3),
    "Error": f"{abs(v_est.dt - TRUE_DT):.3f} s",
    "95% CI Bounds": f"[{v_est.dt_ci95[0]:.3f}, {v_est.dt_ci95[1]:.3f}]",
}, {
    "Parameter": "Drift Velocity v (m/s)",
    "Injected Ground Truth": round(TRUE_VELOCITY, 1),
    "Pipeline Recovered": round(v_est.velocity, 1),
    "Error": f"{abs(v_est.velocity - TRUE_VELOCITY):.1f} m/s",
    "95% CI Bounds": f"[{v_est.velocity_ci95[0]:.1f}, {v_est.velocity_ci95[1]:.1f}]",
}])

print("=== End-to-End Pipeline Validation Summary ===")
print(results_df.to_string(index=False))
"""),
        ]
    )
    (out_dir / "06_validation_pipeline.ipynb").write_text(json.dumps(nb6, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Notebook 7: 07_real_uran4_observation.ipynb
    # -------------------------------------------------------------------------
    nb7 = make_notebook(
        [
            md_cell("""# 07: Authentic URAN-4 Observational Analysis Workflow

This notebook demonstrates the complete observational workflow for analyzing real URAN-4 radio telescope observations, from raw file reading to time-frequency analysis and JSON provenance audit manifest export.
"""),
            code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.spectral_analysis import run_spectral_pipeline
from core.provenance import export_provenance_manifest
from core.config import ProcessingConfig

# 1. Load Demonstration Scintillation Dataset
df = pd.read_csv("data/sample_scintillation.csv")
print("Dataset Head:")
print(df.head())

# Extract signals
signals = {
    "20 MHz Pol A": df["Ch1_Volt"].to_numpy(),
    "20 MHz Pol B": df["Ch2_Volt"].to_numpy(),
    "25 MHz Pol A": df["Ch1_Volt"].to_numpy(),
    "25 MHz Pol B": df["Ch2_Volt"].to_numpy(),
}

# 2. Run Spectral Analysis Pipeline
config = ProcessingConfig(sampling_rate=1.0)
results = run_spectral_pipeline(
    pm_signals=signals,
    fs=1.0,
    lowcut=0.01,
    highcut=0.5,
    window_size=15,
    n_sigmas=3.0,
    apply_smoothing=True,
    config=config
)

# Display Extracted Velocity Estimates
print("\n=== Velocity Estimates Table ===")
for pol_name, v_list in results["velocities"].items():
    for v in v_list:
        print(f"{pol_name} | Freq: {v.peak_freq:.3f} Hz | dt: {v.dt:.2f} s | Velocity: {v.velocity:.1f} m/s | Valid: {v.is_valid} ({v.gating_reason})")

# 3. Export Audit Provenance Manifest
manifest_path = export_provenance_manifest(
    output_path="scratch/uran4_observation_manifest.json",
    input_file_path="data/sample_scintillation.csv",
    config=config,
)
print(f"\nSuccessfully exported audit provenance manifest to: {manifest_path}")
"""),
        ]
    )
    (out_dir / "07_real_uran4_observation.ipynb").write_text(json.dumps(nb7, indent=2), encoding="utf-8")

    print("All 7 educational notebooks created successfully in examples/")


if __name__ == "__main__":
    build_all_notebooks()
