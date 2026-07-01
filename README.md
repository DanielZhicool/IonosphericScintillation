# URAN-4 Ionospheric Scintillation Analyzer

Software suite for primary processing, filtering, and time-frequency analysis (spectroscopy) of radio astronomical signals from the URAN-4 radio telescope.

## Theoretical Tools & Algorithms Used

1. **Hampel Filter (Outlier Detection):** Uses rolling median and median absolute deviation (MAD) to detect and remove industrial impulse noise.
2. **Savitzky-Golay Filter:** Applies polynomial smoothing to reconstruct signal continuity while preserving the physical amplitude of ionospheric scintillations.
3. **PCHIP Interpolation:** Upsamples the signal (x3) using Piecewise Cubic Hermite Interpolating Polynomials to increase detail without introducing ringing artifacts (overshoot) common in splines.
4. **Butterworth Bandpass Filter:** 4th-order SOS (Second-Order Sections) filter used to isolate specific temporal scales ("Small bubbles" 5-150s, "Large clouds" 150-600s).
5. **Continuous Wavelet Transform (CWT):** Uses Generalized Morse Wavelets to analyze the non-stationary frequency content of scintillations over time.
6. **Synchrosqueezing Transform (SWT):** A time-frequency reassignment method used to sharpen the CWT spectrogram, producing extremely narrow and high-resolution spectral lines.
7. **Thomson Multitaper Power Spectral Density (PSD):** Uses DPSS (Slepian) tapers to estimate a low-variance, low-bias power spectrum.
8. **Thomson F-Test:** A statistical test used to detect deterministic harmonic lines against a stochastic red-noise background, determining the fundamental oscillation period ($T_0$) and its harmonics ($2T, 3T$).
9. **Multitaper Cross-Spectral Analysis:** Calculates the co-spectrum (in-phase), quadrature spectrum (phase shift), and magnitude-squared coherence between the 20 MHz and 25 MHz interferometric channels.
10. **Ionospheric Drift Velocity Estimation (IDVE):** Calculates time delays and horizontal drift velocities based on the cross-spectral phase difference and the known physical separation of the telescope beams (2500m).
11. **Red Noise Generation:** Synthesizes Brownian bridge-corrected red noise to seamlessly fill calibration gaps in the data without creating boundary artifacts.
12. **Tukey Windowing:** Applies a tapered cosine window to the edges of the signal to suppress boundary artifacts ("edge fans") during Fourier and Wavelet transforms.

## Installation

1. Install [uv](https://github.com/astral-sh/uv), the fast Python package manager. (e.g., `pip install uv` or via standalone installer).
2. Clone this repository and navigate to the project directory.
3. Run the following command to automatically install all dependencies and launch the application:
   ```bash
   uv run app.py
   ```

## Usage Guide

### 1. Loading Data
<!-- Image placeholder: Data loading -->
![Loading Data]()
* Click **"Load PM6 File"** to load the raw binary observation data.
* Click **"Load REGI Log"** to automatically split the data into individual observation sessions and calibrate calibration gaps using red noise.

### 2. Signal Processing & Filtering
<!-- Image placeholder: Signal processing -->
![Signal Processing]()
* Adjust the **Window Size** and **Sigma** for the Hampel filter to aggressively or softly clean outliers.
* Toggle **Savitzky-Golay smoothing**.
* Select the desired **Spectral Band** (e.g., 5-150s or 150-600s) and click **"Analyze"** to apply the bandpass filters and CWT spectrogram.
* You can manually replace very noisy regions by selecting them on the graph and clicking **"Apply Red Noise to Selection"**.

### 3. Spectral Analysis
<!-- Image placeholder: Spectral Analysis -->
![Spectral Analysis]()
* Click **"Run Spectral Analysis"** on a specific source to compute the Multitaper PSD, F-Test, and Cross-Spectrum.
* View the detected periods ($T_0$, $2T$, $3T$) and estimated horizontal drift velocities between the 20 MHz and 25 MHz beams.

### 4. Exporting Results
<!-- Image placeholder: Exporting -->
![Exporting]()
* Use **"Export Plots"** to save the current view (Raw, Filtered, Spectrogram).
* Use **"Batch Export"** to automatically process and save all charts (Time Domain, PSD, F-Test, Cross-Spectrum) and theoretical logs for multiple sources, channels, and bands at once.

---

## Configuration Parameters (`core/config.py`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `DEFAULT_WINDOW_SIZE` | 15 | Hampel filter window size |
| `DEFAULT_N_SIGMAS` | 3.0 | Deviation threshold for spike removal |
| `SAVGOL_POLYORDER` | 2 | Savitzky-Golay polynomial order |
| `TUKEY_ALPHA` | 0.1 | Edge smoothing degree for Tukey window |
| `MTM_N_TAPERS` | 7 | Number of DPSS tapers for Multitaper method |
| `MTM_NW` | 4.0 | Time-bandwidth product for DPSS windows |
| `FTEST_CONFIDENCE` | 0.99 | Significance level for Fisher's criterion (F-Test) |
| `CROSS_SPECTRUM_DX` | 2500 | Distance between telescope beams (meters) |
| `VELOCITY_N_PEAKS` | 3 | Number of top peaks to use for velocity estimation |