# Scientific Formulation & Theory

This document provides the mathematical foundations, underlying physical assumptions, mathematical notation, implementation defaults, and references for the digital signal processing (DSP), spectral analysis, wavelet transforms, and physical estimation algorithms implemented in the URAN-4 Ionospheric Scintillation Analyzer.

---

## Interactive Notebook Series

Each theoretical formulation below is accompanied by a hands-on, executable Jupyter notebook in the [`examples/`](../examples/) directory mapping mathematical equations directly to working Python code:

| Theory Section | Corresponding Interactive Jupyter Notebook | Primary Focus & Demonstration |
| :--- | :--- | :--- |
| **Section 2: Signal Generation** | [`examples/01_signal_generation.ipynb`](../examples/01_signal_generation.ipynb) | Kolmogorov colored noise, harmonic wave modes, and sub-sample Fourier phase shifting |
| **Section 3: Signal Pre-processing** | [`examples/02_preprocessing.ipynb`](../examples/02_preprocessing.ipynb) | Rolling Hampel outlier filtering, Savitzky-Golay smoothing, Butterworth bandpass, and AR(1) red-noise gap synthesis |
| **Section 3.5: Wavelet Time-Frequency** | [`examples/03_wavelet_analysis.ipynb`](../examples/03_wavelet_analysis.ipynb) | Continuous Wavelet Transform (CWT) vs. Synchrosqueezed Wavelet Transform (SST) ridge sharpening |
| **Section 4.1 & 4.2: Multitaper PSD** | [`examples/04_multitaper_psd.ipynb`](../examples/04_multitaper_psd.ipynb) | Thomson DPSS multitaper spectral estimation, Jackknife 95% CIs, and Thomson F-Test with FDR |
| **Section 4.3 & 4.6: IDVE Velocity** | [`examples/05_cross_spectrum_velocity.ipynb`](../examples/05_cross_spectrum_velocity.ipynb) | Cross-spectral estimation, phase unwrapping, magnitude coherence gating, and WLS drift velocity estimation |
| **Synthetic Validation** | [`examples/06_validation_pipeline.ipynb`](../examples/06_validation_pipeline.ipynb) | Synthetic end-to-end pipeline recovery against ground truth parameters |
| **Full Pipeline Suite** | [`examples/07_complete_pipeline_validation.ipynb`](../examples/07_complete_pipeline_validation.ipynb) | Complete pipeline suite end-to-end verification |
| **Real URAN-4 Workflow** | [`examples/08_real_uran4_observation.ipynb`](../examples/08_real_uran4_observation.ipynb) | End-to-end processing workflow on authentic URAN-4 radio telescope `.PM6` recordings |

---

## 1. Common Mathematical Notation

The following table lists the mathematical symbols, physical variables, and software parameters used throughout this document.

| Symbol | Description | Dimension / Units | Software Default / Implementation |
| :--- | :--- | :--- | :--- |
| $x(t)$, $x_n$ | Discrete input signal at time $t$ or sample index $n$ | $\text{units}$ (e.g., $\text{V}$, $\text{dBm}$, $\text{relative power}$) | Raw or detrended time series |
| $f$ | Temporal frequency | $\text{Hz}$ ($\text{s}^{-1}$) | Discrete frequency bins from FFT / RFFT |
| $f_s$ | Sampling frequency | $\text{Hz}$ ($\text{s}^{-1}$) | $1.0\text{ Hz}$ (or signal sampling rate) |
| $N$ | Total number of discrete time samples | Dimensionless integer | Length of input epoch |
| $n, i$ | Sample time index ($0 \le n < N$) | Dimensionless integer | Array index |
| $W$ | Full window width of rolling Hampel / SavGol filter | Dimensionless integer | $W = 15$ samples ($K = \frac{W-1}{2} = 7$ half-width) |
| $K$ | Number of orthogonal DPSS (Slepian) tapers | Dimensionless integer | $K = 7$ (Multitaper default, $NW=4.0$) |
| $NW$ | Time-halfbandwidth product for DPSS tapers | Dimensionless float | $NW = 4.0$ |
| $h_{n,k}$ | Value of $k$-th Slepian (DPSS) taper at index $n$ | Dimensionless ($\sum_n |h_{n,k}|^2 = 1$) | Generated via `scipy.signal.windows.dpss` |
| $\lambda_k$ | Concentration eigenvalue of $k$-th Slepian taper | Dimensionless float ($0 \le \lambda_k \le 1$) | Computed alongside DPSS tapers |
| $j$ | Imaginary unit ($\sqrt{-1}$) | Imaginary unit | `1j` (Python complex) |
| $\text{MAD}_i$ | Median Absolute Deviation at index $i$ | $\text{units}$ (signal amplitude) | Scaled by $1.4826$ |
| $n_\sigma$ | Outlier detection threshold multiplier | Dimensionless float | $n_\sigma = 3.0$ |
| $PSD(f)$, $\hat{S}(f)$ | Power Spectral Density estimate | $\text{units}^2 / \text{Hz}$ | One-sided power spectrum |
| $S_{xy}(f)$ | Multitaper cross-spectral estimate between channels $x$ and $y$ | $\text{units}_x \cdot \text{units}_y / \text{Hz}$ | Averaged cross-periodograms across DPSS tapers |
| $C_{xy}(f)$ | Magnitude-squared coherence ($0 \le C_{xy} \le 1$) | Dimensionless float | $C_{xy}(f) = |S_{xy}(f)|^2 / [S_{xx}(f) S_{yy}(f)]$ |
| $\theta(f)$ | Cross-spectral phase angle | Degrees ($^\circ$) or Radians | Computed via `np.degrees(np.angle(...))` |
| $\tau$ | Propagation time delay | Seconds ($\text{s}$) | $\tau = \theta / (360^\circ \cdot f)$ or slope $a / 2\pi$ |
| $dx$ | Receiver baseline distance | Meters ($\text{m}$) | $2500\text{ m}$ (URAN-4 baseline) |
| $v$ | Ionospheric irregularity horizontal drift velocity | Meters per second ($\text{m/s}$) | $v = \text{sign}(\tau) \cdot dx / |\tau|$ |

---

## 2. Synthetic Scintillation Signal Generation

### 2.1. Phenomenological Power-Law Scintillation Spectrum
To simulate realistic ionospheric scintillation noise, the power spectral density (PSD) is modeled as flat below a characteristic Fresnel corner frequency and decaying as a power-law at higher frequencies. This phenomenological model captures the propagation of radio waves through a thin turbulent phase screen of electron density irregularities:

$$PSD(f) = \frac{A_0}{\left[1 + \left(\frac{f}{f_F}\right)^2\right]^{p/2}}$$

#### Assumptions:
- **Spatial Turbulence Model:** Ionospheric electron density irregularities follow a power-law spatial spectrum (e.g., Kolmogorov turbulence model).
- **Frozen-Flow (Taylor's Hypothesis):** The irregular structures are static relative to their horizontal drift, meaning spatial structures project directly onto the temporal domain at a rate proportional to the drift velocity.
- **Fresnel Filtering:** Propagation diffractive effects act as a high-pass spatial filter, flattening the temporal power spectrum below the Fresnel frequency $f_F$.

#### Implementation & Software Defaults:
- **Function:** `core.synthetic_generator.generate_power_law_noise()`
- **Algorithm:** Spectral coloring via Discrete Fourier Transform (`np.fft.rfft` and `np.fft.irfft`).
- **Defaults:** Fresnel frequency $f_F = 0.1\text{ Hz}$, spectral index $p = 8/3 \approx 2.67$ (Kolmogorov 3D turbulence projection).
- **Normalization:** Output noise array is normalized to zero mean ($\mu = 0$) and unit variance ($\sigma^2 = 1.0$).

> 📖 **Interactive Notebook Tutorial:** See [`examples/01_signal_generation.ipynb`](../examples/01_signal_generation.ipynb) for a hands-on demonstration of power-law noise generation and frequency-domain phase shifts.

---

### 2.2. Fractional Delay in the Frequency Domain
Applying exact sub-sample temporal delays to a discrete signal is implemented using the Fourier Shift Theorem:

$$Y(f) = X(f) \cdot e^{-j 2\pi f \tau}$$

#### Assumptions:
- **Stationarity & Band-Limiting:** The signal is band-limited below the Nyquist frequency ($f < f_s / 2$).
- **Periodic Boundary Conditions:** The Discrete Fourier Transform (DFT) assumes periodic signals. A phase rotation shifts the signal circularly; signals are tapered or sufficiently long to prevent wrap-around boundary leakage.

#### Implementation & Software Defaults:
- **Function:** `core.synthetic_generator.apply_delay()`
- **Algorithm:** Calculates complex transfer function $H(f) = \exp(-2\pi j f \tau)$ for non-negative frequencies from `np.fft.rfftfreq()`, multiplies $X(f)$ by $H(f)$, and transforms back via `np.fft.irfft()`.
- **Units & Sign:** Delay parameter $\tau$ is in seconds (positive value = time delay / right shift).

> 📖 **Interactive Notebook Tutorial:** See [`examples/01_signal_generation.ipynb`](../examples/01_signal_generation.ipynb) for an executable demonstration of Fourier sub-sample fractional delay synthesis.

---

## 3. Signal Pre-processing & Wavelet Analysis

### 3.1. Hampel Filter (Outlier Detection)
Spike outliers (e.g., impulse industrial interference or telecommunication spikes) are detected using a rolling Hampel filter. At each sample index $i$, the sample is flagged as an outlier if:

$$|x_i - m_i| > n_{\sigma} \cdot \text{MAD}_i$$

where $m_i$ is the rolling median in a centered window of full width $W = 15$ samples ($K = \frac{W-1}{2} = 7$ half-width):

$$m_i = \text{median}(x_{i-K}, \ldots, x_i, \ldots, x_{i+K})$$

and the Median Absolute Deviation (MAD) is scaled to estimate the standard deviation under local normality:

$$\text{MAD}_i = 1.4826 \cdot \text{median}(|x_{i-K} - m_i|, \ldots, |x_{i+K} - m_i|)$$

#### Assumptions:
- **Sparsity of Outliers:** Outliers occupy less than $50\%$ of any given rolling window.
- **Local Normality:** The nominal scintillation background is locally stationary and approximately normally distributed.
- **Normal Consistency:** The scaling factor $1.4826 = 1 / (\Phi^{-1}(0.75))$ correctly maps the median absolute deviation of a normal distribution to its standard deviation ($\sigma \approx 1.4826 \cdot \text{MAD}$).

#### Implementation & Software Defaults:
- **Function:** `core.signal_processing.clean_and_smooth_signal()`
- **Defaults:** Full window width `window_size = 15` samples ($K = 7$ samples on either side), threshold `n_sigmas = 3.0`.
- **Boundary Handling:** Rolling statistics use `center=True, min_periods=1`, so boundary windows are truncated rather than reflect-padded. Flagged outliers are replaced by local medians, followed by backward/forward filling (`bfill().ffill()`).

> 📖 **Interactive Notebook Tutorial:** See [`examples/02_preprocessing.ipynb`](../examples/02_preprocessing.ipynb) for a step-by-step notebook demonstration of Hampel outlier filtering and Savitzky-Golay smoothing.

---

### 3.2. Savitzky-Golay Filter (Smoothing)
Smoothing of the cleaned scintillation signal is performed using a Savitzky-Golay polynomial filter:

$$y_i = \sum_{m=-K}^{K} C_m \cdot x_{i+m}$$

where $C_m$ are polynomial least-squares convolution weights evaluated over a local window of length $W = 2K + 1 = 15$.

#### Assumptions:
- **Local Polynomial Trend:** The underlying physical scintillation is locally smooth and well approximated by a polynomial of degree $d = 2$ over window length $W = 15$.
- **High-Frequency Noise:** Residual additive noise is zero-mean and uncorrelated at sample-to-sample scales.

#### Implementation & Software Defaults:
- **Function:** `core.signal_processing.clean_and_smooth_signal()` (calls `scipy.signal.savgol_filter`)
- **Defaults:** Polynomial degree $d = 2$, window length $W = 15$ samples.

---

### 3.3. PCHIP Monotonic Cubic Interpolation
To increase the sampling rate prior to Continuous Wavelet Transform (CWT) without introducing unphysical oscillations or Gibbs overshoot, Piecewise Cubic Hermite Interpolating Polynomials (PCHIP) are applied:

$$n_{\text{new}} = (N - 1) \cdot \text{factor} + 1$$

$$t_m = \frac{m}{f_s \cdot \text{factor}}, \quad m = 0, 1, \dots, n_{\text{new}} - 1$$

Within each sub-interval $[t_k, t_{k+1}]$, PCHIP constructs a cubic polynomial satisfying the function values and derivative conditions designed to preserve local shape monotonicity:

$$p(t) = d_k \frac{(t_{k+1}-t)^2(2(t-t_k)+h_k)}{h_k^3} + d_{k+1}\frac{(t-t_k)^2(2(t_{k+1}-t)+h_k)}{h_k^3} + \Delta_k \frac{(t_{k+1}-t)^2(t-t_k)}{h_k^2} + \Delta_{k+1}\frac{(t-t_k)^2(t-t_{k+1})}{h_k^2}$$

where $h_k = t_{k+1} - t_k$, and slopes $\Delta_k$ are harmonically weighted to prevent overshoot wherever local slopes change sign.

#### Implementation & Software Defaults:
- **Function:** `core.signal_processing.upsample_pchip()` (uses `scipy.interpolate.pchip_interpolate`)
- **Defaults:** `factor = 3` ($1.0\text{ Hz} \to 3.0\text{ Hz}$).
- **Long-Signal Memory Gating:** Automatically applied for single transits ($N \le 30,000$, `PCHIP_LONG_SIGNAL_THRESHOLD = 30000`) and bypassed for multi-day continuous files to bound heap memory.

---

### 3.4. Zero-Phase Butterworth Bandpass Filtering
To isolate ionospheric wave modes (such as 10–100 s quasi-periodic oscillations) while eliminating baseline receiver drift and Nyquist aliasing noise, a 4th-order digital Butterworth bandpass filter is evaluated in Second-Order Sections (SOS) form:

$$H(z) = \prod_{k=1}^{L} \frac{b_{0k} + b_{1k}z^{-1} + b_{2k}z^{-2}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

#### Processing Steps:
1. **Linear Detrending:** Baseline drift is removed via `scipy.signal.detrend(arr, type='linear')`, removing DC bias and linear slope.
2. **Tukey Window Tapering:** Edge discontinuities are smoothly tapered to zero using a Tukey (cosine-tapered) window:
   $$w(n) = \begin{cases} 
   \frac{1}{2}\left[1 + \cos\left(\pi\left(\frac{2n}{\alpha N} - 1\right)\right)\right], & 0 \le n < \frac{\alpha N}{2} \\
   1, & \frac{\alpha N}{2} \le n \le N\left(1 - \frac{\alpha}{2}\right) \\
   \frac{1}{2}\left[1 + \cos\left(\pi\left(\frac{2(N-1-n)}{\alpha N} - 1\right)\right)\right], & N\left(1 - \frac{\alpha}{2}\right) < n \le N-1
   \end{cases}$$
   with default $\alpha = 0.1$ ($10\%$ taper across endpoints).
3. **Zero-Phase Filtering:** Evaluated via forward-backward filtering (`scipy.signal.sosfiltfilt`), resulting in zero phase distortion ($\text{Im}\{H_{\text{eff}}(\omega)\} = 0$) and an effective 8th-order squared magnitude response $|H(\omega)|^2$.

#### Implementation & Software Defaults:
- **Function:** `core.signal_processing.bandpass_filter()`
- **Defaults:** Passband $0.01 - 0.1\text{ Hz}$ (or $1/150 - 0.2\text{ Hz}$ for wavelets), filter order $4$, Tukey $\alpha = 0.1$.

---

### 3.5. Continuous Wavelet Transform (CWT) & Synchrosqueezing (SST)

#### Continuous Wavelet Transform (CWT):
The Continuous Wavelet Transform projects signal $x(t)$ onto scaled and translated Generalized Morse Wavelets (GMW):

$$W_x(s, t) = \int_{-\infty}^{\infty} x(\tau) \frac{1}{\sqrt{s}} \psi^*\left(\frac{\tau - t}{s}\right) d\tau$$

In the frequency domain, the Generalized Morse Wavelet is defined by parameters $\gamma$ and $\beta$:

$$\Psi_{\beta, \gamma}(\omega) = U(\omega) \cdot a_{\beta, \gamma} \cdot \omega^\beta e^{-\omega^\gamma}$$

where $U(\omega)$ is the Heaviside step function, and $a_{\beta, \gamma} = 2 \left(\frac{e\gamma}{\beta}\right)^{\beta/\gamma}$ is a normalizing constant ensuring peak Fourier amplitude equals 2.

#### Synchrosqueezed Wavelet Transform (SST):
Synchrosqueezing sharpens time-frequency energy localization by reallocating CWT coefficients along the frequency axis based on the instantaneous frequency candidate $\hat{\omega}_x(s, t)$:

$$\hat{\omega}_x(s, t) = \frac{-j}{W_x(s, t)} \frac{\partial W_x(s, t)}{\partial t}$$

$$T_x(\omega, t) = \int_{\{s: W_x(s,t) \ne 0\}} W_x(s, t) \cdot \delta(\omega - \hat{\omega}_x(s, t)) \cdot s^{-3/2} ds$$

#### Implementation & Architecture:
- **Function:** `core.signal_processing.compute_cwt_spectrogram()` (via `ssqueezepy`)
- **Wavelet Presets:** Generalized Morse Wavelet with $\gamma = 3$, $\beta = 30$. Voices per octave $nv = 32$ (Standard / Bubbles preset) or $nv = 64$ (Clouds high-resolution preset).
- **CWT vs. SST Output:** When `use_ssq=True`, the displayed time-frequency map is the reassigned energy matrix $|T_w(f, t)|$; when `use_ssq=False`, the standard continuous transform magnitude $|W_x(f, t)|$ is evaluated.
- **Chunked Processing & Temporal Max-Pooling:** For long recordings ($N \ge 32,768$), signals are processed in 32,768-sample chunks with 16,384-sample overlap. Spectrogram columns are temporal max-pooled targeting $\sim 8,000$ display columns, capping peak intermediate heap allocation at $\sim 220\text{ MB}$ (CWT) / $\sim 540\text{ MB}$ (SST).

> 📖 **Interactive Notebook Tutorial:** See [`examples/03_wavelet_analysis.ipynb`](../examples/03_wavelet_analysis.ipynb) for a comparison of Morlet vs. Generalized Morse Wavelets and SST ridge sharpening.

---

## 4. Spectral & Correlation Analysis

### 4.1. Thomson Multitaper Power Spectral Density
Thomson's multitaper method reduces spectral leakage and variance by combining eigenspectra from orthogonal Slepian window sequences (Discrete Prolate Spheroidal Sequences, DPSS):

$$\hat{S}(f) = \frac{2}{f_s N} \frac{\sum_{k=0}^{K-1} \lambda_k \left| \sum_{n=0}^{N-1} x_n \cdot h_{n,k} \cdot e^{-j 2\pi f n / f_s} \right|^2}{\sum_{k=0}^{K-1} \lambda_k}$$

with the standard DC and Nyquist single-sided normalization factor adjustments.

#### Assumptions:
- **Stationarity:** The detrended signal $x_n$ is a zero-mean stationary random process.
- **Spectral Energy Concentration:** The DPSS tapers $h_{n,k}$ maximize spectral energy concentration within bandwidth $[-W, W]$.
- **Variance Reduction:** Combining eigenspectra from $K$ mutually orthogonal tapers yields significant variance reduction over single-taper periodograms while preserving narrow-band harmonic peaks.

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_multitaper_psd()`
- **Library & Algorithm:** Uses `scipy.signal.windows.dpss` for orthogonal DPSS tapers $h_{n,k}$ and `numpy.fft.rfft` for 1D real-to-complex Discrete Fourier Transforms.
- **Defaults:** Time-halfbandwidth product $NW = 4.0$, number of tapers $K = 7$ (satisfying $K \le 2NW - 1$).
- **Pre-processing:** Detrending (`scipy.signal.detrend`) is applied automatically prior to taper multiplication.

> 📖 **Interactive Notebook Tutorial:** See [`examples/04_multitaper_psd.ipynb`](../examples/04_multitaper_psd.ipynb) for a hands-on comparison of FFT vs. Welch vs. Thomson Multitaper PSD estimation and Jackknife 95% CIs.

---

### 4.2. Thomson F-Test for Harmonic Lines
To detect deterministic periodic oscillations (e.g., Traveling Ionospheric Disturbances or acoustic-gravity waves) against continuous background scintillation, we test whether a sinusoidal line of complex amplitude $\mu(f)$ rises significantly above the local continuous spectrum:

$$F(f) = \frac{(K-1) \cdot \left| \hat{\mu}(f) \right|^2 \sum_{k=0}^{K-1} |H_k(0)|^2}{\sum_{k=0}^{K-1} \left| Y_k(f) - \hat{\mu}(f) H_k(0) \right|^2}$$

where:

$$Y_k(f) = \sum_{n=0}^{N-1} x_n \cdot h_{n,k} \cdot e^{-j 2\pi f n / f_s}$$

$$H_k(0) = \sum_{n=0}^{N-1} h_{n,k}$$

$$\hat{\mu}(f) = \frac{\sum_{k=0}^{K-1} H_k(0) \cdot Y_k(f)}{\sum_{k=0}^{K-1} |H_k(0)|^2}$$

#### Statistical Significance:
- Under the null hypothesis $H_0$ (no sinusoidal component at frequency $f$), $F(f)$ follows an F-distribution with degrees of freedom $d_1 = 2$ and $d_2 = 2K - 2$.
- For statistical confidence level $C = 0.99$ (significance level $\alpha = 1 - C = 0.01$), the critical detection threshold is:
  $$F_{\text{crit}} = F^{-1}_{2, 2K-2}(1 - \alpha) = F^{-1}_{2, 2K-2}(C)$$
  For $K = 7$ tapers ($d_1 = 2, d_2 = 12$), $F_{\text{crit}} = 6.93$ at $99\%$ confidence.

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_ftest()`
- **Critical Threshold:** Critical value is evaluated using `scipy.stats.f.ppf(confidence, df1=2, df2=2*K - 2)`.
- **Defaults:** Confidence level $C = 0.99$ (`confidence = 0.99`), $K = 7$ tapers.

---

### 4.3. Multitaper Cross-Spectral Analysis
For dual-channel observations (e.g., $x_n$ at 20 MHz and $y_n$ at 25 MHz) recorded across baseline distance $dx$, the multitaper cross-spectral estimate is:

$$S_{xy}(f) = \frac{1}{K} \sum_{k=0}^{K-1} X_k(f) \cdot Y_k^*(f)$$

where $X_k(f)$ and $Y_k(f)$ are the DPSS-tapered Fourier transforms of $x_n$ and $y_n$.

#### Derived Spectral Properties:
- **Cross-Spectral Phase:** $\theta(f) = \text{angle}(S_{xy}(f)) \quad [\text{degrees or radians}]$
- **Magnitude-Squared Coherence:** 
  $$C_{xy}(f) = \frac{\left| S_{xy}(f) \right|^2}{S_{xx}(f) \cdot S_{yy}(f)}, \quad 0 \le C_{xy}(f) \le 1$$
- **Single-Peak Delay & Velocity:**
  $$\tau = \frac{\theta(f_0)}{360^\circ \cdot f_0}, \quad v = \text{sign}(\tau) \frac{dx}{|\tau|}$$

#### Assumptions & Scale Invariance:
- Phase angle $\theta(f)$ and magnitude-squared coherence $C_{xy}(f)$ are invariant to overall energy normalization factors.
- **Phase Unwrapping:** When broadband phase regression is evaluated across contiguous frequencies (`enable_phase_regression=True`), phase values are unwrapped (`np.unwrap`) to eliminate $2\pi$ branch jumps across adjacent frequency bins before linear regression.

> 📖 **Interactive Notebook Tutorial:** See [`examples/05_cross_spectrum_velocity.ipynb`](../examples/05_cross_spectrum_velocity.ipynb) for a step-by-step notebook tutorial on cross-spectral phase unwrapping, magnitude coherence, and IDVE velocity calculation.

---

### 4.4. Jackknife 95% Confidence Intervals for Multitaper PSD
To quantify variance in Multitaper spectral estimates without parametric assumptions, non-parametric Jackknife confidence bounds are calculated over the $K$ orthogonal DPSS tapers. The leave-one-out spectral estimate for taper $j$ is:

$$S_{-j}(f) = \frac{1}{K-1} \sum_{k \neq j} S_k(f)$$

The standard error of the log-power spectrum $\log \hat{S}(f)$ is:

$$\text{SE}(\log \hat{S}(f)) = \sqrt{ \frac{K-1}{K} \sum_{j=0}^{K-1} \left( \log S_{-j}(f) - \overline{\log S}(f) \right)^2 }$$

Log-normal 95% confidence interval bounds are then given by:

$$\text{CI}_{95}(f) = \left[ \hat{S}(f) \cdot e^{-1.96 \cdot \text{SE}(f)}, \; \hat{S}(f) \cdot e^{+1.96 \cdot \text{SE}(f)} \right]$$

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_multitaper_psd()`
- **Result Container:** Returns `MultitaperPSDResult` holding `psd`, `log_psd_se`, `ci95_low`, and `ci95_high`.

---

### 4.5. Multiple Testing Correction (Benjamini-Hochberg FDR)
When evaluating the Thomson F-statistic across $M$ discrete frequency bins in the analysis passband $[f_{\text{low}}, f_{\text{high}}]$, multiple hypothesis testing increases false positive peak detections. Raw p-values are computed from the $F(2, 2K-2)$ survival function:

$$p_i = 1 - F_{\text{cdf}}(F_i, 2, 2K-2)$$

The Benjamini-Hochberg (BH) procedure sorts the $M$ passband p-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$ and identifies the maximum index $k_{\max}$ such that:

$$p_{(k)} \le \frac{k}{M} \cdot \alpha_{\text{FDR}}$$

The effective critical F-statistic threshold $F_{\text{FDR}}$ is then computed from $p_{(k_{\max})}$, strictly controlling the False Discovery Rate at $\alpha_{\text{FDR}} = 0.05$.

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_ftest()`
- **Defaults:** `fdr_alpha` defaults to $0.05$.

---

### 4.6. Weighted Linear Phase Regression & Coherence Gating
For broadband scintillation irregularities covering multiple contiguous frequency bins, the unwrapped phase difference $\phi(f)$ exhibits a linear slope with respect to frequency:

$$\phi(f) = a \cdot f + b, \quad \text{where slope } a = -2\pi \tau$$

Weighted Least Squares (WLS) regression is fitted over frequencies where magnitude-squared coherence $C_{xy}(f) \ge C_{\text{min}}$ ($0.7$), using weights $w_i = C_{xy}(f_i)$:

$$\tau = -\frac{a}{2\pi}, \quad v = \text{sign}(\tau) \frac{dx}{|\tau|}$$

Analytical 95% confidence intervals for $\tau$ and $v$ are propagated from the slope standard error $\text{SE}(a)$:

$$\text{SE}(\tau) = \frac{\text{SE}(a)}{2\pi}, \quad \text{SE}(v) = \left|\frac{dx}{\tau^2}\right| \cdot \text{SE}(\tau)$$

To preserve complete observational transparency, calculated values for $v$, $\tau$, $\phi$, and coherence are always reported. If mean coherence falls below $C_{\text{min}} = 0.7$ or velocity exceeds physical thresholds ($>10,000\text{ m/s}$), informative tags (`(low coh)`, `(in-phase)`) are appended directly to the output metrics.

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.estimate_velocities()`
- **Result Container:** Returns a list of `VelocityEstimate` dataclass instances.
- **Defaults:** `coherence_threshold` defaults to $0.7$; `enable_phase_regression` defaults to `False` (single-point peak phase default).

---

## References

1. **Thomson, D. J. (1982).** *Spectrum estimation and harmonic analysis.* Proceedings of the IEEE, 70(9), 1055-1096.
2. **Percival, D. B., & Walden, A. T. (1993).** *Spectral Analysis for Physical Applications.* Cambridge University Press.
3. **Benjamini, Y., & Hochberg, Y. (1995).** *Controlling the false discovery rate: a practical and powerful approach to multiple testing.* Journal of the Royal Statistical Society B, 57(1), 289-300.
4. **Hampel, F. R. (1974).** *The influence curve and its role in robust estimation.* Journal of the American Statistical Association, 69(346), 383-393.
5. **Savitzky, A., & Golay, M. J. (1964).** *Smoothing and differentiation of data by simplified least squares procedures.* Analytical Chemistry, 36(8), 1627-1639.
6. **Fritsch, F. N., & Carlson, R. E. (1980).** *Monotone Piecewise Cubic Interpolation.* SIAM Journal on Numerical Analysis, 17(2), 238-246.
7. **Lilly, J. M., & Olhede, S. C. (2012).** *Generalized Morse Wavelets as a Superfamily of Analytic Wavelets.* IEEE Transactions on Signal Processing, 60(11), 6036-6041.
8. **Daubechies, I., Lu, J., & Wu, H.-T. (2011).** *Synchrosqueezed wavelet transforms: An empirical mode decomposition-like tool.* Applied and Computational Harmonic Analysis, 30(2), 243-261.
9. **Yeh, K. C., & Liu, C. H. (1982).** *Radio wave scintillations in the ionosphere.* Proceedings of the IEEE, 70(4), 324-360.
