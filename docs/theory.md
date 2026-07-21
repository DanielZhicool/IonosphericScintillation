# Scientific Formulation & Theory

This document provides the mathematical foundations, underlying assumptions, mathematical notation, implementation defaults, and references for the digital signal processing (DSP), spectral analysis, and physical estimation algorithms implemented in the URAN-4 Ionospheric Scintillation Analyzer.

---

## 1. Common Mathematical Notation

The following table lists the mathematical symbols, physical variables, and software parameters used throughout this document.

| Symbol | Description | Dimension / Units | Software Default / Implementation |
| :--- | :--- | :--- | :--- |
| $x(t)$, $x_n$ | Discrete input signal at time $t$ or sample index $n$ | $\text{units}$ (e.g., $\text{V}$, $\text{dBm}$, $\text{relative power}$) | Raw or detrended time series |
| $f$ | Temporal frequency | $\text{Hz}$ ($\text{s}^{-1}$) | Discrete frequency bins from FFT / RFFT |
| $fs$ | Sampling frequency | $\text{Hz}$ ($\text{s}^{-1}$) | $1.0\text{ Hz}$ (or signal sampling rate) |
| $N$ | Total number of discrete time samples | Dimensionless integer | Length of input epoch |
| $n, i$ | Sample time index ($0 \le n < N$) | Dimensionless integer | Array index |
| $K$ | Half-width of rolling window OR number of Slepian tapers | Dimensionless integer | $K=15$ (Hampel); $K=7$ (Multitaper) |
| $NW$ | Time-halfbandwidth product for DPSS tapers | Dimensionless float | $NW = 4.0$ |
| $h_{n,k}$ | Value of $k$-th Slepian (DPSS) taper at index $n$ | Dimensionless ($\sum_n |h_{n,k}|^2 = 1$) | Generated via `scipy.signal.windows.dpss` |
| $\lambda_k$ | Concentration eigenvalue of $k$-th Slepian taper | Dimensionless float ($0 \le \lambda_k \le 1$) | Computed alongside DPSS tapers |
| $j$ | Imaginary unit ($\sqrt{-1}$) | Imaginary unit | `1j` (Python complex) |
| $\text{MAD}_i$ | Median Absolute Deviation at index $i$ | $\text{units}$ (signal amplitude) | Scaled by $1.4826$ |
| $n_\sigma$ | Outlier detection threshold multiplier | Dimensionless float | $n_\sigma = 3.0$ |
| $PSD(f)$, $\hat{S}(f)$ | Power Spectral Density estimate | $\text{units}^2 / \text{Hz}$ | One-sided power spectrum |
| $S_{12}(f)$ | Cross-power spectral density between channels 1 and 2 | $\text{units}_1 \cdot \text{units}_2 / \text{Hz}$ | Multitaper cross-spectrum |
| $\theta(f)$ | Cross-spectral phase angle | Degrees ($^\circ$) | Computed via `np.degrees(np.angle(...))` |
| $\tau(f)$ | Propagation time delay at frequency $f$ | Seconds ($\text{s}$) | $\tau(f) = \theta(f) / (360^\circ \cdot f)$ |
| $dx$ | Receiver baseline distance | Meters ($\text{m}$) | Geometry dependent (e.g., $2500\text{ m}$) |
| $v(f)$ | Irregularity horizontal drift velocity | Meters per second ($\text{m/s}$) | $v(f) = dx / |\tau(f)|$ |

---

## 2. Synthetic Scintillation Signal Generation

### 2.1. Power-Law Scintillation Spectrum
To simulate realistic ionospheric scintillation noise, we model the power spectral density (PSD) as flat at low frequencies and decaying as a power-law at high frequencies. This models the propagation of radio waves through a thin phase screen of turbulent irregularities (Fresnel filtering effect):

$$PSD(f) = \frac{A_0}{\left[1 + \left(\frac{f}{f_F}\right)^2\right]^{p/2}}$$

#### Assumptions:
- **Spatial Turbulence Model:** Ionospheric electron density irregularities follow a power-law spatial spectrum (e.g., Kolmogorov turbulence model).
- **Frozen-Flow (Taylor's Hypothesis):** The irregular structures are static relative to their horizontal drift, meaning spatial structures project directly onto the temporal domain at a rate proportional to the drift velocity.
- **Fresnel Filtering:** Propagation diffractive effects act as a high-pass spatial filter, which flattens the temporal power spectrum below the Fresnel frequency $f_F$.

#### Implementation & Software Defaults:
- **Function:** `core.synthetic_generator.generate_power_law_noise()`
- **Algorithm:** Spectral coloring via Discrete Fourier Transform (`np.fft.rfft` and `np.fft.irfft`).
- **Defaults:** Fresnel frequency $f_F = 0.1\text{ Hz}$, spectral index $p = 8/3 \approx 2.67$ (Kolmogorov turbulence value).
- **Normalization:** Output noise array is normalized to zero mean ($\mu = 0$) and unit variance ($\sigma^2 = 1.0$).

---

### 2.2. Fractional Delay in the Frequency Domain
Applying exact sub-sample delays to a discrete signal is implemented using the Fourier Shift Theorem:

$$Y(f) = X(f) \cdot e^{-j 2\pi f \tau}$$

#### Assumptions:
- **Stationarity:** The signal is stationary and band-limited below the Nyquist frequency.
- **Periodic Boundary Conditions:** The Discrete Fourier Transform (DFT) assumes periodic signals. A phase rotation shifts the signal circularly, so signals should be sufficiently long or windowed to avoid wrap-around boundary leakage.

#### Implementation & Software Defaults:
- **Function:** `core.synthetic_generator.apply_delay()`
- **Algorithm:** Calculates complex transfer function $H(f) = \exp(-2\pi j f \tau)$ for non-negative frequencies from `np.fft.rfftfreq()`, multiplies $X(f)$ by $H(f)$, and transforms back via `np.fft.irfft()`.
- **Units & Sign:** Delay parameter $\tau$ is in seconds (positive value = time delay / right shift).

---

## 3. Signal Pre-processing & Cleaning

### 3.1. Hampel Filter (Outlier Detection)
Spike outliers (impulse industrial noise) are detected using a rolling Hampel filter. At each sample index $i$, we evaluate:

$$|x_i - m_i| > n_{\sigma} \cdot \text{MAD}_i$$

where $m_i$ is the rolling median in a window of size $2K + 1$:

$$m_i = \text{median}(x_{i-K}, \ldots, x_i, \ldots, x_{i+K})$$

and the Median Absolute Deviation (MAD) is scaled to estimate the standard deviation of a normal distribution:

$$\text{MAD}_i = 1.4826 \cdot \text{median}(|x_{i-K} - m_i|, \ldots, |x_{i+K} - m_i|)$$

#### Assumptions:
- **Sparsity of Outliers:** Outliers are sparse, occupying less than $50\%$ of any given rolling window.
- **Local Gaussianity:** The clean underlying signal plus nominal noise is locally stationary and approximately normally distributed.
- **Normal Consistency:** The scaling factor $1.4826$ correctly maps the median absolute deviation of a normal distribution to its standard deviation ($\sigma \approx 1.4826 \cdot \text{MAD}$).

#### Implementation & Software Defaults:
- **Function:** `core.signal_processing.hampel_filter()`
- **Defaults:** Window size parameter `window_size` defaults to $15$ ($K = 15$, yielding a full window width $2K+1 = 31$ samples), threshold `n_sigmas` defaults to $3.0$.
- **Boundary Handling:** Edge samples are padded using reflecting boundary conditions to preserve array shape without boundary artifacts.

---

### 3.2. Savitzky-Golay Filter (Smoothing)
Smoothing of the cleaned scintillation signal is performed using a Savitzky-Golay polynomial filter:

$$y_i = \sum_{m=-M}^{M} C_m \cdot x_{i+m}$$

where $C_m$ are polynomial least-squares fitting coefficients.

#### Assumptions:
- **Local Polynomial Trend:** The underlying true signal is locally smooth and can be well approximated by a polynomial of degree $d$ over the window of length $2M + 1$.
- **High-Frequency Noise:** The noise component is high-frequency, uncorrelated, and zero-mean.

#### Implementation & Software Defaults:
- **Function:** `core.signal_processing.clean_and_smooth_signal()` (calls `scipy.signal.savgol_filter`)
- **Defaults:** Polynomial degree $d = 2$. Window length parameter uses the same window width as the Hampel stage ($2M+1 = 31$ samples by default).

---

## 4. Spectral & Correlation Analysis

### 4.1. Thomson Multitaper PSD
Thomson's multitaper method reduces spectral leakage and variance by averaging independent eigenspectra calculated using orthogonal Slepian window sequences (Discrete Prolate Spheroidal Sequences, DPSS):

$$\hat{S}(f) = \frac{1}{K} \sum_{k=0}^{K-1} \lambda_k \cdot \left| \sum_{n=0}^{N-1} x_n \cdot h_{n,k} \cdot e^{-j 2\pi f n / fs} \right|^2$$

#### Assumptions:
- **Stationarity:** The discrete signal $x_n$ is a zero-mean stationary random process.
- **Spectral Leakage Suppression:** The DPSS tapers $h_{n,k}$ are optimized to maximize energy concentration within a narrow frequency band $[-W, W]$.
- **Statistical Independence:** Each of the $K$ tapers yields an independent spectral estimate, allowing variance reduction proportional to $1/K$.

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_multitaper_psd()`
- **Library & Algorithm:** Uses `scipy.signal.windows.dpss` for orthogonal DPSS taper sequences $h_{n,k}$ and `numpy.fft.rfft` for 1D real-to-complex Discrete Fourier Transforms.
- **Defaults:** Time-halfbandwidth product $NW = 4.0$, number of tapers $K = 7$ (derived from $K \le 2NW - 1$).
- **Pre-processing:** Detrending (`scipy.signal.detrend`) is applied automatically prior to taper multiplication.

---

### 4.2. Thomson F-Test for Harmonic Lines
To detect periodic oscillations (e.g., Atmospheric Gravity Waves, Traveling Ionospheric Disturbances) against a continuous background, we test if the amplitude $\mu(f)$ of a sinusoid is statistically significant. The F-statistic is computed as:

$$F(f) = \frac{(K-1) \cdot \left| \mu(f) \right|^2 \sum_{k=0}^{K-1} |H_k(0)|^2}{\sum_{k=0}^{K-1} \left| Y_k(f) - \mu(f) H_k(0) \right|^2}$$

where:

$$Y_k(f) = \sum_{n=0}^{N-1} x_n \cdot h_{n,k} \cdot e^{-j 2\pi f n / fs}$$

$$H_k(0) = \sum_{n=0}^{N-1} h_{n,k}$$

$$\mu(f) = \frac{\sum_{k=0}^{K-1} H_k(0) \cdot Y_k(f)}{\sum_{k=0}^{K-1} |H_k(0)|^2}$$

#### Assumptions:
- **Harmonic Line Model:** The signal at frequency $f$ consists of a deterministic complex line of amplitude $\mu(f)$ plus locally white Gaussian background noise.
- **F-Distribution:** Under the null hypothesis (no line present), the statistic $F(f)$ follows an F-distribution with $2$ and $2K-2$ degrees of freedom. A line is detected if $F(f) > F_{\text{crit}}$ at a confidence level $\alpha$ (e.g., $95\%$).

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_thomson_ftest()`
- **Critical Threshold:** Critical value $F_{\text{crit}}$ is evaluated using `scipy.stats.f.ppf(1 - alpha, df1=2, df2=2*K - 2)`.
- **Defaults:** Significance level $\alpha = 0.05$ ($95\%$ statistical confidence threshold), using $K = 7$ tapers ($df_1 = 2$, $df_2 = 12$).

---

### 4.3. Cross-Spectral Phase, Delay, and Velocity Estimation
For dual-frequency observations (e.g., 20 MHz and 25 MHz) separated by a baseline distance $dx$, the phase difference $\theta(f)$ of the cross-spectrum $S_{12}(f)$ yields the propagation delay $\tau(f)$ and horizontal drift velocity $v(f)$:

$$\theta(f) = \text{angle}\left( S_{12}(f) \right)$$

$$\tau = \frac{\theta(f_0)}{360^\circ \cdot f_0}$$

$$v = \text{sign}(\tau) \frac{dx}{|\tau|}$$

#### Assumptions:
- **One-Dimensional Drift:** The irregularities drift horizontally along the baseline axis of the two receiver beams.
- **Dispersionless propagation:** The time delay $\tau$ is frequency-independent in the scintillation band.
- **No Phase Wrapping:** The delay is small enough such that $|\theta(f)| < 180^\circ$ (i.e., $|\tau| < \frac{1}{2f}$). If wrapping occurs, `np.unwrap` resolves phase $2\pi$ branch jumps across adjacent frequencies.

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.estimate_velocities()`
- **Phase Representation:** Phase $\theta(f)$ is computed in degrees via `np.degrees(np.angle(S12))`.
- **Defaults:** Baseline distance $dx$ defaults to $2500.0\text{ m}$ (URAN-4 receiver configuration).

---

### 4.4. Jackknife 95% Confidence Intervals for Multitaper PSD
To quantify variance in Multitaper spectral estimates, non-parametric Jackknife confidence bounds are calculated over the $K$ orthogonal DPSS tapers. The leave-one-out spectral estimate for taper $j$ is:

$$S_{-j}(f) = \frac{1}{K-1} \sum_{k \neq j} S_k(f)$$

The standard error of the log-power spectrum $\log \hat{S}(f)$ is:

$$\text{SE}(\log \hat{S}(f)) = \sqrt{ \frac{K-1}{K} \sum_{j=0}^{K-1} \left( \log S_{-j}(f) - \overline{\log S}(f) \right)^2 }$$

Log-normal $95\%$ confidence interval bounds are then given by:

$$\text{CI}_{95\%}(f) = \left[ \hat{S}(f) \cdot e^{-1.96 \cdot \text{SE}(f)}, \; \hat{S}(f) \cdot e^{+1.96 \cdot \text{SE}(f)} \right]$$

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_multitaper_psd()`
- **Result Container:** Returns `MultitaperPSDResult` holding `psd`, `log_psd_se`, `ci95_low`, and `ci95_high`.

---

### 4.5. Multiple Testing Correction (Benjamini-Hochberg FDR)
When evaluating the Thomson F-statistic across $M$ discrete frequency bins in the analysis passband $[f_{\text{low}}, f_{\text{high}}]$, multiple hypothesis testing increases false positive peak detections. Raw p-values are computed from the $F(2, 2K-2)$ survival function:

$$p_i = 1 - F_{\text{cdf}}(F_i, 2, 2K-2)$$

The Benjamini-Hochberg (BH) procedure sorts the $M$ p-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$ and identifies the maximum index $k_{\max}$ such that:

$$p_{(k)} \le \frac{k}{M} \cdot \alpha_{\text{FDR}}$$

The effective critical F-statistic threshold $F_{\text{FDR}}$ is then computed from $p_{(k_{\max})}$, strictly controlling False Discovery Rate at $\alpha_{\text{FDR}} = 0.05$.

#### Implementation & Software Defaults:
- **Function:** `core.spectral_analysis.compute_ftest()`
- **Defaults:** `fdr_alpha` defaults to $0.05$.

---

### 4.6. Weighted Linear Phase Regression & Coherence Gating
For broadband scintillation irregularities covering multiple contiguous frequency bins, the phase difference $\phi_{\text{rad}}(f)$ exhibits a linear slope with respect to frequency:

$$\phi_{\text{rad}}(f) = a \cdot f + b, \quad \text{where } a = 2\pi \tau$$

Weighted Least Squares (WLS) regression is fitted over frequencies where magnitude-squared coherence $C_{xy}(f) \ge C_{\text{min}}$ ($0.7$), using weights $w_i = C_{xy}(f_i)$:

$$\tau = \frac{a}{2\pi}, \quad v = \text{sign}(\tau) \frac{dx}{|\tau|}$$

Analytical 95% confidence intervals for $\tau$ and $v$ are propagated from the slope standard error $\text{SE}(a)$:

$$\text{SE}(\tau) = \frac{\text{SE}(a)}{2\pi}, \quad \text{SE}(v) = \left|\frac{dx}{\tau^2}\right| \cdot \text{SE}(\tau)$$

If mean coherence across the peak region falls below $C_{\text{min}} = 0.7$, the velocity estimate is marked invalid (`is_valid = False`).

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
6. **Yeh, K. C., & Liu, C. H. (1982).** *Radio wave scintillations in the ionosphere.* Proceedings of the IEEE, 70(4), 324-360.
