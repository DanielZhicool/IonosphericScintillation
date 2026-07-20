# Performance & Scaling Benchmarks

This document details the execution performance, hardware environment, profiling methodology, and runtime scaling characteristics of the digital signal processing (DSP), spectral analysis, and wavelet transform components in the URAN-4 Ionospheric Scintillation Analyzer.

---

## 1. Hardware & Execution Environment

All benchmark metrics were measured on the following reference host hardware:

| Environment Metric | System Specification |
| :--- | :--- |
| **CPU** | AMD Ryzen 9 5900HX (8 Cores / 16 Threads, Base 3.30 GHz) |
| **System Memory** | 16 GB DDR4 RAM |
| **Operating System** | Microsoft Windows 11 Home (64-bit) |
| **Python Runtime** | Python 3.11.8 |
| **Core Libraries** | NumPy 1.26.4, SciPy 1.13.0, ssqueezepy 0.6.5 |
| **Benchmark Tooling** | `pytest-benchmark` 5.2.3 |

---

## 2. Sample Size Rationale & Methodology

- **Standard Session Size ($N = 2,000$):** Matches a full 33-minute URAN-4 observational session sampled at $fs = 1.0\text{ Hz}$ ($2,000$ seconds).
- **Wavelet Sample Size ($N = 500$):** Continuous Wavelet Transforms (CWT) and Synchrosqueezing (SST) compute continuous multi-scale Generalized Morse Wavelet (GMW) convolutions across dozens of logarithmic frequency voices ($N_{\text{voices}} \approx 32-64$). Operating on $500$-sample sliding sub-windows keeps interactive spectrogram rendering smooth (< 170 ms) while preventing memory bottlenecks.
- **Statistical Repetitions:** Benchmark statistics (Min, Max, Mean, StdDev) are computed across 100 iterations per component using `pytest-benchmark` adaptive calibration to ensure stable estimates.

---

## 3. Core Component Benchmarks ($N = 2,000$)

Empirical performance measured across standard URAN-4 session epochs:

| Benchmark Task | Min Time | Max Time | Mean Time | StdDev | Throughput | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PCHIP 3x Upsampling** | $304.1\ \mu\text{s}$ | $685.5\ \mu\text{s}$ | $333.9\ \mu\text{s}$ | $36.8\ \mu\text{s}$ | $2,995\text{ ops/s}$ | Monotonic cubic hermite interpolation ($2,000 \to 6,000$ pts) |
| **Hampel & Savitzky-Golay** | $2.23\ \text{ms}$ | $2.66\ \text{ms}$ | $2.32\ \text{ms}$ | $94.9\ \mu\text{s}$ | $430.6\text{ ops/s}$ | $K=15$ outlier detection + degree-2 polynomial smoothing |
| **Full Spectral Pipeline** | $53.57\ \text{ms}$ | $61.30\ \text{ms}$ | $55.33\ \text{ms}$ | $2.11\ \text{ms}$ | $18.1\text{ ops/s}$ | 4-channel Multitaper PSD ($K=7$), F-Test, Cross-Spectrum, & IDVE |
| **CWT Spectrogram ($N=500$)** | $136.57\ \text{ms}$ | $163.40\ \text{ms}$ | $145.55\ \text{ms}$ | $10.53\ \text{ms}$ | $6.9\text{ ops/s}$ | Continuous Generalized Morse Wavelet transform |
| **SST Spectrogram ($N=500$)** | $161.80\ \text{ms}$ | $194.93\ \text{ms}$ | $175.87\ \text{ms}$ | $13.49\ \text{ms}$ | $5.7\text{ ops/s}$ | Synchrosqueezed Generalized Morse Wavelet transform |

---

## 4. Runtime Scaling vs. Signal Length

To evaluate algorithmic efficiency as observation windows scale from short epochs ($500$ samples) to ultra-long continuous recordings ($50,000$ samples), execution times were profiled across varying signal lengths $N$ after a initial JIT/cache warm-up phase:

![Benchmark Execution Time Scaling](assets/benchmark_scaling.png)

### Scaling Summary Table

| Signal Length $N$ | Hampel + SavGol Filter | PCHIP 3x Upsampling | Multitaper Spectral Pipeline | CWT Spectrogram | SST Spectrogram |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$500$** | $1.63\ \text{ms}$ | $0.27\ \text{ms}$ | $26.88\ \text{ms}$ | $144.29\ \text{ms}$ | $163.64\ \text{ms}$ |
| **$1,000$** | $1.87\ \text{ms}$ | $0.31\ \text{ms}$ | $37.03\ \text{ms}$ | $151.13\ \text{ms}$ | $171.05\ \text{ms}$ |
| **$2,000$** | $2.50\ \text{ms}$ | $0.38\ \text{ms}$ | $58.06\ \text{ms}$ | $157.32\ \text{ms}$ | $193.02\ \text{ms}$ |
| **$5,000$** | $4.28\ \text{ms}$ | $0.58\ \text{ms}$ | $115.93\ \text{ms}$ | $180.63\ \text{ms}$ | $246.59\ \text{ms}$ |
| **$10,000$** | $7.15\ \text{ms}$ | $0.91\ \text{ms}$ | $217.35\ \text{ms}$ | $194.96\ \text{ms}$ | $335.09\ \text{ms}$ |
| **$50,000$** | $29.44\ \text{ms}$ | $4.96\ \text{ms}$ | $1,091.55\ \text{ms}$ | $1,190.56\ \text{ms}$ | $2,499.46\ \text{ms}$ |

### Algorithmic Complexity Insights
- **Filtering ($O(N)$):** The rolling Hampel median filter and Savitzky-Golay convolution exhibit strict linear scaling, processing $50,000$ samples in under $30\text{ ms}$.
- **PCHIP Interpolation ($O(N)$):** Monotonic piecewise cubic spline evaluation scales linearly, taking less than $5\text{ ms}$ for $50,000$ points.
- **Multitaper Spectral Pipeline ($O(K \cdot N \log N)$):** Dominated by the 1D Fast Fourier Transform (`numpy.fft.rfft`) applied across $K=7$ tapered channels. Scales smoothly, processing $50,000$ samples in $\sim 1.09\text{ s}$.
- **Wavelet Spectrograms ($O(N \cdot M_{\text{scales}} \log N)$):** Multi-scale Generalized Morse Wavelet (GMW) convolutions scale predictably once PyWavelets/ssqueezepy JIT caches are warmed, keeping sub-window ($N \le 2,000$) response times under $200\text{ ms}$.

---

## 5. Reproducing Benchmarks

To run the automated benchmark suite on your local system:

```powershell
uv run pytest --benchmark-only
```

To regenerate the runtime scaling benchmark chart:

```powershell
uv run python scripts/benchmark_scaling.py
```
