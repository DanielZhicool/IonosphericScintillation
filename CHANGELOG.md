# Changelog

All notable changes to the **IonosphericScintillation** project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-27

### Added
- **Direct Local Maxima Cross-Spectral Peak Selection**: Updated cross-spectral peak identification to select distinct local maxima on un-smoothed cross-spectrum arrays, preventing peak merger distortion.
- **Reference-Aligned IDVE Table Formatting**: Updated IDVE table and batch text export outputs to display velocity, time delay, phase angle, and magnitude-squared coherence for all detected harmonic peaks, appending explicit `(low coh)` and `>10,000 m/s (in-phase)` tags.
- **Interactive Educational Notebook Series**: Added 8 complete interactive tutorial notebooks (`examples/01` through `08`) covering signal generation, pre-processing, wavelet spectrograms, multitaper PSD, cross-spectrum velocity estimation, and synthetic/real URAN-4 observation workflows.
- **Enhanced Documentation & Theory Alignment**: Synchronized 99% F-test confidence threshold ($F=6.93$) and added direct notebook tutorial links throughout `docs/theory.md`.

### Changed
- Promoted test suite to 86 unit and benchmark tests covering red-noise seed reproducibility (`seed=42`), peak selection separation, and IDVE table formatting.

---

## [0.1.0] - 2026-07-22

### Added
- **Multitaper Spectral Estimation**: Added DPSS taper sequence generation (`scipy.signal.windows.dpss`), adaptive weighting PSD calculation, and Thomson F-test harmonic detection with Benjamini-Hochberg FDR / Bonferroni multiple testing corrections.
- **Cross-Spectral Velocity & Phase Regression**: Added weighted linear phase slope regression ($\phi(f) = \phi_0 - 2\pi f \tau$), coherence threshold gating ($C_{xy} \ge 0.7$), and analytical WLS 95% confidence intervals.
- **Signal Preprocessing Suite**: Rolling Hampel median outlier rejection, Savitzky-Golay polynomial smoothing, and endpoint-preserved PCHIP monotonic cubic spline upsampling ($n_{\text{new}} = (N-1)\text{factor} + 1$).
- **Time-Frequency Spectrograms**: Continuous Wavelet Transform (CWT) and Synchrosqueezed Wavelet Transform (SST) using Generalized Morse Wavelets (GMW) with memory-bounded sub-window chunking.
- **PySide6 Desktop Application**: Interactive visualization suite with multi-channel time series plotting, spectral analysis tables, coherence gating indicators, and batch export controls.
- **Audit Provenance Manifest Engine**: `core/provenance.py` module generating SHA-256 file checksums, app version, git commit, full `ProcessingConfig` snapshots, and environment library metadata in JSON.
- **Benchmarking Suite**: Comprehensive runtime and peak RAM scaling benchmark suite (`scripts/benchmark_scaling.py`, `tests/test_benchmarks.py`).
- **Modular Test Suite**: Split monolithic test suite into dedicated modules (`test_multitaper.py`, `test_statistical_precision.py`, `test_preprocessing.py`, `test_pchip.py`, `test_velocity.py`, `test_parsers.py`, `test_validation.py`).

### Changed
- Refactor core dataclasses into `core/types.py` (`VelocityEstimate`, `MultitaperPSDResult`, `TimeFrequencyResult`, `ProvenanceManifest`).
- Centralize user parameters into immutable `core/config.py` `ProcessingConfig`.
- Updated comprehensive theoretical documentation (`docs/theory.md`) and benchmark scaling reports (`docs/benchmarks.md`).

### Security & Compliance
- Applied BSD 3-Clause License, `CITATION.cff`, and `CONTRIBUTING.md`.
