import sys

sys.path.insert(0, ".")
import os
import time

import matplotlib.pyplot as plt

from core.signal_processing import clean_and_smooth_signal, compute_cwt_spectrogram, upsample_pchip
from core.spectral_analysis import run_spectral_pipeline
from core.synthetic_generator import generate_synthetic_scintillation


def run_benchmarks():
    fs = 1.0

    # ---------------------------------------------------------
    # Warmup Phase: Trigger JIT compilation & Numba / SciPy caching
    # ---------------------------------------------------------
    print("Warming up JIT compilers and cache...")
    warmup_sig1, warmup_sig2 = generate_synthetic_scintillation(length=500, fs=fs, seed=0)
    _ = clean_and_smooth_signal(warmup_sig1, window_size=15, n_sigmas=3.0, apply_smoothing=True)
    _ = upsample_pchip(warmup_sig1, fs=fs, factor=3)
    _ = run_spectral_pipeline(
        pm_signals={
            "20 MHz Pol A": warmup_sig1,
            "25 MHz Pol A": warmup_sig2,
            "20 MHz Pol B": warmup_sig1,
            "25 MHz Pol B": warmup_sig2,
        },
        fs=fs,
        lowcut=0.01,
        highcut=0.1,
        window_size=15,
        n_sigmas=3.0,
        apply_smoothing=True,
    )
    _ = compute_cwt_spectrogram(warmup_sig1, fs=fs, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=False)
    _ = compute_cwt_spectrogram(warmup_sig1, fs=fs, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=True)
    print("Warmup complete.\n")

    # ---------------------------------------------------------
    # Benchmarking Across Signal Lengths N
    # ---------------------------------------------------------
    lengths = [500, 1000, 2000, 5000, 10000, 50000]

    times_hampel = []
    times_pchip = []
    times_spectral = []
    times_cwt = []
    times_sst = []

    print("Running warmed scaling benchmarks...")

    for length in lengths:
        sig1, sig2 = generate_synthetic_scintillation(
            length=length, fs=fs, f_fresnel=0.1, spectral_index=8.0 / 3.0, seed=42
        )

        # 1. Hampel + SavGol
        t0 = time.perf_counter()
        for _ in range(5):
            _ = clean_and_smooth_signal(sig1, window_size=15, n_sigmas=3.0, apply_smoothing=True)
        t1 = time.perf_counter()
        t_hampel = (t1 - t0) / 5.0 * 1000.0  # ms
        times_hampel.append(t_hampel)

        # 2. PCHIP Upsampling
        t0 = time.perf_counter()
        for _ in range(5):
            _ = upsample_pchip(sig1, fs=fs, factor=3)
        t1 = time.perf_counter()
        t_pchip = (t1 - t0) / 5.0 * 1000.0  # ms
        times_pchip.append(t_pchip)

        # 3. Full Spectral Pipeline
        pm_signals = {
            "20 MHz Pol A": sig1,
            "25 MHz Pol A": sig2,
            "20 MHz Pol B": sig1,
            "25 MHz Pol B": sig2,
        }
        t0 = time.perf_counter()
        for _ in range(3):
            _ = run_spectral_pipeline(
                pm_signals=pm_signals,
                fs=fs,
                lowcut=0.01,
                highcut=0.1,
                window_size=15,
                n_sigmas=3.0,
                apply_smoothing=True,
            )
        t1 = time.perf_counter()
        t_spectral = (t1 - t0) / 3.0 * 1000.0  # ms
        times_spectral.append(t_spectral)

        # 4. CWT Spectrogram
        t0 = time.perf_counter()
        for _ in range(3):
            _ = compute_cwt_spectrogram(sig1, fs=fs, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=False)
        t1 = time.perf_counter()
        t_cwt = (t1 - t0) / 3.0 * 1000.0
        times_cwt.append(t_cwt)

        # 5. SST Spectrogram
        t0 = time.perf_counter()
        for _ in range(3):
            _ = compute_cwt_spectrogram(sig1, fs=fs, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=True)
        t1 = time.perf_counter()
        t_sst = (t1 - t0) / 3.0 * 1000.0
        times_sst.append(t_sst)

        print(
            f"N={length:5d} | Hampel+SavGol: {t_hampel:6.2f} ms | PCHIP: {t_pchip:6.2f} ms | "
            f"Spectral: {t_spectral:6.2f} ms | CWT: {t_cwt:6.2f} ms | SST: {t_sst:6.2f} ms"
        )

    # Plot scaling curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(lengths, times_hampel, "o-", label="Hampel & Savitzky-Golay", linewidth=2)
    ax1.plot(lengths, times_pchip, "s-", label="PCHIP 3x Upsampling", linewidth=2)
    ax1.plot(lengths, times_spectral, "^--", label="Full Multitaper Spectral Pipeline", linewidth=2)
    ax1.set_xlabel("Signal Length N (samples)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.set_title("DSP Operations Execution Time vs Signal Length", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(lengths, times_cwt, "o-", color="purple", label="CWT Spectrogram", linewidth=2)
    ax2.plot(lengths, times_sst, "d--", color="crimson", label="SST Spectrogram", linewidth=2)
    ax2.set_xlabel("Signal Length N (samples)", fontsize=11)
    ax2.set_ylabel("Execution Time (ms)", fontsize=11)
    ax2.set_title("Wavelet (CWT / SST) Execution Time vs Signal Length", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_dir = "docs/assets"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "benchmark_scaling.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nScaling plot saved to {plot_path}")


if __name__ == "__main__":
    run_benchmarks()
