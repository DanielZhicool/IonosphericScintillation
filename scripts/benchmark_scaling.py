import os
import sys

sys.path.insert(0, ".")
import time
import tracemalloc

import matplotlib.pyplot as plt

from core.signal_processing import clean_and_smooth_signal, compute_cwt_spectrogram, upsample_pchip
from core.spectral_analysis import run_spectral_pipeline
from core.synthetic_generator import generate_synthetic_scintillation


def run_benchmarks():
    fs = 1.0

    # ---------------------------------------------------------
    # Warmup Phase: Trigger JIT compilation & Numba / SciPy caching
    # ---------------------------------------------------------
    print("Warming up JIT compilers, PyFFTW wisdom, and SciPy cache...")
    warmup_sig1, warmup_sig2 = generate_synthetic_scintillation(length=500, fs=fs, seed=0)
    for _ in range(2):
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
    # Benchmarking Across Signal Lengths N (at fs = 1.0 Hz)
    # ---------------------------------------------------------
    lengths = [500, 1000, 2000, 5000, 10000, 50000, 86400, 184280, 260793, 432000, 506069]

    times_hampel = []
    times_pchip = []
    times_spectral = []
    times_cwt = []
    times_sst = []

    ram_spectral = []
    ram_cwt = []
    ram_sst = []

    print("Running warmed scaling benchmarks...")

    for length in lengths:
        sig1, sig2 = generate_synthetic_scintillation(
            length=length, fs=fs, f_fresnel=0.1, spectral_index=8.0 / 3.0, seed=42
        )

        n_rep = 1 if length >= 86400 else 3

        # 1. Hampel + SavGol
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = clean_and_smooth_signal(sig1, window_size=15, n_sigmas=3.0, apply_smoothing=True)
        t1 = time.perf_counter()
        t_hampel = (t1 - t0) / n_rep * 1000.0  # ms
        times_hampel.append(t_hampel)

        # 2. PCHIP Upsampling
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = upsample_pchip(sig1, fs=fs, factor=3)
        t1 = time.perf_counter()
        t_pchip = (t1 - t0) / n_rep * 1000.0  # ms
        times_pchip.append(t_pchip)

        # 3. Full Spectral Pipeline
        pm_signals = {
            "20 MHz Pol A": sig1,
            "25 MHz Pol A": sig2,
            "20 MHz Pol B": sig1,
            "25 MHz Pol B": sig2,
        }
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(n_rep):
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
        _, mt_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t_spectral = (t1 - t0) / n_rep * 1000.0  # ms
        times_spectral.append(t_spectral)
        ram_spectral.append(mt_peak / (1024 * 1024))

        # 4. CWT Spectrogram
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = compute_cwt_spectrogram(sig1, fs=fs, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=False)
        t1 = time.perf_counter()
        _, cwt_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t_cwt = (t1 - t0) / n_rep * 1000.0
        times_cwt.append(t_cwt)
        ram_cwt.append(cwt_peak / (1024 * 1024))

        # 5. SST Spectrogram
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = compute_cwt_spectrogram(sig1, fs=fs, lowcut=1.0 / 150.0, highcut=0.2, use_ssq=True)
        t1 = time.perf_counter()
        _, sst_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t_sst = (t1 - t0) / n_rep * 1000.0
        times_sst.append(t_sst)
        ram_sst.append(sst_peak / (1024 * 1024))

        print(
            f"N={length:7d} | Hampel+SavGol: {t_hampel:7.2f} ms | PCHIP: {t_pchip:7.2f} ms | "
            f"Spectral: {t_spectral:8.2f} ms (RAM: {ram_spectral[-1]:6.2f} MB) | "
            f"CWT: {t_cwt:8.2f} ms (RAM: {ram_cwt[-1]:6.2f} MB) | SST: {t_sst:8.2f} ms (RAM: {ram_sst[-1]:6.2f} MB)"
        )

    output_dir = "docs/assets"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot Runtime Scaling curve
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(lengths, times_hampel, "o-", label="Hampel & Savitzky-Golay", linewidth=2)
    ax1.plot(lengths, times_pchip, "s-", label="PCHIP 3x Upsampling", linewidth=2)
    ax1.plot(lengths, times_spectral, "^--", label="Full Multitaper Spectral Pipeline", linewidth=2)
    ax1.set_xlabel("Signal Length N (samples, fs = 1 Hz)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.set_title("DSP Operations Execution Time vs Signal Length", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(lengths, times_cwt, "o-", color="purple", label="CWT Spectrogram", linewidth=2)
    ax2.plot(lengths, times_sst, "d--", color="crimson", label="SST Spectrogram", linewidth=2)
    ax2.set_xlabel("Signal Length N (samples, fs = 1 Hz)", fontsize=11)
    ax2.set_ylabel("Execution Time (ms)", fontsize=11)
    ax2.set_title("Wavelet (CWT / SST) Execution Time vs Signal Length", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "benchmark_scaling.png")
    plt.savefig(plot_path1, dpi=300)
    print(f"\nRuntime scaling plot saved to {plot_path1}")

    # 2. Plot Peak RAM Scaling curve
    fig2, (mb_ax1, mb_ax2) = plt.subplots(1, 2, figsize=(13, 5))

    mb_ax1.plot(lengths, ram_spectral, "^--", color="teal", label="Multitaper Spectral Pipeline", linewidth=2)
    mb_ax1.set_xlabel("Signal Length N (samples, fs = 1 Hz)", fontsize=11)
    mb_ax1.set_ylabel("Peak RAM Allocation (MB)", fontsize=11)
    mb_ax1.set_title("Spectral Pipeline Peak RAM vs Signal Length", fontsize=12, fontweight="bold")
    mb_ax1.grid(True, linestyle="--", alpha=0.6)
    mb_ax1.legend()

    mb_ax2.plot(lengths, ram_cwt, "o-", color="purple", label="CWT Spectrogram", linewidth=2)
    mb_ax2.plot(lengths, ram_sst, "d--", color="crimson", label="SST Spectrogram", linewidth=2)
    mb_ax2.set_xlabel("Signal Length N (samples, fs = 1 Hz)", fontsize=11)
    mb_ax2.set_ylabel("Peak RAM Allocation (MB)", fontsize=11)
    mb_ax2.set_title("Wavelet (CWT / SST) Peak RAM vs Signal Length", fontsize=12, fontweight="bold")
    mb_ax2.grid(True, linestyle="--", alpha=0.6)
    mb_ax2.legend()

    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "benchmark_memory_scaling.png")
    plt.savefig(plot_path2, dpi=300)
    print(f"Peak RAM scaling plot saved to {plot_path2}")


if __name__ == "__main__":
    run_benchmarks()
