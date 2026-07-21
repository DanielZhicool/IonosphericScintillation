"""
Core configuration constants and ProcessingConfig container.
Centralize tunable parameters here for easier experimentation, testing, and documentation.
"""

from dataclasses import dataclass

# Session parsing
GAP_THRESHOLD = 3600  # seconds
SESSION_MERGE_GAP = (
    8000  # seconds – max gap between consecutive transits of the same source to merge them into one session
)
SIDEREAL_DAY = 86164  # seconds (23h 56m 4s)

# Signal cleaning
DEFAULT_WINDOW_SIZE = 15
DEFAULT_N_SIGMAS = 3.0
SAVGOL_POLYORDER = 2
TUKEY_ALPHA = 0.1

# Frequency-Time Analysis (CWT)
PCHIP_FACTOR = 3  # Upsampling multiplier
PCHIP_LONG_SIGNAL_THRESHOLD = 30000  # Skip PCHIP upsampling above this sample count to save memory/time
CWT_NV_BUBBLES = 32  # Voices per octave for bubbles (high frequency)
CWT_NV_CLOUDS = 64  # Voices per octave for clouds (low frequency)
MORSE_GAMMA = 3  # Wavelet symmetry
MORSE_BETA = 30  # Wavelet time-bandwidth
GAUSSIAN_SIGMA_FREQ = 1.0
GAUSSIAN_SIGMA_TIME = 1.0
CWT_DYNAMIC_RANGE_DB = 40.0

# CWT Spectrogram display options (easily toggleable to match reference images)
CWT_SHOW_PERIOD = True  # True to show Period (Sec) on Y-axis, False to show Frequency (mHz)
CWT_SHOW_LINEAR_AMP = True  # True to show linear wavelet amplitude on colorbar, False to show Power (dB)

# Spectral-Correlation Analysis (MTM PSD, F-Test, Cross-Spectrum, IDVE)
MTM_N_TAPERS = 7  # Number of DPSS tapers (Thomson Multitaper)
MTM_NW = 4.0  # Time-bandwidth product for DPSS windows
FTEST_CONFIDENCE = 0.95  # F-test significance level
FDR_ALPHA = 0.05  # False Discovery Rate threshold for multiple testing
CROSS_SPECTRUM_DX = 2500  # Beam separation in meters (model parameter)
VELOCITY_N_PEAKS = 3  # Number of cross-spectral peaks to report
COHERENCE_THRESHOLD = 0.7  # Coherence significance threshold
PHASE_REGRESSION_BANDWIDTH_HZ = 0.02  # Half-bandwidth around peak for weighted phase regression
COMPUTE_JACKKNIFE_CI = True  # Enable Jackknife 95% confidence intervals for Multitaper PSD


@dataclass(frozen=True)
class ProcessingConfig:
    """
    Immutable configuration container for ionospheric scintillation DSP pipelines.
    """

    sampling_rate: float = 1.0
    tukey_alpha: float = TUKEY_ALPHA
    pchip_factor: int = PCHIP_FACTOR
    pchip_long_signal_threshold: int = PCHIP_LONG_SIGNAL_THRESHOLD
    window_size: int = DEFAULT_WINDOW_SIZE
    n_sigmas: float = DEFAULT_N_SIGMAS
    savgol_polyorder: int = SAVGOL_POLYORDER
    mtm_nw: float = MTM_NW
    mtm_n_tapers: int = MTM_N_TAPERS
    ftest_confidence: float = FTEST_CONFIDENCE
    fdr_alpha: float = FDR_ALPHA
    coherence_threshold: float = COHERENCE_THRESHOLD
    cross_spectrum_dx: float = CROSS_SPECTRUM_DX
    velocity_n_peaks: int = VELOCITY_N_PEAKS
    phase_regression_bandwidth_hz: float = PHASE_REGRESSION_BANDWIDTH_HZ
    enable_phase_regression: bool = False
    compute_jackknife_ci: bool = COMPUTE_JACKKNIFE_CI
    cwt_nv_bubbles: int = CWT_NV_BUBBLES
    cwt_nv_clouds: int = CWT_NV_CLOUDS
    morse_gamma: float = MORSE_GAMMA
    morse_beta: float = MORSE_BETA
