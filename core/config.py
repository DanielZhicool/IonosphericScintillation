"""
Core configuration constants. Centralize tunable parameters here for easier
experimentation, testing and documentation.
"""

# Session parsing
GAP_THRESHOLD = 3600  # seconds
SESSION_MERGE_GAP = 8000  # seconds – max gap between consecutive transits of the same source to merge them into one session
SIDEREAL_DAY = 86164  # seconds (23h 56m 4s)

# Signal cleaning
DEFAULT_WINDOW_SIZE = 15
DEFAULT_N_SIGMAS = 3.0
SAVGOL_POLYORDER = 2
TUKEY_ALPHA = 0.1

# Frequency-Time Analysis (CWT)
PCHIP_FACTOR = 3               # Upsampling multiplier
PCHIP_LONG_SIGNAL_THRESHOLD = 30000  # Skip PCHIP upsampling above this sample count to save memory/time
CWT_NV_BUBBLES = 64     # Voices per octave for bubbles (high frequency)
CWT_NV_CLOUDS = 128     # Voices per octave for clouds (low frequency)
MORSE_GAMMA = 3         # Wavelet symmetry
MORSE_BETA = 30         # Wavelet time-bandwidth
GAUSSIAN_SIGMA_FREQ = 1.0
GAUSSIAN_SIGMA_TIME = 1.0
CWT_DYNAMIC_RANGE_DB = 40.0

# CWT Spectrogram display options (easily toggleable to match reference images)
CWT_SHOW_PERIOD = True        # True to show Period (Sec) on Y-axis, False to show Frequency (mHz)
CWT_SHOW_LINEAR_AMP = True    # True to show linear wavelet amplitude on colorbar, False to show Power (dB)

# Spectral-Correlation Analysis (MTM PSD, F-Test, Cross-Spectrum, IDVE)
MTM_N_TAPERS = 7          # Number of DPSS tapers (Thomson Multitaper)
MTM_NW = 4.0              # Time-bandwidth product for DPSS windows
FTEST_CONFIDENCE = 0.95   # F-test significance level
CROSS_SPECTRUM_DX = 2500  # Beam separation in meters (model parameter)
VELOCITY_N_PEAKS = 3      # Number of cross-spectral peaks to report
