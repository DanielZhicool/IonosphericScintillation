"""
Core configuration constants. Centralize tunable parameters here for easier
experimentation, testing and documentation.
"""

# Session parsing
GAP_THRESHOLD = 3600  # seconds

# Signal processing
DEFAULT_WINDOW_SIZE = 15
DEFAULT_N_SIGMAS = 3.0
SAVGOL_POLYORDER = 2

# Decimation and FSST
DECIMATION_Q = 10
LOW_FREQ_HIGHCUT = 0.01
TUKEY_ALPHA = 0.1
NFFT_CAP = 32768
