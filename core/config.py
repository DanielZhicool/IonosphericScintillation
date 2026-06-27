"""
Core configuration constants. Centralize tunable parameters here for easier
experimentation, testing and documentation.
"""

# Session parsing
GAP_THRESHOLD = 3600  # seconds
SIDEREAL_DAY = 86164  # seconds (23h 56m 4s)

# Signal cleaning
DEFAULT_WINDOW_SIZE = 15
DEFAULT_N_SIGMAS = 3.0
SAVGOL_POLYORDER = 2
TUKEY_ALPHA = 0.1

# Frequency-Time Analysis (CWT)
PCHIP_FACTOR = 3        # Upsampling multiplier
CWT_NV = 128            # Voices per octave for wavelet transform
MORSE_GAMMA = 3         # Wavelet symmetry
MORSE_BETA = 60         # Wavelet time-bandwidth
GAUSSIAN_SIGMA_FREQ = 2.5
GAUSSIAN_SIGMA_TIME = 1.0
CWT_DYNAMIC_RANGE_DB = 40.0
