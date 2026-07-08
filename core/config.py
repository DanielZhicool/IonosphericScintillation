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
CWT_NV = 256            # Voices per octave for wavelet transform
MORSE_GAMMA = 3         # Wavelet symmetry
MORSE_BETA = 30         # Wavelet time-bandwidth
GAUSSIAN_SIGMA_FREQ = 1.0
GAUSSIAN_SIGMA_TIME = 1.0
CWT_DYNAMIC_RANGE_DB = 40.0

# Spectral-Correlation Analysis (Steps 10-13)
MTM_N_TAPERS = 7          # Number of DPSS tapers (Thomson Multitaper)
MTM_NW = 4.0              # Time-bandwidth product for DPSS windows
FTEST_CONFIDENCE = 0.95   # F-test significance level
CROSS_SPECTRUM_DX = 2500  # Beam separation in meters (model parameter)
VELOCITY_N_PEAKS = 3      # Number of cross-spectral peaks to report
