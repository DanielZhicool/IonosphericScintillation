BTN_LOAD_PM6 = "1. Load PM6 data"
LBL_PM6_NOT_LOADED = "PM6 file not loaded"
BTN_LOAD_LOGS = "2. Load regi log and split sessions"
CHECK_MARKERS = "Show source markers on main plot"
CHECK_SMOOTH = "Enable smoothing"
BTN_APPLY_NOISE = "3. Manually clean region"
COMBO_BAND_ITEMS = ["Small bubbles (5 - 150 s)", "Large clouds (150 - 600 s)"]
BTN_ANALYZE = "4. Refresh spectrogram"
BTN_EXPORT = "5. Export plots"
LABEL_DISPLAY_CHANNEL = "Displayed channel:"

LABEL_BAND = "Bandpass range:"
MAIN_TAB_NAME = "Full overview"
RAW_DATA_TITLE_TEMPLATE = "[{tab}] Raw data ({channel})"
IONOSPHERIC_TITLE = "Ionospheric scintillations"
SPECTROGRAM_TITLE = "CWT Spectrogram (Morse Wavelet)"
APP_TITLE = "URAN-4 Ionospheric Scintillation Analyzer"
FILE_DIALOG_PM6_TITLE = "Open PM6"
FILE_DIALOG_REGI_TITLE = "Open regi log"
FILTER_PM6 = "PM6 Files (*.PM6);;All Files (*)"
FILTER_TEXT_FILES = "Text Files (*.txt);;All Files (*)"
MSG_NO_LOG_EVENTS_TITLE = "Empty log file"
MSG_NO_LOG_EVENTS_TEXT = "No events could be found in the regi log."
MSG_ANALYSIS_DATA_TOO_SHORT_TITLE = "Insufficient data"
MSG_ANALYSIS_DATA_TOO_SHORT_TEXT = (
    "The signal length is too short for the Large clouds band. Use Small bubbles for shorter observations."
)
MSG_LOG_PROCESSING_RESULT_TITLE = "Result"
MSG_LOG_PROCESSING_ERROR_TITLE = "Processing error"
MSG_OPEN_ERROR_TITLE = "Open error"
MSG_ANALYSIS_ERROR_TITLE = "Analysis error"
MSG_SAVE_ERROR_TITLE = "Save error"

LBL_LOADED_SAMPLES = "Loaded samples: {}"
MSG_CREATED_TABS_TEMPLATE = "Created tabs: {}"
MSG_SKIPPED_TABS_TEMPLATE = "Skipped (no PM6 data):\n{}"
MSG_TAB_CREATION_NONE = "No session tabs were created."
MSG_NO_PM6_SELECTED_TITLE = "No PM6 data"
MSG_NO_PM6_SELECTED_TEXT = "Please load PM6 data before cleaning or splitting sessions."

# Spectral Analysis Tab
BTN_SPECTRAL = "6. Spectral Analysis"
BTN_GLOBAL_SPECTRAL = "7. Global Spectral Analysis"
SPECTRAL_TAB_SUFFIX = " \u27e8Spectral\u27e9"
GLOBAL_SPECTRAL_TAB_NAME = "\u27e8Global Spectral\u27e9"
SPECTRAL_BAND_SMALL = "Small bubbles (5\u2013150 s)"
SPECTRAL_BAND_LARGE = "Large clouds (150\u2013600 s)"
SPECTRAL_TAB_PSD_TITLE = "Multitaper PSD"
SPECTRAL_TAB_FTEST_TITLE = "Thomson F-Test"

SPECTRAL_TAB_VELOCITY_HEADER = "Ionospheric Drift Velocity Estimation"
SPECTRAL_INSUFFICIENT_DATA = "Signal too short for this frequency band."
MSG_SPECTRAL_NO_SOURCE_TITLE = "No source selected"
MSG_SPECTRAL_NO_SOURCE_TEXT = "Navigate to a source tab (not Full overview) to run spectral analysis."
MSG_SPECTRAL_ERROR_TITLE = "Spectral Analysis Error"


CHANNELS = [
    "P1_20A",
    "M1_20A",
    "P2_20B",
    "M2_20B",
    "P3_25A",
    "M3_25A",
    "P4_25B",
    "M4_25B",
    "20 MHz Pol A (P-M)",
    "20 MHz Pol B (P-M)",
    "25 MHz Pol A (P-M)",
    "25 MHz Pol B (P-M)",
]
