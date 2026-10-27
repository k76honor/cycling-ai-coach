import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def get_setting(key: str, default: str = "") -> str:
    """Read configuration from Streamlit Secrets (Cloud) or environment variable (Local .env)."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

# Gemini API Configuration
GEMINI_API_KEY = get_setting("GEMINI_API_KEY", "")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-2.5-flash")

# Strava API Configuration
STRAVA_CLIENT_ID = get_setting("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = get_setting("STRAVA_CLIENT_SECRET", "")
STRAVA_REFRESH_TOKEN = get_setting("STRAVA_REFRESH_TOKEN", "")

# Default Athlete Profile (Garmin Edge 1040 Verified Profile)
DEFAULT_FTP = int(get_setting("DEFAULT_FTP", "196"))          # Watts (3.04 W/kg @ 64.5kg)
DEFAULT_WEIGHT_KG = float(get_setting("DEFAULT_WEIGHT_KG", "64.5"))  # kg
DEFAULT_MAX_HR = int(get_setting("DEFAULT_MAX_HR", "184"))    # bpm (Verified 183bpm peak on 09/05)
DEFAULT_LTHR = int(get_setting("DEFAULT_LTHR", "161"))        # bpm (Lactate Threshold HR)
DEFAULT_REST_HR = int(get_setting("DEFAULT_REST_HR", "49"))   # bpm (Resting HR)
