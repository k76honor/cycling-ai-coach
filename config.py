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
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Strava API Configuration
STRAVA_CLIENT_ID = get_setting("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = get_setting("STRAVA_CLIENT_SECRET", "")
STRAVA_REFRESH_TOKEN = get_setting("STRAVA_REFRESH_TOKEN", "")

# Default Athlete Profile
DEFAULT_FTP = int(get_setting("DEFAULT_FTP", "230"))  # Watts
DEFAULT_WEIGHT_KG = float(get_setting("DEFAULT_WEIGHT_KG", "68.0"))  # kg
DEFAULT_MAX_HR = int(get_setting("DEFAULT_MAX_HR", "190"))  # bpm
DEFAULT_REST_HR = int(get_setting("DEFAULT_REST_HR", "50"))  # bpm
