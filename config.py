import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Strava API Configuration
STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET", "")
STRAVA_REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN", "")

# Default Athlete Profile
DEFAULT_FTP = int(os.getenv("DEFAULT_FTP", "230"))  # Watts
DEFAULT_WEIGHT_KG = float(os.getenv("DEFAULT_WEIGHT_KG", "68.0"))  # kg
DEFAULT_MAX_HR = int(os.getenv("DEFAULT_MAX_HR", "190"))  # bpm
DEFAULT_REST_HR = int(os.getenv("DEFAULT_REST_HR", "50"))  # bpm
