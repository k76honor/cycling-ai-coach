"""
Strava API v3 Client for Cycling Training Data
Handles OAuth2 token refreshing, activity fetching, stream extraction, and fallback to sample data.
"""

from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from datetime import datetime, timezone
from sample_data import generate_sample_cycling_history, generate_sample_streams
from power_metrics import calculate_normalized_power, calculate_intensity_factor, calculate_tss


class StravaClient:
    OAUTH_TOKEN_URL = "https://www.strava.com/oauth/token"
    API_BASE_URL = "https://www.strava.com/api/v3"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token: Optional[str] = None
        self.token_expires_at: int = 0

    def is_configured(self) -> bool:
        """Returns True if Strava API credentials are provided."""
        return bool(self.client_id and self.client_secret and self.refresh_token)

    def refresh_access_token(self) -> Optional[str]:
        """
        Exchanges the refresh_token for a fresh short-lived access_token.
        """
        if not self.is_configured():
            return None

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }

        try:
            resp = requests.post(self.OAUTH_TOKEN_URL, data=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self.access_token = data.get("access_token")
            self.token_expires_at = data.get("expires_at", 0)
            return self.access_token
        except Exception as e:
            print(f"Error refreshing Strava access token: {e}")
            return None

    def get_headers(self) -> Dict[str, str]:
        if not self.access_token:
            self.refresh_access_token()
        return {"Authorization": f"Bearer {self.access_token}"}

    def get_activities(
        self,
        ftp: float = 230.0,
        per_page: int = 50,
        use_sample_if_unconfigured: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetches recent cycling activities from Strava (or returns sample data if unconfigured).
        Filters only cycling-related types: Ride, VirtualRide, EBikeRide, Handcycle.
        """
        if not self.is_configured():
            if use_sample_if_unconfigured:
                return generate_sample_cycling_history(ftp=ftp, days=45)
            return []

        if not self.access_token:
            self.refresh_access_token()
            if not self.access_token:
                if use_sample_if_unconfigured:
                    return generate_sample_cycling_history(ftp=ftp, days=45)
                return []

        try:
            url = f"{self.API_BASE_URL}/athlete/activities"
            params = {"per_page": per_page}
            resp = requests.get(url, headers=self.get_headers(), params=params, timeout=12)
            resp.raise_for_status()
            raw_activities = resp.json()

            # Filter cycling activities
            cycling_types = {"Ride", "VirtualRide", "EBikeRide", "GravelRide", "MountainBikeRide"}
            cycling_activities = [
                act for act in raw_activities
                if act.get("type") in cycling_types or act.get("sport_type") in cycling_types
            ]

            # Enhance activities with NP, IF, and TSS if power is present
            for act in cycling_activities:
                duration_sec = act.get("moving_time", 0)
                np_watts = act.get("weighted_average_watts") or act.get("average_watts", 0)
                
                if np_watts and duration_sec and ftp > 0:
                    act["intensity_factor"] = calculate_intensity_factor(np_watts, ftp)
                    act["tss"] = calculate_tss(duration_sec, np_watts, ftp)
                else:
                    act["intensity_factor"] = 0.0
                    act["tss"] = 0.0

            return cycling_activities
        except Exception as e:
            print(f"Error fetching Strava activities: {e}. Falling back to sample data.")
            if use_sample_if_unconfigured:
                return generate_sample_cycling_history(ftp=ftp, days=45)
            return []

    def get_activity_streams(
        self, activity_id: int, ftp: float = 230.0
    ) -> Dict[str, List[float]]:
        """
        Fetches detailed second-by-second telemetry streams (watts, heartrate, cadence, time).
        Falls back to generated streams if mock or not found.
        """
        if not self.is_configured() or activity_id >= 1000000000:
            return generate_sample_streams(duration_sec=3600, target_watts=ftp * 0.75, ftp=ftp)

        try:
            url = f"{self.API_BASE_URL}/activities/{activity_id}/streams"
            params = {
                "keys": "watts,heartrate,cadence,time,velocity_smooth",
                "key_by_type": "true",
            }
            resp = requests.get(url, headers=self.get_headers(), params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            return {
                "time": data.get("time", {}).get("data", []),
                "watts": data.get("watts", {}).get("data", []),
                "heartrate": data.get("heartrate", {}).get("data", []),
                "cadence": data.get("cadence", {}).get("data", []),
            }
        except Exception as e:
            print(f"Error fetching streams: {e}. Generating sample stream.")
            return generate_sample_streams(duration_sec=3600, target_watts=ftp * 0.75, ftp=ftp)
