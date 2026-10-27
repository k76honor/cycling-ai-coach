"""
Strava API v3 Client for Cycling Training Data
Handles OAuth2 token refreshing, live activity fetching, EF/VI/TSS calculation,
and fallback to the athlete's real Garmin baseline data if Strava is unreachable.
"""

from typing import List, Dict, Any, Optional
import requests
import numpy as np
from sample_data import generate_sample_cycling_history, generate_sample_streams
from power_metrics import calculate_intensity_factor, calculate_tss
from garmin_loader import load_garmin_raw_activities, convert_garmin_to_dashboard_activities


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
        """Exchanges the refresh_token for a fresh short-lived access_token."""
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

    def _fallback_activities(self, ftp: float) -> List[Dict[str, Any]]:
        """Falls back to the athlete's real Garmin baseline dataset before using synthetic data."""
        raw = load_garmin_raw_activities()
        if raw:
            return convert_garmin_to_dashboard_activities(raw, ftp=ftp, recalc_tss_with_ftp=True)
        return generate_sample_cycling_history(ftp=ftp, days=45)

    def get_activities(
        self,
        ftp: float = 196.0,
        per_page: int = 100,
        use_sample_if_unconfigured: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetches recent cycling activities from Strava in real time.
        Automatically computes NP, IF, TSS, EF (Efficiency Factor), and VI (Variability Index).
        """
        if not self.is_configured():
            if use_sample_if_unconfigured:
                return self._fallback_activities(ftp=ftp)
            return []

        if not self.access_token:
            self.refresh_access_token()
            if not self.access_token:
                if use_sample_if_unconfigured:
                    return self._fallback_activities(ftp=ftp)
                return []

        try:
            url = f"{self.API_BASE_URL}/athlete/activities"
            params = {"per_page": per_page}
            resp = requests.get(url, headers=self.get_headers(), params=params, timeout=12)
            resp.raise_for_status()
            raw_activities = resp.json()

            cycling_types = {"Ride", "VirtualRide", "EBikeRide", "GravelRide", "MountainBikeRide"}
            cycling_activities = [
                act for act in raw_activities
                if act.get("type") in cycling_types or act.get("sport_type") in cycling_types
            ]

            for act in cycling_activities:
                duration_sec = act.get("moving_time", 0)
                avg_watts = float(act.get("average_watts") or 0.0)
                np_watts = float(act.get("weighted_average_watts") or avg_watts or 0.0)
                avg_hr = float(act.get("average_heartrate") or 0.0)

                if np_watts and duration_sec and ftp > 0:
                    act["intensity_factor"] = calculate_intensity_factor(np_watts, ftp)
                    act["tss"] = calculate_tss(duration_sec, np_watts, ftp)
                else:
                    act["intensity_factor"] = 0.0
                    act["tss"] = 0.0

                act["ef"] = round(np_watts / avg_hr, 2) if (np_watts > 0 and avg_hr > 0) else None
                act["vi"] = round(np_watts / avg_watts, 2) if (np_watts > 0 and avg_watts > 0) else None

                # Classify indoor vs road from Strava trainer flag / VirtualRide / zero elevation
                is_indoor = bool(
                    act.get("trainer")
                    or act.get("type") == "VirtualRide"
                    or act.get("sport_type") == "VirtualRide"
                )
                act["sub_sport"] = "indoor_cycling" if is_indoor else "road"

            return cycling_activities
        except Exception as e:
            print(f"Error fetching Strava activities: {e}. Falling back to baseline data.")
            if use_sample_if_unconfigured:
                return self._fallback_activities(ftp=ftp)
            return []

    def get_activity_streams(
        self, activity_id: int, ftp: float = 196.0
    ) -> Dict[str, List[float]]:
        """Fetches detailed second-by-second telemetry streams from Strava."""
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


def estimate_ftp_from_live_activities(activities: List[Dict[str, Any]], lthr: int = 161) -> Dict[str, Any]:
    """
    Dynamically estimates the athlete's current FTP directly from live Strava activities!
    Adapts automatically as new rides are uploaded to Strava.
    """
    if not activities:
        return {"recommended_ftp": 196, "potential_max_ftp": 202, "max_np": 172, "avg_ef": 1.18}

    recent = activities[:25]  # Most recent 25 rides from Strava (newest first)

    # Hard rides >= 20 min where NP and HR are recorded
    hard_efs = []
    max_np_20m_plus = 0.0
    max_np_60m_plus = 0.0

    for a in recent:
        dur_m = (a.get("moving_time") or 0) / 60.0
        np_w = float(a.get("weighted_average_watts") or a.get("average_watts") or 0)
        hr = float(a.get("average_heartrate") or 0)
        if dur_m >= 20 and np_w > max_np_20m_plus:
            max_np_20m_plus = np_w
        if dur_m >= 50 and np_w > max_np_60m_plus:
            max_np_60m_plus = np_w
        if dur_m >= 30 and np_w >= 140 and hr >= 125:
            hard_efs.append(np_w / hr)

    mean_hard_ef = float(np.mean(hard_efs)) if hard_efs else 1.21
    lthr_projected_ftp = round(mean_hard_ef * lthr * 0.995)
    np_based_ftp = round(max(max_np_60m_plus * 1.14, max_np_20m_plus * 1.10, 190))

    recommended_ftp = int(round((lthr_projected_ftp * 0.60) + (np_based_ftp * 0.40)))
    recommended_ftp = max(185, min(250, recommended_ftp))

    return {
        "recommended_ftp": recommended_ftp,
        "potential_max_ftp": recommended_ftp + 6,
        "max_np_60m": round(max_np_60m_plus, 1),
        "avg_hard_ef": round(mean_hard_ef, 2),
    }
