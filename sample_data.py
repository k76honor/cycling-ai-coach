"""
Realistic Cycling Activity and Stream Data Generator
Allows immediate testing and demonstration without needing immediate Strava API tokens.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from power_metrics import (
    calculate_normalized_power,
    calculate_intensity_factor,
    calculate_tss,
)


def generate_sample_cycling_history(ftp: float = 230.0, days: int = 45) -> List[Dict[str, Any]]:
    """
    Generates realistic 45-day cycling training history including:
    - Base Endurance rides (Z2)
    - Sweet Spot intervals (Z3-Z4)
    - VO2 Max 3-5min intervals (Z5)
    - Threshold 2x20min (Z4)
    - Active Recovery spins (Z1)
    - Rest days
    """
    np.random.seed(42)
    activities = []
    today = datetime.now(timezone.utc)

    # Workout templates
    templates = [
        {
            "name": "Zone 2 아침 유산소 지구력 라이딩",
            "type": "Ride",
            "duration_min": 90,
            "target_pct": 0.65,
            "variability": 0.08,
            "elev_per_km": 8,
            "speed_kmh": 27.5,
            "cadence": 88,
            "hr_base": 132,
        },
        {
            "name": "주말 동호회 100km 장거리 지구력 라이딩",
            "type": "Ride",
            "duration_min": 190,
            "target_pct": 0.68,
            "variability": 0.15,
            "elev_per_km": 14,
            "speed_kmh": 28.0,
            "cadence": 86,
            "hr_base": 138,
        },
        {
            "name": "스윗스팟 3x15분 인터벌 (Sweet Spot 88-94%)",
            "type": "VirtualRide",
            "duration_min": 65,
            "target_pct": 0.88,
            "variability": 0.18,
            "elev_per_km": 4,
            "speed_kmh": 32.0,
            "cadence": 92,
            "hr_base": 156,
        },
        {
            "name": "VO2 Max 5x3분 고강도 힐클라임 인터벌",
            "type": "Ride",
            "duration_min": 60,
            "target_pct": 0.95,
            "variability": 0.28,
            "elev_per_km": 20,
            "speed_kmh": 25.0,
            "cadence": 90,
            "hr_base": 165,
        },
        {
            "name": "젖산역치 2x20분 지속주 (FTP Threshold)",
            "type": "VirtualRide",
            "duration_min": 75,
            "target_pct": 0.92,
            "variability": 0.12,
            "elev_per_km": 5,
            "speed_kmh": 34.0,
            "cadence": 94,
            "hr_base": 162,
        },
        {
            "name": "다리 털기 Z1 가벼운 회복 라이딩",
            "type": "Ride",
            "duration_min": 45,
            "target_pct": 0.48,
            "variability": 0.05,
            "elev_per_km": 3,
            "speed_kmh": 22.0,
            "cadence": 85,
            "hr_base": 115,
        },
    ]

    # Weekly schedule pattern (0: Mon, 1: Tue, 2: Wed, 3: Thu, 4: Fri, 5: Sat, 6: Sun)
    # Mon: Rest
    # Tue: Sweet Spot or Threshold
    # Wed: Z2 Endurance
    # Thu: VO2 Max or Rest
    # Fri: Recovery or Rest
    # Sat: Long Ride
    # Sun: Z2 Endurance or Rest
    for day_offset in range(days, -1, -1):
        ride_date = today - timedelta(days=day_offset)
        weekday = ride_date.weekday()

        selected_tmpl = None
        if weekday == 0:  # Mon: Rest
            selected_tmpl = None
        elif weekday == 1:  # Tue
            selected_tmpl = templates[2] if day_offset % 2 == 0 else templates[4]
        elif weekday == 2:  # Wed
            selected_tmpl = templates[0]
        elif weekday == 3:  # Thu
            selected_tmpl = templates[3] if day_offset % 3 != 0 else None
        elif weekday == 4:  # Fri
            selected_tmpl = templates[5] if day_offset % 2 == 0 else None
        elif weekday == 5:  # Sat: Long Ride
            selected_tmpl = templates[1]
        elif weekday == 6:  # Sun
            selected_tmpl = templates[0] if day_offset % 2 == 0 else templates[5]

        if selected_tmpl is None:
            continue

        duration_sec = int(selected_tmpl["duration_min"] * 60)
        target_watts = ftp * selected_tmpl["target_pct"]
        
        # Add realistic variation
        avg_watts = round(target_watts * np.random.uniform(0.96, 1.04), 1)
        variability = selected_tmpl["variability"]
        np_watts = round(avg_watts * (1.0 + variability * 0.7), 1)
        
        if_val = calculate_intensity_factor(np_watts, ftp)
        tss_val = calculate_tss(duration_sec, np_watts, ftp)
        
        distance_km = round((duration_sec / 3600.0) * selected_tmpl["speed_kmh"], 2)
        elev_gain = round(distance_km * selected_tmpl["elev_per_km"] * np.random.uniform(0.85, 1.15))
        avg_hr = int(selected_tmpl["hr_base"] + np.random.randint(-4, 6))
        max_hr = int(avg_hr + (25 if "VO2" in selected_tmpl["name"] or "역치" in selected_tmpl["name"] else 15))
        
        activity = {
            "id": 1000000000 + day_offset,
            "name": selected_tmpl["name"],
            "type": selected_tmpl["type"],
            "start_date": ride_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_date_local": ride_date.strftime("%Y-%m-%d %H:%M:%S"),
            "distance": distance_km * 1000,
            "moving_time": duration_sec,
            "elapsed_time": duration_sec + np.random.randint(60, 600),
            "total_elevation_gain": elev_gain,
            "average_speed": round(distance_km * 1000 / duration_sec, 2),
            "max_speed": round((selected_tmpl["speed_kmh"] * 1.5) / 3.6, 2),
            "average_watts": avg_watts,
            "weighted_average_watts": np_watts,
            "max_watts": round(avg_watts * (1.6 if "VO2" in selected_tmpl["name"] else 1.3), 1),
            "average_heartrate": avg_hr,
            "max_heartrate": max_hr,
            "average_cadence": selected_tmpl["cadence"] + np.random.randint(-2, 3),
            "kilojoules": round(avg_watts * duration_sec / 1000.0),
            "device_watts": True,
            "has_heartrate": True,
            "tss": tss_val,
            "intensity_factor": if_val,
        }
        activities.append(activity)

    return sorted(activities, key=lambda x: x["start_date"], reverse=True)


def generate_sample_streams(duration_sec: int = 3600, target_watts: float = 200.0, ftp: float = 230.0) -> Dict[str, List[float]]:
    """
    Generates realistic second-by-second streams for power, heartrate, cadence, time.
    """
    time_series = list(range(duration_sec))
    np.random.seed(int(target_watts) % 100)
    
    # Base power with micro-surges and interval intervals
    noise = np.random.normal(0, target_watts * 0.08, duration_sec)
    power_stream = [max(0.0, round(float(target_watts + noise[i]), 1)) for i in range(duration_sec)]
    
    # Heart rate lag model
    hr_base = 120 + (target_watts / ftp) * 45
    hr_drift = np.linspace(0, 8, duration_sec)  # slight cardiac drift
    hr_noise = np.random.normal(0, 2, duration_sec)
    hr_stream = [round(float(hr_base + hr_drift[i] + hr_noise[i])) for i in range(duration_sec)]
    
    # Cadence
    cad_base = 88
    cad_noise = np.random.normal(0, 3, duration_sec)
    cadence_stream = [max(50, min(120, round(float(cad_base + cad_noise[i])))) for i in range(duration_sec)]
    
    return {
        "time": time_series,
        "watts": power_stream,
        "heartrate": hr_stream,
        "cadence": cadence_stream,
    }
