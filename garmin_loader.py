"""
Garmin Edge 1040 FIT Data Loader, Auto-Persister & Dynamic FTP Estimator
Loads embedded Garmin activities (garmin_activities.json), persists newly uploaded .fit files,
and dynamically estimates the athlete's evolving FTP from recent power & heart-rate data.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import requests

from power_metrics import calculate_intensity_factor, calculate_tss

KST = timezone(timedelta(hours=9))
GARMIN_JSON_PATH = Path(__file__).resolve().parent / "garmin_activities.json"


def sync_file_to_github_if_configured(file_path: Path, commit_message: str) -> bool:
    """
    If GITHUB_TOKEN is set in Streamlit Secrets or environment variables,
    automatically commits the updated JSON file back to k76honor/cycling-ai-coach on GitHub
    so changes survive even full cloud container rebuilds.
    """
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GITHUB_TOKEN" in st.secrets:
                token = str(st.secrets["GITHUB_TOKEN"])
        except Exception:
            pass
    if not token or not file_path.exists():
        return False

    repo = os.getenv("GITHUB_REPO", "k76honor/cycling-ai-coach")
    rel_name = file_path.name
    api_url = f"https://api.github.com/repos/{repo}/contents/{rel_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        get_resp = requests.get(api_url, headers=headers, timeout=8)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
        content_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        payload = {
            "message": commit_message,
            "content": content_b64,
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha
        put_resp = requests.put(api_url, headers=headers, json=payload, timeout=10)
        return put_resp.status_code in (200, 201)
    except Exception as e:
        print(f"GitHub auto-sync skipped: {e}")
        return False


def _get_val(frame, name: str, default=None):
    if frame.has_field(name):
        v = frame.get_value(name)
        return v if v is not None else default
    return default


def _compute_mmp(power_series: List[float], windows=(5, 15, 30, 60, 180, 300, 600, 1200, 1800, 3600)) -> Dict[str, Optional[float]]:
    res = {}
    n = len(power_series)
    if n == 0:
        return res
    arr = np.array(power_series, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    cumsum = np.insert(np.cumsum(arr), 0, 0.0)
    for w in windows:
        if n >= w:
            rolling = (cumsum[w:] - cumsum[:-w]) / w
            res[f"mmp_{w}s"] = round(float(np.max(rolling)), 1)
        else:
            res[f"mmp_{w}s"] = None
    return res


def _compute_decoupling(powers: List[Optional[float]], hrs: List[Optional[float]]) -> Optional[float]:
    valid = [(p, h) for p, h in zip(powers, hrs) if p is not None and h is not None and h > 60 and p >= 0]
    if len(valid) < 600:
        return None
    mid = len(valid) // 2
    first = valid[:mid]
    second = valid[mid:]
    p1 = np.mean([x[0] for x in first])
    h1 = np.mean([x[1] for x in first])
    p2 = np.mean([x[0] for x in second])
    h2 = np.mean([x[1] for x in second])
    if h1 == 0 or h2 == 0 or p1 == 0:
        return None
    ef1 = p1 / h1
    ef2 = p2 / h2
    return round(float(((ef1 - ef2) / ef1) * 100.0), 2)


def parse_uploaded_fit_bytes(
    file_bytes: bytes, filename: str, ftp: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Parses a raw .fit file uploaded via Streamlit file_uploader into our unified Garmin activity dict."""
    try:
        import fitdecode
    except ImportError:
        return None

    session_data = {}
    tiz_session = {}
    workout_name = None
    user_prof = {}
    powers = []
    hrs = []

    try:
        with fitdecode.FitReader(io.BytesIO(file_bytes)) as fit:
            for frame in fit:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue
                if frame.name == "workout":
                    wname = _get_val(frame, "wkt_name")
                    if wname:
                        workout_name = str(wname)
                elif frame.name == "user_profile":
                    for f in frame.fields:
                        if f.value is not None:
                            user_prof[f.name] = f.value
                elif frame.name == "session":
                    for f in frame.fields:
                        if f.value is not None:
                            session_data[f.name] = f.value
                elif frame.name == "time_in_zone":
                    ref = _get_val(frame, "reference_mesg")
                    if ref == "session":
                        for f in frame.fields:
                            if f.value is not None:
                                tiz_session[f.name] = f.value
                elif frame.name == "record":
                    p = _get_val(frame, "power")
                    h = _get_val(frame, "heart_rate")
                    powers.append(p if p is not None else 0)
                    hrs.append(h)
    except Exception as e:
        print(f"Error parsing FIT file {filename}: {e}")
        return None

    if not session_data:
        return None

    start_dt = session_data.get("start_time")
    if isinstance(start_dt, datetime):
        start_kst = start_dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
        date_kst = start_dt.astimezone(KST).strftime("%Y-%m-%d")
    else:
        start_kst = str(start_dt)
        date_kst = filename[:10]

    mmp = _compute_mmp(powers)
    decoupling = _compute_decoupling(powers, hrs)

    avg_p = session_data.get("avg_power")
    np_p = session_data.get("normalized_power")
    avg_hr = session_data.get("avg_heart_rate")
    ef = round(np_p / avg_hr, 2) if (np_p and avg_hr and avg_hr > 0) else None
    vi = round(np_p / avg_p, 3) if (np_p and avg_p and avg_p > 0) else None

    lr = session_data.get("left_right_balance")
    if lr is not None and isinstance(lr, (int, float)) and lr > 32768:
        right_pct = (lr - 32768) / 100.0
        left_pct = round(100.0 - right_pct, 1)
        right_pct = round(right_pct, 1)
        lr_str = f"{left_pct:.1f}% L / {right_pct:.1f}% R"
    else:
        left_pct = None
        right_pct = None
        lr_str = "-"

    return {
        "file": filename,
        "start_kst": start_kst,
        "date": date_kst,
        "sport": str(session_data.get("sport")),
        "sub_sport": str(session_data.get("sub_sport")),
        "sport_profile_name": str(session_data.get("sport_profile_name")),
        "workout_name": workout_name,
        "elapsed_time_s": session_data.get("total_elapsed_time"),
        "timer_time_s": session_data.get("total_timer_time"),
        "distance_m": session_data.get("total_distance"),
        "ascent_m": session_data.get("total_ascent"),
        "descent_m": session_data.get("total_descent"),
        "calories": session_data.get("total_calories"),
        "work_j": session_data.get("total_work"),
        "avg_speed_ms": session_data.get("enhanced_avg_speed") or session_data.get("avg_speed"),
        "max_speed_ms": session_data.get("enhanced_max_speed") or session_data.get("max_speed"),
        "avg_power": avg_p,
        "max_power": session_data.get("max_power"),
        "np": np_p,
        "ftp": session_data.get("threshold_power") or tiz_session.get("functional_threshold_power") or ftp,
        "if": session_data.get("intensity_factor"),
        "tss": session_data.get("training_stress_score"),
        "vi": vi,
        "ef": ef,
        "decoupling_pct": decoupling,
        "avg_hr": avg_hr,
        "max_hr": session_data.get("max_heart_rate"),
        "resting_hr": tiz_session.get("resting_heart_rate"),
        "lthr": tiz_session.get("threshold_heart_rate"),
        "max_hr_setting": tiz_session.get("max_heart_rate"),
        "avg_cadence": session_data.get("avg_cadence"),
        "max_cadence": session_data.get("max_cadence"),
        "avg_resp": session_data.get("enhanced_avg_respiration_rate"),
        "max_resp": session_data.get("enhanced_max_respiration_rate"),
        "training_load": session_data.get("training_load_peak"),
        "aerobic_te": session_data.get("total_training_effect"),
        "anaerobic_te": session_data.get("total_anaerobic_training_effect"),
        "avg_temp": session_data.get("avg_temperature"),
        "max_temp": session_data.get("max_temperature"),
        "workout_feel": session_data.get("workout_feel"),
        "workout_rpe": session_data.get("workout_rpe"),
        "stamina_start": session_data.get("unknown_205"),
        "stamina_end": session_data.get("unknown_206"),
        "left_right_balance": lr,
        "left_pct": left_pct,
        "right_pct": right_pct,
        "lr_str": lr_str,
        "avg_left_torque_eff": session_data.get("avg_left_torque_effectiveness"),
        "avg_right_torque_eff": session_data.get("avg_right_torque_effectiveness"),
        "avg_left_pedal_smooth": session_data.get("avg_left_pedal_smoothness"),
        "avg_right_pedal_smooth": session_data.get("avg_right_pedal_smoothness"),
        "time_in_hr_zone": list(tiz_session.get("time_in_hr_zone", [])) if tiz_session.get("time_in_hr_zone") else None,
        "time_in_power_zone": list(tiz_session.get("time_in_power_zone", [])) if tiz_session.get("time_in_power_zone") else None,
        "hr_zone_bounds": list(tiz_session.get("hr_zone_high_boundary", [])) if tiz_session.get("hr_zone_high_boundary") else None,
        "power_zone_bounds": list(tiz_session.get("power_zone_high_boundary", [])) if tiz_session.get("power_zone_high_boundary") else None,
        "weight_kg": user_prof.get("weight"),
        "mmp": mmp,
    }


def load_garmin_raw_activities(uploaded_files: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Loads the embedded Garmin Edge 1040 activities and merges any newly uploaded .fit files.
    Automatically saves newly uploaded activities to garmin_activities.json!
    """
    raw_list: List[Dict[str, Any]] = []
    full_json_data: Dict[str, Any] = {}
    if GARMIN_JSON_PATH.exists():
        try:
            full_json_data = json.loads(GARMIN_JSON_PATH.read_text(encoding="utf-8"))
            raw_list = full_json_data.get("activities", [])
        except Exception as e:
            print(f"Error reading garmin_activities.json: {e}")

    by_start = {a.get("start_kst", a.get("file")): a for a in raw_list}
    added_new = False

    if uploaded_files:
        for uf in uploaded_files:
            parsed = parse_uploaded_fit_bytes(uf.getvalue(), uf.name)
            if parsed:
                key = parsed.get("start_kst", parsed.get("file"))
                if key not in by_start:
                    added_new = True
                by_start[key] = parsed

    merged = list(by_start.values())
    merged.sort(key=lambda x: x.get("start_kst", ""))

    # Decode L/R balance for each item
    for a in merged:
        lr = a.get("left_right_balance")
        if lr is not None and isinstance(lr, (int, float)) and lr > 32768:
            right_pct = (lr - 32768) / 100.0
            left_pct = 100.0 - right_pct
            a["left_pct"] = round(left_pct, 1)
            a["right_pct"] = round(right_pct, 1)
            a["lr_str"] = f"{left_pct:.1f}% L / {right_pct:.1f}% R"
        else:
            a["left_pct"] = None
            a["right_pct"] = None
            a["lr_str"] = "-"

    if added_new:
        try:
            full_json_data["activities"] = merged
            GARMIN_JSON_PATH.write_text(
                json.dumps(full_json_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            sync_file_to_github_if_configured(
                GARMIN_JSON_PATH, f"Auto-append newly uploaded FIT activities (total: {len(merged)})"
            )
        except Exception as e:
            print(f"Error persisting new FIT activities: {e}")

    return merged


def estimate_dynamic_ftp(garmin_acts: List[Dict[str, Any]], lthr: int = 161) -> Dict[str, Any]:
    """
    Dynamically estimates the athlete's current FTP from the most recent 6 weeks of rides
    by combining:
    1. Recent Garmin FTP detections
    2. Best 20-minute & 60-minute power curves (adjusted for sub-maximal endurance pacing)
    3. Sub-maximal HR-to-Power efficiency at LTHR (e.g. 161 bpm)
    """
    if not garmin_acts:
        return {"recommended_ftp": 196, "potential_max_ftp": 202, "latest_garmin_ftp": 189, "latest_weight": 64.5}

    recent = garmin_acts[-20:]  # Last ~20 rides
    latest_garmin_ftp = next((a["ftp"] for a in reversed(garmin_acts) if a.get("ftp")), 189)
    max_garmin_ftp = max([a.get("ftp") or 0 for a in recent] or [189])
    latest_weight = next((a["weight_kg"] for a in reversed(garmin_acts) if a.get("weight_kg")), 64.5)

    # Best 20m and 60m MMP across recent rides
    best_20m = max([(a.get("mmp", {}).get("mmp_1200s") or 0) for a in garmin_acts] or [189.0])
    best_60m = max([(a.get("mmp", {}).get("mmp_3600s") or 0) for a in garmin_acts] or [159.1])

    # High-intensity efficiency (EF = NP / avg_hr) on hard rides (NP >= 145W, avg_hr >= 128)
    hard_efs = [
        (a["np"] / a["avg_hr"])
        for a in recent
        if a.get("np") and a.get("avg_hr") and a["np"] >= 145 and a["avg_hr"] >= 128
    ]
    mean_hard_ef = float(np.mean(hard_efs)) if hard_efs else 1.21

    # Power projected at LTHR (with 0.98 steady-state conversion factor)
    lthr_projected_ftp = round(mean_hard_ef * lthr * 0.995)
    mmp_based_ftp = round(max(best_20m * 1.03, max_garmin_ftp * 1.02))

    recommended_ftp = int(round((lthr_projected_ftp * 0.55) + (mmp_based_ftp * 0.45)))
    recommended_ftp = max(max_garmin_ftp, min(230, recommended_ftp))
    potential_max_ftp = recommended_ftp + 6

    return {
        "recommended_ftp": recommended_ftp,
        "potential_max_ftp": potential_max_ftp,
        "latest_garmin_ftp": latest_garmin_ftp,
        "max_garmin_ftp": max_garmin_ftp,
        "latest_weight": float(latest_weight),
        "best_20m_w": best_20m,
        "best_60m_w": best_60m,
    }


def convert_garmin_to_dashboard_activities(
    garmin_acts: List[Dict[str, Any]],
    ftp: float = 196.0,
    recalc_tss_with_ftp: bool = False,
) -> List[Dict[str, Any]]:
    """
    Converts Garmin raw activity dicts into the dashboard activity format (newest first),
    compatible with build_pmc_dataframe and coach_agent.
    """
    converted = []
    for idx, a in enumerate(reversed(garmin_acts)):
        sub = a.get("sub_sport", "road")
        prof = a.get("sport_profile_name") or ("실내" if sub == "indoor_cycling" else "도로")
        wkt = a.get("workout_name")
        dist_km = (a.get("distance_m") or 0) / 1000.0
        asc = a.get("ascent_m") or 0

        if wkt:
            ride_name = f"[{prof}] {wkt} ({dist_km:.1f}km)"
        elif sub == "road" and asc >= 1000:
            ride_name = f"[{prof}] 장거리 산악 라이딩 ({dist_km:.1f}km / +{asc}m)"
        elif sub == "road":
            ride_name = f"[{prof}] 야외 로드 라이딩 ({dist_km:.1f}km)"
        else:
            ride_name = f"[{prof}] 인도어 트레이닝 ({dist_km:.1f}km)"

        start_kst = a.get("start_kst", "2026-10-27 19:43:18")
        iso_kst = start_kst.replace(" ", "T") + "+09:00"

        dur_s = float(a.get("timer_time_s") or 0)
        np_w = float(a.get("np") or a.get("avg_power") or 0)
        avg_w = float(a.get("avg_power") or 0)

        if recalc_tss_with_ftp and ftp > 0 and np_w > 0 and dur_s > 0:
            if_val = calculate_intensity_factor(np_w, ftp)
            tss_val = calculate_tss(dur_s, np_w, ftp)
        else:
            if_val = float(a.get("if") or calculate_intensity_factor(np_w, ftp))
            tss_val = float(a.get("tss") or calculate_tss(dur_s, np_w, ftp))

        converted.append({
            "id": 9000000 + idx,
            "name": ride_name,
            "type": "VirtualRide" if sub == "indoor_cycling" else "Ride",
            "sport_type": "VirtualRide" if sub == "indoor_cycling" else "Ride",
            "sub_sport": sub,
            "workout_name": wkt,
            "start_date": iso_kst,
            "start_date_local": iso_kst,
            "distance": float(a.get("distance_m") or 0.0),
            "moving_time": int(dur_s),
            "elapsed_time": int(a.get("elapsed_time_s") or dur_s),
            "total_elevation_gain": float(asc),
            "average_watts": avg_w,
            "weighted_average_watts": np_w,
            "max_watts": float(a.get("max_power") or 0),
            "average_heartrate": float(a.get("avg_hr") or 0),
            "max_heartrate": float(a.get("max_hr") or 0),
            "average_cadence": float(a.get("avg_cadence") or 0),
            "intensity_factor": round(if_val, 3),
            "tss": round(tss_val, 1),
            "garmin_tss": float(a.get("tss") or 0.0),
            "garmin_ftp": a.get("ftp"),
            "ef": a.get("ef"),
            "vi": a.get("vi"),
            "decoupling_pct": a.get("decoupling_pct"),
            "lr_str": a.get("lr_str", "-"),
            "left_pct": a.get("left_pct"),
            "stamina_start": a.get("stamina_start"),
            "stamina_end": a.get("stamina_end"),
            "aerobic_te": a.get("aerobic_te"),
            "anaerobic_te": a.get("anaerobic_te"),
            "mmp": a.get("mmp", {}),
        })

    return converted
