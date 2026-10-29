"""
Cycling Power & Physiological Metrics Engine
Based on Dr. Andrew Coggan & Joe Friel's training methodologies.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


def get_power_zones(ftp: float) -> Dict[str, Dict[str, Any]]:
    """
    Returns Coggan 7 Power Zones based on Functional Threshold Power (FTP).
    Also includes Sweet Spot (88-94% FTP).
    """
    return {
        "Z1_Active_Recovery": {
            "name": "Zone 1: Active Recovery (회복)",
            "range": (0, round(ftp * 0.55)),
            "pct": "< 55%",
            "description": "다리의 피로 회복 및 가벼운 혈액 순환 촉진",
            "color": "#94a3b8",
        },
        "Z2_Endurance": {
            "name": "Zone 2: Endurance (유산소 지구력)",
            "range": (round(ftp * 0.56), round(ftp * 0.75)),
            "pct": "56% - 75%",
            "description": "미토콘드리아 증식, 지방 대사 효율 극대화, 장거리 베이스",
            "color": "#38bdf8",
        },
        "FatMax": {
            "name": "FatMax: 지방연소 최대화 골든존",
            "range": (round(ftp * 0.60), round(ftp * 0.72)),
            "pct": "60% - 72%",
            "description": "젖산 축적 없이 순수 체지방 산화율(g/min)이 최고조에 달하는 구간",
            "color": "#22c55e",
        },
        "Z3_Tempo": {
            "name": "Zone 3: Tempo (템포)",
            "range": (round(ftp * 0.76), round(ftp * 0.90)),
            "pct": "76% - 90%",
            "description": "유산소 파워 향상, 장시간 지속 페이스",
            "color": "#4ade80",
        },
        "Sweet_Spot": {
            "name": "Sweet Spot (스윗스팟)",
            "range": (round(ftp * 0.88), round(ftp * 0.94)),
            "pct": "88% - 94%",
            "description": "피로도 대비 유산소 능력/FTP 향상 효율이 가장 높은 최적 구간",
            "color": "#facc15",
        },
        "Z4_Threshold": {
            "name": "Zone 4: Lactate Threshold (젖산 역치)",
            "range": (round(ftp * 0.91), round(ftp * 1.05)),
            "pct": "91% - 105%",
            "description": "젖산 축적 억제 한계 파워, FTP 직접 향상",
            "color": "#fb923c",
        },
        "Z5_VO2Max": {
            "name": "Zone 5: VO2 Max (최대 산소 섭취량)",
            "range": (round(ftp * 1.06), round(ftp * 1.20)),
            "pct": "106% - 120%",
            "description": "심폐 한계치 자극, 3~5분 고강도 인터벌",
            "color": "#f87171",
        },
        "Z6_Anaerobic": {
            "name": "Zone 6: Anaerobic Capacity (무산소 용량)",
            "range": (round(ftp * 1.21), round(ftp * 1.50)),
            "pct": "121% - 150%",
            "description": "30초~2분 폭발적 파워 및 젖산 내성 훈련",
            "color": "#c084fc",
        },
        "Z7_Neuromuscular": {
            "name": "Zone 7: Neuromuscular Power (신경근 파워/스프린트)",
            "range": (round(ftp * 1.51), round(ftp * 2.50)),
            "pct": "> 150%",
            "description": "최대 스프린트 (<15초), 근신경계 자극",
            "color": "#e879f9",
        },
    }


def calculate_normalized_power(power_stream: List[float]) -> float:
    """
    Calculates Normalized Power (NP) from a second-by-second power stream.
    Formula:
    1. 30-second rolling moving average
    2. Raise to 4th power
    3. Calculate mean of values
    4. Take the 4th root
    """
    if not power_stream or len(power_stream) < 30:
        return float(np.mean(power_stream)) if power_stream else 0.0

    series = pd.Series(power_stream)
    rolling_30s = series.rolling(window=30, min_periods=1).mean()
    np_val = (rolling_30s**4).mean() ** 0.25
    return round(float(np_val), 1)


def calculate_intensity_factor(np_watts: float, ftp: float) -> float:
    """Intensity Factor (IF) = NP / FTP"""
    if ftp <= 0:
        return 0.0
    return round(np_watts / ftp, 2)


def calculate_tss(moving_time_sec: int, np_watts: float, ftp: float) -> float:
    """
    Calculates Training Stress Score (TSS).
    Formula: TSS = (sec * NP * IF) / (FTP * 3600) * 100
    """
    if ftp <= 0 or moving_time_sec <= 0 or np_watts <= 0:
        return 0.0
    intensity_factor = np_watts / ftp
    tss = (moving_time_sec * np_watts * intensity_factor) / (ftp * 3600.0) * 100.0
    return round(float(tss), 1)


def estimate_substrate_burn(
    moving_time_sec: float, avg_watts: float, np_watts: float, ftp: float
) -> Dict[str, float]:
    """
    Estimates total calories (kcal), Fat oxidized (g), and Carbohydrates oxidized (g)
    based on mechanical work (kJ ≈ kcal at ~23.9% human gross efficiency) and Intensity Factor (IF).
    """
    if moving_time_sec <= 0 or (avg_watts <= 0 and np_watts <= 0):
        return {"kcal": 0, "fat_g": 0.0, "carb_g": 0.0, "fat_pct": 0}

    base_w = avg_watts if avg_watts > 0 else np_watts * 0.92
    eff_np = np_watts if np_watts > 0 else base_w
    if_val = eff_np / max(100.0, ftp)

    # Mechanical kJ = Watts * sec / 1000. Human metabolic kcal ≈ kJ * 1.04
    kcal = (base_w * moving_time_sec / 1000.0) * 1.04

    # Fat oxidation energy share vs IF (peaks in Z1/FatMax ~55-68%, drops sharply near FTP)
    if if_val <= 0.60:
        fat_share = 0.65
    elif if_val <= 0.72:  # FatMax Golden Zone
        fat_share = 0.62 - (if_val - 0.60) * 0.50  # 62% -> 56%
    elif if_val <= 0.85:  # Z3 Tempo
        fat_share = 0.56 - (if_val - 0.72) * 1.60  # 56% -> 35%
    elif if_val <= 0.95:  # Sweet Spot / Z4
        fat_share = 0.35 - (if_val - 0.85) * 2.00  # 35% -> 15%
    else:                 # Z4+ / VO2Max
        fat_share = max(0.05, 0.15 - (if_val - 0.95) * 1.00)

    fat_kcal = kcal * fat_share
    carb_kcal = kcal * (1.0 - fat_share)
    # 1g Fat ≈ 9 kcal, 1g Carb ≈ 4 kcal
    return {
        "kcal": round(kcal),
        "fat_g": round(fat_kcal / 9.0, 1),
        "carb_g": round(carb_kcal / 4.0, 1),
        "fat_pct": round(fat_share * 100),
    }


def calculate_power_distribution(power_stream: List[float], ftp: float) -> Dict[str, float]:
    """
    Calculates percentage of time spent in each Coggan Power Zone (Z1 to Z7).
    """
    if not power_stream:
        return {}

    zones = get_power_zones(ftp)
    arr = np.array(power_stream)
    total_points = len(arr)
    
    distribution = {}
    for key, zinfo in zones.items():
        if key in ("Sweet_Spot", "FatMax"):
            continue  # Overlaps standard zones, tracked separately
        low, high = zinfo["range"]
        if key == "Z7_Neuromuscular":
            count = np.sum(arr >= low)
        else:
            count = np.sum((arr >= low) & (arr < high))
        distribution[zinfo["name"]] = round(float((count / total_points) * 100.0), 1)
        
    return distribution


def calculate_aerobic_decoupling(
    power_stream: List[float], hr_stream: List[float]
) -> Optional[float]:
    """
    Calculates Aerobic Decoupling (Pw:HR Drift).
    Compares (NP / Avg HR) of first half vs second half.
    Decoupling > 5% indicates aerobic fatigue / cardiac drift.
    """
    if not power_stream or not hr_stream:
        return None
    
    n = min(len(power_stream), len(hr_stream))
    if n < 1200:  # Need at least 20 minutes for meaningful decoupling
        return None
        
    half = n // 2
    p1, hr1 = power_stream[:half], hr_stream[:half]
    p2, hr2 = power_stream[half:n], hr_stream[half:n]
    
    mean_hr1 = np.mean(hr1)
    mean_hr2 = np.mean(hr2)
    if mean_hr1 == 0 or mean_hr2 == 0:
        return None
        
    np1 = calculate_normalized_power(p1)
    np2 = calculate_normalized_power(p2)
    
    ratio1 = np1 / mean_hr1
    ratio2 = np2 / mean_hr2
    
    if ratio1 == 0:
        return None
        
    drift_pct = ((ratio1 - ratio2) / ratio1) * 100.0
    return round(float(drift_pct), 2)


def calculate_pmc(daily_tss_series: pd.Series) -> pd.DataFrame:
    """
    Calculates Performance Management Chart (PMC) metrics:
    - CTL (Chronic Training Load / 체력): 42-day exponential weighted moving average
    - ATL (Acute Training Load / 피로도): 7-day exponential weighted moving average
    - TSB (Training Stress Balance / 컨디션, 폼): CTL - ATL

    daily_tss_series: Index should be daily DatetimeIndex with complete calendar days.
    """
    df = pd.DataFrame({"tss": daily_tss_series})
    df["tss"] = df["tss"].fillna(0.0)

    # Exponential time constants
    # Using decay alpha = 1 - exp(-1 / days) or standard pandas ewm span
    df["ctl"] = df["tss"].ewm(span=42, adjust=False).mean()
    df["atl"] = df["tss"].ewm(span=7, adjust=False).mean()
    df["tsb"] = df["ctl"] - df["atl"]

    return df.round(1)


def project_pmc_future(pmc_df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Projects CTL, ATL, and TSB for the next `days` days assuming rest (0 TSS/day),
    helping the rider plan tapering or recovery.
    """
    if pmc_df.empty:
        return pd.DataFrame()
    last_date = pmc_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    combined_tss = pd.concat([
        pmc_df["tss"],
        pd.Series(0.0, index=future_dates),
    ])
    full_pmc = calculate_pmc(combined_tss)
    return full_pmc.iloc[-(days + 1):]


def interpret_tsb(tsb: float) -> Dict[str, str]:
    """
    Interprets Training Stress Balance (TSB / Form) for training recommendations.
    """
    if tsb > 15:
        return {
            "status": "Transitional / Detraining Risk (과도한 휴식)",
            "advice": "피로는 완전히 풀렸으나 훈련 자극이 적어 유산소 체력이 감소할 수 있습니다. 템포런이나 인터벌로 훈련 부하를 점진적으로 늘려주세요.",
            "badge_color": "warning",
            "zone": "High Rest",
        }
    elif 5 <= tsb <= 15:
        return {
            "status": "Race Ready / Peak Form (최상의 컨디션)",
            "advice": "대회나 최고 기록(PB) 도전에 가장 이상적인 컨디션입니다. 몸이 매우 가볍고 파워 출력이 최상에 도달해 있습니다.",
            "badge_color": "success",
            "zone": "Fresh / Peak",
        }
    elif -10 <= tsb < 5:
        return {
            "status": "Maintenance / Neutral (유지 및 준비)",
            "advice": "피로와 체력이 균형을 이루고 있습니다. 중강도 훈련이나 주말 중장거리 라이딩을 소화하기에 적합합니다.",
            "badge_color": "info",
            "zone": "Neutral",
        }
    elif -30 <= tsb < -10:
        return {
            "status": "Optimal Building / Productive Fatigue (효과적인 훈련 구간)",
            "advice": "체력(CTL)을 가장 효과적으로 키우는 최적의 빌드업 상태입니다. 영양 섭취와 수면을 철저히 관리하면서 계획된 인터벌 훈련을 이어가세요.",
            "badge_color": "primary",
            "zone": "Productive Training",
        }
    else:
        return {
            "status": "Overreaching / High Injury Risk (과훈련 경고)",
            "advice": "피로도(ATL)가 체력(CTL) 대비 과도하게 높습니다. 오버트레이닝 및 부상 위험이 높으므로 즉시 완전 휴식 또는 Z1 가벼운 회복 라이딩으로 전환하세요.",
            "badge_color": "danger",
            "zone": "Overreaching",
        }
