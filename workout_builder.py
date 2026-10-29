"""
Garmin Edge 1040 (.fit) & Zwift/Intervals.icu (.zwo) Workout File Generator
Generates binary-compliant Garmin FIT workout files and ZWO XML files
tailored to the athlete's live FTP, LTHR, Max HR, and current TSB.
"""

import struct
from datetime import datetime, timezone
from typing import List, Dict, Any


# Garmin FIT CRC-16 lookup table
_FIT_CRC_TABLE = (
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
    0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
)


def _fit_crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        tmp = _FIT_CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _FIT_CRC_TABLE[byte & 0xF]
        tmp = _FIT_CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _FIT_CRC_TABLE[(byte >> 4) & 0xF]
    return crc


def _encode_fixed_str(text: str, length: int) -> bytes:
    raw = text.encode("utf-8")[: length - 1]
    return raw + b"\x00" * (length - len(raw))


def build_garmin_workout_fit_bytes(workout_name: str, steps: List[Dict[str, Any]]) -> bytes:
    """
    Builds a binary Garmin .fit Workout file (file_type=5) that can be placed directly into
    Garmin Edge 1040 `Garmin/NewFiles` or `Garmin/Workouts`.

    Each step in `steps` is a dict:
      - name: str (step label)
      - duration_sec: int
      - low_w: int (target low watts)
      - high_w: int (target high watts)
      - intensity: int (0=active, 1=rest, 2=warmup, 3=cooldown)
    """
    records = bytearray()

    # FIT Epoch is 1989-12-31 00:00:00 UTC
    fit_epoch = datetime(1989, 12, 31, tzinfo=timezone.utc)
    now_ts = int((datetime.now(timezone.utc) - fit_epoch).total_seconds())

    # 1. Definition Message: file_id (Global Mesg Num = 0, Local = 0)
    # Fields (4-byte aligned first, then 2-byte, then 1-byte):
    # - serial_number (field 3, uint32z, size 4, base_type 0x8C)
    # - time_created (field 4, uint32, size 4, base_type 0x86)
    # - manufacturer (field 1, uint16, size 2, base_type 0x84)
    # - product (field 2, uint16, size 2, base_type 0x84)
    # - type (field 0, enum/uint8, size 1, base_type 0x00)
    records.extend(struct.pack(
        "<BBBHB" + "3B" * 5,
        0x40,  # Definition header, local mesg 0
        0,     # reserved
        0,     # little-endian
        0,     # Global mesg num = 0 (file_id)
        5,     # 5 fields
        3, 4, 0x8C,  # serial_number (uint32z)
        4, 4, 0x86,  # time_created (uint32)
        1, 2, 0x84,  # manufacturer (uint16)
        2, 2, 0x84,  # product (uint16)
        0, 1, 0x00,  # type (enum)
    ))
    # Data Message: file_id (type=5 -> workout, manufacturer=1 -> Garmin, product=3843 -> Edge 1040)
    records.extend(struct.pack(
        "<BIIHHB",
        0x00,       # Data header, local mesg 0
        12345678,   # serial_number
        now_ts,     # time_created
        1,          # manufacturer = 1 (garmin)
        3843,       # product = Edge 1040
        5,          # type = 5 (workout)
    ))

    # 2. Definition Message: workout (Global Mesg Num = 26, Local = 1)
    # Fields:
    # - capabilities (field 5, uint32z, size 4, base_type 0x8C) -> 0x00000020
    # - num_valid_steps (field 6, uint16, size 2, base_type 0x84)
    # - sport (field 4, enum, size 1, base_type 0x00) -> 2 = cycling
    # - wkt_name (field 8, string, size 24, base_type 0x07)
    records.extend(struct.pack(
        "<BBBHB" + "3B" * 4,
        0x41,  # Definition header, local mesg 1
        0,
        0,     # little-endian
        26,    # Global mesg num = 26 (workout)
        4,     # 4 fields
        5, 4, 0x8C,   # capabilities
        6, 2, 0x84,   # num_valid_steps
        4, 1, 0x00,   # sport
        8, 24, 0x07,  # wkt_name (24-byte string)
    ))
    records.extend(struct.pack(
        "<BIHB",
        0x01,          # Data header, local mesg 1
        0x00000020,    # capabilities
        len(steps),    # num_valid_steps
        2,             # sport = 2 (cycling)
    ))
    records.extend(_encode_fixed_str(workout_name, 24))

    # 3. Definition Message: workout_step (Global Mesg Num = 27, Local = 2)
    # Fields:
    # - duration_value (field 2, uint32, size 4, base_type 0x86) -> milliseconds
    # - target_value (field 4, uint32, size 4, base_type 0x86) -> 0 = custom range
    # - custom_target_value_low (field 5, uint32, size 4, base_type 0x86) -> watts + 1000
    # - custom_target_value_high (field 6, uint32, size 4, base_type 0x86) -> watts + 1000
    # - message_index (field 254, uint16, size 2, base_type 0x84)
    # - wkt_step_name (field 0, string, size 16, base_type 0x07)
    # - duration_type (field 1, enum, size 1, base_type 0x00) -> 0 = time
    # - target_type (field 3, enum, size 1, base_type 0x00) -> 4 = power
    # - intensity (field 7, enum, size 1, base_type 0x00) -> 0=active, 1=rest, 2=warmup, 3=cooldown
    records.extend(struct.pack(
        "<BBBHB" + "3B" * 9,
        0x42,  # Definition header, local mesg 2
        0,
        0,     # little-endian
        27,    # Global mesg num = 27 (workout_step)
        9,     # 9 fields
        2, 4, 0x86,    # duration_value
        4, 4, 0x86,    # target_value
        5, 4, 0x86,    # custom_target_value_low
        6, 4, 0x86,    # custom_target_value_high
        254, 2, 0x84,  # message_index
        0, 16, 0x07,   # wkt_step_name
        1, 1, 0x00,    # duration_type
        3, 1, 0x00,    # target_type
        7, 1, 0x00,    # intensity
    ))

    for idx, st_item in enumerate(steps):
        step_name = _encode_fixed_str(st_item.get("name", f"Step {idx+1}"), 16)
        dur_ms = int(round(st_item["duration_sec"] * 1000))
        # In Garmin FIT standard, custom power target in Watts is encoded as (Watts + 1000)
        low_val = int(round(st_item["low_w"])) + 1000
        high_val = int(round(st_item["high_w"])) + 1000
        intensity = int(st_item.get("intensity", 0))

        records.extend(struct.pack(
            "<BIIIIH",
            0x02,       # Data header, local mesg 2
            dur_ms,     # duration_value (ms)
            0,          # target_value = 0 (use custom range)
            low_val,    # custom_target_value_low (Watts + 1000)
            high_val,   # custom_target_value_high (Watts + 1000)
            idx,        # message_index
        ))
        records.extend(step_name)
        records.extend(struct.pack(
            "<BBB",
            0,          # duration_type = 0 (time)
            4,          # target_type = 4 (power)
            intensity,  # intensity
        ))

    data_size = len(records)
    # 14-byte FIT header: header_size(14), protocol_version(0x10), profile_version(2116), data_size, b".FIT"
    header_without_crc = struct.pack("<BBHI4s", 14, 0x10, 2116, data_size, b".FIT")
    header_crc = _fit_crc16(header_without_crc)
    header = header_without_crc + struct.pack("<H", header_crc)

    full_body = header + bytes(records)
    file_crc = _fit_crc16(full_body)
    return full_body + struct.pack("<H", file_crc)


def build_zwift_zwo_xml(workout_name: str, description: str, ftp: float, steps: List[Dict[str, Any]]) -> str:
    """
    Builds a Zwift (.zwo) XML string compatible with Zwift, Intervals.icu, Rouvy, and TrainingPeaks.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<workout_file>",
        "    <author>AI Cycling Coach</author>",
        f"    <name>{workout_name}</name>",
        f"    <description>{description}</description>",
        "    <sportType>bike</sportType>",
        "    <tags><tag name=\"AI_Coach\"/></tags>",
        "    <workout>",
    ]
    for s in steps:
        dur = int(s["duration_sec"])
        p_low = round(s["low_w"] / max(1.0, ftp), 3)
        p_high = round(s["high_w"] / max(1.0, ftp), 3)
        p_mid = round((p_low + p_high) / 2.0, 3)
        cad = s.get("cadence", 90)
        inten = s.get("intensity", 0)
        if inten == 2:
            lines.append(f'        <Warmup Duration="{dur}" PowerLow="{p_low}" PowerHigh="{p_high}" Cadence="{cad}"/>')
        elif inten == 3:
            lines.append(f'        <Cooldown Duration="{dur}" PowerLow="{p_high}" PowerHigh="{p_low}" Cadence="{cad}"/>')
        else:
            lines.append(f'        <SteadyState Duration="{dur}" Power="{p_mid}" Cadence="{cad}"/>')

    lines.extend([
        "    </workout>",
        "</workout_file>",
    ])
    return "\n".join(lines)


def get_preset_workouts(ftp: float, lthr: int = 161, max_hr: int = 184) -> Dict[str, Dict[str, Any]]:
    """
    Returns tailored preset workouts for the athlete (FTP 196W, LTHR 161bpm, MaxHR 184bpm).
    Includes FatMax 90m Surge, FatMax 60m Standard, SST 3x12m, and Z4 4x8m Threshold.
    """
    # 1. FatMax 90m Surge (10m Warmup + 3 x [20m @ 128~137W + 20s Surge @ 230~260W] + 9m FatMax + 10m Z1 Cooldown = 90m)
    fm90_steps = [
        {"name": "10m Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.62), "intensity": 2, "cadence": 85, "hr_guide": f"~{round(max_hr*0.65)} bpm"},
    ]
    for i in range(1, 4):
        fm90_steps.append({
            "name": f"FatMax 20m #{i}",
            "duration_sec": 1200,  # 20m @ 65~70% FTP (128~137W)
            "low_w": round(ftp * 0.65),
            "high_w": round(ftp * 0.70),
            "intensity": 0,
            "cadence": 90,
            "hr_guide": f"{round(max_hr*0.65)}~{round(max_hr*0.75)} bpm (<138bpm)",
        })
        fm90_steps.append({
            "name": f"20s Surge #{i}",
            "duration_sec": 20,    # 20s @ 117~133% FTP (230~260W)
            "low_w": round(ftp * 1.17),
            "high_w": round(ftp * 1.33),
            "intensity": 0,
            "cadence": 95,
            "hr_guide": "20초 신경근 자극 (무산소 젖산 축적 전 종료)",
        })
    fm90_steps.append({
        "name": "FatMax Finish",
        "duration_sec": 540,       # 9m @ 65~70% FTP (128~137W)
        "low_w": round(ftp * 0.65),
        "high_w": round(ftp * 0.70),
        "intensity": 0,
        "cadence": 90,
        "hr_guide": f"{round(max_hr*0.65)}~{round(max_hr*0.75)} bpm (<138bpm)",
    })
    fm90_steps.append({
        "name": "10m Z1 Cooldown", "duration_sec": 600, "low_w": round(ftp * 0.48), "high_w": round(ftp * 0.56), "intensity": 3, "cadence": 85, "hr_guide": "혈압안정 쿨다운 (<125 bpm)"
    })

    # 2. FatMax 60m Standard (10m Warmup + 40m FatMax 125~138W + 10m Z1 Cooldown)
    fm60_steps = [
        {"name": "10m Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.62), "intensity": 2, "cadence": 85, "hr_guide": f"~{round(max_hr*0.65)} bpm"},
        {"name": "FatMax 20m #1", "duration_sec": 1200, "low_w": round(ftp * 0.64), "high_w": round(ftp * 0.68), "intensity": 0, "cadence": 90, "hr_guide": f"{round(max_hr*0.65)}~{round(max_hr*0.72)} bpm"},
        {"name": "FatMax 20m #2", "duration_sec": 1200, "low_w": round(ftp * 0.66), "high_w": round(ftp * 0.70), "intensity": 0, "cadence": 90, "hr_guide": f"{round(max_hr*0.68)}~{round(max_hr*0.75)} bpm (<138bpm)"},
        {"name": "10m Z1 Cooldown", "duration_sec": 600, "low_w": round(ftp * 0.48), "high_w": round(ftp * 0.56), "intensity": 3, "cadence": 85, "hr_guide": "혈압안정 쿨다운 (<125 bpm)"},
    ]

    # 3. Sweet Spot (SST 3 x 12m @ 172~184W + 10m Z1 Cooldown for BP stability)
    sst_steps = [
        {"name": "10m Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.70), "intensity": 2, "cadence": 88, "hr_guide": "~130 bpm"},
    ]
    for i in range(1, 4):
        sst_steps.append({
            "name": f"SST 12m #{i}",
            "duration_sec": 720,  # 12m @ 88~94% FTP (172~184W)
            "low_w": round(ftp * 0.88),
            "high_w": round(ftp * 0.94),
            "intensity": 0,
            "cadence": 90,
            "hr_guide": f"{round(lthr*0.91)}~{round(lthr*0.97)} bpm",
        })
        if i < 3:
            sst_steps.append({
                "name": f"Z1 Rest 4m #{i}",
                "duration_sec": 240,  # 4m
                "low_w": round(ftp * 0.50),
                "high_w": round(ftp * 0.55),
                "intensity": 1,
                "cadence": 85,
                "hr_guide": "< 125 bpm",
            })
    sst_steps.append({
        "name": "10m Z1 Cooldown", "duration_sec": 600, "low_w": round(ftp * 0.48), "high_w": round(ftp * 0.56), "intensity": 3, "cadence": 85, "hr_guide": "알닥톤 맞춤 10분 혈압안정 쿨다운 (<125 bpm)"
    })

    # 4. Z4 Threshold Cruise (4 x 8m @ 188~204W + 10m Z1 Cooldown)
    z4_steps = [
        {"name": "10m Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.72), "intensity": 2, "cadence": 90, "hr_guide": "~132 bpm"},
    ]
    for i in range(1, 5):
        z4_steps.append({
            "name": f"Z4 8m #{i}",
            "duration_sec": 480,  # 8m @ 96~104% FTP (188~204W)
            "low_w": round(ftp * 0.96),
            "high_w": round(ftp * 1.04),
            "intensity": 0,
            "cadence": 92,
            "hr_guide": f"{round(lthr*0.96)}~{lthr+2} bpm",
        })
        if i < 4:
            z4_steps.append({
                "name": f"Z1 Rest 4m #{i}",
                "duration_sec": 240,  # 4m
                "low_w": round(ftp * 0.50),
                "high_w": round(ftp * 0.55),
                "intensity": 1,
                "cadence": 85,
                "hr_guide": "< 125 bpm",
            })
    z4_steps.append({
        "name": "10m Z1 Cooldown", "duration_sec": 600, "low_w": round(ftp * 0.48), "high_w": round(ftp * 0.56), "intensity": 3, "cadence": 85, "hr_guide": "알닥톤 맞춤 10분 혈압안정 쿨다운 (<125 bpm)"
    })

    # 5. Z1 Active Recovery & BP Stability 45m (for fatigued / consecutive ride days)
    rec_steps = [
        {"name": "10m Easy Spin", "duration_sec": 600, "low_w": round(ftp * 0.45), "high_w": round(ftp * 0.52), "intensity": 2, "cadence": 85, "hr_guide": "< 115 bpm (가벼운 관절 이완)"},
        {"name": "25m Z1 Recovery", "duration_sec": 1500, "low_w": round(ftp * 0.48), "high_w": round(ftp * 0.55), "intensity": 0, "cadence": 88, "hr_guide": "110~122 bpm (젖산 제거 및 하체 혈류 촉진)"},
        {"name": "10m BP Cooldown", "duration_sec": 600, "low_w": round(ftp * 0.42), "high_w": round(ftp * 0.50), "intensity": 3, "cadence": 85, "hr_guide": "< 115 bpm (이완기 혈압 안정 마무리)"},
    ]

    t1 = f"🔥 FatMax 90분 서지(Surge) 코스 ({round(ftp*0.65)}~{round(ftp*0.70)}W 크루징 + 20분마다 20초 {round(ftp*1.17)}~{round(ftp*1.33)}W 서지 3회)"
    t2 = f"🔥 FatMax 60분 정석 인도어 코스 ({round(ftp*0.64)}~{round(ftp*0.70)}W 평일 체지방 감량 & 유산소 효율 극대화)"
    t3 = f"⚡ SST 스윗스팟 3x12분 + 10분 혈압안정 쿨다운 ({round(ftp*0.88)}~{round(ftp*0.94)}W + 알닥톤 복용 맞춤 10분 Z1 쿨다운)"
    t4 = f"🚀 Z4 역치 4x8분 (FTP 210W 돌파) + 10분 쿨다운 ({round(ftp*0.96)}~{round(ftp*1.04)}W + 10분 Z1 쿨다운)"
    t5 = f"🌿 Z1 액티브 리커버리 & 혈압안정 45분 회복 코스 ({round(ftp*0.48)}~{round(ftp*0.55)}W 피로 해소 & 오금·혈압 케어)"

    presets_raw = {
        t1: {
            "id": "fatmax_90_surge",
            "short_label": "🔥 FatMax 90분 서지",
            "base_title": t1,
            "title": t1,
            "fit_name": "FatMax_90m_Surge",
            "short_name": "FatMax_90m_Surge",
            "duration_min": 90,
            "est_tss": 69,
            "est_fat_g": 54.5,
            "est_fat_pct": 58,
            "est_carb_g": 88.0,
            "est_kcal": round((ftp * 0.67 * 5400 / 1000.0) * 1.04),
            "aerobic_te": "3.5 (유산소·지방대사 대폭 향상)",
            "anaerobic_te": "1.4 (20초 서지 신경근 자극)",
            "ftp_gain_score": "★★★☆☆ (베이스·EF 극대화)",
            "joint_bp_safety": "🟢 낮음 (권장 회복 ~18시간 / 10분 혈압안정 쿨다운)",
            "summary": f"FatMax 스위트스팟({round(ftp*0.65)}~{round(ftp*0.70)}W, 심박 120~137bpm)으로 20분씩 순항하면서 20분마다 딱 20초간 {round(ftp*1.17)}~{round(ftp*1.33)}W 서지(가속)를 3회 넣어 지루함을 없애고 지방분해 호르몬 분비 및 페달링 효율(EF)을 동시에 끌어올립니다. (예상 지방연소 ~54.5g)",
            "description": f"FatMax 스위트스팟({round(ftp*0.65)}~{round(ftp*0.70)}W, 심박 120~137bpm)으로 20분씩 순항하면서 20분마다 딱 20초간 {round(ftp*1.17)}~{round(ftp*1.33)}W 서지(가속)를 3회 넣어 지루함을 없애고 지방분해 호르몬 분비 및 페달링 효율(EF)을 동시에 끌어올립니다. (예상 지방연소 ~54.5g)",
            "steps": fm90_steps,
        },
        t2: {
            "id": "fatmax_60",
            "short_label": "🔥 FatMax 60분 정석",
            "base_title": t2,
            "title": t2,
            "fit_name": "FatMax_60m",
            "short_name": "FatMax_60m",
            "duration_min": 60,
            "est_tss": 44,
            "est_fat_g": 36.2,
            "est_fat_pct": 61,
            "est_carb_g": 52.0,
            "est_kcal": round((ftp * 0.65 * 3600 / 1000.0) * 1.04),
            "aerobic_te": "2.9 (유산소 지방대사 유지·강화)",
            "anaerobic_te": "0.2 (무산소 피로 없음)",
            "ftp_gain_score": "★★☆☆☆ (유산소 효율 EF 개선)",
            "joint_bp_safety": "🟢 매우 안전 (권장 회복 ~12시간 / 매일 수행 가능)",
            "summary": f"평일 퇴근 후 60분 동안 심박수를 {round(max_hr*0.75)}bpm(138bpm 미만 캡) 이하로 유지하며 {round(ftp*0.64)}~{round(ftp*0.70)}W 구간에서 근손실과 피로 누적 없이 순수 체지방만 태우는 정석 코스입니다. (예상 지방연소 ~36.2g)",
            "description": f"평일 퇴근 후 60분 동안 심박수를 {round(max_hr*0.75)}bpm(138bpm 미만 캡) 이하로 유지하며 {round(ftp*0.64)}~{round(ftp*0.70)}W 구간에서 근손실과 피로 누적 없이 순수 체지방만 태우는 정석 코스입니다. (예상 지방연소 ~36.2g)",
            "steps": fm60_steps,
        },
        t3: {
            "id": "sst_210",
            "short_label": "⚡ SST 스윗스팟 3x12분",
            "base_title": t3,
            "title": t3,
            "fit_name": "SST_3x12m",
            "short_name": "SST_3x12m",
            "duration_min": 64,
            "est_tss": 66,
            "est_fat_g": 24.5,
            "est_fat_pct": 34,
            "est_carb_g": 107.0,
            "est_kcal": round((ftp * 0.78 * 3840 / 1000.0) * 1.04),
            "aerobic_te": "3.8 (젖산역치 및 유산소 파워 향상)",
            "anaerobic_te": "0.8 (역치 하단 자극)",
            "ftp_gain_score": "★★★★☆ (FTP 210W 돌파 핵심)",
            "joint_bp_safety": "🟡 보통 (권장 회복 ~24시간 / 케이던스 88~90rpm 유지)",
            "summary": f"FTP 196W → 210W 달성을 위한 핵심 스윗스팟 훈련! {round(ftp*0.88)}~{round(ftp*0.94)}W 구간을 12분씩 3세트 소화하고, 알닥톤 복용 중 운동 후 이완기 혈압 반등을 예방하는 10분 Z1 혈압안정 쿨다운(심박 125bpm 이하)으로 마무리합니다.",
            "description": f"FTP 196W → 210W 달성을 위한 핵심 스윗스팟 훈련! {round(ftp*0.88)}~{round(ftp*0.94)}W 구간을 12분씩 3세트 소화하고, 알닥톤 복용 중 운동 후 이완기 혈압 반등을 예방하는 10분 Z1 혈압안정 쿨다운(심박 125bpm 이하)으로 마무리합니다.",
            "steps": sst_steps,
        },
        t4: {
            "id": "z4_threshold",
            "short_label": "🚀 Z4 역치 4x8분",
            "base_title": t4,
            "title": t4,
            "fit_name": "Z4_4x8m",
            "short_name": "Z4_4x8m",
            "duration_min": 64,
            "est_tss": 74,
            "est_fat_g": 18.8,
            "est_fat_pct": 25,
            "est_carb_g": 126.0,
            "est_kcal": round((ftp * 0.81 * 3840 / 1000.0) * 1.04),
            "aerobic_te": "4.2 (FTP 및 젖산 제거 한계 돌파)",
            "anaerobic_te": "1.6 (고강도 역치 자극)",
            "ftp_gain_score": "★★★★★ (FTP 210W 직접 돌파 1위)",
            "joint_bp_safety": "🟠 주의 (권장 회복 ~36시간 / 전해질 600ml+ 필수)",
            "summary": f"실측 LTHR({lthr}bpm) 부근에서 젖산 제거 능력과 FTP(210W 돌파)를 직접 끌어올리는 {round(ftp*0.96)}~{round(ftp*1.04)}W 8분 인터벌 4세트 + 10분 Z1 혈압안정 쿨다운입니다.",
            "description": f"실측 LTHR({lthr}bpm) 부근에서 젖산 제거 능력과 FTP(210W 돌파)를 직접 끌어올리는 {round(ftp*0.96)}~{round(ftp*1.04)}W 8분 인터벌 4세트 + 10분 Z1 혈압안정 쿨다운입니다.",
            "steps": z4_steps,
        },
        t5: {
            "id": "z1_recovery",
            "short_label": "🌿 Z1 회복 45분",
            "base_title": t5,
            "title": t5,
            "fit_name": "Z1_Recovery_45m",
            "short_name": "Z1_Recovery_45m",
            "duration_min": 45,
            "est_tss": 21,
            "est_fat_g": 21.8,
            "est_fat_pct": 66,
            "est_carb_g": 25.0,
            "est_kcal": round((ftp * 0.50 * 2700 / 1000.0) * 1.04),
            "aerobic_te": "1.8 (젖산 제거 및 모세혈관 회복)",
            "anaerobic_te": "0.0 (관절·신경계 휴식)",
            "ftp_gain_score": "★☆☆☆☆ (초과회복 유도)",
            "joint_bp_safety": "🟢 최상 (오금 이완 & 이완기 혈압 안정화 특화)",
            "summary": f"연속 주행 피로 누적일이나 고강도 훈련 다음 날, 좌측 오금 부담을 덜고 하체 모세혈관 혈류를 촉진하여 피로도(ATL)와 이완기 혈압을 안정시키는 {round(ftp*0.48)}~{round(ftp*0.55)}W 45분 회복 코스입니다.",
            "description": f"연속 주행 피로 누적일이나 고강도 훈련 다음 날, 좌측 오금 부담을 덜고 하체 모세혈관 혈류를 촉진하여 피로도(ATL)와 이완기 혈압을 안정시키는 {round(ftp*0.48)}~{round(ftp*0.55)}W 45분 회복 코스입니다.",
            "steps": rec_steps,
        },
    }
    # Scale fat_g and carb_g dynamically if user changes FTP from baseline 196W
    ftp_scale = max(100.0, float(ftp)) / 196.0
    for w_item in presets_raw.values():
        w_item["est_fat_g"] = round(w_item["est_fat_g"] * ftp_scale, 1)
        w_item["est_carb_g"] = round(w_item["est_carb_g"] * ftp_scale, 1)
    return presets_raw


def get_ranked_preset_workouts(
    ftp: float,
    lthr: int = 161,
    max_hr: int = 184,
    tsb: float = 0.0,
    ctl: float = 0.0,
    atl: float = 0.0,
    recent_activities: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Dynamically ranks the 5 preset workouts (1st to 5th) based on the athlete's real-time
    physiological state (TSB, ATL, CTL, recent hard rides, and consecutive riding days),
    and computes simulated tomorrow CTL/ATL/TSB for each course.
    """
    raw_presets = get_preset_workouts(ftp=ftp, lthr=lthr, max_hr=max_hr)
    by_id = {v["id"]: v for v in raw_presets.values()}

    # Analyze recent 3 days of rides if available
    recent_hard_tss = 0.0
    consec_days = 0
    if recent_activities:
        ride_dates = sorted({a.get("start_date_local", "")[:10] for a in recent_activities if a.get("start_date_local")}, reverse=True)
        if ride_dates:
            latest_act = recent_activities[0]
            recent_hard_tss = float(latest_act.get("tss") or 0.0)
            try:
                d0 = datetime.strptime(ride_dates[0], "%Y-%m-%d").date()
                today_d = datetime.now(timezone.utc).astimezone().date()
                if (today_d - d0).days <= 1:
                    consec_days = 1
                    for idx_d in range(1, len(ride_dates)):
                        prev_d = datetime.strptime(ride_dates[idx_d], "%Y-%m-%d").date()
                        if (d0 - prev_d).days == 1:
                            consec_days += 1
                            d0 = prev_d
                        else:
                            break
            except Exception:
                pass

    # Determine ranking order by physiological state
    if tsb < -12 or consec_days >= 4 or (atl - ctl) > 15:
        order = ["z1_recovery", "fatmax_60", "fatmax_90_surge", "sst_210", "z4_threshold"]
        reason = (
            f"현재 피로도(ATL `{atl:.1f}`) 누적 또는 연속 주행(`{consec_days}일`) 영향으로 컨디션(`TSB {tsb:+.1f}`) 관리가 최우선입니다. "
            f"**🌿 Z1 액티브 리커버리** 또는 **🔥 FatMax 60분**을 1·2순위로 배치했습니다."
        )
    elif recent_hard_tss >= 75 and tsb < 8:
        order = ["fatmax_60", "fatmax_90_surge", "z1_recovery", "sst_210", "z4_threshold"]
        reason = (
            f"직전 라이딩 부하(`{recent_hard_tss:.0f} TSS`)가 높았으므로 오늘은 관절·신경계 회복과 체지방 연소를 겸하는 "
            f"**🔥 FatMax 60분 정석 코스**를 추천 1순위로 자동 배치했습니다."
        )
    elif tsb >= 8:
        order = ["sst_210", "fatmax_90_surge", "z4_threshold", "fatmax_60", "z1_recovery"]
        reason = (
            f"컨디션 폼(`TSB {tsb:+.1f}`)이 매우 상쾌한 골든타임입니다! FTP 210W 돌파 자극을 주는 **⚡ SST 스윗스팟 3x12분**과 "
            f"지방연소·서지를 결합한 **🔥 FatMax 90분 서지 코스**를 1·2순위로 자동 배치했습니다."
        )
    elif tsb >= 0:
        order = ["fatmax_90_surge", "sst_210", "fatmax_60", "z4_threshold", "z1_recovery"]
        reason = (
            f"체력(`CTL {ctl:.1f}`)과 피로도(`ATL {atl:.1f}`)가 이상적인 균형(`TSB {tsb:+.1f}`) 상태입니다. "
            f"체지방 감량과 유산소 엔진을 동시에 극대화하는 **🔥 FatMax 90분 서지 코스**를 1순위로 자동 배치했습니다."
        )
    else:
        order = ["fatmax_60", "fatmax_90_surge", "sst_210", "z1_recovery", "z4_threshold"]
        reason = (
            f"현재 효과적인 훈련 부하 구간(`TSB {tsb:+.1f}`)이므로 피로 과누적 없이 지방 대사율을 높이는 "
            f"**🔥 FatMax 60분 / 90분 코스**를 1·2순위로 자동 배치했습니다."
        )

    medals = ["🥇 [추천 1순위]", "🥈 [추천 2순위]", "🥉 [추천 3순위]", "4순위:", "5순위:"]
    ranked_dict = {}
    alpha_ctl = 2.0 / (42.0 + 1.0)
    alpha_atl = 2.0 / (7.0 + 1.0)

    for rank_idx, wid in enumerate(order):
        item = dict(by_id[wid])
        labeled_title = f"{medals[rank_idx]} {item['base_title']}"
        item["title"] = labeled_title
        item["rank"] = rank_idx + 1
        item["rank_badge"] = medals[rank_idx]

        # Simulate tomorrow's CTL, ATL, TSB if this workout is completed today
        w_tss = float(item["est_tss"])
        next_ctl = ctl + (w_tss - ctl) * alpha_ctl
        next_atl = atl + (w_tss - atl) * alpha_atl
        next_tsb = next_ctl - next_atl
        item["next_ctl"] = round(next_ctl, 1)
        item["next_atl"] = round(next_atl, 1)
        item["next_tsb"] = round(next_tsb, 1)
        item["ctl_delta"] = round(next_ctl - ctl, 1)
        item["atl_delta"] = round(next_atl - atl, 1)
        item["tsb_delta"] = round(next_tsb - tsb, 1)

        ranked_dict[labeled_title] = item

    return {
        "workouts": ranked_dict,
        "reason": reason,
    }
