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
    Includes FatMax 60m, FatMax 90m Surge, SST 210W Breakthrough, and Z4 Threshold.
    """
    # 1. FatMax 60m Standard
    fm60_steps = [
        {"name": "Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.65), "intensity": 2, "cadence": 85, "hr_guide": f"~{round(max_hr*0.65)} bpm"},
        {"name": "FatMax Core 1", "duration_sec": 1200, "low_w": round(ftp * 0.64), "high_w": round(ftp * 0.68), "intensity": 0, "cadence": 90, "hr_guide": f"{round(max_hr*0.66)}~{round(max_hr*0.72)} bpm"},
        {"name": "FatMax Peak", "duration_sec": 1500, "low_w": round(ftp * 0.68), "high_w": round(ftp * 0.72), "intensity": 0, "cadence": 90, "hr_guide": f"{round(max_hr*0.70)}~{round(max_hr*0.75)} bpm (<138bpm)"},
        {"name": "Z1 Cooldown", "duration_sec": 300, "low_w": round(ftp * 0.45), "high_w": round(ftp * 0.55), "intensity": 3, "cadence": 85, "hr_guide": "< 115 bpm"},
    ]

    # 2. FatMax 90m Surge (5 x [14m30s @ 65% + 30s @ 85%])
    fm90_steps = [
        {"name": "Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.65), "intensity": 2, "cadence": 85, "hr_guide": f"~{round(max_hr*0.65)} bpm"},
    ]
    for i in range(1, 6):
        fm90_steps.append({
            "name": f"FatMax Z2 #{i}",
            "duration_sec": 870,  # 14m 30s
            "low_w": round(ftp * 0.63),
            "high_w": round(ftp * 0.67),
            "intensity": 0,
            "cadence": 90,
            "hr_guide": f"{round(max_hr*0.65)}~{round(max_hr*0.74)} bpm",
        })
        fm90_steps.append({
            "name": f"30s Surge #{i}",
            "duration_sec": 30,   # 30s at 85% FTP
            "low_w": round(ftp * 0.83),
            "high_w": round(ftp * 0.87),
            "intensity": 0,
            "cadence": 95,
            "hr_guide": "카테콜아민 분비 자극 (무산소 역치 미만)",
        })
    fm90_steps.append({
        "name": "Z1 Cooldown", "duration_sec": 300, "low_w": round(ftp * 0.45), "high_w": round(ftp * 0.55), "intensity": 3, "cadence": 85, "hr_guide": "< 115 bpm"
    })

    # 3. Sweet Spot (SST 3 x 12m + 10m Z1 Cooldown for BP stability)
    sst_steps = [
        {"name": "Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.70), "intensity": 2, "cadence": 88, "hr_guide": "~130 bpm"},
    ]
    for i in range(1, 4):
        sst_steps.append({
            "name": f"SST #{i} (90%)",
            "duration_sec": 720,  # 12m
            "low_w": round(ftp * 0.88),
            "high_w": round(ftp * 0.93),
            "intensity": 0,
            "cadence": 90,
            "hr_guide": f"{round(lthr*0.91)}~{round(lthr*0.97)} bpm ({round(lthr*0.91)}~{round(lthr*0.97)}bpm)",
        })
        if i < 3:
            sst_steps.append({
                "name": f"Recovery #{i}",
                "duration_sec": 240,  # 4m
                "low_w": round(ftp * 0.50),
                "high_w": round(ftp * 0.55),
                "intensity": 1,
                "cadence": 85,
                "hr_guide": "< 125 bpm",
            })
    sst_steps.append({
        "name": "BP Z1 Cooldown", "duration_sec": 600, "low_w": round(ftp * 0.48), "high_w": round(ftp * 0.55), "intensity": 3, "cadence": 85, "hr_guide": "10분 혈압안정 쿨다운 (<115 bpm)"
    })

    # 4. Z4 Threshold Cruise (4 x 8m @ 96~100% FTP + 10m Z1 Cooldown)
    z4_steps = [
        {"name": "Warmup", "duration_sec": 600, "low_w": round(ftp * 0.50), "high_w": round(ftp * 0.72), "intensity": 2, "cadence": 90, "hr_guide": "~132 bpm"},
    ]
    for i in range(1, 5):
        z4_steps.append({
            "name": f"Z4 FTP #{i}",
            "duration_sec": 480,  # 8m
            "low_w": round(ftp * 0.96),
            "high_w": round(ftp * 1.01),
            "intensity": 0,
            "cadence": 92,
            "hr_guide": f"{round(lthr*0.96)}~{lthr+2} bpm",
        })
        if i < 4:
            z4_steps.append({
                "name": f"Rest #{i}",
                "duration_sec": 240,  # 4m
                "low_w": round(ftp * 0.50),
                "high_w": round(ftp * 0.55),
                "intensity": 1,
                "cadence": 85,
                "hr_guide": "< 125 bpm",
            })
    z4_steps.append({
        "name": "BP Z1 Cooldown", "duration_sec": 600, "low_w": round(ftp * 0.48), "high_w": round(ftp * 0.55), "intensity": 3, "cadence": 85, "hr_guide": "10분 혈압안정 쿨다운 (<115 bpm)"
    })

    return {
        "🔥 [추천 1순위] FatMax 90분 서지(Surge) 지방연소 극대화 코스": {
            "id": "fatmax_90_surge",
            "title": "🔥 [추천 1순위] FatMax 90분 서지(Surge) 지방연소 극대화 코스",
            "fit_name": "FatMax_90m_Surge",
            "short_name": "FatMax_90m_Surge",
            "duration_min": 90,
            "est_tss": 69,
            "est_fat_g": 54,
            "summary": f"14분 30초 FatMax({round(ftp*0.65)}W) 순항 후 30초 중강도 서지({round(ftp*0.85)}W)를 5회 반복하여 지루함을 없애고 카테콜아민(지방분해 호르몬) 분비와 후연소 효과를 극대화합니다. (예상 지방연소 ~54g, 69 TSS)",
            "description": f"14분 30초 FatMax({round(ftp*0.65)}W) 순항 후 30초 중강도 서지({round(ftp*0.85)}W)를 5회 반복하여 지루함을 없애고 카테콜아민(지방분해 호르몬) 분비와 후연소 효과를 극대화합니다. (예상 지방연소 ~54g, 69 TSS)",
            "steps": fm90_steps,
        },
        "🔥 [평일 60분] FatMax 정석 유산소 지방연소 코스": {
            "id": "fatmax_60",
            "title": "🔥 [평일 60분] FatMax 정석 유산소 지방연소 코스",
            "fit_name": "FatMax_60m",
            "short_name": "FatMax_60m",
            "duration_min": 60,
            "est_tss": 44,
            "est_fat_g": 36,
            "summary": f"평일 퇴근 후 60분 동안 심박수를 {round(max_hr*0.75)}bpm(138bpm) 이하로 유지하며 {round(ftp*0.66)}W → {round(ftp*0.70)}W 구간에서 순수 체지방 연소율을 최고조로 유지합니다. (예상 지방연소 ~36g, 44 TSS)",
            "description": f"평일 퇴근 후 60분 동안 심박수를 {round(max_hr*0.75)}bpm(138bpm) 이하로 유지하며 {round(ftp*0.66)}W → {round(ftp*0.70)}W 구간에서 순수 체지방 연소율을 최고조로 유지합니다. (예상 지방연소 ~36g, 44 TSS)",
            "steps": fm60_steps,
        },
        "⚡ [FTP 210W 돌파] 스윗스팟 (SST 3x12분 + 10분 혈압안정 쿨다운)": {
            "id": "sst_210",
            "title": "⚡ [FTP 210W 돌파] 스윗스팟 (SST 3x12분 + 10분 혈압안정 쿨다운)",
            "fit_name": "SST_3x12m",
            "short_name": "SST_3x12m",
            "duration_min": 64,
            "est_tss": 66,
            "est_fat_g": 24,
            "summary": f"FTP 196W → 210W 달성을 위한 핵심 훈련! {round(ftp*0.88)}~{round(ftp*0.93)}W 구간을 12분씩 3세트 소화하고, 운동 후 이완기 혈압 반등을 막는 10분 Z1 쿨다운으로 마무리합니다. (66 TSS)",
            "description": f"FTP 196W → 210W 달성을 위한 핵심 훈련! {round(ftp*0.88)}~{round(ftp*0.93)}W 구간을 12분씩 3세트 소화하고, 운동 후 이완기 혈압 반등을 막는 10분 Z1 쿨다운으로 마무리합니다. (66 TSS)",
            "steps": sst_steps,
        },
        "🚀 [역치 강화] Z4 크루즈 인터벌 (4x8분 + 10분 혈압안정 쿨다운)": {
            "id": "z4_threshold",
            "title": "🚀 [역치 강화] Z4 크루즈 인터벌 (4x8분 + 10분 혈압안정 쿨다운)",
            "fit_name": "Z4_4x8m",
            "short_name": "Z4_4x8m",
            "duration_min": 64,
            "est_tss": 74,
            "est_fat_g": 19,
            "summary": f"LTHR({lthr}bpm) 부근에서 젖산 제거 능력을 극대화하는 {round(ftp*0.96)}~{round(ftp*1.01)}W 8분 인터벌 4세트 + 10분 Z1 혈압안정 쿨다운입니다. (74 TSS)",
            "description": f"LTHR({lthr}bpm) 부근에서 젖산 제거 능력을 극대화하는 {round(ftp*0.96)}~{round(ftp*1.01)}W 8분 인터벌 4세트 + 10분 Z1 혈압안정 쿨다운입니다. (74 TSS)",
            "steps": z4_steps,
        },
    }
