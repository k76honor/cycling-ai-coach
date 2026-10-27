"""
Athlete Memory and Chat History Manager
Provides persistent storage for conversation history, physiological profile, and coach notes across sessions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

MEMORY_FILE = Path(__file__).resolve().parent / "athlete_memory.json"

DEFAULT_PROFILE = {
    "ftp": 196,
    "weight_kg": 64.5,
    "max_hr": 184,
    "lthr": 161,
    "rest_hr": 49,
}

DEFAULT_NOTES = [
    "생년월일 및 신체 프로필: 1976년 1월 9일생(만 50세 남성, 168cm), 체중 66.6kg -> 64.5kg(-2.1kg 감량), 실전 권장 FTP 196W(3.04 W/kg, 단기 목표 210W+), LTHR 161bpm, 최대심박 184bpm, 안정시 심박 49~55bpm",
    "기기 및 누적 베이스: Garmin Edge 1040 + Forerunner 955 + 양발 파워미터 + OMRON 혈압계 (최근 8주 40회 2,061.1km, 획득고도 18,105m, 총 4,836.5 TSS / 피크 MMP: 1초 737W, 1분 353W, 5분 238W, 20분 189W[NP 203W])",
    "훈련 철학 (심박수 150bpm 캡 금지 & 파워 향상 우선): 2026년 4월 심장초음파 정밀검진 정상(운동선수 심장의 전기적 위양성). 무리한 150bpm 하드캡 대신 LTHR 161bpm / Max 184bpm 기준으로 평일 SST(173~186W)·Z4 역치(186~206W) 파워 향상 훈련을 적극 수행하되 종료 전 10분 Z1 쿨다운 필수",
    "혈압(모닝 서지) & 알닥톤(25mg) 수분 프로토콜: 이완기 혈압 상승(90~100mmHg)의 주원인은 4~6일 연속 주행 피로 및 알닥톤 복용 중 장거리 탈수(10/25 94km 직후 140/91 -> 수분 보충 후 익일 아침 이완기 78 회복). 장거리 시 15~20분마다 150~200ml 전해질·수분 섭취 및 주 1~2회 완전 휴식(Zero Bike) 준수",
    "피팅 & 좌측 오금(슬와근) 건 보호: 클릿 최후방 한계 세팅 + 좌측 페달링 편중(10월 53~53.5% L : 46.5~47% R)으로 하사점(6시) 좌측 무릎 뒤 당김 이력 -> 안장 높이 단독 2~3mm 하향 권장 및 오른발 12시~3시 지그시 밟기 큐잉",
]


def load_memory() -> Dict[str, Any]:
    """Loads persistent athlete memory, profile, and messages from disk."""
    if not MEMORY_FILE.exists():
        initial_data = {
            "profile": DEFAULT_PROFILE.copy(),
            "athlete_notes": DEFAULT_NOTES.copy(),
            "messages": [],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_memory(initial_data)
        return initial_data

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        if "profile" not in data or not isinstance(data["profile"], dict):
            data["profile"] = DEFAULT_PROFILE.copy()
        else:
            for k, v in DEFAULT_PROFILE.items():
                data["profile"].setdefault(k, v)
        if "athlete_notes" not in data:
            data["athlete_notes"] = DEFAULT_NOTES.copy()
        if "messages" not in data:
            data["messages"] = []
        return data
    except Exception as e:
        print(f"Error loading memory: {e}")
        return {
            "profile": DEFAULT_PROFILE.copy(),
            "athlete_notes": DEFAULT_NOTES.copy(),
            "messages": [],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def get_athlete_profile() -> Dict[str, Any]:
    """Retrieves saved athlete profile (FTP, Weight, Max HR, LTHR, Rest HR)."""
    mem = load_memory()
    prof = mem.get("profile", DEFAULT_PROFILE.copy())
    for k, v in DEFAULT_PROFILE.items():
        prof.setdefault(k, v)
    return prof


def update_athlete_profile(
    ftp: int,
    weight_kg: float,
    max_hr: Optional[int] = None,
    lthr: Optional[int] = None,
    rest_hr: Optional[int] = None,
) -> None:
    """Updates and saves athlete profile to persistent storage."""
    mem = load_memory()
    prof = mem.get("profile", DEFAULT_PROFILE.copy())
    prof["ftp"] = int(ftp)
    prof["weight_kg"] = float(weight_kg)
    if max_hr is not None:
        prof["max_hr"] = int(max_hr)
    if lthr is not None:
        prof["lthr"] = int(lthr)
    if rest_hr is not None:
        prof["rest_hr"] = int(rest_hr)
    mem["profile"] = prof
    save_memory(mem, sync_github=True)


def save_memory(data: Dict[str, Any], sync_github: bool = False) -> None:
    """Saves memory data to JSON file (and syncs to GitHub if configured when profile/notes change)."""
    try:
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if sync_github:
            from garmin_loader import sync_file_to_github_if_configured
            sync_file_to_github_if_configured(MEMORY_FILE, "Auto-sync updated athlete profile & notes")
    except Exception as e:
        print(f"Error saving memory: {e}")


def add_athlete_note(note: str) -> None:
    """Adds a key athlete note (injury, goal, preference) to persistent memory."""
    note = note.strip()
    if not note:
        return
    mem = load_memory()
    if note not in mem["athlete_notes"]:
        mem["athlete_notes"].append(note)
        save_memory(mem)


def remove_athlete_note(index: int) -> None:
    """Removes an athlete note by index."""
    mem = load_memory()
    if 0 <= index < len(mem["athlete_notes"]):
        mem["athlete_notes"].pop(index)
        save_memory(mem)


def append_chat_message(role: str, content: str) -> None:
    """Appends a new chat message to persistent history."""
    mem = load_memory()
    mem["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    if len(mem["messages"]) > 60:
        mem["messages"] = mem["messages"][-60:]
    save_memory(mem)


def clear_chat_history() -> None:
    """Clears chat history while keeping athlete notes intact."""
    mem = load_memory()
    mem["messages"] = []
    save_memory(mem)


def auto_extract_athlete_insight(agent, user_message: str, bot_response: str) -> None:
    """
    Uses Gemini to extract important permanent insights about the athlete
    and adds to notes.
    """
    if not agent or not agent.is_available():
        return

    prompt = f"""
다음은 사이클리스트와 AI 코치 간의 대화입니다.
대화 내용 중 코치가 앞으로 장기적으로 기억해야 할 선수의 '새로운 영구적 정보(목표 대회/일정, 특정 부상이나 통증 부위, 신체적 제약, 기어비/자전거 변경 등)'가 있는지 확인하세요.

[사용자 질문]: {user_message}
[코치 답변]: {bot_response}

만약 기억해둘 만한 핵심 정보가 있다면 한국어로 1줄(40자 내외)의 간결한 메모 문장으로만 출력하세요.
기억해둘 만한 새로운 사실이 없거나 단순 일상 잡담/일회성 계산 질문이라면 아무것도 쓰지 말고 오직 'NONE'이라고만 답하세요.
"""
    try:
        resp = agent._call_gemini(prompt, temperature=0.1)
        if resp:
            insight = resp.strip()
            if insight and insight != "NONE" and "NONE" not in insight and len(insight) > 4:
                add_athlete_note(insight)
    except Exception as e:
        print(f"Auto-extract insight error: {e}")
