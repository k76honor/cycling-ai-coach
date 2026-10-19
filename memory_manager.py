"""
Athlete Memory and Chat History Manager
Provides persistent storage for conversation history and coach notes across sessions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

MEMORY_FILE = Path(__file__).resolve().parent / "athlete_memory.json"

DEFAULT_NOTES = [
    "목표: FTP 향상 및 그란폰도 완주",
    "신체 특이사항: 무릎 통증 예방을 위해 가벼운 기어비 및 케이던스 85rpm 이상 유지 권장",
]


def load_memory() -> Dict[str, Any]:
    """Loads persistent athlete memory and messages from disk."""
    if not MEMORY_FILE.exists():
        initial_data = {
            "athlete_notes": DEFAULT_NOTES.copy(),
            "messages": [],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_memory(initial_data)
        return initial_data

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        if "athlete_notes" not in data:
            data["athlete_notes"] = DEFAULT_NOTES.copy()
        if "messages" not in data:
            data["messages"] = []
        return data
    except Exception as e:
        print(f"Error loading memory: {e}")
        return {
            "athlete_notes": DEFAULT_NOTES.copy(),
            "messages": [],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def save_memory(data: Dict[str, Any]) -> None:
    """Saves memory data to JSON file."""
    try:
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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
    # Keep up to 60 messages to manage context
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
    (e.g. newly stated injury, target race date, equipment change) and adds to notes.
    """
    if not agent or not agent.is_available():
        return

    prompt = f"""
다음은 사이클리스트와 AI 코치 간의 대화입니다.
대화 내용 중 코치가 앞으로 장기적으로 기억해야 할 선수의 '새로운 영구적 정보(목표 대회/일정, 특정 부상이나 통증 부위, 신체적 제약, 기어비/자전거 변경 등)'가 있는지 확인하세요.

[사용자 질문]: {user_message}
[코치 답변]: {bot_response}

만약 기억해둘 만한 핵심 정보가 있다면 한국어로 1줄(30자 내외)의 간결한 메모 문장으로만 출력하세요. (예: "2026년 5월 설악 그란폰도 208km 완주 목표", "안장 높이 조절 후 햄스트링 당김 증상 있음")
기억해둘 만한 새로운 사실이 없거나 단순 일상 잡담/일회성 계산 질문이라면 아무것도 쓰지 말고 오직 'NONE'이라고만 답하세요.
"""
    try:
        resp = agent.client.models.generate_content(
            model=agent.model_name,
            contents=prompt,
            config={"temperature": 0.1},
        )
        insight = resp.text.strip()
        if insight and insight != "NONE" and "NONE" not in insight and len(insight) > 4:
            add_athlete_note(insight)
    except Exception as e:
        print(f"Auto-extract insight error: {e}")
