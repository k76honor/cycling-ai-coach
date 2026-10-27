"""
AI Cycling Coach Agent
Powered by Google Gemini (gemini-3.8-flash) & Cycling Physiology Expertise
(Coggan Power Zones, Friel Periodization, Seiler Polarized Training).
"""

from typing import List, Dict, Any, Optional
import json
import pandas as pd
from power_metrics import get_power_zones, interpret_tsb


COACH_SYSTEM_PROMPT = """
당신은 라이더님(1976년 1월 9일생, 만 50세 남성, 168cm, 64.5kg)의 전담 스포츠 생리학자이자 사이클링 파워 코치(AI Cycling Coach)입니다.
앤드류 코건(Andrew Coggan) 박사의 7단계 파워존 훈련 이론, 조 프리엘(Joe Friel)의 주기화 이론, 스티븐 세일러(Stephen Seiler)의 양극화 훈련 원칙, 그리고 심혈관·바이오메카닉스 데이터를 통합하여 코칭합니다.

[라이더 핵심 생리학·의학·피팅 프로필 (반드시 숙지할 것)]
1. 기본 프로필 및 파워 역치:
   - 생년월일: 1976년 1월 9일생 (만 50세 남성), 체중 64.5kg (9월 66.6kg에서 2.1kg 감량)
   - 현재 기준 FTP: 196W (3.04 W/kg, 잠재 올아웃 200~205W / 단기 목표 210W+)
   - 심박 지표: 젖산역치 심박수(LTHR) 161 bpm, 최대심박수(Max HR) 184 bpm (실측 1분 평균 181.8bpm 검증 완료), 안정시 심박수 49~55 bpm
   - 보유 장비: Garmin Edge 1040, Forerunner 955 / 가슴 심박계, 양발형 파워미터, OMRON 블루투스 혈압계

2. [중요 교정 지침] 심박수 캡(150 bpm) 강요 금지 & 파워 향상 중심 코칭:
   - 라이더님은 2026년 4월 심장초음파(Echocardiography) 정밀 검진에서 구조적/해부학적 심근 비후가 전혀 없는 '정상(운동선수 심장의 전기적 위양성)' 판정을 받았습니다.
   - 라이더님의 목표는 **지속 순항 파워(TT) 및 FTP 향상(196W -> 210W+)**입니다.
   - 과거 잘못된 설정(LTHR 150bpm 오인)처럼 **무조건 심박수 150 bpm 이하로 타라고 하드캡(Hard Cap)을 씌우지 마세요!** LTHR이 161 bpm이고 최대심박이 184 bpm이므로, 150 bpm으로 제한하면 Sweet Spot(173~186W, 심박 148~158bpm)이나 Z4 역치(186~206W, 심박 156~165bpm), VO2max(210~235W, 심박 165~175bpm) 훈련이 불가능해져 파워가 오르지 않습니다.
   - 평일 인도어/단거리 파워 향상 워크아웃에서는 목표 와트(Sweet Spot, Z4 Threshold, VO2max)를 적극적으로 밟도록 지도하되, 운동 후 혈압 반등 예방을 위해 **종료 전 10분 이상 Z1(100~110W, 115bpm 이하) 쿨다운**을 필수로 넣으세요.
   - 주말 90~100km+ 장거리 투어에서만 후반 탈수·교감신경 과부하 방지를 위해 평지/순항 시 Z2~Z3(심박 125~148bpm) 효율 주행을 권장하고, 언덕에서는 LTHR(161bpm) 내외까지 자연스럽게 사용하는 유연한 페이싱을 제시하세요.

3. 혈압(모닝 서지 & 이완기 혈압) 및 알닥톤(Spironolactone 25mg) 실전 피드백:
   - 라이더님은 **알닥톤 25mg을 아침에 평소대로 복용하며, 실제 라이딩해 본 결과 운동 중 퍼포먼스나 컨디션에 별다른 영향이 없음**을 확인했습니다. 따라서 알닥톤 복용을 이유로 겁을 주거나 까다로운 제약(까치발 운동 강요 등)을 두지 마세요.
   - 이완기 혈압 상승(90~100 mmHg)은 휴식 없는 4~6일 연속 주행 피로 누적이나 3시간 반 이상 장거리 주행 시 수분 부족에서 기인하므로, 장거리 주행 시 기본적인 수분·전해질 섭취와 주 1~2회 휴식일 배분만 자연스럽게 챙기세요.

4. 바이오메카닉스(클릿 최후방 세팅 & 안장 높이 미조정 상태):
   - 현재 클릿을 신발 뒤꿈치 방향 **최후방 한계선**까지 밀어둔 상태이며, **안장 높이는 아직 낮추지 않은(조정 전) 상태**입니다.
   - 장거리 주행에서 좌우 밸런스가 **53~53.5% L / 46.5~47% R**로 왼쪽에 쏠릴 때 왼쪽 무릎 뒤(오금/슬와근) 당김이 발생한 적이 있으므로, 향후 라이딩에서 오금 당김이 재발하면 안장 높이를 2~3mm 낮추도록 안내하고 평소에는 오른발 12시~3시 방향 페달링 밸런스(50:50)와 업힐 케이던스(75rpm+)에 집중하도록 코칭하세요.

[응답 스타일]
- 모호한 말 대신 정확한 목표 파워(W), 케이던스(RPM), 심박 구간(bpm), 수분 섭취량(ml), 안장 조정 수치(mm)를 제시하세요.
- 라이더님의 파워 향상 열정을 적극 지지하면서, 혈압·수분·무릎 건 보호를 과학적으로 뒷받침하는 든든한 전담 코치 어조로 한국어로 답변하세요.
"""


class CyclingCoachAgent:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Gemini Client: {e}")
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _call_gemini(
        self,
        prompt: str,
        system_instruction: str = COACH_SYSTEM_PROMPT,
        temperature: float = 0.5,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Calls Gemini API with automatic model fallback and optional multimodal attachments (images/PDF/files)."""
        if not self.is_available():
            return None

        candidate_models = [self.model_name]
        for m in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash"]:
            if m not in candidate_models:
                candidate_models.append(m)

        contents_payload: Any = prompt
        if attachments:
            try:
                from google.genai import types
                parts: List[Any] = [prompt]
                for att in attachments:
                    if att.get("data") and att.get("mime_type"):
                        parts.append(
                            types.Part.from_bytes(
                                data=att["data"],
                                mime_type=att["mime_type"],
                            )
                        )
                contents_payload = parts
            except Exception as e:
                print(f"Failed to build multimodal parts: {e}")
                contents_payload = prompt

        for model in candidate_models:
            try:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=contents_payload,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": temperature,
                    },
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                print(f"Gemini call with {model} failed: {e}. Trying next candidate...")
                continue
        return None

    def diagnose_training_state(
        self,
        ftp: float,
        weight_kg: float,
        recent_activities: List[Dict[str, Any]],
        pmc_latest: Dict[str, float],
    ) -> str:
        """
        Generates a comprehensive diagnostic report on recent training load, fitness, and recovery state.
        """
        # Calculate summary statistics
        total_rides = len(recent_activities)
        total_km = sum(act.get("distance", 0) for act in recent_activities) / 1000.0
        total_hours = sum(act.get("moving_time", 0) for act in recent_activities) / 3600.0
        total_tss = sum(act.get("tss", 0) for act in recent_activities)
        avg_tss_per_ride = total_tss / max(1, total_rides)

        ctl = pmc_latest.get("ctl", 0.0)
        atl = pmc_latest.get("atl", 0.0)
        tsb = pmc_latest.get("tsb", 0.0)
        tsb_status = interpret_tsb(tsb)
        w_per_kg = round(ftp / max(30.0, weight_kg), 2)

        prompt = f"""
다음 라이더의 최근 45일간 사이클링 훈련 데이터와 생체 파워 지표를 종합 분석하여 '훈련 진단 리포트'를 작성해주세요.

[라이더 프로필]
- 기준 FTP: {ftp} W (체중 {weight_kg} kg 기준: {w_per_kg} W/kg)
- 최근 라이딩 횟수: 총 {total_rides}회
- 누적 거리 및 시간: {total_km:.1f} km / {total_hours:.1f} 시간
- 누적 TSS: {total_tss:.0f} (라이딩당 평균 TSS: {avg_tss_per_ride:.1f})
- 현재 훈련 지표 (PMC):
  * CTL (장기 체력): {ctl:.1f}
  * ATL (단기 피로도): {atl:.1f}
  * TSB (현재 컨디션/폼): {tsb:+.1f} -> 상태: {tsb_status['status']}

[요청 사항]
1. 📊 현재 컨디션 및 체력 평가 (CTL vs ATL 균형 및 TSB 해석)
2. ⚡ 파워 및 훈련 강도 분배 진단 (볼륨, 강도, 주간 누적 부하의 적절성)
3. 🎯 핵심 강점과 보완점 (지구력 베이스, 젖산역치 유지력, 회복 관리 등)
4. 💡 다음 1~2주간 최우선 집중 권고사항

가독성이 높은 마크다운 형식으로 작성해주세요.
"""
        gemini_resp = self._call_gemini(prompt, temperature=0.4)
        if gemini_resp:
            return gemini_resp

        # Fallback offline diagnostic report
        return f"""### 📊 AI 사이클링 코치 훈련 진단 리포트 (기본 분석 모드)

#### 1. 📈 현재 체력 & 컨디션 상태 (PMC 분석)
- **현재 FTP**: **{ftp}W** ({w_per_kg} W/kg)
- **체력 (CTL)**: **{ctl:.1f}** | **피로도 (ATL)**: **{atl:.1f}**
- **컨디션 지수 (TSB)**: **{tsb:+.1f}** ({tsb_status['status']})
> **코치 진단**: {tsb_status['advice']}

#### 2. 🚴 최근 훈련 볼륨 & 강도
- 최근 분석 라이딩: 총 **{total_rides}회**, 누적 거리 **{total_km:.1f}km**, 주행 시간 **{total_hours:.1f}시간**
- 누적 TSS: **{total_tss:.0f}** (라이딩 1회당 평균 TSS **{avg_tss_per_ride:.1f}**)
- 훈련 강도가 꾸준히 누적되며 체력 베이스(CTL)가 탄탄히 형성되어 가고 있는 단계입니다.

#### 3. 🎯 추천 훈련 방향
1. **유산소 기반(Zone 2) 확보**: 전체 주간 라이딩 시간의 약 70~80%는 56~75% FTP({round(ftp*0.56)}W~{round(ftp*0.75)}W) 영역에서 케이던스 85~95rpm으로 소화하세요.
2. **질적 인터벌(Sweet Spot & Threshold)**: 주 1~2회 {round(ftp*0.88)}W~{round(ftp*0.94)}W 스윗스팟 구간을 타겟팅하여 피로를 최소화하면서 FTP 임계치를 효율적으로 높이세요.
3. **철저한 회복**: TSB가 -20 이하로 급락할 때는 고강도 라이딩을 쉬고 완전 휴식 또는 Z1({round(ftp*0.55)}W 이하) 회복 페달링을 진행하세요.

*(Tip: 사이드바에 Gemini API Key를 입력하시면 실시간 맞춤형 심층 분석을 받으실 수 있습니다.)*
"""

    def generate_training_plan(
        self,
        target_goal: str,
        duration_weeks: int,
        hours_per_week: float,
        ftp: float,
        weight_kg: float,
        current_tsb: float,
    ) -> str:
        """
        Generates a structured multi-week cycling training plan.
        """
        zones = get_power_zones(ftp)
        prompt = f"""
다음 조건의 사이클리스트를 위한 체계적인 {duration_weeks}주간 맞춤형 사이클링 훈련 플랜을 설계해주세요.

[라이더 정보 및 목표]
- 목표 종목/이벤트: {target_goal}
- 훈련 기간: {duration_weeks}주
- 주당 가용 훈련 시간: 약 {hours_per_week}시간
- 현재 FTP: {ftp} W (체중 {weight_kg} kg 기준: {round(ftp/max(30.0, weight_kg), 2)} W/kg)
- 현재 컨디션 폼(TSB): {current_tsb:+.1f}

[파워존 참고 기준]
- Z1 회복: ~ {zones['Z1_Active_Recovery']['range'][1]} W
- Z2 지구력: {zones['Z2_Endurance']['range'][0]} ~ {zones['Z2_Endurance']['range'][1]} W
- Z3 템포: {zones['Z3_Tempo']['range'][0]} ~ {zones['Z3_Tempo']['range'][1]} W
- Sweet Spot: {zones['Sweet_Spot']['range'][0]} ~ {zones['Sweet_Spot']['range'][1]} W
- Z4 젖산역치: {zones['Z4_Threshold']['range'][0]} ~ {zones['Z4_Threshold']['range'][1]} W
- Z5 VO2Max: {zones['Z5_VO2Max']['range'][0]} ~ {zones['Z5_VO2Max']['range'][1]} W

[플랜 작성 규칙]
1. 주기화(Periodization): {duration_weeks}주 전체의 메소스케줄(Base -> Build -> Peak/Taper)을 명시하세요.
2. 대표 주간(Microcycle)의 월~일 일별 워크아웃을 구체적으로 작성하세요.
   - 요일별 훈련 목적 (예: 화-스윗스팟 인터벌, 수-Z2 리커버리, 토-장거리 팩라이딩, 월-휴식 등)
   - 각 워크아웃의 워밍업, 메인 세트(와트, 케이던스, 반복 횟수, 휴식 시간), 쿨다운 상세 스펙
   - 예상 TSS 및 소요 시간
3. 영양 섭취(라이딩 전/중 카보하이드레이트 보충) 및 휴식 팁을 포함하세요.
"""
        gemini_plan = self._call_gemini(prompt, temperature=0.4)
        if gemini_plan:
            return gemini_plan

        # Fallback structured plan template
        z2_low, z2_high = zones['Z2_Endurance']['range']
        ss_low, ss_high = zones['Sweet_Spot']['range']
        vo2_low, vo2_high = zones['Z5_VO2Max']['range']
        
        return f"""### 📅 {target_goal} 대비 {duration_weeks}주 맞춤형 훈련 플랜

#### 1. 🗺️ 주기화 (Periodization) 로드맵
- **1~{max(1, duration_weeks//2)}주차 [Base & Sweet Spot Phase]**: 유산소 모세혈관 및 미토콘드리아 확장, 스윗스팟({ss_low}~{ss_high}W)을 통한 FTP 토대 구축
- **{max(1, duration_weeks//2)+1}~{max(2, duration_weeks-1)}주차 [Build Phase]**: 젖산 내성 및 VO2 Max({vo2_low}~{vo2_high}W) 고강도 인터벌 추가, 실전 레이스 파워 시뮬레이션
- **{duration_weeks}주차 [Peak & Taper Phase]**: 훈련 볼륨은 50%로 축소, 인터벌의 강도는 유지하여 최상의 폼(TSB +5~+15) 달성

---

#### 2. 🚴 표준 주간(Microcycle) 훈련표 (주당 약 {hours_per_week}시간 기준)

| 요일 | 훈련 세션명 | 목표 파워존 & 와트 | 시간 | 예상 TSS | 핵심 포인트 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **월요일** | **완전 휴식 (Rest Day)** | - | - | 0 | 폼롤러 스트레칭 & 고단백 식단 |
| **화요일** | **스윗스팟 인터벌 (SST 3x15m)** | Sweet Spot ({ss_low}~{ss_high}W) | 70분 | ~65 | 케이던스 90rpm 일정하게 유지 |
| **수요일** | **유산소 지구력 라이딩 (Z2)** | Zone 2 ({z2_low}~{z2_high}W) | 90분 | ~55 | 코로 호흡 가능한 대화 페이스 |
| **목요일** | **VO2 Max 힐클라임 인터벌** | Zone 5 ({vo2_low}~{vo2_high}W) | 60분 | ~60 | 3분 파워 전개 x 5회 (휴식 3분) |
| **금요일** | **가벼운 회복 스핀 (Active Recovery)** | Zone 1 (< {round(ftp*0.55)}W) | 45분 | ~20 | 가벼운 기어비로 95rpm 이상 회전 |
| **토요일** | **주말 장거리 엔듀런스 라이딩** | Zone 2~3 ({z2_low}~{zones['Z3_Tempo']['range'][1]}W) | 180분 | ~130 | 45분마다 탄수화물(30~60g) 섭취 |
| **일요일** | **자율 라이딩 또는 휴식** | Zone 1~2 | 60분 | ~30 | 다리 상태에 따라 강도 조절 |

---

#### 3. 📝 화요일 메인 워크아웃 상세 가이드: `Sweet Spot 3 x 15분`
1. **워밍업 (15분)**: 100W에서 180W까지 점진적 빌드업 + 100rpm 스핀업 30초 x 3회
2. **메인 세트 (55분)**:
   - 1세트: **15분**간 **{round((ss_low+ss_high)/2)}W** 유지 (케이던스 88~92rpm) -> 회복 5분 (Z1 120W)
   - 2세트: **15분**간 **{round((ss_low+ss_high)/2)}W** 유지 -> 회복 5분
   - 3세트: **15분**간 **{round((ss_low+ss_high)/2)}W** 유지 -> 회복 5분
3. **쿨다운 (10분)**: 가벼운 페달링으로 심박수 110bpm 이하 안정화

*(Tip: 사이드바에 Gemini API Key를 입력하시면 더 정밀한 개인 맞춤 인터벌 계획을 생성할 수 있습니다.)*
"""

    def consult(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        athlete_context: Dict[str, Any],
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Interactive cycling consultation chatbot.
        Answers user training/fatigue/gear/nutrition questions with live athlete context and multimodal attachments.
        """
        ftp = athlete_context.get("ftp", 196)
        weight_kg = athlete_context.get("weight_kg", 64.5)
        lthr = athlete_context.get("lthr", 161)
        max_hr = athlete_context.get("max_hr", 184)
        ctl = athlete_context.get("ctl", 50.0)
        atl = athlete_context.get("atl", 55.0)
        tsb = athlete_context.get("tsb", -5.0)
        recent_ride = athlete_context.get("last_ride_summary", "최근 라이딩 기록 없음")
        athlete_notes = athlete_context.get("athlete_notes", [])
        notes_str = "\n".join([f"- {n}" for n in athlete_notes]) if athlete_notes else "- 특이사항 없음"

        extra_file_context = ""
        if attachments:
            text_extracts = [a["text_summary"] for a in attachments if a.get("text_summary")]
            binary_names = [a["name"] for a in attachments if a.get("data")]
            if binary_names:
                extra_file_context += f"\n[첨부된 이미지/문서 파일 ({len(binary_names)}개): {', '.join(binary_names)}]\n- 첨부된 이미지(가민/스트라바 캡처, 파워/심박 그래프, 자세/장비 사진 등) 또는 문서를 시각적으로 정밀 분석하여 답변에 반영하세요.\n"
            if text_extracts:
                extra_file_context += "\n[첨부된 데이터 파일(.fit/CSV/JSON) 분석 내용]\n" + "\n".join(text_extracts) + "\n"

        context_prompt = f"""
[코치가 기억하고 있는 라이더 핵심 메모 (장기 기억)]
{notes_str}

[현재 상담 라이더의 실시간 데이터]
- FTP: {ftp}W (체중 {weight_kg}kg, W/kg: {round(ftp/max(1.0, weight_kg), 2)})
- 심박 기준: LTHR {lthr}bpm / 최대심박 {max_hr}bpm
- 현재 PMC 지표: CTL(체력)={ctl:.1f}, ATL(피로도)={atl:.1f}, TSB(컨디션)={tsb:+.1f}
- 가장 최근 라이딩: {recent_ride}
{extra_file_context}
[라이더의 질문]
"{user_message}"

코치님은 위 장기 기억(선수의 목표, 부상 이력, 특이사항), 실시간 파워 데이터, 그리고 첨부된 이미지/파일이 있다면 그 내용을 모두 종합하여 전담 코치로서 맥락이 이어지는 깊이 있는 맞춤 조언을 제공해주세요.
"""

        # Build contents with chat history
        messages = []
        for msg in chat_history[-6:]:  # Keep last 3 turns
            role = "user" if msg["role"] == "user" else "model"
            messages.append(f"{role.upper()}: {msg['content']}")
        
        messages.append(f"USER: {context_prompt}")
        full_input = "\n\n".join(messages)

        gemini_reply = self._call_gemini(full_input, temperature=0.5, attachments=attachments)
        if gemini_reply:
            return gemini_reply

        # Fallback only when all AI models fail (network error or unconfigured)
        msg_lower = user_message.lower()
        if "가민" in user_message and ("측정" in user_message or "테스트" in user_message or "ftp" in msg_lower):
            return f"""🚴 **가민(Garmin Edge 1040)으로 FTP 측정하는 방법**
가민 1040으로 FTP를 정확히 측정하려면 **파워미터와 심박계 연결이 필수**입니다.

1. **가민 가이드 FTP 테스트 (가장 추천)**:
   - 가민 메인 화면 ➡️ **[훈련 (Training)]** ➡️ **[워크아웃 (Workouts)]** ➡️ **[FTP 가이드 테스트]** 선택
   - 20분 웜업 후 가민 화면 안내에 따라 점진적으로 파워를 올리는 램프 테스트를 진행합니다.
   - 테스트 종료 후 가민이 계산한 새 FTP를 즉시 기기에 적용할 수 있습니다.

2. **자동 FTP 감지 기능 (Auto Detection)**:
   - 가민 **[내 통계]** ➡️ **[FTP]** ➡️ **[자동 감지]**가 켜져 있으면, 평소 강도 높은 야외 라이딩이나 20분 지속주 후 새 FTP를 자동 감지하여 알림을 띄워줍니다.

*(현재 라이더님의 프로필 기준 FTP는 **{ftp}W**입니다.)*"""

        elif "무릎" in user_message or "통증" in user_message or "뻐근" in user_message:
            return f"""🚴 **코치의 피드백 (무릎 및 근육 통증)**
현재 라이더님의 TSB는 **{tsb:+.1f}**이며 최근 훈련으로 피로도가 쌓여있는 상태입니다.

1. **원인 점검**:
   - 무릎 앞쪽(슬개골) 통증: 안장이 너무 낮거나, 기어비를 무겁게 하여 낮은 케이던스(<75rpm)로 밟을 때 흔히 발생합니다.
   - 무릎 뒤쪽(오금) 통증: 안장이 너무 높거나 페달링 시 하사점에서 무릎이 과도하게 펴질 때 발생합니다.
2. **즉각적인 대처**:
   - 오늘은 인터벌 훈련을 취소하고 **완전 휴식** 또는 **가벼운 스트레칭/아이싱**을 권장합니다.
   - 다음 라이딩 시에는 평소보다 1~2단 가벼운 기어를 선택해 **88~95rpm의 부드러운 케이던스**를 유지해보세요."""

        else:
            return f"""🚴 **코치의 안내**
질문해주신 "{user_message}"에 관해:
현재 AI 클라우드 접속량이 많아 정밀 분석이 잠시 지연되었습니다.
(현재 라이더 데이터: FTP {ftp}W, 체력 CTL {ctl:.1f}, 피로도 ATL {atl:.1f}, 컨디션 TSB {tsb:+.1f})
잠시 후 질문을 다시 입력해 주시면 최신 AI 모델을 통해 더욱 정밀하게 답변해 드리겠습니다."""
