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
당신은 세계적인 사이클링 전문 파워 코치(AI Cycling Coach)입니다.
앤드류 코건(Andrew Coggan) 박사의 7단계 파워존 훈련 이론, 조 프리엘(Joe Friel)의 '사이클리스트 트레이닝 바이블', 그리고 스티븐 세일러(Stephen Seiler)의 양극화 훈련(80/20 Polarized Training) 원칙을 철저히 따릅니다.

[당신의 역할 및 태도]
1. 과학적 근거에 기반한 명쾌하고 전문적인 코칭: 파워(Watt), 체중당 파워(W/kg), TSS(훈련 스트레스 점수), IF(강도 계수), NP(정규화 파워), CTL(체력)/ATL(피로도)/TSB(컨디션) 수치를 바탕으로 분석하고 조언합니다.
2. 부상 방지와 과훈련(Overreaching) 예방 최우선: TSB가 -20 이하로 떨어지거나 피로도가 높을 때는 무리한 고강도 훈련을 엄격히 만류하고 회복 훈련(Z1 Active Recovery)이나 완전 휴식을 처방합니다.
3. 구체적이고 실행 가능한 워크아웃 제공: "열심히 타세요" 대신 목표 파워(W), 목표 케이던스(RPM), 인터벌 구성(세트, 시간, 휴식 비율)을 정확한 수치로 제공합니다.
4. 친절하고 동기부여 넘치는 어조: 한국어로 자연스럽고 전문적인 스포츠 코치의 말투로 응답합니다.
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
        if self.is_available():
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "system_instruction": COACH_SYSTEM_PROMPT,
                        "temperature": 0.4,
                    },
                )
                return response.text
            except Exception as e:
                print(f"Gemini API call failed: {e}")

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
        if self.is_available():
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "system_instruction": COACH_SYSTEM_PROMPT,
                        "temperature": 0.4,
                    },
                )
                return response.text
            except Exception as e:
                print(f"Gemini API call failed for training plan: {e}")

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
    ) -> str:
        """
        Interactive cycling consultation chatbot.
        Answers user training/fatigue/gear/nutrition questions with live athlete context.
        """
        ftp = athlete_context.get("ftp", 230)
        weight_kg = athlete_context.get("weight_kg", 68.0)
        ctl = athlete_context.get("ctl", 50.0)
        atl = athlete_context.get("atl", 55.0)
        tsb = athlete_context.get("tsb", -5.0)
        recent_ride = athlete_context.get("last_ride_summary", "최근 라이딩 기록 없음")
        athlete_notes = athlete_context.get("athlete_notes", [])
        notes_str = "\n".join([f"- {n}" for n in athlete_notes]) if athlete_notes else "- 특이사항 없음"

        context_prompt = f"""
[코치가 기억하고 있는 라이더 핵심 메모 (장기 기억)]
{notes_str}

[현재 상담 라이더의 실시간 데이터]
- FTP: {ftp}W (체중 {weight_kg}kg, W/kg: {round(ftp/max(1.0, weight_kg), 2)})
- 현재 PMC 지표: CTL(체력)={ctl:.1f}, ATL(피로도)={atl:.1f}, TSB(컨디션)={tsb:+.1f}
- 가장 최근 라이딩: {recent_ride}

[라이더의 질문]
"{user_message}"

코치님은 위 장기 기억(선수의 목표, 부상 이력, 특이사항)과 현재 실시간 파워 데이터를 항상 인지하고, 전담 코치로서 맥락이 이어지는 깊이 있는 맞춤 조언을 제공해주세요.
"""

        if self.is_available():
            try:
                # Build contents with chat history
                messages = []
                for msg in chat_history[-6:]:  # Keep last 3 turns
                    role = "user" if msg["role"] == "user" else "model"
                    messages.append(f"{role.upper()}: {msg['content']}")
                
                messages.append(f"USER: {context_prompt}")
                full_input = "\n\n".join(messages)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_input,
                    config={
                        "system_instruction": COACH_SYSTEM_PROMPT,
                        "temperature": 0.5,
                    },
                )
                return response.text
            except Exception as e:
                print(f"Gemini API chat error: {e}")

        # Fallback heuristic consultation answers
        msg_lower = user_message.lower()
        if "무릎" in user_message or "통증" in user_message or "뻐근" in user_message:
            return f"""🚴 **코치의 피드백 (무릎 및 근육 통증)**
현재 라이더님의 TSB는 **{tsb:+.1f}**이며 최근 훈련으로 피로도가 쌓여있는 상태입니다.

1. **원인 점검**:
   - 무릎 앞쪽(슬개골) 통증: 안장이 너무 낮거나, 기어비를 무겁게 하여 낮은 케이던스(<75rpm)로 밟을 때 흔히 발생합니다.
   - 무릎 뒤쪽(오금) 통증: 안장이 너무 높거나 페달링 시 하사점에서 무릎이 과도하게 펴질 때 발생합니다.
2. **즉각적인 대처**:
   - 오늘은 인터벌 훈련을 취소하고 **완전 휴식** 또는 **가벼운 스트레칭/아이싱**을 권장합니다.
   - 다음 라이딩 시에는 평소보다 1~2단 가벼운 기어를 선택해 **88~95rpm의 부드러운 케이던스**를 유지해보세요. 통증이 지속되면 전문가의 클릿 피팅 점검을 권장합니다."""

        elif "ftp" in msg_lower or "올리" in user_message or "향상" in user_message:
            return f"""📈 **코치의 FTP 향상 로드맵**
현재 FTP는 **{ftp}W**({round(ftp/weight_kg, 2)} W/kg)이며, CTL은 **{ctl:.1f}**입니다.

1. **스윗스팟(Sweet Spot) 훈련 비중 확대**:
   - 88~94% FTP({round(ftp*0.88)}~{round(ftp*0.94)}W) 구간은 피로 발생 대비 미토콘드리아와 젖산 완충 능력을 가장 효율적으로 증대시킵니다.
   - 주 2회: `2x20분` 또는 `3x15분` 스윗스팟 세션을 수행해보세요.
2. **Zone 2 유산소 베이스 확장**:
   - 고강도 인터벌만으로는 한계가 옵니다. 주말 2.5~3시간의 Z2({round(ftp*0.56)}~{round(ftp*0.75)}W) 지속주로 심장 용적과 모세혈관망을 넓혀야 임계치가 올라갑니다.
3. **4주 주기화**: 3주 빌드업 후 1주는 회복주(TSS 40% 감축)를 가져야 초회복(Supercompensation)이 일어나 FTP가 상승합니다."""

        elif "오늘" in user_message and ("탈까" in user_message or "훈련" in user_message or "어떻게" in user_message):
            status = interpret_tsb(tsb)
            return f"""🎯 **오늘의 맞춤형 훈련 제안**
현재 폼 지수(TSB)는 **{tsb:+.1f}** ({status['status']})입니다.

- **권장 훈련**:
  * TSB가 {tsb:+.1f}로 {status['zone']} 영역에 위치하고 있습니다.
  * {status['advice']}
- **추천 세션**:
  * 만약 컨디션이 상쾌하다면: **스윗스팟({round(ftp*0.9)}W) 2 x 20분**
  * 다리가 다소 무겁다면: **Zone 2 ({round(ftp*0.65)}W) 60분 지구력 라이딩**
  * 페달링 케이던스는 **90rpm** 전후를 목표로 하세요!"""

        else:
            return f"""🚴 **코치의 답변**
라이더님의 현재 데이터(FTP {ftp}W, 체력 CTL {ctl:.1f}, 피로도 ATL {atl:.1f}, 컨디션 TSB {tsb:+.1f})를 기준으로 답변드립니다.

질문해주신 "{user_message}"에 관해:
- 체계적인 사이클링 훈련에서는 무조건 오래 타거나 세게 타는 것보다 **'강도(Intensity)와 휴식(Recovery)의 균형'**이 가장 중요합니다.
- 현재 누적 피로도(ATL)를 체크하면서 주간 총 TSS의 80%는 Zone 2 유산소 영역에, 20%는 Sweet Spot 및 Threshold 인터벌에 투자하는 것을 추천합니다.
- 궁금한 특정 워크아웃, 인터벌 구성, 보급 전략(카보 로딩) 등에 대해 편하게 더 물어보세요!"""
