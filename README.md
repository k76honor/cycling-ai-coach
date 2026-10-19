# 🚴 AI 사이클링 코치 & 파워 트레이닝 플래너

가민(Garmin) 및 스트라바(Strava) 라이딩 기록을 연동하여 **파워(FTP), TSS, 체력(CTL)/피로도(ATL)/컨디션(TSB)**를 정밀 분석하고, 목표(그란폰도, FTP 향상, 힐클라임 등)에 맞춘 **주기화 훈련 플랜 생성 및 1:1 실시간 코칭 상담**을 제공하는 AI 에이전트 시스템입니다.

---

## 🌟 핵심 기능

1. **📊 과학적 파워 및 생체 지표 엔진 (Coggan & Friel 모델)**
   - **기준 FTP 기반 Coggan 7개 파워존** (Z1 회복 ~ Z7 신경근 파워 + Sweet Spot 88~94%)
   - **NP(Normalized Power), IF(Intensity Factor), TSS(Training Stress Score)** 자동 계산
   - **PMC (Performance Management Chart)**: 42일 장기 체력(CTL), 7일 단기 피로도(ATL), 실시간 컨디션 폼(TSB) 추적
   - **유산소 분리율 (Aerobic Decoupling / Pw:HR Drift)** 분석

2. **📅 목표 맞춤형 주기화 훈련 플랜 (Periodized Plan Generator)**
   - 목표 종목별 설계: FTP 향상, 100km+ 그란폰도, 힐클라임(W/kg), 크리테리움, 시즌오프 베이스
   - 기간(4~12주) 및 주당 가용 시간(4~16시간) 반영
   - 세부 인터벌(워밍업, 세트수, 목표 와트, 케이던스, 휴식시간, 쿨다운) 스펙 및 TSS 계산 제공

3. **💬 AI 사이클링 코치 1:1 대화 상담 (Google Gemini 연동)**
   - 최신 `gemini-3.8-flash` 기반 지능형 코칭
   - 실시간 라이더 데이터(FTP, 현재 TSB, 최근 라이딩 부하)를 컨텍스트로 자동 주입하여 답변
   - 근육통/무릎 통증 대처, 페달링 케이던스 교정, 대회 전 테이퍼링 및 보급(카보로딩) 전략 등 즉각 상담

4. **📡 유연한 데이터 연동 및 즉시 체험 모드**
   - **Strava API v3 연동**: 가민 워치/엣지에서 스트라바로 자동 업로드된 최신 라이딩 데이터 실시간 동기화
   - **샘플 데이터 모드**: API 키가 없어도 45일간의 실감나는 인터벌/엔듀런스/회복 라이딩 데이터로 모든 기능을 즉시 체험 가능

---

## 🚀 빠른 시작 가이드

### 1. 가상환경 및 패키지 실행

`uv` 패키지 관리자를 통해 생성된 가상환경으로 간편하게 실행할 수 있습니다:

```bash
cd C:\Users\k76ho\cycling-ai-coach

# 가상환경의 Streamlit 앱 실행
.venv\Scripts\streamlit run app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501`로 대시보드가 실행됩니다.

---

## 🔑 API 설정 (선택 사항)

### 1) Google Gemini API (AI 코칭 고도화)
- [Google AI Studio](https://aistudio.google.com/)에서 무료로 API 키를 발급받아 사이드바 또는 `.env`에 입력합니다.
- 키가 없어도 내장된 룰 기반 코칭 엔진으로 모든 기능이 정상 동작합니다.

### 2) Strava API (내 실제 가민 라이딩 데이터 연동)
1. 가민 커넥트(Garmin Connect) 앱 설정에서 **Strava 자동 동기화**를 켭니다 (대부분의 라이더가 이미 사용 중).
2. [Strava Settings > My API App](https://www.strava.com/settings/api)에 접속하여 새 앱을 생성합니다.
3. 발급받은 `Client ID`, `Client Secret`, 그리고 OAuth `Refresh Token`을 사이드바 또는 `.env`에 저장합니다.
