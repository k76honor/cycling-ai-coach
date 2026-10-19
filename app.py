"""
AI Cycling Coach & Strava Training Dashboard
Built with Streamlit, Plotly, Strava API, and Google Gemini.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    STRAVA_CLIENT_ID,
    STRAVA_CLIENT_SECRET,
    STRAVA_REFRESH_TOKEN,
    DEFAULT_FTP,
    DEFAULT_WEIGHT_KG,
)
from strava_client import StravaClient
from power_metrics import (
    get_power_zones,
    calculate_pmc,
    interpret_tsb,
    calculate_power_distribution,
    calculate_aerobic_decoupling,
)
from coach_agent import CyclingCoachAgent
from memory_manager import (
    load_memory,
    add_athlete_note,
    remove_athlete_note,
    append_chat_message,
    clear_chat_history,
    auto_extract_athlete_insight,
    get_athlete_profile,
    update_athlete_profile,
)

# Page configuration
st.set_page_config(
    page_title="AI 사이클링 코치 & 훈련 계획기",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        color: white;
        margin-bottom: 12px;
        border-left: 5px solid #38bdf8;
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    /* 채팅 입력창이 항상 화면 아래쪽에 고정되도록 스타일링 */
    div[data-testid="stChatInput"] {
        position: sticky !important;
        bottom: 12px !important;
        z-index: 999 !important;
        border-radius: 12px !important;
        box-shadow: 0 -2px 16px rgba(0, 0, 0, 0.3) !important;
        background-color: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR -----------------
st.sidebar.title("🚴 사이클링 코치 설정")

# Athlete Profile Section
st.sidebar.subheader("👤 라이더 프로필")
saved_profile = get_athlete_profile()
init_ftp = int(saved_profile.get("ftp", DEFAULT_FTP))
init_weight = float(saved_profile.get("weight_kg", DEFAULT_WEIGHT_KG))

ftp_input = st.sidebar.number_input(
    "기준 FTP (Watts)", min_value=100, max_value=500, value=init_ftp, step=5,
    help="1시간 동안 지속할 수 있는 최대 평균 파워(FTP)"
)
weight_input = st.sidebar.number_input(
    "체중 (kg)", min_value=40.0, max_value=130.0, value=init_weight, step=0.5
)

# Auto-save whenever user changes FTP or Weight
if ftp_input != init_ftp or weight_input != init_weight:
    update_athlete_profile(ftp=ftp_input, weight_kg=weight_input)

w_per_kg = round(ftp_input / max(1.0, weight_input), 2)
st.sidebar.caption(f"⚡ **체중당 파워 (W/kg)**: `{w_per_kg} W/kg`")
st.sidebar.caption("💾 *설정하신 FTP와 체중은 자동 저장되어 다음 접속 시에도 유지됩니다.*")

st.sidebar.divider()

# Data Source Section
st.sidebar.subheader("📡 데이터 소스 연동")
strava_has_credentials = bool(STRAVA_CLIENT_ID and STRAVA_REFRESH_TOKEN)
data_options = ["🔗 Strava API 연동", "🧪 샘플 라이딩 데이터로 즉시 체험"]
data_mode = st.sidebar.radio(
    "데이터 수집 방식",
    data_options,
    index=0 if strava_has_credentials else 1,
)

s_client_id = STRAVA_CLIENT_ID
s_client_secret = STRAVA_CLIENT_SECRET
s_refresh_token = STRAVA_REFRESH_TOKEN

if data_mode == "🔗 Strava API 연동":
    s_client_id = st.sidebar.text_input("Client ID", value=STRAVA_CLIENT_ID)
    s_client_secret = st.sidebar.text_input("Client Secret", value=STRAVA_CLIENT_SECRET, type="password")
    s_refresh_token = st.sidebar.text_input("Refresh Token", value=STRAVA_REFRESH_TOKEN, type="password")
    
    with st.sidebar.expander("❓ Strava API 키 발급 방법"):
        st.markdown(
            """
            1. [Strava Settings > My API App](https://www.strava.com/settings/api)에 접속합니다.
            2. Application을 생성하고 **Client ID**와 **Client Secret**을 발급받습니다.
            3. OAuth 인증을 통해 발급받은 **Refresh Token**을 입력하면 가민/스마트워치에서 스트라바로 올라간 라이딩 데이터가 자동 연동됩니다.
            """
        )

st.sidebar.divider()

# AI Coach (Gemini) Settings
st.sidebar.subheader("🧠 AI 코치 (Gemini) 설정")
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=GEMINI_API_KEY,
    type="password",
    help="Google AI Studio에서 발급받은 Gemini API Key를 입력하세요.",
)
model_name = st.sidebar.selectbox(
    "Gemini 모델",
    ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
    index=0,
)

if not api_key_input:
    st.sidebar.info("💡 API Key 없이도 내장된 전문 코칭 규칙 엔진으로 기본 기능이 완벽히 작동합니다. 키를 입력하시면 실시간 심층 코칭이 활성화됩니다.")

# Initialize AI Coach Agent
coach_agent = CyclingCoachAgent(api_key=api_key_input or None, model_name=model_name)

# ----------------- DATA PREPARATION -----------------
@st.cache_data(ttl=300)
def load_activities(is_strava: bool, c_id: str, c_sec: str, r_tok: str, ftp: float):
    client = StravaClient(c_id, c_sec, r_tok) if is_strava else StravaClient()
    return client.get_activities(ftp=ftp, per_page=60)

activities = load_activities(
    is_strava=(data_mode == "🔗 Strava API 연동"),
    c_id=s_client_id if data_mode == "🔗 Strava API 연동" else "",
    c_sec=s_client_secret if data_mode == "🔗 Strava API 연동" else "",
    r_tok=s_refresh_token if data_mode == "🔗 Strava API 연동" else "",
    ftp=ftp_input,
)

# Build daily TSS series for PMC
def build_pmc_dataframe(act_list: list) -> pd.DataFrame:
    if not act_list:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=45, freq="D")
        return calculate_pmc(pd.Series(0.0, index=dates))

    df_act = pd.DataFrame(act_list)
    df_act["date"] = pd.to_datetime(df_act["start_date"]).dt.tz_convert(None).dt.date
    daily_tss = df_act.groupby("date")["tss"].sum()

    # Create full calendar index
    min_date = daily_tss.index.min()
    max_date = datetime.now().date()
    full_idx = pd.date_range(start=min_date, end=max_date, freq="D").date
    daily_tss = daily_tss.reindex(full_idx, fill_value=0.0)
    daily_tss.index = pd.DatetimeIndex(daily_tss.index)

    return calculate_pmc(daily_tss)

pmc_df = build_pmc_dataframe(activities)
latest_pmc = {
    "ctl": float(pmc_df["ctl"].iloc[-1]) if not pmc_df.empty else 0.0,
    "atl": float(pmc_df["atl"].iloc[-1]) if not pmc_df.empty else 0.0,
    "tsb": float(pmc_df["tsb"].iloc[-1]) if not pmc_df.empty else 0.0,
}
tsb_status = interpret_tsb(latest_pmc["tsb"])

# Header
st.title("🚴 AI 사이클링 코치 & 파워 훈련 플래너")
st.markdown(
    "가민/스트라바 라이딩 기록을 바탕으로 **FTP, 파워존, 훈련 부하(TSS), 체력(CTL)/피로도(ATL)/컨디션(TSB)**를 분석하고, "
    "목표에 맞춘 **맞춤형 훈련 계획과 실시간 코칭 상담**을 제공합니다."
)

# Top KPI Summary Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
with kpi_col1:
    st.metric("기준 FTP", f"{ftp_input} W", f"{w_per_kg} W/kg")
with kpi_col2:
    recent_4w_tss = (
        sum(act.get("tss", 0) for act in activities[:20]) / 4.0 if activities else 0
    )
    st.metric("주간 평균 TSS", f"{recent_4w_tss:.0f}", "최근 4주")
with kpi_col3:
    st.metric("체력 (CTL)", f"{latest_pmc['ctl']:.1f}", "장기 부하 (42일)")
with kpi_col4:
    st.metric("피로도 (ATL)", f"{latest_pmc['atl']:.1f}", "단기 부하 (7일)")
with kpi_col5:
    tsb_val = latest_pmc["tsb"]
    st.metric(
        "컨디션 (TSB)",
        f"{tsb_val:+.1f}",
        tsb_status["zone"],
        delta_color="normal" if tsb_val >= -20 else "inverse",
    )

st.divider()

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 훈련 분석 & 대시보드",
    "📅 맞춤형 훈련 플랜 생성기",
    "💬 AI 코치 1:1 상담실",
])

# ----------------- TAB 1: DASHBOARD -----------------
with tab1:
    st.subheader("📈 훈련 부하 및 컨디션 관리 차트 (PMC: Performance Management Chart)")
    st.caption("Dr. Andrew Coggan의 피로도-체력 모델: 파란선(체력 CTL), 주황선(피로도 ATL), 녹색/빨간 막대(컨디션 TSB)")

    # Plotly PMC Chart
    fig_pmc = go.Figure()
    
    # TSB Bars
    colors = ["#22c55e" if val >= 0 else "#ef4444" for val in pmc_df["tsb"]]
    fig_pmc.add_trace(
        go.Bar(
            x=pmc_df.index,
            y=pmc_df["tsb"],
            name="컨디션 (TSB)",
            marker_color=colors,
            opacity=0.65,
            yaxis="y1",
        )
    )

    # CTL Line (Fitness)
    fig_pmc.add_trace(
        go.Scatter(
            x=pmc_df.index,
            y=pmc_df["ctl"],
            name="체력 (CTL / Fitness)",
            line=dict(color="#0284c7", width=3),
            yaxis="y1",
        )
    )

    # ATL Line (Fatigue)
    fig_pmc.add_trace(
        go.Scatter(
            x=pmc_df.index,
            y=pmc_df["atl"],
            name="피로도 (ATL / Fatigue)",
            line=dict(color="#f97316", width=2, dash="dot"),
            yaxis="y1",
        )
    )

    fig_pmc.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="날짜", showgrid=True),
        yaxis=dict(title="TSS / 점수", showgrid=True),
        hovermode="x unified",
        template="plotly_dark",
    )
    st.plotly_chart(fig_pmc, use_container_width=True)

    # Diagnostic Banner & Button
    col_diag_text, col_diag_btn = st.columns([4, 1])
    with col_diag_text:
        st.info(f"**현재 상태 코치 한줄 진단**: {tsb_status['advice']}")
    with col_diag_btn:
        run_diag = st.button("🔍 AI 종합 진단 리포트 생성", use_container_width=True)

    if run_diag:
        with st.spinner("AI 코치가 최근 라이딩 데이터와 파워 지표를 정밀 진단 중입니다..."):
            diag_result = coach_agent.diagnose_training_state(
                ftp=ftp_input,
                weight_kg=weight_input,
                recent_activities=activities,
                pmc_latest=latest_pmc,
            )
            st.markdown(diag_result)

    st.divider()

    # Two columns for Power Zones and Recent Activities
    col_pz, col_recent = st.columns([1, 1])

    with col_pz:
        st.subheader("⚡ Coggan 7 파워존 기준표")
        zones = get_power_zones(ftp_input)
        zone_data = []
        for k, v in zones.items():
            low, high = v["range"]
            zone_data.append({
                "파워 존": v["name"].split(":")[1].strip() if ":" in v["name"] else v["name"],
                "비율": v["pct"],
                "와트 (W)": f"{low}W ~ {high}W",
                "훈련 목적": v["description"],
            })
        st.dataframe(pd.DataFrame(zone_data), use_container_width=True, hide_index=True)

    with col_recent:
        st.subheader("🚴 최근 라이딩 활동 내역")
        if activities:
            recent_table = []
            for act in activities[:10]:
                recent_table.append({
                    "날짜": act["start_date_local"][:10],
                    "라이딩명": act["name"],
                    "거리(km)": round(act["distance"] / 1000.0, 1),
                    "시간(분)": round(act["moving_time"] / 60.0),
                    "NP (와트)": round(act.get("weighted_average_watts") or act.get("average_watts", 0)),
                    "IF": round(act.get("intensity_factor", 0.0), 2),
                    "TSS": round(act.get("tss", 0.0), 1),
                })
            st.dataframe(pd.DataFrame(recent_table), use_container_width=True, hide_index=True)
        else:
            st.write("표시할 활동 내역이 없습니다.")


# ----------------- TAB 2: TRAINING PLAN GENERATOR -----------------
with tab2:
    st.subheader("📅 목표 맞춤형 사이클링 훈련 플랜 설계")
    st.markdown("목표 이벤트와 주당 가용 시간을 설정하면 현재 FTP와 TSB 컨디션을 반영한 체계적인 주기화 훈련표를 설계합니다.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        plan_goal = st.selectbox(
            "목표 이벤트 / 종목",
            [
                "FTP 향상 (스윗스팟 및 역치 집중)",
                "그란폰도 / 메디오폰도 완주 (100km+ 지구력 & 페이싱)",
                "힐클라임 대회 (W/kg 강화 & VO2 Max)",
                "크리테리움 / 단거리 로드 레이스 (무산소 파워)",
                "시즌 오프 유산소 베이스 빌딩 (Z2 엔듀런스)",
            ],
            index=0,
        )
    with col_p2:
        plan_weeks = st.select_slider("훈련 기간 (주)", options=[4, 6, 8, 12], value=8)
    with col_p3:
        plan_hours = st.slider("주당 가용 훈련 시간", min_value=4.0, max_value=16.0, value=7.5, step=0.5)

    generate_plan_btn = st.button("🚀 AI 맞춤형 훈련 플랜 생성하기", type="primary", use_container_width=True)

    if generate_plan_btn:
        with st.spinner("AI 코치가 생리학적 원리를 바탕으로 주기화 훈련표를 생성 중입니다..."):
            plan_text = coach_agent.generate_training_plan(
                target_goal=plan_goal,
                duration_weeks=plan_weeks,
                hours_per_week=plan_hours,
                ftp=ftp_input,
                weight_kg=weight_input,
                current_tsb=latest_pmc["tsb"],
            )
            st.session_state["generated_plan"] = plan_text

    if "generated_plan" in st.session_state:
        st.markdown(st.session_state["generated_plan"])
        st.download_button(
            "📥 훈련 플랜 마크다운 다운로드",
            data=st.session_state["generated_plan"],
            file_name=f"cycling_training_plan_{plan_weeks}weeks.md",
            mime="text/markdown",
        )


# ----------------- TAB 3: CHATBOT CONSULTATION -----------------
with tab3:
    st.subheader("💬 AI 사이클링 코치 1:1 실시간 상담실")
    st.caption("훈련 루틴, 오늘의 피로도 대처, 인터벌 페이싱, 보급(카보로딩), 페달링 테크닉 등 무엇이든 질문하세요.")

    # Load persistent memory
    mem_data = load_memory()
    athlete_notes = mem_data.get("athlete_notes", [])

    # Memory & Notes Management Expander
    with st.expander("🧠 코치가 기억하고 있는 라이더 핵심 메모 (장기 기억)", expanded=False):
        st.markdown(
            "코치는 브라우저를 닫거나 며칠 뒤에 다시 들어와도 아래 등록된 **선수 특이사항, 부상 이력, 목표**를 영구적으로 기억하고 상담에 반영합니다."
        )
        if athlete_notes:
            for idx, note in enumerate(athlete_notes):
                col_n_text, col_n_del = st.columns([8, 1])
                col_n_text.markdown(f"• **{note}**")
                if col_n_del.button("삭제", key=f"del_note_{idx}"):
                    remove_athlete_note(idx)
                    st.rerun()
        else:
            st.info("등록된 특이사항 메모가 없습니다. 아래에서 새로 추가해 보세요.")

        # Add new note input
        col_new_in, col_new_btn = st.columns([6, 1])
        new_note_text = col_new_in.text_input(
            "새 메모 추가",
            placeholder="예: 2026년 5월 설악 그란폰도 완주 목표, 오른쪽 무릎 외측 통증 주의",
            label_visibility="collapsed",
        )
        if col_new_btn.button("➕ 추가", use_container_width=True):
            if new_note_text.strip():
                add_athlete_note(new_note_text)
                st.rerun()

    # Chat Controls (Clear History)
    col_chat_title, col_chat_clear = st.columns([5, 1])
    with col_chat_clear:
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True, help="저장된 대화 내역을 모두 지우고 새 대화를 시작합니다 (핵심 메모는 유지됨)"):
            clear_chat_history()
            st.session_state["messages"] = []
            st.rerun()

    # Quick prompt buttons
    st.write("💡 **자주 묻는 추천 질문:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_prompt = None
    if q_col1.button("🦵 오늘 다리가 뻐근한데 타도 될까?", use_container_width=True):
        quick_prompt = "오늘 라이딩 후 허벅지가 뻐근하고 피로한데 내일 계획된 인터벌을 타도 될까요?"
    if q_col2.button("📈 FTP 20W 올리는 스윗스팟 루틴", use_container_width=True):
        quick_prompt = "현재 FTP에서 20W를 올리고 싶은데 스윗스팟 인터벌 루틴과 주간 구성을 어떻게 해야 할까요?"
    if q_col3.button("🚴 무릎 안 아픈 케이던스 요령", use_container_width=True):
        quick_prompt = "평소 케이던스가 75rpm 정도로 낮은 편인데 무릎 통증 없이 90rpm으로 올리는 방법이 궁금합니다."
    if q_col4.button("🍌 그란폰도 대회 전 테이퍼링과 보급", use_container_width=True):
        quick_prompt = "100km 그란폰도 대회가 2주 남았는데 훈련 볼륨 테이퍼링과 시합 당일 파워젤 보급 전략을 알려주세요."

    # Initialize messages from persistent storage
    if "messages" not in st.session_state or not st.session_state["messages"]:
        saved_msgs = mem_data.get("messages", [])
        if saved_msgs:
            st.session_state["messages"] = saved_msgs
        else:
            default_greeting = {
                "role": "assistant",
                "content": f"안녕하세요, 라이더님! 🚴 AI 사이클링 코치입니다.\n현재 라이더님의 FTP는 **{ftp_input}W**({w_per_kg} W/kg), 컨디션 폼(TSB)은 **{latest_pmc['tsb']:+.1f}**입니다.\n오늘 라이딩 피드백, 인터벌 페이싱, 맞춤형 훈련 고민 등 무엇이든 편하게 물어보세요!",
            }
            st.session_state["messages"] = [default_greeting]
            append_chat_message(default_greeting["role"], default_greeting["content"])

    # Display chat history inside a fixed-height scrollable container
    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Handle input (Always fixed & docked right at the bottom of the screen)
    user_input = st.chat_input("코치에게 훈련 상담 질문을 입력하세요...")
    query = quick_prompt or user_input

    if query:
        # Append user message to UI and disk
        st.session_state["messages"].append({"role": "user", "content": query})
        append_chat_message("user", query)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        # Build context including athlete notes
        last_ride = activities[0] if activities else {}
        last_ride_summary = (
            f"{last_ride.get('name', '라이딩')} (거리: {round(last_ride.get('distance', 0)/1000, 1)}km, "
            f"NP: {round(last_ride.get('weighted_average_watts', 0))}W, TSS: {round(last_ride.get('tss', 0), 1)})"
            if last_ride else "기록 없음"
        )
        athlete_context = {
            "ftp": ftp_input,
            "weight_kg": weight_input,
            "ctl": latest_pmc["ctl"],
            "atl": latest_pmc["atl"],
            "tsb": latest_pmc["tsb"],
            "last_ride_summary": last_ride_summary,
            "athlete_notes": load_memory().get("athlete_notes", []),
        }

        # Generate response inside chat container
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("코치가 선수의 장기 기억과 파워 데이터를 종합 분석 중입니다..."):
                    answer = coach_agent.consult(
                        user_message=query,
                        chat_history=st.session_state["messages"][:-1],
                        athlete_context=athlete_context,
                    )
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    append_chat_message("assistant", answer)

                    # Auto-extract new insights about the athlete to long-term memory
                    auto_extract_athlete_insight(coach_agent, query, answer)
        st.rerun()
