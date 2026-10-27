"""
AI Cycling Coach & Garmin Edge 1040 / Strava Training Dashboard
Built with Streamlit, Plotly, Garmin FIT Parser, Strava API, and Google Gemini.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    STRAVA_CLIENT_ID,
    STRAVA_CLIENT_SECRET,
    STRAVA_REFRESH_TOKEN,
    DEFAULT_FTP,
    DEFAULT_WEIGHT_KG,
    DEFAULT_MAX_HR,
    DEFAULT_LTHR,
    DEFAULT_REST_HR,
)
from strava_client import StravaClient
from garmin_loader import (
    load_garmin_raw_activities,
    convert_garmin_to_dashboard_activities,
    estimate_dynamic_ftp,
)
from power_metrics import (
    get_power_zones,
    calculate_pmc,
    interpret_tsb,
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
    page_title="AI 사이클링 코치 & 가민 1040 분석기",
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
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
st.sidebar.subheader("👤 라이더 생리학 프로필")
saved_profile = get_athlete_profile()
init_ftp = int(saved_profile.get("ftp", DEFAULT_FTP))
init_weight = float(saved_profile.get("weight_kg", DEFAULT_WEIGHT_KG))
init_max_hr = int(saved_profile.get("max_hr", DEFAULT_MAX_HR))
init_lthr = int(saved_profile.get("lthr", DEFAULT_LTHR))
init_rest_hr = int(saved_profile.get("rest_hr", DEFAULT_REST_HR))

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    ftp_input = st.number_input(
        "기준 FTP (W)", min_value=100, max_value=500, value=init_ftp, step=1,
        help="권장 실전 FTP: 195~198W (잠재 올아웃 FTP: 200~205W)"
    )
with col_sb2:
    weight_input = st.number_input(
        "체중 (kg)", min_value=40.0, max_value=130.0, value=init_weight, step=0.1
    )

col_hr1, col_hr2, col_hr3 = st.sidebar.columns(3)
with col_hr1:
    lthr_input = st.number_input(
        "LTHR (역치)", min_value=120, max_value=200, value=init_lthr, step=1,
        help="젖산역치 심박수 (실측 검증: 161 bpm)"
    )
with col_hr2:
    max_hr_input = st.number_input(
        "최대심박", min_value=140, max_value=220, value=init_max_hr, step=1,
        help="최대심박수 (9/5 실측 최고: 183~184 bpm)"
    )
with col_hr3:
    rest_hr_input = st.number_input(
        "안정심박", min_value=35, max_value=90, value=init_rest_hr, step=1
    )

if (
    ftp_input != init_ftp
    or round(weight_input, 1) != round(init_weight, 1)
    or max_hr_input != init_max_hr
    or lthr_input != init_lthr
    or rest_hr_input != init_rest_hr
):
    update_athlete_profile(
        ftp=ftp_input,
        weight_kg=round(weight_input, 1),
        max_hr=max_hr_input,
        lthr=lthr_input,
        rest_hr=rest_hr_input,
    )

w_per_kg = round(ftp_input / max(1.0, weight_input), 2)
st.sidebar.caption(
    f"⚡ **체중당 FTP**: `{w_per_kg} W/kg` | ❤️ **LTHR**: `{lthr_input} bpm` (Max `{max_hr_input}`)"
)

# Pre-load garmin activities so we can show dynamic FTP estimation in sidebar
_init_garmin_acts = load_garmin_raw_activities()
dyn_est = estimate_dynamic_ftp(_init_garmin_acts, lthr=lthr_input)
st.sidebar.info(
    f"🤖 **최근 주행 기반 동적 산출**\n"
    f"• 권장 실전 FTP: **`{dyn_est['recommended_ftp']} W`** (잠재 `{dyn_est['potential_max_ftp']} W`)\n"
    f"• 가민 최근 감지: `{dyn_est['latest_garmin_ftp']} W` (최고 `{dyn_est['max_garmin_ftp']} W`)\n"
    f"• 최근 기록 체중: `{dyn_est['latest_weight']} kg`"
)
if st.sidebar.button("🔄 최신 주행 분석값(FTP·체중) 자동 적용", use_container_width=True):
    update_athlete_profile(
        ftp=dyn_est["recommended_ftp"],
        weight_kg=dyn_est["latest_weight"],
        max_hr=max_hr_input,
        lthr=lthr_input,
        rest_hr=rest_hr_input,
    )
    st.rerun()

st.sidebar.divider()

# Data Source Section
st.sidebar.subheader("📡 데이터 소스 연동")
data_options = [
    "⌚ 가민 Edge 1040 실측 데이터 (40회 + .fit 업로드)",
    "🔗 Strava API 연동",
    "🧪 샘플 라이딩 데이터로 즉시 체험",
]
data_mode = st.sidebar.radio("데이터 수집 방식", data_options, index=0)

uploaded_fit_files = None
recalc_tss = False
s_client_id = STRAVA_CLIENT_ID
s_client_secret = STRAVA_CLIENT_SECRET
s_refresh_token = STRAVA_REFRESH_TOKEN

if data_mode.startswith("⌚"):
    uploaded_fit_files = st.sidebar.file_uploader(
        "📂 새 가민 .fit 파일 추가 업로드 (선택)",
        type=["fit"],
        accept_multiple_files=True,
        help="Edge 1040의 Garmin\\Activities 폴더에서 새 .fit 파일을 드래그 앤 드롭하면 기존 40회 데이터에 즉시 병합됩니다.",
    )
    recalc_tss = st.sidebar.checkbox(
        "현재 입력된 FTP로 전체 TSS/IF 재계산",
        value=False,
        help="체크 시 가민 파일에 기록된 당시 TSS 대신 현재 사이드바의 FTP 기준으로 TSS를 일괄 재계산합니다.",
    )
elif data_mode == "🔗 Strava API 연동":
    s_client_id = st.sidebar.text_input("Client ID", value=STRAVA_CLIENT_ID)
    s_client_secret = st.sidebar.text_input("Client Secret", value=STRAVA_CLIENT_SECRET, type="password")
    s_refresh_token = st.sidebar.text_input("Refresh Token", value=STRAVA_REFRESH_TOKEN, type="password")

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
    [
        "gemini-2.5-flash (추천 / 고속·정밀)",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    ],
    index=0,
)
model_name_clean = model_name.split(" ")[0].strip()

coach_agent = CyclingCoachAgent(api_key=api_key_input or None, model_name=model_name_clean)

# ----------------- DATA PREPARATION -----------------
garmin_raw_acts = load_garmin_raw_activities(uploaded_fit_files)

@st.cache_data(ttl=300)
def load_strava_or_sample(is_strava: bool, c_id: str, c_sec: str, r_tok: str, ftp: float):
    client = StravaClient(c_id, c_sec, r_tok) if is_strava else StravaClient()
    return client.get_activities(ftp=ftp, per_page=60)

if data_mode.startswith("⌚"):
    activities = convert_garmin_to_dashboard_activities(
        garmin_raw_acts, ftp=ftp_input, recalc_tss_with_ftp=recalc_tss
    )
else:
    activities = load_strava_or_sample(
        is_strava=(data_mode == "🔗 Strava API 연동"),
        c_id=s_client_id if data_mode == "🔗 Strava API 연동" else "",
        c_sec=s_client_secret if data_mode == "🔗 Strava API 연동" else "",
        r_tok=s_refresh_token if data_mode == "🔗 Strava API 연동" else "",
        ftp=ftp_input,
    )


def build_pmc_dataframe(act_list: list) -> pd.DataFrame:
    if not act_list:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=45, freq="D")
        return calculate_pmc(pd.Series(0.0, index=dates))

    df_act = pd.DataFrame(act_list)
    df_act["date"] = pd.to_datetime(df_act["start_date"]).dt.date
    daily_tss = df_act.groupby("date")["tss"].sum()

    min_date = daily_tss.index.min()
    max_date = max(daily_tss.index.max(), datetime.now().date())
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
st.title("🚴 AI 사이클링 코치 & 가민 Edge 1040 퍼포먼스 분석기")
st.markdown(
    "가민 엣지 1040 실측 `.fit` 데이터(40회, **2,061.1km / +18,105m**)를 기반으로 "
    "**FTP(W/kg), 심박·파워존, 유산소 효율(EF), 좌우 밸런스, 체력(CTL)/피로도(ATL)/컨디션(TSB)**을 정밀 분석합니다."
)

# Top KPI Summary Cards (6 Columns)
total_km_all = sum(a.get("distance", 0) for a in activities) / 1000.0
total_elev_all = sum(a.get("total_elevation_gain", 0) for a in activities)
total_tss_all = sum(a.get("tss", 0) for a in activities)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("기준 FTP", f"{ftp_input} W", f"{w_per_kg} W/kg (@{weight_input}kg)")
with k2:
    st.metric("역치 심박 (LTHR)", f"{lthr_input} bpm", f"Max {max_hr_input} / 안정 {rest_hr_input}")
with k3:
    st.metric(f"누적 주행 ({len(activities)}회)", f"{total_km_all:,.1f} km", f"고도 +{total_elev_all:,.0f} m")
with k4:
    st.metric("체력 (CTL)", f"{latest_pmc['ctl']:.1f}", f"총 {total_tss_all:,.0f} TSS")
with k5:
    st.metric("피로도 (ATL)", f"{latest_pmc['atl']:.1f}", "단기 부하 (7일)")
with k6:
    tsb_val = latest_pmc["tsb"]
    st.metric(
        "컨디션 (TSB)",
        f"{tsb_val:+.1f}",
        tsb_status["zone"],
        delta_color="normal" if tsb_val >= -20 else "inverse",
    )

st.divider()

# Main Tabs (4 Tabs)
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 훈련 부하 & PMC 대시보드",
    "🏆 가민 1040 심층 분석 (파워커브·효율·밸런스)",
    "📅 맞춤형 훈련 플랜 생성기",
    "💬 AI 코치 1:1 상담실",
])

# ----------------- TAB 1: PMC DASHBOARD -----------------
with tab1:
    st.subheader("📈 훈련 부하 및 컨디션 관리 차트 (PMC: Performance Management Chart)")
    st.caption("파란선: 장기 체력(CTL, 42일) | 주황 점선: 단기 피로도(ATL, 7일) | 막대: 컨디션 폼(TSB = 전일 CTL - 전일 ATL)")

    fig_pmc = go.Figure()
    colors = ["#22c55e" if val >= 0 else "#ef4444" for val in pmc_df["tsb"]]
    fig_pmc.add_trace(
        go.Bar(
            x=pmc_df.index,
            y=pmc_df["tsb"],
            name="컨디션 (TSB)",
            marker_color=colors,
            opacity=0.65,
        )
    )
    fig_pmc.add_trace(
        go.Scatter(
            x=pmc_df.index,
            y=pmc_df["ctl"],
            name="체력 (CTL / Fitness)",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig_pmc.add_trace(
        go.Scatter(
            x=pmc_df.index,
            y=pmc_df["atl"],
            name="피로도 (ATL / Fatigue)",
            line=dict(color="#f97316", width=2, dash="dot"),
        )
    )
    fig_pmc.update_layout(
        height=370,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="날짜", showgrid=True),
        yaxis=dict(title="TSS / 지수", showgrid=True),
        hovermode="x unified",
        template="plotly_dark",
    )
    st.plotly_chart(fig_pmc, use_container_width=True)

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

    col_pz, col_hrz = st.columns(2)
    with col_pz:
        st.subheader(f"⚡ Coggan 7 파워존 (FTP {ftp_input}W 기준)")
        zones = get_power_zones(ftp_input)
        zone_data = []
        for k, v in zones.items():
            low, high = v["range"]
            zone_data.append({
                "파워 존": v["name"].split(":")[1].strip() if ":" in v["name"] else v["name"],
                "FTP 비율": v["pct"],
                "목표 와트 (W)": f"{low}W ~ {high}W",
                "훈련 목적": v["description"],
            })
        st.dataframe(pd.DataFrame(zone_data), use_container_width=True, hide_index=True)

    with col_hrz:
        st.subheader(f"❤️ 심박수 훈련 존 (LTHR {lthr_input}bpm / Max {max_hr_input}bpm)")
        hr_zones_table = [
            {
                "심박 존": "Zone 1 (회복 / 웜업)",
                "%LTHR 기준": f"< 81% LTHR",
                "심박수 (%LTHR 추천)": f"~ {round(lthr_input * 0.81)} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.50)} ~ {round(max_hr_input * 0.60)} bpm",
                "훈련 효과": "적극적 회복 및 혈류 촉진",
            },
            {
                "심박 존": "Zone 2 (유산소 지구력)",
                "%LTHR 기준": "81% ~ 89% LTHR",
                "심박수 (%LTHR 추천)": f"{round(lthr_input * 0.81)} ~ {round(lthr_input * 0.89)} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.60)} ~ {round(max_hr_input * 0.70)} bpm",
                "훈련 효과": "지방 대사 및 기초체력(Base) 양성",
            },
            {
                "심박 존": "Zone 3 (템포 / 에어로빅)",
                "%LTHR 기준": "90% ~ 93% LTHR",
                "심박수 (%LTHR 추천)": f"{round(lthr_input * 0.90)} ~ {round(lthr_input * 0.93)} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.70)} ~ {round(max_hr_input * 0.80)} bpm",
                "훈련 효과": "유산소 용량 및 크루징 속도 향상",
            },
            {
                "심박 존": "Zone 4 (젖산역치 / 한계치)",
                "%LTHR 기준": "94% ~ 99% LTHR",
                "심박수 (%LTHR 추천)": f"{round(lthr_input * 0.94)} ~ {lthr_input} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.80)} ~ {round(max_hr_input * 0.90)} bpm",
                "훈련 효과": "젖산 제거 능력 및 FTP 직접 향상",
            },
            {
                "심박 존": "Zone 5 (VO2 Max / 무산소)",
                "%LTHR 기준": f"100%+ LTHR (~Max)",
                "심박수 (%LTHR 추천)": f"{lthr_input + 1} ~ {max_hr_input} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.90)} ~ {max_hr_input} bpm",
                "훈련 효과": "최대 산소 섭취량 및 고강도 인터벌",
            },
        ]
        st.dataframe(pd.DataFrame(hr_zones_table), use_container_width=True, hide_index=True)

    st.subheader("🚴 최근 라이딩 활동 요약 (최근 12회)")
    if activities:
        recent_table = []
        for act in activities[:12]:
            recent_table.append({
                "날짜": act["start_date_local"][:16].replace("T", " "),
                "라이딩 / 워크아웃": act["name"],
                "거리(km)": round(act["distance"] / 1000.0, 1),
                "이동시간(분)": round(act["moving_time"] / 60.0, 1),
                "획득고도(m)": int(act.get("total_elevation_gain", 0)),
                "평균 파워(W)": round(act.get("average_watts", 0)),
                "NP(W)": round(act.get("weighted_average_watts") or act.get("average_watts", 0)),
                "평균/최대 심박": f"{int(act.get('average_heartrate') or 0)} / {int(act.get('max_heartrate') or 0)}",
                "케이던스": int(act.get("average_cadence") or 0),
                "EF(효율)": act.get("ef") or "-",
                "IF": round(act.get("intensity_factor", 0.0), 2),
                "TSS": round(act.get("tss", 0.0), 1),
            })
        st.dataframe(pd.DataFrame(recent_table), use_container_width=True, hide_index=True)


# ----------------- TAB 2: GARMIN 1040 DEEP DIVE -----------------
with tab2:
    st.subheader("🏆 가민 Edge 1040 실측 데이터 심층 분석 (2026.09.01 ~ 10.27)")

    if not garmin_raw_acts:
        st.warning("가민 활동 데이터를 불러올 수 없습니다.")
    else:
        road_acts = [a for a in garmin_raw_acts if a.get("sub_sport") == "road"]
        ind_acts = [a for a in garmin_raw_acts if a.get("sub_sport") == "indoor_cycling"]

        road_km = sum((a.get("distance_m") or 0) for a in road_acts) / 1000.0
        ind_km = sum((a.get("distance_m") or 0) for a in ind_acts) / 1000.0
        road_h = sum((a.get("timer_time_s") or 0) for a in road_acts) / 3600.0
        ind_h = sum((a.get("timer_time_s") or 0) for a in ind_acts) / 3600.0
        road_ef = np.mean([a["ef"] for a in road_acts if a.get("ef")])
        ind_ef = np.mean([a["ef"] for a in ind_acts if a.get("ef")])
        road_cad = np.mean([a["avg_cadence"] for a in road_acts if a.get("avg_cadence")])
        ind_cad = np.mean([a["avg_cadence"] for a in ind_acts if a.get("avg_cadence")])

        c_sum1, c_sum2, c_sum3 = st.columns(3)
        with c_sum1:
            st.info(
                f"🛣️ **야외 도로 라이딩 (`도로`, {len(road_acts)}회)**\n"
                f"- 누적 거리: **{road_km:,.1f} km** ({road_h:.1f}시간)\n"
                f"- 누적 획득고도: **18,105 m** (회당 평균 862m)\n"
                f"- 평균 NP: **148.2 W** | 평균 EF: **{road_ef:.2f}** | 평균 케이던스: **{road_cad:.1f} rpm**"
            )
        with c_sum2:
            st.info(
                f"🏠 **실내 스마트 트레이너 (`실내`, {len(ind_acts)}회)**\n"
                f"- 누적 거리: **{ind_km:,.1f} km** ({ind_h:.1f}시간)\n"
                f"- 구조화 워크아웃: **스프린트(7회), 기초체력(5회), 템포(2회), 한계치(2회), 회복(2회), VO2Max(1회)**\n"
                f"- 평균 NP: **127.2 W** | 평균 EF: **{ind_ef:.2f}** | 평균 케이던스: **{ind_cad:.1f} rpm**"
            )
        with c_sum3:
            st.success(
                f"🎯 **핵심 피지컬 진화 요약**\n"
                f"- 체중: **66.6 kg → 64.5 kg (-2.1 kg 감량)**\n"
                f"- 실전 적정 FTP: **196 W (3.04 W/kg)** (잠재 올아웃 **200~205 W**)\n"
                f"- 심박 검증: **LTHR 161 bpm / 최대심박 184 bpm** (9/5 1분 평균 181.8bpm 실측)"
            )

        # 1) Weekly Volume, Elevation & TSS Chart
        st.markdown("#### 1️⃣ 주간 주행 거리(도로 vs 실내), 획득고도(m) 및 주간 TSS 추이")
        weeks = defaultdict(list)
        for a in garmin_raw_acts:
            dt = datetime.strptime(a["date"], "%Y-%m-%d")
            monday = dt - timedelta(days=dt.weekday())
            weeks[monday].append(a)

        sorted_mondays = sorted(weeks.keys())
        w_labels = [f"{i+1}주차 ({m.strftime('%m/%d')})" for i, m in enumerate(sorted_mondays)]
        w_road_km = [sum((a.get("distance_m") or 0)/1000.0 for a in weeks[m] if a.get("sub_sport") == "road") for m in sorted_mondays]
        w_ind_km = [sum((a.get("distance_m") or 0)/1000.0 for a in weeks[m] if a.get("sub_sport") == "indoor_cycling") for m in sorted_mondays]
        w_elev = [sum((a.get("ascent_m") or 0) for a in weeks[m]) for m in sorted_mondays]
        w_tss = [sum((a.get("tss") or 0) for a in weeks[m]) for m in sorted_mondays]

        fig_wk = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=("주간 거리 (km) & 획득고도 (m)", "주간 누적 훈련 스트레스 점수 (Weekly TSS)"),
        )
        fig_wk.add_trace(go.Bar(x=w_labels, y=w_road_km, name="야외 도로 (km)", marker_color="#38bdf8"), row=1, col=1, secondary_y=False)
        fig_wk.add_trace(go.Bar(x=w_labels, y=w_ind_km, name="실내 트레이너 (km)", marker_color="#818cf8"), row=1, col=1, secondary_y=False)
        fig_wk.add_trace(
            go.Scatter(x=w_labels, y=w_elev, name="획득고도 (m)", mode="lines+markers", line=dict(color="#fbbf24", width=3)),
            row=1, col=1, secondary_y=True,
        )
        fig_wk.add_trace(go.Bar(x=w_labels, y=w_tss, name="주간 TSS", marker_color="#f43f5e"), row=2, col=1)
        fig_wk.update_layout(barmode="stack", height=520, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_wk, use_container_width=True)

        # 2) Peak Power Curve (MMP) & Zone Distributions
        st.markdown("#### 2️⃣ 피크 파워 커브 (1초 ~ 60분 MMP) 및 심박·파워 존 체류 분포")
        col_mmp, col_zdist = st.columns([1.15, 1])

        dur_keys = [
            ("1초(Max)", "max_power"), ("5초", "mmp_5s"), ("15초", "mmp_15s"), ("30초", "mmp_30s"),
            ("1분", "mmp_60s"), ("3분", "mmp_180s"), ("5분", "mmp_300s"), ("10분", "mmp_600s"),
            ("20분", "mmp_1200s"), ("30분", "mmp_1800s"), ("60분", "mmp_3600s"),
        ]
        sept_acts = [a for a in garmin_raw_acts if a["date"].startswith("2026-09")]
        oct_acts = [a for a in garmin_raw_acts if a["date"].startswith("2026-10")]

        s_peaks = [max([(a.get("max_power") if k == "max_power" else (a.get("mmp", {}).get(k) or 0)) or 0 for a in sept_acts]) for _, k in dur_keys]
        o_peaks = [max([(a.get("max_power") if k == "max_power" else (a.get("mmp", {}).get(k) or 0)) or 0 for a in oct_acts]) for _, k in dur_keys]
        all_peaks = [max(s, o) for s, o in zip(s_peaks, o_peaks)]

        with col_mmp:
            fig_mmp = go.Figure()
            fig_mmp.add_trace(go.Scatter(
                x=[d[0] for d in dur_keys], y=s_peaks, mode="lines+markers",
                name="9월 최고 파워 (W)", line=dict(color="#38bdf8", width=3),
            ))
            fig_mmp.add_trace(go.Scatter(
                x=[d[0] for d in dur_keys], y=o_peaks, mode="lines+markers",
                name="10월 최고 파워 (W)", line=dict(color="#f43f5e", width=3, dash="dash"),
            ))
            fig_mmp.update_layout(
                title="구간별 최고 파워 커브 (9월 vs 10월)",
                xaxis_title="지속 시간", yaxis_title="파워 (Watts)",
                height=360, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_mmp, use_container_width=True)

        with col_zdist:
            pwr_z = np.zeros(7)
            for a in garmin_raw_acts:
                tiz = a.get("time_in_power_zone")
                if tiz:
                    for i in range(1, min(8, len(tiz))):
                        if tiz[i] is not None:
                            pwr_z[i - 1] += float(tiz[i]) / 3600.0
            pwr_pct = (pwr_z / max(0.001, np.sum(pwr_z))) * 100.0
            pwr_labels = ["Z1 회복", "Z2 지구력", "Z3 템포", "Z4 역치", "Z5 VO2Max", "Z6 무산소", "Z7 신경근"]
            fig_pz = go.Figure(go.Bar(
                x=pwr_pct, y=pwr_labels, orientation="h",
                text=[f"{p:.1f}% ({h:.1f}h)" for p, h in zip(pwr_pct, pwr_z)],
                textposition="auto",
                marker_color=["#94a3b8", "#38bdf8", "#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7"],
            ))
            fig_pz.update_layout(
                title="8주 누적 파워 존(Z1~Z7) 체류 비율",
                xaxis_title="비율 (%)", height=360, template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pz, use_container_width=True)

        # 3) Aerobic Efficiency (EF) & Pedaling Balance
        st.markdown("#### 3️⃣ 유산소 효율 지수(EF = NP / 평균심박) 향상 추이 & 좌/우 페달링 밸런스")
        col_ef, col_lr = st.columns(2)

        with col_ef:
            fig_ef = go.Figure()
            r_ef_acts = [a for a in road_acts if a.get("ef") and (a.get("timer_time_s") or 0) >= 1800]
            i_ef_acts = [a for a in ind_acts if a.get("ef") and (a.get("timer_time_s") or 0) >= 1800]
            fig_ef.add_trace(go.Scatter(
                x=[a["date"] for a in r_ef_acts],
                y=[a["ef"] for a in r_ef_acts],
                mode="markers+lines",
                name="야외 도로 EF (NP/HR)",
                marker=dict(size=10, color="#fb923c"),
                line=dict(color="#fb923c", width=2),
            ))
            fig_ef.add_trace(go.Scatter(
                x=[a["date"] for a in i_ef_acts],
                y=[a["ef"] for a in i_ef_acts],
                mode="markers",
                name="실내 트레이너 EF",
                marker=dict(size=9, color="#a78bfa", symbol="diamond"),
            ))
            fig_ef.update_layout(
                title="유산소 효율 지수 (EF = NP / HR) 추이",
                xaxis_title="날짜", yaxis_title="EF (W/bpm)",
                height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        with col_lr:
            lr_acts = [a for a in garmin_raw_acts if a.get("left_pct") is not None]
            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(
                x=[a["date"] for a in lr_acts],
                y=[a["left_pct"] for a in lr_acts],
                mode="lines+markers",
                name="왼발 파워 비중 (%)",
                marker=dict(size=9, color="#38bdf8"),
                line=dict(color="#38bdf8", width=2),
            ))
            fig_lr.add_hline(y=50.0, line_dash="dash", line_color="#22c55e", annotation_text="50.0% 대칭 기준선")
            fig_lr.update_layout(
                title="좌/우 페달링 밸런스 추이 (왼발 %)",
                xaxis_title="날짜", yaxis_title="왼발 비중 (%)",
                height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_lr, use_container_width=True)

        # 4) Full 40-Ride Detailed Table
        st.markdown(f"#### 4️⃣ 전체 가민 라이딩 상세 로그 (총 {len(garmin_raw_acts)}회)")
        full_rows = []
        for idx, a in enumerate(reversed(garmin_raw_acts), 1):
            dist_km = (a.get("distance_m") or 0) / 1000.0
            dur_m = (a.get("timer_time_s") or 0) / 60.0
            spd = (a.get("avg_speed_ms") or 0) * 3.6
            prof = a.get("sport_profile_name") or ""
            wkt = f" [{a['workout_name']}]" if a.get("workout_name") else ""
            full_rows.append({
                "순번": len(garmin_raw_acts) - idx + 1,
                "일시 (KST)": a.get("start_kst", "")[:16],
                "구분 [워크아웃]": f"{prof}{wkt}",
                "이동시간(분)": round(dur_m, 1),
                "거리(km)": round(dist_km, 1),
                "평균속도(km/h)": round(spd, 1),
                "획득고도(m)": int(a.get("ascent_m") or 0),
                "평균파워(W)": a.get("avg_power") or 0,
                "NP(W)": a.get("np") or 0,
                "최대파워(W)": a.get("max_power") or 0,
                "IF": round(float(a.get("if") or 0), 2),
                "TSS": round(float(a.get("tss") or 0), 1),
                "평균/최대심박": f"{a.get('avg_hr') or '-'} / {a.get('max_hr') or '-'}",
                "케이던스(rpm)": a.get("avg_cadence") or 0,
                "효율(EF)": a.get("ef") or "-",
                "디커플링(%)": a.get("decoupling_pct") if a.get("decoupling_pct") is not None else "-",
                "스태미나(시작→종료)": f"{a.get('stamina_start')}→{a.get('stamina_end')}%" if a.get("stamina_end") is not None else "-",
                "좌/우 밸런스": a.get("lr_str", "-"),
            })
        st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)


# ----------------- TAB 3: TRAINING PLAN GENERATOR -----------------
with tab3:
    st.subheader("📅 목표 맞춤형 사이클링 훈련 플랜 설계")
    st.markdown("목표 이벤트와 주당 가용 시간을 설정하면 현재 FTP와 TSB 컨디션을 반영한 체계적인 주기화 훈련표를 설계합니다.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        plan_goal = st.selectbox(
            "목표 이벤트 / 종목",
            [
                "FTP 향상 & 3.1+ W/kg 달성 (스윗스팟 및 역치 집중)",
                "그란폰도 / 장거리 산악 완주 (100km+ 지구력 & 업힐 케이던스)",
                "힐클라임 대회 (W/kg 강화 & VO2 Max)",
                "크리테리움 / 단거리 로드 레이스 (무산소 파워)",
                "동계 인도어 유산소 베이스 빌딩 (Z2 엔듀런스)",
            ],
            index=0,
        )
    with col_p2:
        plan_weeks = st.select_slider("훈련 기간 (주)", options=[4, 6, 8, 12], value=8)
    with col_p3:
        plan_hours = st.slider("주당 가용 훈련 시간", min_value=4.0, max_value=16.0, value=9.0, step=0.5)

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


# ----------------- TAB 4: CHATBOT CONSULTATION -----------------
with tab4:
    st.subheader("💬 AI 사이클링 코치 1:1 실시간 상담실")
    st.caption("가민 엣지 1040의 40회 주행 기록, FTP(196W), 심박존(LTHR 161bpm), 좌우 밸런스 데이터를 모두 기억하고 상담합니다.")

    mem_data = load_memory()
    athlete_notes = mem_data.get("athlete_notes", [])

    with st.expander("🧠 코치가 기억하고 있는 라이더 핵심 메모 (장기 기억)", expanded=False):
        st.markdown(
            "코치는 브라우저를 닫거나 다시 접속해도 아래 등록된 **가민 1040 분석 요약, 신체 지표, 목표**를 영구적으로 기억하고 상담에 반영합니다."
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

        col_new_in, col_new_btn = st.columns([6, 1])
        new_note_text = col_new_in.text_input(
            "새 메모 추가",
            placeholder="예: 2027년 설악 그란폰도 참가 목표, 업힐 시 케이던스 75rpm 이상 유지하기",
            label_visibility="collapsed",
        )
        if col_new_btn.button("➕ 추가", use_container_width=True):
            if new_note_text.strip():
                add_athlete_note(new_note_text)
                st.rerun()

    col_chat_title, col_chat_clear = st.columns([5, 1])
    with col_chat_clear:
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            clear_chat_history()
            st.session_state["messages"] = []
            st.rerun()

    st.write("💡 **자주 묻는 추천 질문:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_prompt = None
    if q_col1.button("🦵 10월 왼발 밸런스(53.5%) 교정법", use_container_width=True):
        quick_prompt = "9월에는 좌우 페달링 밸런스가 50:50이었는데 10월 야외 산악 라이딩에서 왼발 비중이 52~53.5%로 높아졌습니다. 클릿과 스트레칭을 어떻게 점검하면 좋을까요?"
    if q_col2.button("📈 FTP 196W → 210W 돌파 루틴", use_container_width=True):
        quick_prompt = "현재 FTP 196W(3.04 W/kg)에서 210W(3.25 W/kg)로 올리기 위해 평일 인도어 워크아웃과 주말 라이딩을 어떻게 구성할까요?"
    if q_col3.button("⛰️ 장거리 업힐 케이던스 & 스태미나", use_container_width=True):
        quick_prompt = "획득고도 2,000m 이상 장거리 산악 라이딩에서 평균 케이던스가 62~65rpm으로 떨어지고 후반 스태미나가 30% 이하로 고갈되는데 개선 요령을 알려주세요."
    if q_col4.button("❤️ LTHR 161bpm 기준 인도어 심박 관리", use_container_width=True):
        quick_prompt = "젖산역치 심박수(LTHR) 161bpm, 최대심박수 184bpm 기준으로 실내 기초체력(Z2)과 한계치(Z4) 워크아웃을 탈 때 적정 심박 범위를 알려주세요."

    if "messages" not in st.session_state or not st.session_state["messages"]:
        saved_msgs = mem_data.get("messages", [])
        if saved_msgs:
            st.session_state["messages"] = saved_msgs
        else:
            default_greeting = {
                "role": "assistant",
                "content": (
                    f"안녕하세요, 라이더님! 🚴 **가민 Edge 1040** 40회 실측 라이딩 데이터(**2,061.1km / +18,105m**)가 연동된 AI 코치입니다.\n"
                    f"현재 프로필은 **FTP {ftp_input}W ({w_per_kg} W/kg)**, **LTHR {lthr_input}bpm (Max {max_hr_input}bpm)**, 컨디션 폼(TSB)은 **{latest_pmc['tsb']:+.1f}**입니다.\n"
                    f"궁금하신 훈련 루틴이나 컨디션 관리법을 편하게 물어보세요!"
                ),
            }
            st.session_state["messages"] = [default_greeting]
            append_chat_message(default_greeting["role"], default_greeting["content"])

    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("코치에게 훈련 상담 질문을 입력하세요...")
    query = quick_prompt or user_input

    if query:
        st.session_state["messages"].append({"role": "user", "content": query})
        append_chat_message("user", query)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        last_ride = activities[0] if activities else {}
        last_ride_summary = (
            f"{last_ride.get('name', '라이딩')} (날짜: {last_ride.get('start_date_local', '')[:10]}, "
            f"거리: {round(last_ride.get('distance', 0)/1000, 1)}km, "
            f"NP: {round(last_ride.get('weighted_average_watts', 0))}W, "
            f"평균심박: {int(last_ride.get('average_heartrate') or 0)}bpm, TSS: {round(last_ride.get('tss', 0), 1)})"
            if last_ride else "기록 없음"
        )
        athlete_context = {
            "ftp": ftp_input,
            "weight_kg": weight_input,
            "lthr": lthr_input,
            "max_hr": max_hr_input,
            "rest_hr": rest_hr_input,
            "ctl": latest_pmc["ctl"],
            "atl": latest_pmc["atl"],
            "tsb": latest_pmc["tsb"],
            "last_ride_summary": last_ride_summary,
            "athlete_notes": load_memory().get("athlete_notes", []),
        }

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("코치가 가민 1040 실측 데이터와 장기 기억을 종합 분석 중입니다..."):
                    answer = coach_agent.consult(
                        user_message=query,
                        chat_history=st.session_state["messages"][:-1],
                        athlete_context=athlete_context,
                    )
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    append_chat_message("assistant", answer)
                    auto_extract_athlete_insight(coach_agent, query, answer)
        st.rerun()
