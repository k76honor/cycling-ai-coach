"""
AI Cycling Coach & Live Strava Training Dashboard (with Garmin .fit Baseline Reference)
Built with Streamlit, Plotly, Strava API v3, and Google Gemini.
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
import importlib
import garmin_loader as _gl_mod
import power_metrics as _pm_mod
import workout_builder as _wb_mod

importlib.reload(_gl_mod)
importlib.reload(_pm_mod)
importlib.reload(_wb_mod)

from strava_client import StravaClient, estimate_ftp_from_live_activities
from garmin_loader import (
    load_garmin_raw_activities,
    convert_garmin_to_dashboard_activities,
    parse_uploaded_fit_bytes,
)
from power_metrics import (
    get_power_zones,
    calculate_pmc,
    interpret_tsb,
    estimate_substrate_burn,
    project_pmc_future,
)
from workout_builder import (
    get_preset_workouts,
    build_garmin_workout_fit_bytes,
    build_zwift_zwo_xml,
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
    page_title="AI 사이클링 코치 & Strava 훈련 플래너",
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
        help="1시간 동안 지속할 수 있는 최대 평균 파워(FTP). 언제든 직접 수정하거나 아래 Strava 자동 산출값으로 갱신할 수 있습니다."
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

st.sidebar.divider()

# Data Source Section (Strava is #1 Default!)
st.sidebar.subheader("📡 데이터 소스 연동")
data_options = [
    "🔗 Strava 실시간 자동 연동 (기본)",
    "📋 가민 .fit 정밀 분석 데이터 (로컬/업로드 통합)",
]
data_mode = st.sidebar.radio("데이터 수집 방식", data_options, index=0)

s_client_id = STRAVA_CLIENT_ID
s_client_secret = STRAVA_CLIENT_SECRET
s_refresh_token = STRAVA_REFRESH_TOKEN

if data_mode.startswith("🔗"):
    if st.sidebar.button("🔄 Strava 최신 라이딩 즉시 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    with st.sidebar.expander("⚙️ Strava API 연결 설정", expanded=not bool(STRAVA_CLIENT_ID and STRAVA_REFRESH_TOKEN)):
        s_client_id = st.text_input("Client ID", value=STRAVA_CLIENT_ID)
        s_client_secret = st.text_input("Client Secret", value=STRAVA_CLIENT_SECRET, type="password")
        s_refresh_token = st.text_input("Refresh Token", value=STRAVA_REFRESH_TOKEN, type="password")

with st.sidebar.expander("📤 가민 Edge 1040 .fit 파일 추가 업로드", expanded=False):
    uploaded_fit_files = st.file_uploader(
        ".fit 원본 파일 업로드 (복수 선택 가능, GitHub 자동 동기화)",
        type=["fit"],
        accept_multiple_files=True,
    )

# ----------------- DATA PREPARATION -----------------
@st.cache_data(ttl=300)
def load_strava_activities(c_id: str, c_sec: str, r_tok: str, ftp: float):
    client = StravaClient(c_id, c_sec, r_tok)
    return client.get_activities(ftp=ftp, per_page=100)

garmin_ref_acts = load_garmin_raw_activities(uploaded_files=uploaded_fit_files)

if data_mode.startswith("🔗"):
    activities = load_strava_activities(
        c_id=s_client_id,
        c_sec=s_client_secret,
        r_tok=s_refresh_token,
        ftp=ftp_input,
    )
else:
    activities = convert_garmin_to_dashboard_activities(
        garmin_ref_acts, ftp=ftp_input, recalc_tss_with_ftp=True
    )

# Dynamic FTP Estimator in Sidebar (computed directly from live Strava rides!)
live_ftp_est = estimate_ftp_from_live_activities(activities, lthr=lthr_input)
st.sidebar.info(
    f"🤖 **최근 Strava 주행 기반 추정 FTP**\n"
    f"• 권장 실전 FTP: **`{live_ftp_est['recommended_ftp']} W`** (잠재 `{live_ftp_est['potential_max_ftp']} W`)\n"
    f"• 고강도 주행 평균 EF: `{live_ftp_est['avg_hard_ef']}`\n"
    f"• 50분+ 최고 NP: `{live_ftp_est['max_np_60m']} W`"
)
if st.sidebar.button("⚡ Strava 산출 권장 FTP 자동 적용", use_container_width=True):
    update_athlete_profile(
        ftp=live_ftp_est["recommended_ftp"],
        weight_kg=round(weight_input, 1),
        max_hr=max_hr_input,
        lthr=lthr_input,
        rest_hr=rest_hr_input,
    )
    st.rerun()

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
        "gemini-3.5-flash-lite (추천 ⭐ / 0.9초 초고속)",
        "gemini-3.6-flash (정밀 심층 분석 / 4초)",
        "gemini-3.5-flash (장문 훈련 플랜 특화)",
    ],
    index=0,
)
model_name_clean = model_name.split(" ")[0].strip()

coach_agent = CyclingCoachAgent(api_key=api_key_input or None, model_name=model_name_clean)


def build_pmc_dataframe(act_list: list) -> pd.DataFrame:
    if not act_list:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=45, freq="D")
        return calculate_pmc(pd.Series(0.0, index=dates))

    df_act = pd.DataFrame(act_list)
    # Use start_date_local[:10] for accurate KST local date grouping
    df_act["date"] = pd.to_datetime(df_act["start_date_local"].str[:10]).dt.date
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
st.title("🚴 AI 사이클링 코치 & Strava 실시간 파워 훈련 플래너")
st.markdown(
    "가민 엣지 1040에서 **Strava로 자동 동기화되는 실시간 라이딩 데이터**를 기반으로 "
    "**FTP(W/kg), 심박·파워존, 유산소 효율(EF), 체력(CTL)/피로도(ATL)/컨디션(TSB)**을 분석하고 맞춤형 코칭을 제공합니다."
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
    st.metric(f"연동 라이딩 ({len(activities)}회)", f"{total_km_all:,.1f} km", f"고도 +{total_elev_all:,.0f} m")
with k4:
    st.metric("체력 (CTL)", f"{latest_pmc['ctl']:.1f}", f"누적 {total_tss_all:,.0f} TSS")
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
    "🏆 주간 볼륨·유산소 효율 & FIT 참고치",
    "📅 맞춤형 훈련 플랜 생성기",
    "💬 AI 코치 1:1 상담실",
])

# ----------------- TAB 1: PMC DASHBOARD -----------------
with tab1:
    st.subheader("📈 훈련 부하 및 컨디션 관리 차트 (PMC: Performance Management Chart + 향후 7일 예측)")
    st.caption("파란선: 장기 체력(CTL, 42일) | 주황 점선: 단기 피로도(ATL, 7일) | 막대: 컨디션 폼(TSB = 전일 CTL - 전일 ATL) | 연한 점선: 휴식 시 향후 7일 회복 궤적")

    future_pmc_df = project_pmc_future(pmc_df, days=7, planned_daily_tss=0.0)
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
    if not future_pmc_df.empty:
        fut_colors = ["#86efac" if val >= 0 else "#fca5a5" for val in future_pmc_df["tsb"]]
        fig_pmc.add_trace(
            go.Bar(
                x=future_pmc_df.index,
                y=future_pmc_df["tsb"],
                name="예측 TSB (향후 7일 휴식 가정)",
                marker_color=fut_colors,
                opacity=0.35,
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
    if not future_pmc_df.empty:
        proj_ctl_x = [pmc_df.index[-1]] + list(future_pmc_df.index)
        proj_ctl_y = [float(pmc_df["ctl"].iloc[-1])] + list(future_pmc_df["ctl"])
        proj_atl_y = [float(pmc_df["atl"].iloc[-1])] + list(future_pmc_df["atl"])
        fig_pmc.add_trace(
            go.Scatter(
                x=proj_ctl_x,
                y=proj_ctl_y,
                name="예측 CTL (7일)",
                line=dict(color="#7dd3fc", width=2, dash="dash"),
            )
        )
        fig_pmc.add_trace(
            go.Scatter(
                x=proj_ctl_x,
                y=proj_atl_y,
                name="예측 ATL (7일)",
                line=dict(color="#fdba74", width=2, dash="dashdot"),
            )
        )
    fig_pmc.update_layout(
        height=380,
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
        st.info(
            f"**현재 상태 코치 한줄 진단**: {tsb_status['advice']}\n\n"
            f"🔥 **FatMax 골든존 요약**: 파워 **`{round(ftp_input * 0.60)}~{round(ftp_input * 0.72)}W`** (스위트스팟 `{round(ftp_input * 0.65)}~{round(ftp_input * 0.70)}W`), "
            f"심박수 **`{round(max_hr_input * 0.65)}~{round(max_hr_input * 0.75)}bpm`** (상한 캡 `{round(max_hr_input * 0.75) + 1}bpm` 미만)"
        )
    with col_diag_btn:
        run_diag = st.button("🔍 AI 종합 진단 리포트 생성", use_container_width=True)

    if run_diag:
        with st.spinner("AI 코치가 Strava 최신 라이딩 데이터와 파워 지표를 정밀 진단 중입니다..."):
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
        st.subheader(f"⚡ Coggan 파워존 + FatMax (FTP {ftp_input}W 기준)")
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
        st.subheader(f"❤️ 심박수 훈련 존 + FatMax (LTHR {lthr_input}bpm / Max {max_hr_input}bpm)")
        hr_zones_table = [
            {
                "심박 존": "Zone 1 (회복 / 웜업)",
                "%LTHR 기준": "< 81% LTHR",
                "심박수 (%LTHR 추천)": f"~ {round(lthr_input * 0.81)} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.50)} ~ {round(max_hr_input * 0.60)} bpm",
                "훈련 효과": "적극적 회복 및 혈류 촉진 (쿨다운 125bpm 이하 필수)",
            },
            {
                "심박 존": "🔥 FatMax 골든존 (지방연소 극대화)",
                "%LTHR 기준": "75% ~ 85% LTHR",
                "심박수 (%LTHR 추천)": f"{round(lthr_input * 0.75)} ~ {round(lthr_input * 0.85)} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.65)} ~ {round(max_hr_input * 0.75)} bpm",
                "훈련 효과": f"분당 지방 산화율 최고점 (상한 캡 {round(max_hr_input * 0.75) + 1}bpm 미만 유지)",
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
                "%LTHR 기준": "100%+ LTHR (~Max)",
                "심박수 (%LTHR 추천)": f"{lthr_input + 1} ~ {max_hr_input} bpm",
                "심박수 (%MaxHR 참고)": f"{round(max_hr_input * 0.90)} ~ {max_hr_input} bpm",
                "훈련 효과": "최대 산소 섭취량 및 고강도 인터벌",
            },
        ]
        st.dataframe(pd.DataFrame(hr_zones_table), use_container_width=True, hide_index=True)

    st.subheader(f"🚴 실시간 연동 라이딩 활동 내역 및 지방·탄수화물 연소량 추정 (총 {len(activities)}회)")
    if activities:
        recent_table = []
        for act in activities:
            avg_w = float(act.get("average_watts") or 0.0)
            np_w = float(act.get("weighted_average_watts") or avg_w)
            mov_s = float(act.get("moving_time") or 0.0)
            if_val = float(act.get("intensity_factor") or (np_w / ftp_input if ftp_input > 0 else 0.0))
            sub = estimate_substrate_burn(avg_watts=avg_w, duration_seconds=mov_s, intensity_factor=if_val)
            recent_table.append({
                "날짜": act["start_date_local"][:16].replace("T", " "),
                "라이딩명": act["name"],
                "구분": "실내" if act.get("sub_sport") == "indoor_cycling" else "야외 도로",
                "거리(km)": round(act["distance"] / 1000.0, 1),
                "이동시간(분)": round(mov_s / 60.0, 1),
                "획득고도(m)": int(act.get("total_elevation_gain", 0)),
                "평균 파워(W)": round(avg_w),
                "NP(W)": round(np_w),
                "평균/최대 심박": f"{int(act.get('average_heartrate') or 0)} / {int(act.get('max_heartrate') or 0)}",
                "케이던스": int(act.get("average_cadence") or 0),
                "EF(효율)": act.get("ef") or "-",
                "IF": round(if_val, 2),
                "TSS": round(act.get("tss", 0.0), 1),
                "소모열량(kcal)": sub["kcal"],
                "🔥지방 / 🍚탄수(g)": f"🔥{sub['fat_g']}g ({sub['fat_pct']}%) / 🍚{sub['carb_g']}g",
            })
        st.dataframe(pd.DataFrame(recent_table), use_container_width=True, hide_index=True)


# ----------------- TAB 2: LIVE STRAVA DEEP DIVE + FIT BASELINE REFERENCE -----------------
with tab2:
    st.subheader("🏆 실시간 Strava 주간 볼륨·유산소 효율 추이 & 가민 `.fit` 정밀 분석 참고치")

    if activities:
        # Sort chronological for trend charts
        chrono_acts = list(reversed(activities))
        road_live = [a for a in chrono_acts if a.get("sub_sport") != "indoor_cycling"]
        ind_live = [a for a in chrono_acts if a.get("sub_sport") == "indoor_cycling"]

        r_km = sum(a.get("distance", 0) for a in road_live) / 1000.0
        i_km = sum(a.get("distance", 0) for a in ind_live) / 1000.0
        r_h = sum(a.get("moving_time", 0) for a in road_live) / 3600.0
        i_h = sum(a.get("moving_time", 0) for a in ind_live) / 3600.0
        r_ef_vals = [a["ef"] for a in road_live if a.get("ef")]
        i_ef_vals = [a["ef"] for a in ind_live if a.get("ef")]
        r_cad_vals = [a["average_cadence"] for a in road_live if a.get("average_cadence")]
        i_cad_vals = [a["average_cadence"] for a in ind_live if a.get("average_cadence")]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(
                f"🛣️ **야외 도로 라이딩 ({len(road_live)}회 - Strava 실시간)**\n"
                f"- 누적 거리: **{r_km:,.1f} km** ({r_h:.1f}시간)\n"
                f"- 누적 획득고도: **{sum(a.get('total_elevation_gain', 0) for a in road_live):,.0f} m**\n"
                f"- 평균 EF(효율): **{np.mean(r_ef_vals):.2f}** | 평균 케이던스: **{np.mean(r_cad_vals):.1f} rpm**"
                if r_ef_vals and r_cad_vals else f"🛣️ **야외 도로 라이딩 ({len(road_live)}회)**"
            )
        with c2:
            st.info(
                f"🏠 **실내 스마트 트레이너 ({len(ind_live)}회 - Strava 실시간)**\n"
                f"- 누적 거리: **{i_km:,.1f} km** ({i_h:.1f}시간)\n"
                f"- 누적 TSS: **{sum(a.get('tss', 0) for a in ind_live):,.0f} TSS**\n"
                f"- 평균 EF(효율): **{np.mean(i_ef_vals):.2f}** | 평균 케이던스: **{np.mean(i_cad_vals):.1f} rpm**"
                if i_ef_vals and i_cad_vals else f"🏠 **실내 트레이너 ({len(ind_live)}회)**"
            )
        with c3:
            st.success(
                f"📌 **2026.09~10 `.fit` 원본 정밀 검증 기준치**\n"
                f"- 체중 감량: **66.6 kg → 64.5 kg (-2.1 kg)**\n"
                f"- 실전 권장 FTP: **196 W (3.04 W/kg)** | 잠재 **200~205 W**\n"
                f"- 심박 검증: **LTHR 161 bpm / 최대심박 184 bpm** (9/5 1분 평균 181.8bpm)"
            )

        # 1) Live Weekly Volume, Elevation & TSS from Strava
        st.markdown("#### 1️⃣ [Strava 실시간] 주간 주행 거리(도로 vs 실내), 획득고도(m) 및 주간 TSS 추이")
        weeks = defaultdict(list)
        for a in chrono_acts:
            d_str = a["start_date_local"][:10]
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            monday = dt - timedelta(days=dt.weekday())
            weeks[monday].append(a)

        sorted_mondays = sorted(weeks.keys())
        w_labels = [f"{i+1}주차 ({m.strftime('%m/%d')})" for i, m in enumerate(sorted_mondays)]
        w_road_km = [sum(a.get("distance", 0)/1000.0 for a in weeks[m] if a.get("sub_sport") != "indoor_cycling") for m in sorted_mondays]
        w_ind_km = [sum(a.get("distance", 0)/1000.0 for a in weeks[m] if a.get("sub_sport") == "indoor_cycling") for m in sorted_mondays]
        w_elev = [sum(a.get("total_elevation_gain", 0) for a in weeks[m]) for m in sorted_mondays]
        w_tss = [sum(a.get("tss", 0) for a in weeks[m]) for m in sorted_mondays]

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
        fig_wk.update_layout(barmode="stack", height=500, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_wk, use_container_width=True)

        # 2) Live Aerobic Efficiency (EF) & Cadence vs NP from Strava
        st.markdown("#### 2️⃣ [Strava 실시간] 유산소 효율 지수(EF = NP / 심박수) 추이 & 실내·야외 케이던스 비교")
        col_ef, col_cad = st.columns(2)
        with col_ef:
            fig_ef = go.Figure()
            r_ef_acts = [a for a in road_live if a.get("ef") and (a.get("moving_time") or 0) >= 1800]
            i_ef_acts = [a for a in ind_live if a.get("ef") and (a.get("moving_time") or 0) >= 1800]
            fig_ef.add_trace(go.Scatter(
                x=[a["start_date_local"][:10] for a in r_ef_acts],
                y=[a["ef"] for a in r_ef_acts],
                text=[a["name"] for a in r_ef_acts],
                mode="markers+lines",
                name="야외 도로 EF (NP/HR)",
                marker=dict(size=10, color="#fb923c"),
                line=dict(color="#fb923c", width=2),
            ))
            fig_ef.add_trace(go.Scatter(
                x=[a["start_date_local"][:10] for a in i_ef_acts],
                y=[a["ef"] for a in i_ef_acts],
                text=[a["name"] for a in i_ef_acts],
                mode="markers",
                name="실내 트레이너 EF",
                marker=dict(size=9, color="#a78bfa", symbol="diamond"),
            ))
            fig_ef.update_layout(
                title="유산소 효율 지수 (EF = NP / 평균심박) 변화 추이",
                xaxis_title="날짜", yaxis_title="EF (W/bpm)",
                height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        with col_cad:
            fig_cad = go.Figure()
            fig_cad.add_trace(go.Scatter(
                x=[a["average_cadence"] for a in road_live if a.get("average_cadence")],
                y=[a.get("weighted_average_watts") or a.get("average_watts") for a in road_live if a.get("average_cadence")],
                text=[f"{a['start_date_local'][:10]} {a['name']}" for a in road_live if a.get("average_cadence")],
                mode="markers",
                name="야외 도로",
                marker=dict(size=10, color="#38bdf8"),
            ))
            fig_cad.add_trace(go.Scatter(
                x=[a["average_cadence"] for a in ind_live if a.get("average_cadence")],
                y=[a.get("weighted_average_watts") or a.get("average_watts") for a in ind_live if a.get("average_cadence")],
                text=[f"{a['start_date_local'][:10]} {a['name']}" for a in ind_live if a.get("average_cadence")],
                mode="markers",
                name="실내 트레이너",
                marker=dict(size=10, color="#a78bfa", symbol="diamond"),
            ))
            fig_cad.update_layout(
                title="평균 케이던스(RPM) vs 정규화 파워(NP)",
                xaxis_title="평균 케이던스 (RPM)", yaxis_title="NP (Watts)",
                height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_cad, use_container_width=True)

    st.divider()

    # 3) Reference Baseline from Garmin .fit files
    with st.expander(f"📋 [정밀 분석] 가민 .fit 원본 ({len(garmin_ref_acts)}회) 피크 파워 커브(MMP) & 양발 좌/우 밸런스", expanded=True):
        st.caption("원본 .fit 파일에서만 추출 가능한 초 단위 피크 파워 커브(MMP)와 양발 파워미터 좌/우 밸런스 데이터입니다. (사이드바에서 새 .fit 파일을 업로드하면 즉시 반영됩니다)")
        if garmin_ref_acts:
            col_ref1, col_ref2 = st.columns(2)
            dur_keys = [
                ("1초(Max)", "max_power"), ("5초", "mmp_5s"), ("15초", "mmp_15s"), ("30초", "mmp_30s"),
                ("1분", "mmp_60s"), ("3분", "mmp_180s"), ("5분", "mmp_300s"), ("10분", "mmp_600s"),
                ("20분", "mmp_1200s"), ("30분", "mmp_1800s"), ("60분", "mmp_3600s"),
            ]
            months = sorted({a["date"][:7] for a in garmin_ref_acts if a.get("date")})
            if len(months) >= 2:
                m_prev, m_curr = months[-2], months[-1]
                prev_acts = [a for a in garmin_ref_acts if a["date"].startswith(m_prev)]
                curr_acts = [a for a in garmin_ref_acts if a["date"].startswith(m_curr)]
                label_prev, label_curr = f"{m_prev} 최고 파워 (W)", f"{m_curr} 최고 파워 (W)"
            else:
                half = max(1, len(garmin_ref_acts) // 2)
                prev_acts = garmin_ref_acts[:half]
                curr_acts = garmin_ref_acts[half:] or garmin_ref_acts
                label_prev, label_curr = "전반기 최고 파워 (W)", "후반기 최고 파워 (W)"

            s_peaks = [max([((a.get("max_power") if k == "max_power" else (a.get("mmp", {}).get(k) or 0)) or 0) for a in prev_acts] or [0]) for _, k in dur_keys]
            o_peaks = [max([((a.get("max_power") if k == "max_power" else (a.get("mmp", {}).get(k) or 0)) or 0) for a in curr_acts] or [0]) for _, k in dur_keys]

            with col_ref1:
                fig_mmp = go.Figure()
                fig_mmp.add_trace(go.Scatter(
                    x=[d[0] for d in dur_keys], y=s_peaks, mode="lines+markers",
                    name=label_prev, line=dict(color="#38bdf8", width=3),
                ))
                fig_mmp.add_trace(go.Scatter(
                    x=[d[0] for d in dur_keys], y=o_peaks, mode="lines+markers",
                    name=label_curr, line=dict(color="#f43f5e", width=3, dash="dash"),
                ))
                fig_mmp.update_layout(
                    title="구간별 피크 파워 커브 (1초 ~ 60분 MMP)",
                    xaxis_title="지속 시간", yaxis_title="파워 (Watts)",
                    height=340, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_mmp, use_container_width=True)

            with col_ref2:
                lr_acts = [a for a in garmin_ref_acts if a.get("left_pct") is not None]
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=[a["date"] for a in lr_acts],
                    y=[a["left_pct"] for a in lr_acts],
                    mode="lines+markers",
                    name="왼발 파워 비중 (%)",
                    marker=dict(size=8, color="#38bdf8"),
                    line=dict(color="#38bdf8", width=2),
                ))
                fig_lr.add_hline(y=50.0, line_dash="dash", line_color="#22c55e", annotation_text="50.0% 대칭 기준선")
                fig_lr.update_layout(
                    title="좌/우 페달링 밸런스 추이",
                    xaxis_title="날짜", yaxis_title="왼발 비중 (%)",
                    height=340, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_lr, use_container_width=True)


# ----------------- TAB 3: TRAINING PLAN GENERATOR & WORKOUT FILE EXPORTER -----------------
with tab3:
    st.subheader("⚡ 원클릭 가민 Edge 1040 (`.fit`) & Zwift/Intervals.icu (`.zwo`) 워크아웃 파일 생성기")
    st.markdown(
        f"현재 설정된 **FTP `{ftp_input}W`**, **LTHR `{lthr_input}bpm`**, **최대심박 `{max_hr_input}bpm`**에 맞춰 와트(W) 타겟이 자동 계산된 실전 워크아웃 파일을 즉시 다운로드할 수 있습니다."
    )

    st.success(
        f"**📋 탑재된 4대 실전 맞춤형 워크아웃 요약 (가민 순정 바이너리 `.fit` 즉시 다운로드)**\n"
        f"- 🔥 **1. FatMax 90분 서지(Surge) 코스**: `{round(ftp_input*0.65)}~{round(ftp_input*0.70)}W` 크루징 + 20분마다 20초 `{round(ftp_input*1.17)}~{round(ftp_input*1.33)}W` 서지 3회 (지루함 방지 & 지방분해 호르몬 자극)\n"
        f"- 🔥 **2. FatMax 60분 정석 인도어 코스**: `{round(ftp_input*0.64)}~{round(ftp_input*0.70)}W` 평일 체지방 감량 & 유산소 효율(EF) 극대화 (심박 138bpm 미만 캡)\n"
        f"- ⚡ **3. SST 스윗스팟 3x12분 + 10분 혈압안정 쿨다운**: `{round(ftp_input*0.88)}~{round(ftp_input*0.94)}W` + 알닥톤 복용 맞춤 10분 Z1 쿨다운\n"
        f"- 🚀 **4. Z4 역치 4x8분 (FTP 210W 돌파) + 10분 쿨다운**: `{round(ftp_input*0.96)}~{round(ftp_input*1.04)}W` + 10분 Z1 혈압안정 쿨다운\n\n"
        f"**📲 가민 Edge 1040에 넣는 법 (1분 완료)**:\n"
        f"1. 아래에서 원하는 코스를 선택하고 **`📥 가민 Edge 1040 전용 워크아웃 (.fit) 다운로드`** 버튼을 누릅니다.\n"
        f"2. 가민 Edge 1040을 USB 케이블로 PC에 연결한 뒤 **`Garmin\\NewFiles`** 폴더 안에 다운로드한 `.fit` 파일을 복사해 넣습니다.\n"
        f"3. USB 연결을 해제하고 Edge 1040 전원을 켜면 **[훈련(Training) > 워크아웃(Workouts)]** 메뉴에 즉시 생성되어 주행 중 목표 와트 범위와 20초 서지 알람을 자동으로 띄워줍니다!"
    )

    presets = get_preset_workouts(ftp=ftp_input, lthr=lthr_input, max_hr=max_hr_input)
    preset_titles = list(presets.keys())
    selected_title = st.selectbox("🎯 다운로드할 맞춤 워크아웃 선택", preset_titles, index=0)
    selected_wo = presets[selected_title]
    wo_desc = selected_wo.get("description") or selected_wo.get("summary", "")
    wo_short = selected_wo.get("short_name") or selected_wo.get("fit_name", "Garmin_Workout")

    st.info(f"💡 **선택한 코스 코치 설계 의도**: {wo_desc}")

    wo_steps = selected_wo["steps"]
    step_rows = []
    bar_x = []
    bar_y = []
    bar_colors = []
    bar_texts = []
    for idx_s, stp in enumerate(wo_steps, 1):
        dur_m = round(stp["duration_sec"] / 60.0, 1)
        mid_w = round((stp["low_w"] + stp["high_w"]) / 2.0)
        step_rows.append({
            "순서": f"Step {idx_s}",
            "구간명": stp["name"],
            "지속 시간(분)": f"{dur_m}분",
            "목표 파워(W)": f"{stp['low_w']}W ~ {stp['high_w']}W",
            "평균 타겟": f"{mid_w}W ({round(mid_w / ftp_input * 100)}% FTP)",
            "심박/코칭 가이드": stp.get("hr_guide", "-"),
        })
        bar_x.append(f"{idx_s}. {stp['name']} ({dur_m}m)")
        bar_y.append(mid_w)
        if stp["intensity"] in ("warmup", "cooldown", "rest", 1, 2, 3):
            bar_colors.append("#38bdf8")
        elif mid_w >= ftp_input * 0.94:
            bar_colors.append("#ef4444")
        elif mid_w >= ftp_input * 0.85:
            bar_colors.append("#f97316")
        else:
            bar_colors.append("#22c55e")
        bar_texts.append(f"{stp['low_w']}~{stp['high_w']}W")

    col_wo_chart, col_wo_table = st.columns([3, 2])
    with col_wo_chart:
        fig_wo = go.Figure(
            go.Bar(
                x=bar_x,
                y=bar_y,
                marker_color=bar_colors,
                text=bar_texts,
                textposition="auto",
                name="목표 파워 (W)",
            )
        )
        fig_wo.add_hline(y=ftp_input, line_dash="dash", line_color="#f43f5e", annotation_text=f"FTP {ftp_input}W")
        fig_wo.update_layout(
            title=f"워크아웃 구간별 목표 파워 프로파일 ({wo_short})",
            xaxis_title="구간 (Step)",
            yaxis_title="목표 파워 (Watts)",
            height=320,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_wo, use_container_width=True)

    with col_wo_table:
        st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True, height=320)

    fit_bytes = build_garmin_workout_fit_bytes(workout_name=wo_short, steps=wo_steps)
    zwo_xml = build_zwift_zwo_xml(
        workout_name=wo_short,
        description=wo_desc,
        ftp=ftp_input,
        steps=wo_steps,
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "📥 가민 Edge 1040 전용 워크아웃 (.fit) 다운로드",
            data=fit_bytes,
            file_name=f"{wo_short}_FTP{ftp_input}W.fit",
            mime="application/octet-stream",
            type="primary",
            use_container_width=True,
        )
        st.caption("📌 **가민 1040 넣는 법**: Edge 1040을 PC에 USB로 연결 → `Garmin\\NewFiles` 폴더에 다운로드한 `.fit` 파일을 복사 후 연결 해제하면 **훈련 > 워크아웃**에 즉시 생성됩니다!")
    with dl_col2:
        st.download_button(
            "📥 Zwift / Intervals.icu 워크아웃 (.zwo) 다운로드",
            data=zwo_xml.encode("utf-8"),
            file_name=f"{wo_short}_FTP{ftp_input}W.zwo",
            mime="application/xml",
            use_container_width=True,
        )
        st.caption("📌 **Zwift / Intervals.icu 넣는 법**: `Documents\\Zwift\\Workouts` 폴더에 넣거나 Intervals.icu 워크아웃 빌더에 업로드하면 가민 커넥트 달력과 무선 동기화할 수도 있습니다.")

    st.divider()

    st.subheader("📅 AI 다주간 주기화 훈련 플랜 설계기")
    st.markdown("목표 이벤트와 주당 가용 시간을 설정하면 현재 FTP와 TSB 컨디션을 반영한 체계적인 다주간 주기화 훈련표를 설계합니다.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        plan_goal = st.selectbox(
            "목표 이벤트 / 종목",
            [
                "FTP 향상 & 3.1+ W/kg 달성 (스윗스팟 및 역치 집중)",
                "FatMax 체지방 감량 & 유산소 엔진 극대화 (64.5kg → 62.5kg 목표)",
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

    generate_plan_btn = st.button("🚀 AI 맞춤형 다주간 훈련 플랜 생성하기", type="primary", use_container_width=True)

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
    st.caption("Strava 실시간 라이딩 기록과 가민 1040 정밀 분석 참고치(FTP 196W, LTHR 161bpm, 피크 파워, 좌우 밸런스)를 모두 반영해 상담합니다.")

    mem_data = load_memory()
    athlete_notes = mem_data.get("athlete_notes", [])

    with st.expander("🧠 코치가 기억하고 있는 라이더 핵심 메모 (장기 기억 & FIT 참고치)", expanded=False):
        st.markdown(
            "코치는 브라우저를 닫거나 다시 접속해도 아래 등록된 **가민 1040 분석 참고치, 신체 지표, 목표**를 영구적으로 기억하고 상담에 반영합니다."
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
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
    quick_prompt = None
    if q_col1.button("🔥 FatMax 90분 서지 & 체지방 감량", use_container_width=True):
        quick_prompt = "현재 FTP 196W, 최대심박 184bpm(LTHR 161bpm) 기준에서 근육 손실 없이 내장지방과 체지방만 가장 빠르게 태우는 FatMax 골든존(118~141W, 심박 120~137bpm) 90분 라이딩 방법과 식단·보급 타이밍을 알려주세요."
    if q_col2.button("📈 FTP 196W → 210W 돌파 루틴", use_container_width=True):
        quick_prompt = "불필요한 150bpm 심박 제한 없이, 실제 LTHR 161bpm(최대 184bpm)과 현재 FTP 196W를 기준으로 FTP를 210W 이상으로 끌어올릴 수 있는 평일 인도어/단거리 인터벌과 주말 라이딩 구성을 짜주세요."
    if q_col3.button("🦵 클릿 최후방 & 좌측 오금 당김 교정", use_container_width=True):
        quick_prompt = "현재 클릿을 신발 최후방으로 밀어둔 상태이고 10월 장거리 주행에서 왼발 비중(53~53.5%)이 높아지며 왼쪽 무릎 뒤(오금/슬와근)가 당겼습니다. 안장 높이와 페달링을 어떻게 교정할까요?"
    if q_col4.button("💧 알닥톤 25mg & 장거리 혈압·수분 관리", use_container_width=True):
        quick_prompt = "알닥톤 25mg 복용 중인데 90~100km 장거리 라이딩이나 고강도 훈련을 타면서 운동 후 이완기 혈압 반등(90mmHg+)과 탈수를 막으려면 수분·전해질 섭취와 쿨다운을 어떻게 해야 할까요?"
    if q_col5.button("❤️ LTHR 161bpm 기준 파워·심박 매칭", use_container_width=True):
        quick_prompt = "젖산역치 심박수(LTHR) 161bpm, 최대심박수 184bpm, FTP 196W 기준으로 FatMax, Z2 지구력, Sweet Spot, Z4 역치 훈련 시 각각 적정 와트(W)와 심박수(bpm) 범위를 정리해 주세요."

    if "messages" not in st.session_state or not st.session_state["messages"]:
        saved_msgs = mem_data.get("messages", [])
        if saved_msgs:
            st.session_state["messages"] = saved_msgs
        else:
            default_greeting = {
                "role": "assistant",
                "content": (
                    f"안녕하세요, 라이더님! 🚴 **Strava 실시간 연동** 및 **가민 Edge 1040 정밀 분석 참고치**가 결합된 AI 코치입니다.\n"
                    f"현재 프로필은 **FTP {ftp_input}W ({w_per_kg} W/kg)**, **LTHR {lthr_input}bpm (Max {max_hr_input}bpm)**, 컨디션 폼(TSB)은 **{latest_pmc['tsb']:+.1f}**입니다.\n"
                    f"가민으로 라이딩을 마치시면 Strava를 통해 자동으로 최신 기록이 반영됩니다!"
                ),
            }
            st.session_state["messages"] = [default_greeting]
            append_chat_message(default_greeting["role"], default_greeting["content"])

    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("images"):
                    img_cols = st.columns(min(3, len(msg["images"])))
                    for idx_img, img_bytes in enumerate(msg["images"]):
                        img_cols[idx_img % len(img_cols)].image(img_bytes, use_container_width=True)

    user_input = st.chat_input(
        "질문 입력 또는 클립보드 이미지 붙여넣기(Ctrl+V) / 파일 첨부(+ 아이콘 또는 드래그)...",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "webp", "gif", "pdf", "csv", "fit", "txt", "json"],
    )

    query_text = None
    uploaded_chat_files = []
    if quick_prompt:
        query_text = quick_prompt
    elif user_input is not None:
        if isinstance(user_input, str):
            query_text = user_input
        else:
            query_text = getattr(user_input, "text", "") or ""
            uploaded_chat_files = list(getattr(user_input, "files", []) or [])
            if not query_text.strip() and uploaded_chat_files:
                query_text = "첨부한 이미지/파일을 사이클링 코치 관점에서 정밀 분석해 주세요."

    if query_text or uploaded_chat_files:
        attachments_payload = []
        preview_images = []
        attached_names = []

        for uf in uploaded_chat_files:
            fname = getattr(uf, "name", "attachment")
            ftype = (getattr(uf, "type", "") or "").lower()
            raw_bytes = uf.getvalue()
            attached_names.append(fname)
            lower_name = fname.lower()

            if lower_name.endswith(".fit"):
                parsed_fit = parse_uploaded_fit_bytes(raw_bytes, filename=fname, ftp=float(ftp_input))
                if parsed_fit:
                    fit_summary = (
                        f"• FIT 파일 [{fname}]: 날짜={parsed_fit.get('date', '')}, "
                        f"거리={round((parsed_fit.get('distance_m') or 0)/1000, 2)}km, "
                        f"이동시간={round((parsed_fit.get('timer_time_s') or 0)/60, 1)}분, "
                        f"획득고도={parsed_fit.get('ascent_m', 0)}m, "
                        f"평균파워={parsed_fit.get('avg_power', 0)}W, NP={parsed_fit.get('np', 0)}W, "
                        f"최대파워={parsed_fit.get('max_power', 0)}W, 평균심박={parsed_fit.get('avg_hr', 0)}bpm, "
                        f"최대심박={parsed_fit.get('max_hr', 0)}bpm, 케이던스={parsed_fit.get('avg_cadence', 0)}rpm, "
                        f"좌우밸런스(L/R)={parsed_fit.get('lr_str', 'N/A')}, "
                        f"토크효율(L/R)={parsed_fit.get('avg_left_torque_eff', 'N/A')}/{parsed_fit.get('avg_right_torque_eff', 'N/A')}%, "
                        f"페달스무스니스(L/R)={parsed_fit.get('avg_left_pedal_smooth', 'N/A')}/{parsed_fit.get('avg_right_pedal_smooth', 'N/A')}%, "
                        f"스태미나(시작→종료)={parsed_fit.get('stamina_start', 'N/A')}→{parsed_fit.get('stamina_end', 'N/A')}%, "
                        f"피크파워(MMP)={parsed_fit.get('mmp', {})}"
                    )
                    attachments_payload.append({"name": fname, "text_summary": fit_summary})
                else:
                    attachments_payload.append({"name": fname, "text_summary": f"• FIT 파일 [{fname}]: 파싱 실패"})
            elif any(lower_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]) or ftype.startswith("image/"):
                mime = ftype if ftype.startswith("image/") else "image/png"
                attachments_payload.append({"name": fname, "mime_type": mime, "data": raw_bytes})
                preview_images.append(raw_bytes)
            elif lower_name.endswith(".pdf") or ftype == "application/pdf":
                attachments_payload.append({"name": fname, "mime_type": "application/pdf", "data": raw_bytes})
            else:
                try:
                    txt_content = raw_bytes.decode("utf-8", errors="replace")[:8000]
                    attachments_payload.append(
                        {"name": fname, "text_summary": f"• 파일 [{fname}] 내용 발췌:\n```\n{txt_content}\n```"}
                    )
                except Exception:
                    pass

        display_query = query_text
        if attached_names:
            display_query = f"{query_text}\n\n📎 *첨부 파일: {', '.join(attached_names)}*"

        msg_entry = {"role": "user", "content": display_query}
        if preview_images:
            msg_entry["images"] = preview_images
        st.session_state["messages"].append(msg_entry)
        append_chat_message("user", display_query)

        with chat_container:
            with st.chat_message("user"):
                st.markdown(display_query)
                if preview_images:
                    img_cols = st.columns(min(3, len(preview_images)))
                    for idx_img, img_bytes in enumerate(preview_images):
                        img_cols[idx_img % len(img_cols)].image(img_bytes, use_container_width=True)

        last_ride = activities[0] if activities else {}
        last_ride_summary = (
            f"{last_ride.get('name', '라이딩')} (날짜: {last_ride.get('start_date_local', '')[:10]}, "
            f"거리: {round(last_ride.get('distance', 0)/1000, 1)}km, "
            f"NP: {round(last_ride.get('weighted_average_watts') or last_ride.get('average_watts') or 0)}W, "
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
                with st.spinner("코치가 이미지/첨부파일 및 Strava·가민 데이터를 종합 분석 중입니다..."):
                    clean_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["messages"][:-1]
                    ]
                    answer = coach_agent.consult(
                        user_message=query_text,
                        chat_history=clean_history,
                        athlete_context=athlete_context,
                        attachments=attachments_payload,
                    )
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    append_chat_message("assistant", answer)
                    auto_extract_athlete_insight(coach_agent, display_query, answer)
        st.rerun()
