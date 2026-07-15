"""
PaiT - Edge AI Personal Trainer (Streamlit)

사용 예:
    streamlit run app.py

세션 상태(st.session_state.page)를 기준으로 화면을 전환하는 단일 진입점입니다.
각 화면의 렌더링 로직은 pages/*.py 에 있습니다.
"""

from pathlib import Path

import streamlit as st

from components.sidebar import render_sidebar
from pages import detect, exercise, goal, result, routine, splash, workout
from utils.constants import (
    APP_NAME,
    CSS_PATH,
    PAGE_DETECT,
    PAGE_EXERCISE,
    PAGE_GOAL,
    PAGE_HISTORY,
    PAGE_RESULT,
    PAGE_ROUTINE,
    PAGE_SETTINGS,
    PAGE_SPLASH,
    PAGE_WORKOUT,
    POSE_SUPPORTED_EXERCISES,
)
from utils.data import clear_history, load_history
from utils.session import init_session_state

PAGE_RENDERERS = {
    PAGE_SPLASH: splash.render,
    PAGE_GOAL: goal.render,
    PAGE_ROUTINE: routine.render,
    PAGE_DETECT: detect.render,
    PAGE_EXERCISE: exercise.render,
    PAGE_WORKOUT: workout.render,
    PAGE_RESULT: result.render,
}


def inject_css():
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_history():
    st.subheader("History")

    history = load_history()  # database/history_log.json (앱 재시작 후에도 유지되는 영구 기록)
    if not history:
        st.info("아직 완료한 운동 기록이 없습니다.")
        return

    rows = []
    for entry in reversed(history):  # 최신 기록이 위로
        minutes, seconds = divmod(entry.get("elapsed_sec", 0), 60)
        rows.append(
            {
                "날짜": entry.get("date", "-"),
                "운동명": entry.get("name", "-"),
                "Rep": entry.get("reps_completed", 0),
                "Set": entry.get("sets_completed", 0),
                "시간": f"{minutes:02d}:{seconds:02d}",
                "칼로리(kcal)": entry.get("calories", 0),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    total_kcal = round(sum(e.get("calories", 0) for e in history), 1)
    st.caption(f"총 {len(history)}회 운동, 누적 칼로리 {total_kcal} kcal")


def render_settings():
    st.subheader("Settings")
    st.markdown(f"**앱 이름**: {APP_NAME}")
    st.markdown("**모델**: MobileNetV2 (ONNX Runtime, on-device)")
    st.markdown(f"**포즈 분석**: MediaPipe Pose — {', '.join(sorted(POSE_SUPPORTED_EXERCISES))} 실시간 지원")
    if st.button("운동 기록 초기화", key="settings_clear_history"):
        st.session_state.history = []
        clear_history()
        st.success("기록을 초기화했습니다.")


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🏋️", layout="centered")
    init_session_state()
    inject_css()
    render_sidebar()

    page = st.session_state.page
    if page == PAGE_HISTORY:
        render_history()
    elif page == PAGE_SETTINGS:
        render_settings()
    else:
        renderer = PAGE_RENDERERS.get(page, splash.render)
        renderer()


if __name__ == "__main__":
    main()
