"""Streamlit session_state 초기화 및 페이지 전환 헬퍼."""

import streamlit as st

from utils.constants import PAGE_SPLASH


def init_session_state():
    defaults = {
        "page": PAGE_SPLASH,
        "selected_goal": None,
        "selected_body_part": None,
        "routine": None,
        "current_exercise_key": None,
        "detected_exercise": None,
        "weight": 65.0,
        "workout_log": [],
        "rep_count": 0,
        "set_count": 1,
        "elapsed_sec": 0,
        "workout_start_time": None,
        "pose_feedback": "",
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(page: str):
    st.session_state.page = page
    st.rerun()


def reset_workout():
    """Home으로 돌아갈 때: 목표/루틴/이번 세션 기록까지 전부 초기화합니다."""
    st.session_state.selected_goal = None
    st.session_state.selected_body_part = None
    st.session_state.routine = None
    st.session_state.current_exercise_key = None
    st.session_state.detected_exercise = None
    st.session_state.workout_log = []
    st.session_state.rep_count = 0
    st.session_state.set_count = 1
    st.session_state.elapsed_sec = 0
    st.session_state.workout_start_time = None
    st.session_state.pose_feedback = ""


def reset_for_next_exercise():
    """루틴의 다음 운동으로 넘어갈 때: 목표/루틴/이번 세션 누적 기록(workout_log)은
    유지하고, 방금 끝낸 운동과 관련된 상태만 초기화합니다."""
    st.session_state.current_exercise_key = None
    st.session_state.detected_exercise = None
    st.session_state.rep_count = 0
    st.session_state.set_count = 1
    st.session_state.elapsed_sec = 0
    st.session_state.workout_start_time = None
    st.session_state.pose_feedback = ""
