"""실시간 운동 화면.

두 가지 모드를 제공합니다.
- AI 자세 분석: MediaPipe Pose(components.camera.PoseWorkoutProcessor)가 웹캠으로
  Rep을 자동 카운트하고 자세 피드백을 계산합니다.
- 수동 카운팅: 카메라 권한 없이 [세트 완료] 버튼 하나로 세트 단위를 직접 기록합니다
  (예: 카메라를 쓸 수 없는 환경, 프라이버시 등의 이유). Rep 단위 카운팅은 하지 않고
  현재 세트만 크게 표시합니다.

두 모드 모두 Rep/Set/운동시간 표시와 세트·종료 버튼 흐름은 동일하게 동작하며, 이
부분을 st.fragment(run_every=1)로 감싸 1초마다 자동 갱신합니다. AI 모드에서는
webrtc_streamer() 호출을 프래그먼트 바깥(전체 rerun에서 딱 1번)에 두어, 프래그먼트가
자기 영역만 재실행해도 웹캠 컴포넌트가 재마운트되지 않습니다 (예전에 수동
sleep+rerun 폴링을 쓰다가 웹캠이 재마운트되어 스켈레톤이 안 그려지던 버그를 이
방식으로 근본적으로 해결했습니다).
"""

import time
from datetime import datetime

import streamlit as st

from components.camera import render_workout_stream
from components.cards import routine_progress
from utils.constants import PAGE_RESULT, WORKOUT_MODE_AI, WORKOUT_MODE_MANUAL
from utils.data import append_history, load_exercise_db
from utils.session import go_to

MODE_AI = WORKOUT_MODE_AI
MODE_MANUAL = WORKOUT_MODE_MANUAL


def render():
    exercise_key = st.session_state.current_exercise_key
    if not exercise_key:
        st.warning("먼저 운동을 선택해주세요.")
        return

    exercise_db = load_exercise_db()
    exercise_data = exercise_db.get(exercise_key)
    if exercise_data is None:
        st.error("운동 정보를 찾을 수 없습니다. 처음부터 다시 진행해주세요.")
        return

    total_reps = exercise_data["reps"]
    total_sets = exercise_data["sets"]

    # 이번 운동을 처음 시작하는 시점인지 판별 (세트 전환이 아니라 '새 운동' 시작인 경우에만
    # 카운터를 리셋합니다 - 안 그러면 이전에 하다 만 운동의 Rep이 남아있는 상태로 새
    # 운동이 시작되는 버그가 있었습니다).
    is_fresh_start = st.session_state.workout_start_time is None
    if is_fresh_start:
        st.session_state.workout_start_time = time.time()
        st.session_state.rep_count = 0

    routine = st.session_state.routine or []
    if routine:
        done_keys = {entry["exercise_key"] for entry in st.session_state.workout_log}
        with st.container(key="routine-header"):
            routine_progress(len(done_keys), len(routine))

    mode = st.segmented_control(
        "모드",
        [MODE_AI, MODE_MANUAL],
        default=MODE_AI,
        required=True,
        key="workout_mode",
        label_visibility="collapsed",
    )

    if mode == MODE_AI:
        ctx = render_workout_stream(exercise_key, total_reps)  # webrtc_streamer 호출 - 전체 rerun에서 1번만
        if is_fresh_start and ctx.video_processor:
            ctx.video_processor.reset()
        _live_panel_ai(ctx, exercise_key, exercise_data, total_reps, total_sets)
    else:
        _live_panel_manual(exercise_key, exercise_data, total_reps, total_sets)


def _render_rep_display(rep: int, total_reps: int, set_count: int, total_sets: int):
    st.markdown(f'<div class="pait-rep-huge">{rep}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pait-rep-sub">/ {total_reps} Reps</div>', unsafe_allow_html=True)
    st.progress(min(rep / total_reps, 1.0) if total_reps else 0.0)
    st.markdown(f'<div class="pait-set-small">Set {set_count} / {total_sets}</div>', unsafe_allow_html=True)


def _finish_workout(exercise_key: str, exercise_data: dict, set_count: int, completed_reps: int):
    # 칼로리 = 운동시간(분) x MET x 체중(kg) / 60
    duration_min = st.session_state.elapsed_sec / 60
    met = exercise_data.get("met", 3.5)
    weight_kg = st.session_state.get("weight") or 65.0
    calories = round(duration_min * met * weight_kg / 60, 1)

    log_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exercise_key": exercise_key,
        "name": exercise_data["name"],
        "sets_completed": set_count,
        "reps_completed": completed_reps,
        "elapsed_sec": st.session_state.elapsed_sec,
        "calories": calories,
    }
    st.session_state.workout_log.append(log_entry)
    st.session_state.history.append(log_entry)
    append_history(log_entry)  # database/history_log.json에 영구 저장
    st.session_state.workout_start_time = None
    go_to(PAGE_RESULT)  # 프래그먼트 안이지만 기본 st.rerun()은 전체 스코프라 정상 이동


@st.fragment(run_every=1)
def _live_panel_ai(ctx, exercise_key: str, exercise_data: dict, total_reps: int, total_sets: int):
    if not ctx.state.playing:
        if ctx.state.signalling:
            st.info("카메라 연결 중입니다...")
        else:
            st.info("위 START 버튼을 눌러 카메라를 켜주세요.")

    # session_state로 동기화하지 않고 ctx에서 매 틱 직접 읽습니다 (스테일 방지).
    rep = ctx.video_processor.rep_count if ctx.video_processor else 0
    feedback = ctx.video_processor.feedback if ctx.video_processor else ""

    st.session_state.rep_count = rep
    st.session_state.pose_feedback = feedback
    st.session_state.elapsed_sec = int(time.time() - st.session_state.workout_start_time)

    set_count = st.session_state.set_count
    _render_rep_display(rep, total_reps, set_count, total_sets)

    feedback_class = "pait-feedback-warn" if feedback else "pait-feedback-good"
    feedback_text = feedback or "GOOD"
    st.markdown(f'<div class="pait-feedback-line {feedback_class}">{feedback_text}</div>', unsafe_allow_html=True)

    set_complete = total_reps > 0 and rep >= total_reps
    with st.bottom:
        if not set_complete:
            pass  # 세트 중엔 버튼 없음
        elif set_count < total_sets:
            if st.button("다음 세트", width="stretch", key="workout_next_set_ai"):
                st.session_state.set_count += 1
                st.session_state.rep_count = 0
                if ctx.video_processor:
                    ctx.video_processor.reset()
        else:
            if st.button("운동 종료", width="stretch", key="workout_finish_ai"):
                completed_reps = (set_count - 1) * total_reps + rep  # 라이브 rep 사용 (스테일 방지)
                _finish_workout(exercise_key, exercise_data, set_count, completed_reps)


@st.fragment(run_every=1)
def _live_panel_manual(exercise_key: str, exercise_data: dict, total_reps: int, total_sets: int):
    st.session_state.elapsed_sec = int(time.time() - st.session_state.workout_start_time)

    # 버튼 처리를 먼저 수행해서, 클릭 직후 렌더링에 최신 set 값이 바로 반영되도록 합니다
    # (렌더링을 먼저 하면 이번 클릭의 효과가 다음 1초 후 자동 갱신 때까지 화면에 안 보이는
    # 한 박자 밀림이 생깁니다).
    set_count = st.session_state.set_count
    is_last_set = set_count >= total_sets
    button_label = "운동 종료" if is_last_set else "세트 완료"

    with st.bottom:
        if st.button(button_label, width="stretch", key="workout_set_complete_manual"):
            if is_last_set:
                completed_reps = total_sets * total_reps  # Rep 단위는 기록하지 않으므로 목표치로 계산
                _finish_workout(exercise_key, exercise_data, set_count, completed_reps)
            else:
                st.session_state.set_count += 1

    # 버튼 처리 이후 최신 값으로 다시 읽어서 표시합니다.
    set_count = st.session_state.set_count
    st.markdown(f'<div class="pait-rep-huge">Set {set_count}/{total_sets}</div>', unsafe_allow_html=True)
    st.progress(min(set_count / total_sets, 1.0) if total_sets else 0.0)

    warnings = exercise_data.get("warnings") or []
    if warnings:
        st.markdown('<p class="pait-subtle" style="text-align:center;">주의사항</p>', unsafe_allow_html=True)
        for warning in warnings:
            st.markdown(f'<p class="pait-subtle" style="text-align:center;">· {warning}</p>', unsafe_allow_html=True)
