"""운동 결과 화면.

session_state['workout_log']에 workout.py가 저장한 실제 운동시간/Rep/Set/칼로리
(MET 공식 기반)를 그대로 집계해서 보여줍니다. 완료 기록은 database/history_log.json에도
영구 저장되어 History 화면에서 다시 볼 수 있습니다.
"""

import streamlit as st

from components.cards import routine_progress, stat_box
from utils.constants import PAGE_DETECT, PAGE_SPLASH
from utils.data import load_exercise_db
from utils.session import go_to, reset_for_next_exercise, reset_workout


def render():
    st.markdown("## Great Job!")

    log = st.session_state.workout_log
    if not log:
        st.info("완료한 운동이 없습니다.")
    else:
        total_sec = sum(entry["elapsed_sec"] for entry in log)
        total_reps = sum(entry["reps_completed"] for entry in log)
        total_sets = sum(entry["sets_completed"] for entry in log)
        total_kcal = round(sum(entry.get("calories", 0) for entry in log), 1)

        minutes, seconds = divmod(total_sec, 60)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stat_box(f"{minutes:02d}:{seconds:02d}", "운동시간")
        with col2:
            stat_box(f"{total_kcal}", "칼로리(kcal)")
        with col3:
            stat_box(f"{total_reps}", "총 Rep")
        with col4:
            stat_box(f"{total_sets}", "총 Set")

        st.caption(" · ".join(f"{e['name']} {e['sets_completed']}세트/{e['reps_completed']}회" for e in log))

    routine = st.session_state.routine or []
    done_keys = {entry["exercise_key"] for entry in log}
    next_exercise = next((ex for ex in routine if ex["key"] not in done_keys), None)

    if routine:
        with st.container(key="routine-header"):
            routine_progress(len(done_keys), len(routine))

    st.markdown("---")
    if next_exercise:
        exercise_db = load_exercise_db()
        next_data = exercise_db[next_exercise["key"]]
        st.markdown("### 다음 추천 운동")
        st.markdown(f"## {next_data['name']}")
        st.caption(f"{next_exercise['sets']} Sets x {next_exercise['reps']} Reps")
    else:
        st.markdown("### 오늘 루틴을 모두 완료했습니다!")

    with st.bottom:
        if next_exercise:
            if st.button("다음 운동", width="stretch", key="result_next_exercise"):
                reset_for_next_exercise()
                go_to(PAGE_DETECT)
            st.markdown('<p class="pait-subtle" style="text-align:center;">홈으로 돌아가려면 아래를 눌러주세요</p>', unsafe_allow_html=True)
            if st.button("홈", key="result_home"):
                reset_workout()
                go_to(PAGE_SPLASH)
        else:
            if st.button("홈", width="stretch", key="result_home"):
                reset_workout()
                go_to(PAGE_SPLASH)
