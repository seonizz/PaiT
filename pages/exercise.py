"""운동 정보(상세) 화면.

detect.py에서 저장한 session_state['detected_exercise']를 기준으로
exercise_db.json에서 운동명/난이도/주운동근/보조운동근/추천 세트-렙/주의사항/팁을
불러와 표시합니다.

Priority 4: 실시간 Rep 카운팅/자세 피드백은 lat_pulldown/chest_press/shoulder_press
3개만 완성도 있게 튜닝했습니다(POSE_SUPPORTED_EXERCISES). 나머지 운동은 여기 정보
화면까지만 제공하고, 실시간 운동(workout) 화면으로는 보내지 않습니다.
"""

import streamlit as st

from components.cards import exercise_detail_card, exercise_thumbnail
from utils.constants import PAGE_DETECT, PAGE_WORKOUT, POSE_SUPPORTED_EXERCISES
from utils.data import load_exercise_db
from utils.session import go_to, reset_for_next_exercise


def render():
    st.subheader("운동 정보")

    detected = st.session_state.get("detected_exercise")
    if not detected:
        st.warning("먼저 기구를 인식해주세요.")
        return

    exercise_key = detected["exercise_key"]
    confidence = detected.get("confidence")

    exercise_db = load_exercise_db()
    exercise_data = exercise_db.get(exercise_key)
    if exercise_data is None:
        st.error("운동 정보를 찾을 수 없습니다. 기구를 다시 인식해주세요.")
        if st.button("기구 다시 인식하기", width="stretch", key="exercise_redetect_missing"):
            reset_for_next_exercise()
            go_to(PAGE_DETECT)
        return

    exercise_thumbnail(exercise_key, exercise_data.get("image"))
    if confidence is not None:
        st.caption(f"인식 신뢰도: {confidence * 100:.0f}%")
    exercise_detail_card(exercise_data)

    st.markdown("---")

    if exercise_key in POSE_SUPPORTED_EXERCISES:
        if st.button("운동 시작", width="stretch", key="exercise_start"):
            st.session_state.current_exercise_key = exercise_key
            st.session_state.rep_count = 0
            st.session_state.set_count = 1
            st.session_state.elapsed_sec = 0
            st.session_state.workout_start_time = None
            st.session_state.pose_feedback = ""
            go_to(PAGE_WORKOUT)
    else:
        st.info("이 기구는 아직 실시간 Rep 카운팅을 지원하지 않습니다. 운동 정보를 참고해서 직접 진행해주세요.")
        if st.button("다음 기구 인식하기", width="stretch", key="exercise_next_detect"):
            reset_for_next_exercise()
            go_to(PAGE_DETECT)
