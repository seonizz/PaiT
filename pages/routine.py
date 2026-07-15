"""오늘의 루틴 화면.

3초 안에 "지금 뭘 할 차례인지"를 파악할 수 있도록, 아직 안 한 운동 중 첫 번째를
'현재 운동'으로 크게 강조하고 나머지는 작은 목록으로만 보여줍니다.
"""

import streamlit as st

from components.cards import current_exercise_card, next_exercise_row, routine_progress
from utils.constants import PAGE_DETECT
from utils.data import load_exercise_db
from utils.session import go_to


def render():
    st.subheader("오늘의 루틴")

    routine = st.session_state.routine
    if not routine:
        st.warning("먼저 목표와 운동 부위를 선택해주세요.")
        return

    done_keys = {entry["exercise_key"] for entry in st.session_state.workout_log}

    with st.container(key="routine-header"):
        routine_progress(len(done_keys), len(routine))

    exercise_db = load_exercise_db()

    remaining = [ex for ex in routine if ex["key"] not in done_keys]
    if remaining:
        current = remaining[0]
        current_order = routine.index(current) + 1
        current_exercise_card(current_order, current["key"], exercise_db[current["key"]], current["sets"], current["reps"])

        for ex in remaining[1:]:
            order = routine.index(ex) + 1
            next_exercise_row(order, exercise_db[ex["key"]])
    else:
        st.markdown("오늘 루틴을 모두 완료했습니다.")

    with st.bottom:
        if remaining and st.button("시작하기", width="stretch", key="routine_start"):
            # 실제로 어떤 기구를 하게 될지는 detect.py의 실물 촬영 인식 결과로 정해지므로
            # 여기서는 특정 운동을 미리 지정하지 않고 기구 인식 화면으로만 이동합니다.
            go_to(PAGE_DETECT)
