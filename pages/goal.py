"""목표(벌크업/다이어트/자세교정) 및 운동 부위 선택 화면."""

import streamlit as st

from components.cards import goal_card
from utils.constants import BODY_PARTS, GOALS, PAGE_ROUTINE
from utils.data import build_routine
from utils.session import go_to


def render():
    st.subheader("오늘의 목표를 선택하세요")

    cols = st.columns(3)
    for col, goal in zip(cols, GOALS):
        with col:
            selected = st.session_state.selected_goal == goal["key"]
            if goal_card(goal["title"], goal["desc"], selected, key=f"goal_{goal['key']}"):
                st.session_state.selected_goal = goal["key"]
                st.rerun()

    st.markdown("---")
    st.subheader("체중")
    st.session_state.weight = st.number_input(
        "체중(kg) - 칼로리 계산에 사용됩니다",
        min_value=30.0,
        max_value=200.0,
        value=float(st.session_state.get("weight") or 65.0),
        step=0.5,
        key="goal_weight_input",
    )

    st.markdown("---")
    st.subheader("운동 부위")
    body_part = st.radio(
        "운동 부위 선택",
        BODY_PARTS,
        index=BODY_PARTS.index(st.session_state.selected_body_part) if st.session_state.selected_body_part else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="goal_body_part_radio",
    )
    st.session_state.selected_body_part = body_part

    disabled = st.session_state.selected_goal is None
    with st.bottom:
        if st.button("선택 완료", width="stretch", disabled=disabled, key="goal_confirm"):
            st.session_state.routine = build_routine(st.session_state.selected_goal, st.session_state.selected_body_part)
            go_to(PAGE_ROUTINE)
