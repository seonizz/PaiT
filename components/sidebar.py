"""좌측 사이드바 네비게이션 (Home / Workout / History / Settings)."""

import streamlit as st

from utils.constants import APP_NAME, SIDEBAR_ITEMS
from utils.session import go_to


def render_sidebar():
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption("Edge AI Personal Trainer")
        st.markdown("---")

        current_page = st.session_state.get("page")
        active_key = None
        for item in SIDEBAR_ITEMS:
            is_active = current_page == item["key"] or (
                item["key"] == "goal" and current_page in {"goal", "routine", "detect", "exercise", "workout", "result"}
            )
            if is_active:
                active_key = item["key"]
            with st.container(key=f"sidebar-{item['key']}"):
                if st.button(item["label"], key=f"sidebar_{item['key']}", width="stretch"):
                    go_to(item["key"])

        # CSS만으로는 "현재 활성 항목"을 알 수 없으므로(모든 항목이 동일한 마크업),
        # 매 렌더링마다 활성 항목의 st-key- 클래스만 골라 스타일을 주입합니다.
        if active_key:
            st.markdown(
                f"""
                <style>
                .st-key-sidebar-{active_key} button {{
                    background-color: #262626 !important;
                    font-weight: 700;
                    border-left: 3px solid var(--pait-accent);
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
