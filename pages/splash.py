"""스플래시(시작) 화면."""

import streamlit as st

from utils.constants import APP_NAME, APP_SUBTITLE, APP_TAGLINE, PAGE_GOAL
from utils.session import go_to


def render():
    st.markdown(f'<div class="pait-splash-logo">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pait-splash-tagline">{APP_TAGLINE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pait-splash-subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        if st.button("운동 시작", width="stretch", key="splash_start"):
            go_to(PAGE_GOAL)
