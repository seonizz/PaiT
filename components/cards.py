"""재사용 카드 컴포넌트: 목표 선택 카드, 운동 카드, 통계 박스, 배지, 진행률."""

from pathlib import Path

import streamlit as st

from utils.constants import BASE_DIR, THUMBNAIL_DIR


def goal_card(title: str, desc: str, selected: bool, key: str) -> bool:
    """목표(벌크업/다이어트/자세교정) 선택 카드. 클릭되면 True를 반환합니다."""
    css_class = "pait-goal-card selected" if selected else "pait-goal-card"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="pait-title">{title}</div>
            <div class="pait-subtle">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.button("선택", key=key, width="stretch")


def exercise_thumbnail(exercise_key: str, image_path: str | None = None):
    """exercise_db.json의 image 필드(있으면)를 우선 쓰고, 없으면 기존 경로 규칙으로 대체합니다."""
    path = BASE_DIR / image_path if image_path else THUMBNAIL_DIR / f"{exercise_key}.jpg"
    if path.exists():
        st.image(str(path), width="stretch")


def current_exercise_card(order: int, exercise_key: str, exercise_data: dict, sets: int, reps: int):
    """루틴 화면에서 '현재 운동'을 크게 강조해서 보여줍니다."""
    exercise_thumbnail(exercise_key, exercise_data.get("image"))
    st.markdown(
        f"""
        <div class="pait-card">
            <div class="pait-subtle">현재 운동 · {order}번째</div>
            <div class="pait-title" style="font-size:1.7rem;">{exercise_data['name']}</div>
            <span class="pait-badge">{sets} Sets</span>
            <span class="pait-badge">{reps} Reps</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def next_exercise_row(order: int, exercise_data: dict):
    """루틴 화면에서 아직 순서가 오지 않은 운동을 작게 나열합니다."""
    st.markdown(
        f'<p class="pait-subtle">{order}. {exercise_data["name"]}</p>',
        unsafe_allow_html=True,
    )


def stat_box(value: str, label: str):
    st.markdown(
        f"""
        <div class="pait-stat-box">
            <div class="pait-stat-value">{value}</div>
            <div class="pait-stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(is_good: bool, good_text: str = "GOOD", warn_text: str = "WARNING"):
    """자세/신뢰도 등 상태를 GOOD(녹색) / WARNING(주황색) 배지로 표시합니다."""
    css_class = "pait-badge-good" if is_good else "pait-badge-warn"
    label = good_text if is_good else warn_text
    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


def routine_progress(done: int, total: int):
    """루틴 전체 진행률("오늘 운동 done/total")을 표시합니다."""
    if total <= 0:
        return
    st.progress(min(done / total, 1.0), text=f"오늘 운동 {done}/{total}")


def exercise_detail_card(exercise_data: dict):
    """운동 정보(exercise) 페이지에서 쓰이는 상세 카드."""
    description = exercise_data.get("description", "")
    st.markdown(
        f"""
        <div class="pait-card">
            <div class="pait-title">{exercise_data['name']} <span class="pait-subtle">({exercise_data['name_kr']})</span></div>
            <p class="pait-subtle" style="margin-top:0.2rem;">{description}</p>
            <span class="pait-badge-outline">난이도 {exercise_data['difficulty']}</span>
            <span class="pait-badge-outline">추천 {exercise_data['sets']} Sets x {exercise_data['reps']} Reps</span>
            <p style="margin-top:0.7rem;"><b>주 운동근</b><br>{exercise_data['target_muscle']}</p>
            <p><b>보조 운동근</b><br>{exercise_data['secondary_muscle']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**사용법**")
    for step in exercise_data["tips"]:
        st.markdown(f"- {step}")

    st.markdown("**주의사항**")
    for warning in exercise_data["warnings"]:
        st.markdown(f"- {warning}")
