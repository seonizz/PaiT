"""exercise_db.json / routine_db.json 로더 (캐시 적용) + history_log.json 영구 저장."""

import json

import streamlit as st

from utils.constants import EXERCISE_DB_PATH, HISTORY_LOG_PATH, ROUTINE_DB_PATH


@st.cache_data
def load_exercise_db() -> dict:
    return json.loads(EXERCISE_DB_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_routine_db() -> dict:
    return json.loads(ROUTINE_DB_PATH.read_text(encoding="utf-8"))


def build_routine(goal: str, body_part: str) -> list[dict]:
    """routine_db.json에서 (목표, 부위)에 맞는 운동 목록을 가져옵니다."""
    routine_db = load_routine_db()
    return routine_db[goal][body_part]["exercises"]


def load_history() -> list[dict]:
    """history_log.json에서 과거 운동 기록을 불러옵니다. 파일이 없으면 빈 목록."""
    if not HISTORY_LOG_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_LOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def append_history(entry: dict) -> None:
    """운동 완료 기록 한 건을 history_log.json에 영구 저장합니다."""
    history = load_history()
    history.append(entry)
    HISTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_LOG_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_history() -> None:
    HISTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_LOG_PATH.write_text("[]", encoding="utf-8")
