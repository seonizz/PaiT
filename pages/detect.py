"""기구 인식 화면 - export/model_fp32.onnx로 실제 인식.

사진을 촬영하면 즉시 ONNX 추론을 실행해 결과를 화면에 보여주지만, 화면이 자동으로
넘어가지는 않습니다("왜 화면이 넘어갔지?" 하는 어색함을 피하기 위해). 인식된 운동명과
신뢰도를 사용자가 직접 확인한 뒤, 신뢰도가 충분할 때만 활성화되는 [운동 시작] 버튼을
눌러야 다음 화면으로 이동합니다.

ONNX 추론은 매 rerun마다 현재 사진으로 다시 계산합니다(캐싱하지 않음) — 재촬영 시
이전 결과가 남아있는 스테일 버그를 피하기 위한 의도적인 선택입니다.

카메라를 아예 쓸 수 없는 상황(권한 거부, 카메라 없는 기기 등)을 위해 운동을 직접
선택해서 넘어가는 대체 경로도 제공합니다. 이 경로로 넘어가면 워크아웃 화면의 모드도
자동으로 "수동 카운팅"으로 맞춰줍니다(카메라가 안 되면 자세 분석도 못 쓰니까).
"""

import streamlit as st

from components.cards import exercise_thumbnail, status_badge
from components.camera import render_camera_input, run_onnx_detection
from utils.constants import GOOD_CONFIDENCE_THRESHOLD, PAGE_EXERCISE, THUMBNAIL_DIR, WORKOUT_MODE_MANUAL
from utils.data import load_exercise_db
from utils.session import go_to

CAMERA_KEY = "detect_camera_input"
EXAMPLE_THUMBNAIL = THUMBNAIL_DIR / "smith_machine.jpg"


def render():
    st.subheader("기구 인식")
    st.markdown("**기구를 화면 가운데 맞춰주세요**")

    image = render_camera_input(key=CAMERA_KEY)

    result = None
    exercise_data = None
    good_enough = False

    if image is not None:
        try:
            with st.spinner("인식 중..."):
                result = run_onnx_detection(image)
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001 - 손상된 이미지 등 예상 못한 오류도 앱이 죽지 않게 처리
            st.error(f"기구를 인식하지 못했습니다: {e}")
            st.info("사진이 흐릿하거나 손상되었을 수 있습니다. 다시 촬영해주세요.")

    if result is not None:
        exercise_db = load_exercise_db()
        exercise_data = exercise_db.get(result["exercise_key"])
        confidence = result["confidence"]

        if exercise_data is None:
            st.error("인식된 기구 정보를 찾을 수 없습니다. 다시 촬영해주세요.")
        else:
            good_enough = confidence >= GOOD_CONFIDENCE_THRESHOLD
            col1, col2 = st.columns([1, 1.6])
            with col1:
                exercise_thumbnail(result["exercise_key"], exercise_data.get("image"))
            with col2:
                st.markdown(f"## {exercise_data['name']}")
                status_badge(good_enough, good_text="GOOD", warn_text="WARNING")
                st.progress(min(confidence, 1.0), text=f"Confidence {confidence * 100:.0f}%")
            if not good_enough:
                st.caption("신뢰도가 낮습니다. 기구가 잘 보이도록 다시 촬영해주세요.")
    elif image is None:
        with st.expander("촬영 예시 보기", expanded=True):
            if EXAMPLE_THUMBNAIL.exists():
                st.image(str(EXAMPLE_THUMBNAIL), width="stretch")
            st.caption("예시: 이렇게 기구 전체가 프레임 안에 들어오도록 촬영해주세요.")
        st.info("카메라 화면이 안 뜨나요? 브라우저 주소창의 카메라 아이콘에서 권한을 허용해주세요.")

    exercise_db = load_exercise_db()
    with st.expander("카메라를 사용할 수 없나요? 직접 선택하기"):
        manual_key = st.selectbox(
            "운동 선택",
            options=list(exercise_db.keys()),
            format_func=lambda k: exercise_db[k]["name"],
            key="detect_manual_select",
            label_visibility="collapsed",
        )
        manual_clicked = st.button("선택한 운동으로 진행", width="stretch", key="detect_manual_confirm")

    with st.bottom:
        clicked = st.button("운동 시작", width="stretch", key="detect_start", disabled=not good_enough)

    if clicked and exercise_data is not None:
        st.session_state["detected_exercise"] = {
            "exercise_key": result["exercise_key"],
            "confidence": result["confidence"],
        }
        st.session_state.current_exercise_key = result["exercise_key"]

        # 다음에 detect 화면에 다시 오면 새 사진을 요구하도록 위젯 상태를 비웁니다.
        if CAMERA_KEY in st.session_state:
            del st.session_state[CAMERA_KEY]

        go_to(PAGE_EXERCISE)

    if manual_clicked:
        st.session_state["detected_exercise"] = {
            "exercise_key": manual_key,
            "confidence": None,  # 카메라 인식이 아니라 직접 선택했다는 표시
        }
        st.session_state.current_exercise_key = manual_key
        st.session_state["workout_mode"] = WORKOUT_MODE_MANUAL  # 카메라가 안 되면 자세 분석도 못 쓰니 기본을 수동으로

        if CAMERA_KEY in st.session_state:
            del st.session_state[CAMERA_KEY]

        go_to(PAGE_EXERCISE)
