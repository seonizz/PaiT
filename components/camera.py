"""웹캠/사진 입력 + ONNX 기구 인식 + MediaPipe Pose 실시간 운동 분석.

이 환경에 설치되는 mediapipe(0.10.x, Windows wheel)는 예전에 흔히 쓰이던
``mp.solutions.pose`` 레거시 API를 포함하지 않고, 새 Tasks API(``mediapipe.tasks``)만
제공합니다. 그래서 PoseLandmarker는 Tasks API로 구현하고, 스켈레톤 연결선/랜드마크
이름도 ``mp.solutions.drawing_utils`` 대신 직접 정의해서 그립니다.

구성:
- render_camera_input() / run_onnx_detection(): 기구 인식(detect 페이지)용 정지 사진 처리
- PoseWorkoutProcessor: streamlit-webrtc VideoProcessorBase. 프레임마다
  1) MediaPipe Pose로 스켈레톤을 그리고
  2) 운동 종류별 관절 각도를 계산해 Rep을 자동 카운트하고
  3) 팔꿈치 각도 / 허리 기울기 / 반동 여부를 규칙 기반으로 판정해 피드백 문구를 만듭니다.
- render_workout_stream(): workout 페이지 진입점 (webrtc_streamer 실행 + session_state 동기화)
"""

import time
import urllib.request
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from utils.constants import CLASSES_PATH, MODEL_PATH, POSE_MODEL_PATH, POSE_MODEL_URL

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# --- 정지 사진 기반 기구 인식 (detect 페이지) -------------------------------------

# export_onnx.py(학습 파이프라인)의 VAL_TRANSFORM과 반드시 동일한 전처리여야 합니다.
# torch/torchvision을 배포 환경(Streamlit Community Cloud 등)에 설치하지 않기 위해
# PIL/NumPy만으로 동일한 연산(Resize 짧은 변 255 -> CenterCrop 224 -> [0,1] 정규화 ->
# ImageNet mean/std 정규화)을 재현합니다. 실제 이미지로 torchvision 결과와 diff=0 확인함.
ONNX_INPUT_NAME = "input"
ONNX_IMAGE_SIZE = 224
ONNX_RESIZE_SIZE = int(ONNX_IMAGE_SIZE * 1.14)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def render_camera_input(key: str = "detect_camera_input"):
    """기구 인식(detect 페이지)용 정지 사진 촬영."""
    captured = st.camera_input("기구를 카메라에 비춰주세요", key=key)
    if captured is None:
        return None
    return Image.open(captured)


@st.cache_resource
def _load_onnx_session():
    import onnxruntime as ort

    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def _preprocess_for_onnx(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    w, h = image.size
    if w < h:
        new_w, new_h = ONNX_RESIZE_SIZE, int(ONNX_RESIZE_SIZE * h / w)
    else:
        new_h, new_w = ONNX_RESIZE_SIZE, int(ONNX_RESIZE_SIZE * w / h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    left = int(round((new_w - ONNX_IMAGE_SIZE) / 2.0))
    top = int(round((new_h - ONNX_IMAGE_SIZE) / 2.0))
    image = image.crop((left, top, left + ONNX_IMAGE_SIZE, top + ONNX_IMAGE_SIZE))

    arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr[np.newaxis, ...].astype(np.float32)  # NCHW


def run_onnx_detection(image: Image.Image) -> dict:
    """detect 페이지에서 더미 데이터 대신 이 함수를 사용하도록 연결하면 실제 인식으로 전환됩니다."""
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("export_onnx.py를 먼저 실행해 모델을 생성하세요.")

    class_keys = CLASSES_PATH.read_text(encoding="utf-8").splitlines()
    session = _load_onnx_session()

    tensor = _preprocess_for_onnx(image)
    outputs = session.run(None, {ONNX_INPUT_NAME: tensor})
    logits = outputs[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    idx = int(np.argmax(probs))
    return {
        "exercise_key": class_keys[idx],
        "confidence": float(probs[idx]),
    }


# --- MediaPipe Pose (Tasks API) 실시간 운동 분석 ----------------------------------

# mp.solutions.pose.PoseLandmark 열거형이 없으므로 표준 33포인트 인덱스를 직접 정의.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# mp.solutions.pose.POSE_CONNECTIONS 와 동일한 표준 33포인트 연결선(스켈레톤 그리기용).
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)

JOINT_LANDMARKS = {
    "elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}

# 완성도를 높인 3개 운동(lat_pulldown/chest_press/shoulder_press)은 운동마다
# 실제 가동범위가 달라서 각각 다른 각도 임계값을 씁니다.
#   joint      : 추적할 관절(elbow/knee)
#   down/up    : Rep 카운팅 히스테리시스(도). 펴짐(up) -> 굽힘(down) -> 펴짐(up) = 1회
#   min_angle  : 이보다 작게(너무 접히면) 굽히면 "더 내려주세요" 피드백
EXERCISE_ANGLE_CONFIG = {
    # 풀다운: 위에서 거의 편 팔(160도 이상)로 시작해 가슴 앞까지 깊게 당겨옵니다.
    "lat_pulldown": {"joint": "elbow", "down": 90.0, "up": 160.0, "min_angle": 45.0},
    # 체스트 프레스: 가슴 앞(약 90도)에서 시작해 끝까지 밀어내되, 풀다운만큼 깊게 접히지 않습니다.
    "chest_press": {"joint": "elbow", "down": 100.0, "up": 155.0, "min_angle": 75.0},
    # 숄더 프레스: 어깨 높이(약 90도)에서 머리 위로 밀어 올립니다.
    "shoulder_press": {"joint": "elbow", "down": 100.0, "up": 160.0, "min_angle": 75.0},
}
# 나머지 운동(기구 인식 + 운동 정보만 제공)은 workout 화면까지 오지 않지만,
# 혹시 직접 접근하더라도 동작은 하도록 관절 기반 기본값을 둡니다.
DEFAULT_ANGLE_CONFIG = {"joint": "elbow", "down": 100.0, "up": 155.0, "min_angle": 60.0}
_LOWER_BODY_DEFAULT = {"joint": "knee", "down": 100.0, "up": 155.0, "min_angle": 60.0}
for _key in ("leg_press", "leg_curl_extension", "hip_abduction", "smith_machine"):
    EXERCISE_ANGLE_CONFIG.setdefault(_key, _LOWER_BODY_DEFAULT)
for _key in ("cable_row_crossover", "assisted_pullup", "pec_deck"):
    EXERCISE_ANGLE_CONFIG.setdefault(_key, DEFAULT_ANGLE_CONFIG)

# 자세 피드백 임계값 (전 운동 공통 규칙)
TORSO_LEAN_MAX_DEG = 20.0  # 수직 기준 이보다 기울면 "허리를 세워주세요"
BOUNCE_VELOCITY_DEG_PER_SEC = 220.0  # 각속도가 이보다 크면 "반동을 줄여주세요"
FEEDBACK_COOLDOWN_SEC = 1.5  # 같은 문구가 너무 자주 깜빡이지 않도록 최소 유지 시간


def ensure_pose_model() -> Path:
    """pose_landmarker_lite.task 모델이 없으면 다운로드합니다."""
    if not POSE_MODEL_PATH.exists():
        POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POSE_MODEL_URL, str(POSE_MODEL_PATH))
    return POSE_MODEL_PATH


def _calculate_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """세 점(a-b-c)이 이루는 b 지점의 각도(도)를 계산합니다."""
    a_arr, b_arr, c_arr = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - np.arctan2(
        a_arr[1] - b_arr[1], a_arr[0] - b_arr[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def _torso_lean_angle(landmarks) -> float:
    """어깨-엉덩이 중점을 잇는 선이 수직에서 얼마나 기울었는지(도)를 계산합니다."""
    ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    lh, rh = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]

    mid_shoulder = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    mid_hip = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def _draw_skeleton(img: np.ndarray, landmarks) -> None:
    h, w = img.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for i, j in POSE_CONNECTIONS:
        cv2.line(img, points[i], points[j], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)


class PoseWorkoutProcessor(VideoProcessorBase):
    """실시간 프레임마다 Pose 추정 -> 스켈레톤 오버레이 -> Rep 카운트 -> 자세 피드백."""

    def __init__(self):
        self.exercise_key = "lat_pulldown"  # 메인 스레드(workout 페이지)가 매 rerun마다 갱신
        self.set_target_reps = 12
        self.rep_count = 0
        self.feedback = ""

        self._state = "up"  # "up"(펴짐) / "down"(굽힘)
        self._prev_angle = None
        self._prev_time = time.time()
        self._last_feedback_time = 0.0
        self._frame_index = 0  # detect_for_video용 단조증가 타임스탬프(ms 대신 프레임 번호 사용)

        # 백그라운드(WebRTC worker) 스레드에서 초기화가 실패해도 콘솔에만 찍히고
        # 화면엔 아무 단서가 안 남는 경우가 있어, 에러를 저장해뒀다가 프레임에 직접 표시합니다.
        self._init_error = None
        try:
            model_path = ensure_pose_model()
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
        except Exception as e:  # noqa: BLE001
            self._landmarker = None
            self._init_error = f"Pose init failed: {e}"

    def _feedback_rule(self, joint_angle: float, torso_angle: float, angular_velocity: float, min_angle: float) -> str:
        # 우선순위: 반동(안전) > 허리 자세 > 관절 가동범위
        if angular_velocity > BOUNCE_VELOCITY_DEG_PER_SEC:
            return "반동을 줄여주세요"
        if torso_angle > TORSO_LEAN_MAX_DEG:
            return "허리를 세워주세요"
        if joint_angle < min_angle:
            return "더 내려주세요"
        return ""

    def reset(self):
        """세트가 바뀔 때 workout 페이지에서 호출해 카운터를 초기화합니다."""
        self.rep_count = 0
        self._state = "up"
        self._prev_angle = None
        self.feedback = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self._init_error:
            cv2.putText(img, self._init_error[:70], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # detect_for_video는 프레임마다 '엄격히 증가하는' 타임스탬프를 요구합니다.
            # time.time() 기반 ms값은 프레임 간격이 짧을 때 같은 값으로 반올림되어
            # 예외를 던질 수 있어, 항상 1씩 늘어나는 프레임 번호를 대신 사용합니다.
            self._frame_index += 1
            result = self._landmarker.detect_for_video(mp_image, self._frame_index)

            now = time.time()
            dt = max(now - self._prev_time, 1e-3)

            landmark_count = len(result.pose_landmarks[0]) if result.pose_landmarks else 0
            cv2.putText(
                img,
                f"Pose landmarks: {landmark_count}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
            )

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                _draw_skeleton(img, landmarks)

                config = EXERCISE_ANGLE_CONFIG.get(self.exercise_key, DEFAULT_ANGLE_CONFIG)
                joint = config["joint"]
                p1, p2, p3 = JOINT_LANDMARKS[joint]
                a = (landmarks[p1].x, landmarks[p1].y)
                b = (landmarks[p2].x, landmarks[p2].y)
                c = (landmarks[p3].x, landmarks[p3].y)
                joint_angle = _calculate_angle(a, b, c)
                torso_angle = _torso_lean_angle(landmarks)

                angular_velocity = 0.0
                if self._prev_angle is not None:
                    angular_velocity = abs(joint_angle - self._prev_angle) / dt
                self._prev_angle = joint_angle
                self._prev_time = now

                # Rep 카운팅: 펴짐(up) -> 굽힘(down) -> 펴짐(up) = 1회.
                # (프레스/풀 방향에 상관없이 굽힘-폄 사이클 하나를 1회로 셉니다)
                if self._state == "up" and joint_angle < config["down"]:
                    self._state = "down"
                elif self._state == "down" and joint_angle > config["up"]:
                    self._state = "up"
                    self.rep_count = min(self.rep_count + 1, self.set_target_reps)

                message = self._feedback_rule(joint_angle, torso_angle, angular_velocity, config["min_angle"])
                if message and (now - self._last_feedback_time) > FEEDBACK_COOLDOWN_SEC:
                    self.feedback = message
                    self._last_feedback_time = now
                elif not message:
                    self.feedback = ""

                cv2.putText(
                    img, f"Angle: {joint_angle:.0f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.putText(
                    img,
                    f"Rep: {self.rep_count}/{self.set_target_reps}",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                if self.feedback:
                    cv2.putText(img, self.feedback, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                self._prev_time = now

        except Exception as e:  # noqa: BLE001 - 실패 원인을 화면에 바로 노출
            cv2.putText(img, f"recv error: {e}"[:80], (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def render_workout_stream(exercise_key: str, target_reps: int):
    """workout 페이지에서 호출: 실시간 스트리밍을 띄우고 Rep/피드백을 session_state에 반영합니다."""
    ctx = webrtc_streamer(
        key="pait-workout-pose",
        video_processor_factory=PoseWorkoutProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.exercise_key = exercise_key
        ctx.video_processor.set_target_reps = target_reps
        st.session_state.rep_count = ctx.video_processor.rep_count
        st.session_state.pose_feedback = ctx.video_processor.feedback

    return ctx
