"""PaiT 전역 상수 정의."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EXERCISE_DB_PATH = BASE_DIR / "database" / "exercise_db.json"
ROUTINE_DB_PATH = BASE_DIR / "database" / "routine_db.json"
HISTORY_LOG_PATH = BASE_DIR / "database" / "history_log.json"
MODEL_PATH = BASE_DIR / "export" / "model_fp32.onnx"
CLASSES_PATH = BASE_DIR / "export" / "classes.txt"
CSS_PATH = BASE_DIR / "assets" / "css" / "style.css"
THUMBNAIL_DIR = BASE_DIR / "assets" / "thumbnails"

POSE_MODEL_PATH = BASE_DIR / "assets" / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

APP_NAME = "PaiT"
APP_TAGLINE = "Train Smarter, Anywhere."
APP_SUBTITLE = "Edge AI Personal Trainer"

GOALS = [
    {"key": "벌크업", "title": "벌크업", "desc": "근육량을 늘리고 싶다면"},
    {"key": "다이어트", "title": "다이어트", "desc": "체지방을 줄이고 싶다면"},
    {"key": "자세교정", "title": "자세교정", "desc": "바른 자세를 만들고 싶다면"},
]

BODY_PARTS = ["상체", "하체", "전신"]

PAGE_SPLASH = "splash"
PAGE_GOAL = "goal"
PAGE_ROUTINE = "routine"
PAGE_DETECT = "detect"
PAGE_EXERCISE = "exercise"
PAGE_WORKOUT = "workout"
PAGE_RESULT = "result"
PAGE_HISTORY = "history"
PAGE_SETTINGS = "settings"

SIDEBAR_ITEMS = [
    {"key": PAGE_SPLASH, "label": "Home"},
    {"key": PAGE_GOAL, "label": "Workout"},
    {"key": PAGE_HISTORY, "label": "History"},
    {"key": PAGE_SETTINGS, "label": "Settings"},
]

# 루틴 예상 소요시간/칼로리 계산용 데모 근사치 (실측 데이터 아님)
SEC_PER_REP = 3
DEFAULT_REST_SEC = 60
KCAL_PER_REP = 0.35

# 실시간 Rep 카운팅/자세 피드백을 완성도 있게 튜닝한 운동만 workout(실시간 운동) 화면으로 보냅니다.
# 나머지 운동은 기구 인식 + 운동 정보 화면까지만 제공합니다.
POSE_SUPPORTED_EXERCISES = {"lat_pulldown", "chest_press", "shoulder_press"}

# 기구 인식 신뢰도 배지 기준
GOOD_CONFIDENCE_THRESHOLD = 0.7

# 운동 화면 모드 (detect.py의 카메라-없음 대체 흐름과 workout.py 토글이 같은 값을 공유합니다)
WORKOUT_MODE_AI = "AI 자세 분석"
WORKOUT_MODE_MANUAL = "수동 카운팅"
