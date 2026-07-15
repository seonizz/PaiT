# PaiT — Personal AI Trainer

**Edge AI Personal Trainer**: 카메라로 헬스장 기구를 인식하고, 실시간으로 자세와 Rep 수를 분석해주는 온디바이스 AI 운동 보조 웹앱입니다. Streamlit 기반으로 만들어졌으며, 별도 서버 없이 로컬 브라우저(모바일 포함)에서 바로 동작합니다.

## 주요 기능

- **기구 인식**: MobileNetV2(ONNX) 모델로 사진 한 장만으로 헬스장 기구 10종을 실시간 인식 (Confidence 표시)
- **실시간 자세 분석**: MediaPipe Pose로 웹캠에서 관절 스켈레톤을 추적해 Rep을 자동으로 카운트하고, 자세 피드백(반동/허리 자세/가동범위)을 실시간으로 제공 — Lat Pulldown / Chest Press / Shoulder Press 3종에 집중 튜닝
- **수동 카운팅 모드**: 카메라를 쓸 수 없는 환경에서도 `[+1 Rep]` 버튼으로 직접 세트를 기록할 수 있는 대체 모드
- **맞춤 루틴 추천**: 목표(벌크업 / 다이어트 / 자세교정)와 운동 부위(상체 / 하체 / 전신)에 맞춰 오늘의 루틴을 자동 구성
- **운동 기록 및 칼로리 계산**: 세트/Rep/운동시간을 기반으로 MET 공식으로 칼로리를 계산하고, 완료 기록을 영구 저장해 History 화면에서 조회
- **미니멀 UI**: 화면당 단일 Primary 버튼, 상단 루틴 진행률 표시, 하단 고정 CTA 등 3초 안에 상태를 파악할 수 있는 모바일 친화적 UI

## 기술 스택

| 영역 | 기술 |
|---|---|
| App / UI | Streamlit, streamlit-webrtc |
| 기구 인식 | PyTorch(학습) → ONNX Runtime(추론), MobileNetV2 전이학습 |
| 자세 분석 | MediaPipe Tasks API (Pose Landmarker) |
| 이미지 처리 | OpenCV, Pillow |
| 데이터 | JSON 기반 로컬 DB (exercise_db / routine_db / history_log) |

## 실행 방법

```bash
git clone https://github.com/seonizz/PaiT.git
cd PaiT
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속하면 됩니다. 모바일에서 테스트하려면 같은 네트워크에서 `http://<PC의 IP>:8501` 로 접속하세요.

> 최초 실행 시 MediaPipe Pose 모델(`pose_landmarker_lite.task`)이 없으면 자동으로 다운로드됩니다.


