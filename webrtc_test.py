"""
streamlit-webrtc 웹캠 실시간 스트리밍 테스트.

PaiT의 workout/detect 페이지에 실시간 웹캠을 붙이기 전, WebRTC 파이프라인
(브라우저 카메라 -> 서버 -> 프레임 처리 -> 브라우저) 자체가 이 환경에서
정상 동작하는지 확인하기 위한 독립 테스트 스크립트입니다.

사용 예:
    streamlit run webrtc_test.py

확인 포인트:
- START를 누르면 카메라 권한 요청 후 실시간 영상이 뜨는지
- 좌측 상단 FPS 숫자가 실시간으로 갱신되는지 (서버까지 프레임이 왕복하고 있다는 증거)
- '그레이스케일 변환' 체크박스로 서버에서 프레임을 실제로 가공하는지
- 같은 Wi-Fi의 폰 브라우저(Network URL)에서도 카메라가 붙는지
"""

import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# 로컬 네트워크 밖(외부 STUN)에서도 접속을 시도할 수 있도록 공개 STUN 서버 사용.
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FrameProcessor(VideoProcessorBase):
    """수신 프레임마다 FPS를 계산해 오버레이하고, 필요 시 그레이스케일로 변환합니다."""

    def __init__(self):
        self.grayscale = False
        self._frame_count = 0
        self._last_tick = time.time()
        self._fps = 0.0

    def _update_fps(self):
        self._frame_count += 1
        elapsed = time.time() - self._last_tick
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_tick = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._update_fps()

        if self.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.putText(img, f"FPS: {self._fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # TODO(PaiT 연동): 여기서 프레임을 ONNX 기구 인식 / MediaPipe Pose에 넘기면 됩니다.
        # 예: pred = components.camera.run_onnx_detection(Image.fromarray(img[:, :, ::-1]))

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="WebRTC 웹캠 테스트", page_icon="📷")
    st.title("streamlit-webrtc 실시간 웹캠 테스트")
    st.caption("아래 START 버튼을 눌러 카메라 권한을 허용하면 실시간 스트리밍이 시작됩니다.")

    ctx = webrtc_streamer(
        key="pait-webrtc-test",
        video_processor_factory=FrameProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.grayscale = st.checkbox("그레이스케일 변환 (서버 처리 확인용)", value=False)

    if ctx.state.playing:
        st.success("스트리밍 중입니다. 영상 좌측 상단 FPS가 계속 바뀌면 정상입니다.")
    elif ctx.state.signalling:
        st.warning("연결 중입니다 (STUN/ICE 협상)...")
    else:
        st.info("START를 눌러 스트리밍을 시작하세요.")


if __name__ == "__main__":
    main()
