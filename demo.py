"""
웹캠 화면을 실시간으로 분류해 기구 이름과 신뢰도를 화면에 표시합니다.

사용 예:
    python demo.py --model export/model_fp32.onnx --classes export/classes.txt

조작:
    q 키를 누르면 종료합니다.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from train import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

INPUT_NAME = "input"

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def load_classes(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def predict(session: ort.InferenceSession, frame_bgr: np.ndarray) -> tuple[int, float]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    tensor = VAL_TRANSFORM(image).unsqueeze(0).numpy().astype(np.float32)

    outputs = session.run(None, {INPUT_NAME: tensor})
    probs = softmax(outputs[0][0])
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("export/model_fp32.onnx"))
    parser.add_argument("--classes", type=Path, default=Path("export/classes.txt"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.5, help="이 값 미만이면 '알 수 없음'으로 표시")
    args = parser.parse_args()

    classes = load_classes(args.classes)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(index={args.camera_index})를 열 수 없습니다.")

    print("웹캠 추론 시작. 'q'를 누르면 종료합니다.")

    prev_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        pred, confidence = predict(session, frame)
        label = classes[pred] if confidence >= args.min_confidence else "알 수 없음"

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        text = f"{label} ({confidence * 100:.1f}%)"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Gym Equipment Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
