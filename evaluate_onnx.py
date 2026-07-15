"""
export_onnx.py로 만든 FP32 / INT8 ONNX 모델을 val 셋으로 검증합니다.
정확도(accuracy)와 추론 속도(latency), 모델 파일 크기를 비교 출력합니다.

사용 예:
    python evaluate_onnx.py --data-dir dataset --export-dir export
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torchvision import datasets

from train import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: F401 (참고용 상수)
from export_onnx import VAL_TRANSFORM, INPUT_NAME


def load_val_dataset(data_dir: Path):
    return datasets.ImageFolder(data_dir / "val", transform=VAL_TRANSFORM)


def build_session(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def evaluate_model(session: ort.InferenceSession, dataset, num_warmup: int = 10):
    correct, total = 0, 0
    latencies = []

    for idx in range(len(dataset)):
        tensor, label = dataset[idx]
        input_array = tensor.unsqueeze(0).numpy().astype(np.float32)

        start = time.perf_counter()
        outputs = session.run(None, {INPUT_NAME: input_array})
        elapsed = time.perf_counter() - start

        if idx >= num_warmup:
            latencies.append(elapsed)

        pred = int(np.argmax(outputs[0], axis=1)[0])
        correct += int(pred == label)
        total += 1

    accuracy = correct / total
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else float("nan")
    return accuracy, avg_latency_ms


def format_row(name: str, size_mb: float, accuracy: float, latency_ms: float) -> str:
    return f"{name:<12} | {size_mb:>10.2f} MB | {accuracy * 100:>8.2f} % | {latency_ms:>10.2f} ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--export-dir", type=Path, default=Path("export"))
    parser.add_argument("--num-warmup", type=int, default=10)
    args = parser.parse_args()

    fp32_path = args.export_dir / "model_fp32.onnx"
    int8_path = args.export_dir / "model_int8.onnx"

    for path in (fp32_path, int8_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없습니다. 먼저 export_onnx.py를 실행하세요.")

    dataset = load_val_dataset(args.data_dir)
    print(f"Val samples: {len(dataset)}, classes: {dataset.classes}")

    print(f"{'model':<12} | {'size':>13} | {'accuracy':>10} | {'avg latency':>13}")
    print("-" * 58)

    for name, path in (("FP32", fp32_path), ("INT8", int8_path)):
        session = build_session(path)
        accuracy, latency_ms = evaluate_model(session, dataset, args.num_warmup)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(format_row(name, size_mb, accuracy, latency_ms))


if __name__ == "__main__":
    main()
