"""
export_onnx.py로 만든 ONNX 모델을 val 셋에 대해 평가하고 confusion matrix를 그려 저장합니다.

사용 예:
    python plot_confusion_matrix.py --model export/model_fp32.onnx --data-dir dataset
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix
from torchvision import datasets

from export_onnx import INPUT_NAME, VAL_TRANSFORM


def load_val_dataset(data_dir: Path):
    return datasets.ImageFolder(data_dir / "val", transform=VAL_TRANSFORM)


def predict_all(session: ort.InferenceSession, dataset) -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    for idx in range(len(dataset)):
        tensor, label = dataset[idx]
        input_array = tensor.unsqueeze(0).numpy().astype(np.float32)
        outputs = session.run(None, {INPUT_NAME: input_array})
        pred = int(np.argmax(outputs[0], axis=1)[0])
        y_true.append(label)
        y_pred.append(pred)
    return y_true, y_pred


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: Path, normalize: bool):
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), max(5, len(classes) * 0.7)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:{fmt}}"
            color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("export/model_fp32.onnx"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("confusion_matrix.png"))
    parser.add_argument("--normalize", action="store_true", help="행 기준으로 비율(0~1)로 정규화해서 표시")
    args = parser.parse_args()

    dataset = load_val_dataset(args.data_dir)
    classes = dataset.classes
    print(f"Val samples: {len(dataset)}, classes: {classes}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    y_true, y_pred = predict_all(session, dataset)

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plot_confusion_matrix(cm, classes, args.output, args.normalize)


if __name__ == "__main__":
    main()
