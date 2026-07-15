"""
train.py로 학습한 best_model.pt를 ONNX로 변환하고 INT8(static, QDQ) 양자화까지 수행합니다.

사용 예:
    python export_onnx.py --checkpoint best_model.pt --data-dir dataset

MobileNetV2는 Conv 레이어 비중이 커서 dynamic quantization(Linear/LSTM 가중치만 양자화)으로는
효과가 거의 없습니다. 그래서 calibration 데이터로 활성값 range를 측정하는 static quantization(QDQ)을 사용합니다.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image
from torchvision import transforms

from train import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


def export_fp32_onnx(model, onnx_path: Path):
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    print(f"Exported FP32 ONNX to {onnx_path}")


def collect_calibration_images(data_dir: Path, num_images: int) -> list[Path]:
    all_images = list((data_dir / "train").rglob("*.*"))
    all_images = [p for p in all_images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not all_images:
        raise RuntimeError(f"{data_dir / 'train'} 아래에 calibration용 이미지가 없습니다.")
    random.shuffle(all_images)
    return all_images[:num_images]


class GymCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path]):
        self.image_paths = image_paths
        self._iter = iter(self.image_paths)

    def get_next(self):
        path = next(self._iter, None)
        if path is None:
            return None
        img = Image.open(path).convert("RGB")
        tensor = VAL_TRANSFORM(img).unsqueeze(0).numpy().astype(np.float32)
        return {INPUT_NAME: tensor}

    def rewind(self):
        self._iter = iter(self.image_paths)


def quantize_int8(fp32_path: Path, int8_path: Path, calib_images: list[Path]):
    reader = GymCalibrationDataReader(calib_images)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"Exported INT8 ONNX to {int8_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("export"))
    parser.add_argument("--num-calib-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = args.output_dir / "model_fp32.onnx"
    int8_path = args.output_dir / "model_int8.onnx"

    model, classes = load_model(args.checkpoint)
    export_fp32_onnx(model, fp32_path)

    calib_images = collect_calibration_images(args.data_dir, args.num_calib_images)
    print(f"Using {len(calib_images)} calibration images")
    quantize_int8(fp32_path, int8_path, calib_images)

    classes_path = args.output_dir / "classes.txt"
    classes_path.write_text("\n".join(classes), encoding="utf-8")
    print(f"Saved class list to {classes_path}")


if __name__ == "__main__":
    main()
