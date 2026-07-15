"""
dataset/train/<class>/ 이미지에 오프라인 데이터 증강을 적용해 클래스당 목표 장수까지 채웁니다.
(train.py의 실시간 증강과 별개로, 실제 파일을 늘려서 저장합니다.)

적용 기법 (증강 이미지마다 무작위로 조합):
- 좌우반전 (50% 확률)
- 밝기 조절 (0.7 ~ 1.3배)
- 가우시안 블러
- 랜덤 회전 (±15도)

사용 예:
    python augment_dataset.py --target 100
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    brightness_factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    blur_radius = rng.uniform(0.5, 1.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    return img


def augment_class(class_dir: Path, target: int, rng: random.Random) -> int:
    originals = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    current = len(originals)
    needed = target - current

    if not originals:
        print(f"  건너뜀 (원본 이미지 없음): {class_dir.name}")
        return 0
    if needed <= 0:
        print(f"  건너뜀 (이미 {current}장, 목표 {target}장 충족): {class_dir.name}")
        return 0

    for i in range(needed):
        src_path = rng.choice(originals)
        with Image.open(src_path) as img:
            augmented = augment_image(img, rng)

        dst_path = class_dir / f"aug_{i:04d}_{src_path.stem}.jpg"
        while dst_path.exists():
            dst_path = class_dir / f"aug_{i:04d}_{rng.randint(0, 999999)}_{src_path.stem}.jpg"
        augmented.save(dst_path, quality=95)

    print(f"  {class_dir.name}: {current}장 -> {target}장 ({needed}장 증강 생성)")
    return needed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--target", type=int, default=100, help="클래스별 목표 총 이미지 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    class_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    print(f"{len(class_dirs)}개 클래스를 target={args.target} 로 증강합니다.")

    total_generated = 0
    for class_dir in class_dirs:
        total_generated += augment_class(class_dir, args.target, rng)

    print(f"완료. 총 {total_generated}장의 증강 이미지를 생성했습니다.")


if __name__ == "__main__":
    main()
