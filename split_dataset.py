"""
dataset/train/<class>/ 에 모아둔 이미지 중 일부를 dataset/val/<class>/ 로 분리합니다.

사용 예:
    python split_dataset.py --val-ratio 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_class(train_dir: Path, val_dir: Path, val_ratio: float, seed: int, copy: bool):
    images = [p for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if len(images) < 2:
        print(f"  건너뜀 (이미지 {len(images)}장, 최소 2장 필요): {train_dir.name}")
        return 0

    rng = random.Random(seed)
    rng.shuffle(images)

    num_val = max(1, round(len(images) * val_ratio))
    num_val = min(num_val, len(images) - 1)  # train에 최소 1장은 남김
    val_images = images[:num_val]

    val_dir.mkdir(parents=True, exist_ok=True)
    for src in val_images:
        dst = val_dir / src.name
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    print(f"  {train_dir.name}: train {len(images) - num_val}장 / val {num_val}장")
    return num_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2, help="val로 옮길 비율 (기본 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="이동 대신 복사 (train 폴더에 원본을 남겨둠)",
    )
    args = parser.parse_args()

    train_root = args.data_dir / "train"
    val_root = args.data_dir / "val"

    if not train_root.exists():
        raise FileNotFoundError(f"train 폴더가 없습니다: {train_root}")

    class_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    print(f"{len(class_dirs)}개 클래스에 대해 val_ratio={args.val_ratio} 로 분할합니다.")

    total_val = 0
    for class_dir in class_dirs:
        total_val += split_class(
            train_dir=class_dir,
            val_dir=val_root / class_dir.name,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy=args.copy,
        )

    print(f"완료. 총 {total_val}장을 val로 {'복사' if args.copy else '이동'}했습니다.")


if __name__ == "__main__":
    main()
