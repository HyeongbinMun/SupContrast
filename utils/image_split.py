import os
import shutil
import random
from pathlib import Path
from typing import Tuple

def split_dataset(
    source_dir: str,
    target_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> None:
    random.seed(seed)

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    classes = [d.name for d in source_dir.iterdir() if d.is_dir()]

    for split in ['train', 'val']:
        for cls in classes:
            (target_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        img_files = list((source_dir / cls).glob('*'))
        random.shuffle(img_files)

        split_idx = int(len(img_files) * train_ratio)
        train_files = img_files[:split_idx]
        val_files = img_files[split_idx:]

        for file in train_files:
            shutil.copy(file, target_dir / 'train' / cls / file.name)

        for file in val_files:
            shutil.copy(file, target_dir / 'val' / cls / file.name)

    print(f"Dataset split complete. Train ratio: {train_ratio:.2f}")
    print(f"Output path: {target_dir}")

split_dataset(
    source_dir='/ssd/hbmun/supcon/etri',
    target_dir='/ssd/hbmun/supcon/etri_linear',
    train_ratio=0.8
)