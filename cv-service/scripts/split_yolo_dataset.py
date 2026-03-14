#!/usr/bin/env python3
"""
Split a YOLO dataset into train/val subsets with deterministic sampling.

Why this script exists
----------------------
For this repo, images were annotated first and placed entirely under
`datasets/durian/images/train` and `datasets/durian/labels/train`.
Training YOLO requires a validation subset as well, and doing the split
manually is error-prone because images and labels must move together.

This script enforces three business rules:
1. image/label pairs must always stay in sync
2. the split must be reproducible via a fixed random seed
3. existing `val` content should be cleared before regenerating, so the
   result reflects the current dataset instead of accumulating stale files
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "durian"


@dataclass(frozen=True)
class DatasetPair:
    """A matched image/label pair identified by the same basename."""

    stem: str
    image_path: Path
    label_path: Path


def _iter_pairs(images_dir: Path, labels_dir: Path) -> list[DatasetPair]:
    """
    Collect image/label pairs from the train subset.

    The split intentionally uses only the current `train` subset as the source
    of truth. This makes the script idempotent when re-run after the user moves
    every file back to train to regenerate the split.
    """

    image_map = {
        path.stem: path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_map = {
        path.stem: path
        for path in sorted(labels_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    missing_labels = sorted(set(image_map) - set(label_map))
    extra_labels = sorted(set(label_map) - set(image_map))
    if missing_labels or extra_labels:
        problems: list[str] = []
        if missing_labels:
            problems.append(f"missing labels for: {', '.join(missing_labels[:10])}")
        if extra_labels:
            problems.append(f"labels without images: {', '.join(extra_labels[:10])}")
        raise RuntimeError("; ".join(problems))

    return [
        DatasetPair(stem=stem, image_path=image_map[stem], label_path=label_map[stem])
        for stem in sorted(image_map)
    ]


def _move_all_files(source_dir: Path, target_dir: Path, suffixes: Iterable[str]) -> None:
    """Move matching files from source to target if they exist."""

    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        shutil.move(str(path), target_dir / path.name)


def _prepare_output_dirs(dataset_root: Path) -> dict[str, Path]:
    """Create required YOLO directories and return them."""

    paths = {
        "images_train": dataset_root / "images" / "train",
        "images_val": dataset_root / "images" / "val",
        "labels_train": dataset_root / "labels" / "train",
        "labels_val": dataset_root / "labels" / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_source_state(paths: dict[str, Path]) -> None:
    """
    Move any existing val files back into train before re-splitting.

    This keeps the split deterministic. Without this normalization, a rerun
    would sample from an already-split train subset and silently distort the
    train/val ratio.
    """

    _move_all_files(paths["images_val"], paths["images_train"], IMAGE_EXTENSIONS)
    _move_all_files(paths["labels_val"], paths["labels_train"], {".txt"})


def split_dataset(dataset_root: Path, *, val_ratio: float, seed: int) -> dict[str, int]:
    """
    Split the dataset into train/val by moving files.

    Args:
        dataset_root: Root directory containing `images/` and `labels/`.
        val_ratio: Fraction of pairs to move into validation.
        seed: Random seed for deterministic sampling.
    """

    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1")

    paths = _prepare_output_dirs(dataset_root)
    _normalize_source_state(paths)

    pairs = _iter_pairs(paths["images_train"], paths["labels_train"])
    total = len(pairs)
    if total < 2:
        raise RuntimeError("Need at least 2 labeled images to create train/val split")

    val_count = max(1, round(total * val_ratio))
    if val_count >= total:
        val_count = total - 1

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    val_stems = {pair.stem for pair in shuffled[:val_count]}

    for pair in pairs:
        if pair.stem not in val_stems:
            continue
        shutil.move(str(pair.image_path), paths["images_val"] / pair.image_path.name)
        shutil.move(str(pair.label_path), paths["labels_val"] / pair.label_path.name)

    train_count = total - val_count
    return {"total": total, "train": train_count, "val": val_count}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the YOLO dataset into train/val subsets deterministically."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio. Example: 0.2 means 20%% validation data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = split_dataset(args.dataset_root, val_ratio=args.val_ratio, seed=args.seed)
    print(
        "split complete:",
        f"total={result['total']}",
        f"train={result['train']}",
        f"val={result['val']}",
        f"seed={args.seed}",
        sep=" ",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
