#!/usr/bin/env python3
"""
Rename dataset images sequentially by index (safe two-pass).

Why this exists
---------------
During dataset collection it's common to end up with filenames like
`image copy 12.png`, mixed extensions, or non-natural sort ordering
(`1.jpg`, `10.png`, `2.jpg`...). Some training pipelines and humans
prefer a clean sequence like `1.png`, `2.jpg`, ...

This script:
1) builds a deterministic ordering of images in a directory
2) generates a rename plan (old -> new)
3) applies the rename in *two passes* to avoid name collisions
4) optionally renames paired label files (YOLO format) with the same basename
5) writes a CSV mapping for audit/rollback

Safety notes
------------
- Default is dry-run. Use `--apply` to actually rename.
- It writes a mapping CSV (`rename_map_*.csv`) in the target directory.
- If you have external annotation manifests (e.g. COCO JSON) referencing
  image filenames, you must update them separately.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class RenameItem:
    """A single rename operation (image plus optional label)."""

    src: Path
    dst: Path
    label_src: Optional[Path]
    label_dst: Optional[Path]


def _extract_first_int(text: str) -> Optional[int]:
    """
    Extract the first integer in a filename (if any).

    This is used only to build a stable "human expected" ordering where
    names like `image copy 83.png` naturally sort near `83.*`.
    """

    match = re.search(r"(\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _iter_images(directory: Path, exts: set[str]) -> list[Path]:
    """List image files (non-recursive) in `directory`."""

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    images: list[Path] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in exts:
            continue
        images.append(entry)
    return images


def _sort_key(path: Path) -> tuple[int, int, str]:
    """
    Deterministic sorting:
    - files with any integer come first
    - then by that integer value
    - then by lowercase filename (tie-break)
    """

    number = _extract_first_int(path.stem)
    if number is None:
        return (1, 2**31 - 1, path.name.lower())
    return (0, number, path.name.lower())


def _compute_width(count: int, start: int, width: int, auto_width: bool) -> int:
    """Decide numeric padding width."""

    if width < 0:
        raise ValueError("--width must be >= 0")
    if width > 0:
        return width
    if not auto_width:
        return 0
    max_index = start + max(0, count - 1)
    return len(str(max_index))


def build_plan(
    images_dir: Path,
    *,
    labels_dir: Optional[Path],
    start: int,
    width: int,
    auto_width: bool,
    prefix: str,
) -> tuple[list[RenameItem], Path]:
    """
    Build rename plan and mapping CSV path.

    Args:
        images_dir: Directory containing images to rename.
        labels_dir: Optional paired labels directory (YOLO .txt by basename).
        start: Starting index (default 1).
        width: Zero-padding width. 0 means no padding unless `auto_width`.
        auto_width: If true and width==0, pad to the max index length.
        prefix: Prefix before the index (usually empty).
    """

    images = sorted(_iter_images(images_dir, SUPPORTED_IMAGE_EXTS), key=_sort_key)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    resolved_width = _compute_width(len(images), start, width, auto_width)
    ts = time.strftime("%Y%m%d_%H%M%S")
    mapping_csv = images_dir / f"rename_map_{ts}.csv"

    plan: list[RenameItem] = []
    for i, src in enumerate(images, start=start):
        index_str = str(i).zfill(resolved_width) if resolved_width else str(i)
        dst_name = f"{prefix}{index_str}{src.suffix.lower()}"
        dst = images_dir / dst_name

        label_src = None
        label_dst = None
        if labels_dir is not None:
            candidate = labels_dir / f"{src.stem}.txt"
            if candidate.exists():
                label_src = candidate
                label_dst = labels_dir / f"{prefix}{index_str}.txt"

        plan.append(RenameItem(src=src, dst=dst, label_src=label_src, label_dst=label_dst))

    # Sanity check: destination uniqueness.
    dst_set = {item.dst.name for item in plan}
    if len(dst_set) != len(plan):
        raise RuntimeError("Destination names are not unique; aborting.")

    return plan, mapping_csv


def _write_mapping_csv(mapping_csv: Path, plan: Iterable[RenameItem]) -> None:
    """Write a mapping CSV for audit/rollback."""

    with mapping_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["src", "dst", "label_src", "label_dst"], extrasaction="ignore"
        )
        writer.writeheader()
        for item in plan:
            writer.writerow(
                {
                    "src": str(item.src),
                    "dst": str(item.dst),
                    "label_src": str(item.label_src) if item.label_src else "",
                    "label_dst": str(item.label_dst) if item.label_dst else "",
                }
            )


def apply_plan(plan: list[RenameItem], *, mapping_csv: Path) -> None:
    """
    Apply the rename plan in two passes to avoid collisions.

    The two-pass strategy prevents problems like:
    - A.png -> 1.png while 1.png already exists and is also being renamed.
    """

    tmp_token = uuid.uuid4().hex[:8]
    tmp_items: list[tuple[Path, Path]] = []

    def tmp_name(p: Path) -> Path:
        return p.with_name(f".__tmp_rename_{tmp_token}__{p.name}")

    # First pass: move everything to unique temporary names.
    for item in plan:
        tmp_src = item.src
        tmp_dst = tmp_name(item.src)
        os.replace(tmp_src, tmp_dst)
        tmp_items.append((tmp_dst, item.dst))

        if item.label_src and item.label_dst:
            tmp_label_dst = tmp_name(item.label_src)
            os.replace(item.label_src, tmp_label_dst)
            tmp_items.append((tmp_label_dst, item.label_dst))

    # Second pass: move temp names to final names.
    for tmp_src, final_dst in tmp_items:
        os.replace(tmp_src, final_dst)

    _write_mapping_csv(mapping_csv, plan)


def load_plan_from_mapping_csv(mapping_csv: Path, *, reverse: bool) -> list[RenameItem]:
    """
    Load a rename plan from a previously generated mapping CSV.

    Args:
        mapping_csv: Path to `rename_map_*.csv` generated by this script.
        reverse: If true, swap src/dst to undo a previous rename.
    """

    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    plan: list[RenameItem] = []
    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"src", "dst", "label_src", "label_dst"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"Invalid mapping CSV header. Required columns: {sorted(required)}"
            )

        for row in reader:
            src = Path(row["src"])
            dst = Path(row["dst"])
            label_src = Path(row["label_src"]) if row.get("label_src") else None
            label_dst = Path(row["label_dst"]) if row.get("label_dst") else None

            if reverse:
                src, dst = dst, src
                label_src, label_dst = label_dst, label_src

            plan.append(RenameItem(src=src, dst=dst, label_src=label_src, label_dst=label_dst))

    dst_set = {item.dst for item in plan}
    if len(dst_set) != len(plan):
        raise RuntimeError("Destination paths are not unique; aborting.")

    return plan


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename dataset images sequentially by index (safe two-pass).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("server/python-cv-service/datasets/durian/images/train"),
        help="Target images directory (non-recursive).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("server/python-cv-service/datasets/durian/labels/train"),
        help="Paired labels directory (YOLO .txt). Use --no-labels to disable.",
    )
    parser.add_argument("--no-labels", action="store_true", help="Disable label renaming.")
    parser.add_argument("--start", type=int, default=1, help="Starting index.")
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Zero-padding width. 0 means no padding unless --auto-width is set.",
    )
    parser.add_argument(
        "--auto-width",
        action="store_true",
        help="Automatically pad to the max index length when --width=0.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix before the numeric index (e.g. 'train_').",
    )
    parser.add_argument("--apply", action="store_true", help="Actually rename files.")
    parser.add_argument(
        "--undo-csv",
        type=Path,
        default=None,
        help="Undo a previous rename using a mapping CSV generated by this script.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many planned operations to print.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    images_dir: Path = args.images_dir
    labels_dir: Optional[Path]
    if args.no_labels:
        labels_dir = None
    else:
        labels_dir = args.labels_dir
        if labels_dir is not None and not labels_dir.exists():
            # Labels directory is optional; treat as disabled if it doesn't exist.
            labels_dir = None

    if args.undo_csv:
        plan = load_plan_from_mapping_csv(args.undo_csv, reverse=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        mapping_csv = args.undo_csv.parent / f"undo_map_{ts}.csv"
    else:
        plan, mapping_csv = build_plan(
            images_dir,
            labels_dir=labels_dir,
            start=args.start,
            width=args.width,
            auto_width=args.auto_width,
            prefix=args.prefix,
        )

    print(f"Images dir: {images_dir}")
    if labels_dir:
        print(f"Labels dir: {labels_dir}")
    print(f"Total images: {len(plan)}")
    print(f"Mapping CSV: {mapping_csv}")
    print("")

    to_show = max(0, min(args.show, len(plan)))
    for item in plan[:to_show]:
        print(f"- {item.src.name} -> {item.dst.name}")
        if item.label_src and item.label_dst:
            print(f"  label: {item.label_src.name} -> {item.label_dst.name}")
    if len(plan) > to_show:
        print(f"... ({len(plan) - to_show} more)")

    if not args.apply:
        if args.undo_csv:
            print("\nUndo mode is also dry-run by default. Re-run with --apply to rename.")
        else:
            print("\nDry-run only. Re-run with --apply to rename.")
        return 0

    # Abort if any destination already exists and is not the same file.
    existing = [item.dst for item in plan if item.dst.exists()]
    if existing:
        # This can happen if you already renamed once. We fail fast to prevent overwrite.
        print("\nRefusing to overwrite existing destination files:", file=sys.stderr)
        for p in existing[:20]:
            print(f"- {p}", file=sys.stderr)
        if len(existing) > 20:
            print(f"... ({len(existing) - 20} more)", file=sys.stderr)
        return 2

    apply_plan(plan, mapping_csv=mapping_csv)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
