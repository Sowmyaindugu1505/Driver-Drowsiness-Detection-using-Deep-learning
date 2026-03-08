"""Prepare both eye and yawn datasets from a single archive.zip.

Output folders:
- data/{train,valid,test}/{open,closed}
- data_yawn/{train,valid,test}/{yawn,no_yawn}
"""

from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path

from app_config import ARCHIVE_PATH, EXTRACT_DIR, EYE_DATA_DIR, YAWN_DATA_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SPLIT_RATIOS = {"train": 0.7, "valid": 0.15, "test": 0.15}
SEED = 42

ALIASES = {
    "open": {"open", "eye_open", "eyes_open"},
    "closed": {"closed", "eye_closed", "eyes_closed", "close"},
    "yawn": {"yawn", "yawning"},
    "no_yawn": {"no_yawn", "not_yawn", "no-yawn", "non_yawn", "no yawn", "normal"},
}


def normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def find_class_dirs(root: Path, aliases: set[str]) -> list[Path]:
    alias_set = {normalize(a) for a in aliases}
    return [p for p in root.rglob("*") if p.is_dir() and normalize(p.name) in alias_set]


def collect_images(directories: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        files.extend(
            [
                p
                for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
    return files


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    items = files[:]
    random.Random(SEED).shuffle(items)
    total = len(items)
    train_end = int(total * SPLIT_RATIOS["train"])
    valid_end = train_end + int(total * SPLIT_RATIOS["valid"])
    return {
        "train": items[:train_end],
        "valid": items[train_end:valid_end],
        "test": items[valid_end:],
    }


def reset_output(base: Path, class_names: list[str]) -> None:
    if base.exists():
        shutil.rmtree(base)
    for split in SPLIT_RATIOS:
        for class_name in class_names:
            (base / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_split(split_map: dict[str, list[Path]], output_dir: Path, class_name: str) -> None:
    for split, files in split_map.items():
        for i, source in enumerate(files):
            dst = output_dir / split / class_name / f"{source.stem}_{i}{source.suffix.lower()}"
            shutil.copy2(source, dst)


def unzip_archive() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"archive.zip not found at: {ARCHIVE_PATH}")

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)


def build_task(output_dir: Path, spec: dict[str, set[str]]) -> None:
    reset_output(output_dir, list(spec.keys()))

    for class_name, aliases in spec.items():
        class_dirs = find_class_dirs(EXTRACT_DIR, aliases)
        if not class_dirs:
            raise FileNotFoundError(f"Missing class folders for '{class_name}' (aliases: {sorted(aliases)})")

        files = collect_images(class_dirs)
        if not files:
            raise FileNotFoundError(f"No images found for class '{class_name}'")

        split_map = split_files(files)
        copy_split(split_map, output_dir, class_name)
        print(
            f"{output_dir.name}/{class_name}: "
            f"train={len(split_map['train'])}, valid={len(split_map['valid'])}, test={len(split_map['test'])}"
        )


def main() -> None:
    unzip_archive()
    build_task(EYE_DATA_DIR, {"open": ALIASES["open"], "closed": ALIASES["closed"]})
    build_task(YAWN_DATA_DIR, {"yawn": ALIASES["yawn"], "no_yawn": ALIASES["no_yawn"]})
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
