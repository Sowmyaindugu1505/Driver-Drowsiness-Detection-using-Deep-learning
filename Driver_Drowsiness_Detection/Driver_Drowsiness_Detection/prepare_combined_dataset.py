"""Unzip archive.zip and prepare eye + yawn datasets for training.

Expected classes anywhere inside extracted archive folder:
- Eye classes: open, closed
- Yawn classes: yawn, no_yawn (or similar aliases)
"""

from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = PROJECT_DIR / "archive.zip"
EXTRACT_DIR = PROJECT_DIR / "datasets" / "archive_extracted"

EYE_OUTPUT = PROJECT_DIR / "data"
YAWN_OUTPUT = PROJECT_DIR / "data_yawn"

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
    normalized = {normalize(a) for a in aliases}
    matched = []
    for directory in root.rglob("*"):
        if directory.is_dir() and normalize(directory.name) in normalized:
            matched.append(directory)
    return matched


def collect_images(directories: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(path)
    return files


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    files = files[:]
    random.Random(SEED).shuffle(files)
    total = len(files)
    train_end = int(total * SPLIT_RATIOS["train"])
    valid_end = train_end + int(total * SPLIT_RATIOS["valid"])
    return {
        "train": files[:train_end],
        "valid": files[train_end:valid_end],
        "test": files[valid_end:],
    }


def reset_output(base: Path, class_names: list[str]) -> None:
    if base.exists():
        shutil.rmtree(base)
    for split in SPLIT_RATIOS:
        for class_name in class_names:
            (base / split / class_name).mkdir(parents=True, exist_ok=True)


def write_split(split_files_map: dict[str, list[Path]], output: Path, class_name: str) -> None:
    for split, files in split_files_map.items():
        for index, source in enumerate(files):
            dst = output / split / class_name / f"{source.stem}_{index}{source.suffix.lower()}"
            shutil.copy2(source, dst)


def unzip_archive() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"archive.zip not found at: {ARCHIVE_PATH}")

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)


def prepare_task(output_dir: Path, spec: dict[str, set[str]]) -> None:
    reset_output(output_dir, list(spec.keys()))

    for class_name, alias_set in spec.items():
        class_dirs = find_class_dirs(EXTRACT_DIR, alias_set)
        if not class_dirs:
            raise FileNotFoundError(
                f"Could not find class directories for '{class_name}' with aliases: {sorted(alias_set)}"
            )

        files = collect_images(class_dirs)
        if not files:
            raise FileNotFoundError(f"No images found for class '{class_name}'")

        split_map = split_files(files)
        write_split(split_map, output_dir, class_name)
        print(
            f"{output_dir.name}/{class_name}: "
            f"train={len(split_map['train'])}, valid={len(split_map['valid'])}, test={len(split_map['test'])}"
        )


def main() -> None:
    unzip_archive()
    prepare_task(EYE_OUTPUT, {"open": ALIASES["open"], "closed": ALIASES["closed"]})
    prepare_task(YAWN_OUTPUT, {"yawn": ALIASES["yawn"], "no_yawn": ALIASES["no_yawn"]})
    print("Prepared both eye and yawn datasets successfully.")


if __name__ == "__main__":
    main()
