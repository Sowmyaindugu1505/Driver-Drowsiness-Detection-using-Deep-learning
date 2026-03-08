from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path

from config import ARCHIVE_PATH, EXTRACT_DIR, EYE_DATA_DIR, YAWN_DATA_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SPLIT_RATIOS = {"train": 0.7, "valid": 0.15, "test": 0.15}
SEED = 42

ALIASES = {
    "eyes": {
        "open": {"open", "eye_open", "eyes_open"},
        "closed": {"closed", "eye_closed", "eyes_closed", "close"},
    },
    "yawns": {
        "yawn": {"yawn", "yawning", "mouth_open"},
        "no_yawn": {"no_yawn", "not_yawn", "no-yawn", "non_yawn", "no yawn", "normal"},
    },
}


def normalize(value: str) -> str:
    """Normalize string to lowercase with underscores instead of spaces/hyphens."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def find_class_dirs(root: Path, aliases: set[str]) -> list[Path]:
    """Find all directories inside root whose name matches one of the aliases."""
    normalized_aliases = {normalize(a) for a in aliases}
    return [
        path for path in root.rglob("*")
        if path.is_dir() and normalize(path.name) in normalized_aliases
    ]


def collect_images(directories: list[Path]) -> list[Path]:
    """Collect all valid image files from a list of directories."""
    files: list[Path] = []
    for directory in directories:
        files.extend([
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
    return files


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    """Shuffle and split a list of files into train/valid/test dictionaries."""
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


def reset_output_dirs(base: Path, class_names: list[str]) -> None:
    """Clear and recreate the required folder structure (train/valid/test -> classes)."""
    if base.exists():
        shutil.rmtree(base)
    for split in SPLIT_RATIOS:
        for class_name in class_names:
            (base / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_split(split_map: dict[str, list[Path]], output_dir: Path, class_name: str) -> None:
    """Copy the actual files to their respective train/valid/test class directories."""
    for split, files in split_map.items():
        for i, source in enumerate(files):
            dst = output_dir / split / class_name / f"{class_name}_{i}{source.suffix.lower()}"
            shutil.copy2(source, dst)


def extract_archive() -> None:
    """Extract archive.zip if it exists."""
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Missing zip file at: {ARCHIVE_PATH.resolve()}")

    print(f"Extracting {ARCHIVE_PATH.name}...")
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)


def build_dataset(output_dir: Path, spec: dict[str, set[str]], feature_name: str) -> None:
    """Core logic to find images, split them, and save to target directory."""
    print(f"\nBuilding dataset for: {feature_name.upper()}")
    reset_output_dirs(output_dir, list(spec.keys()))

    for class_name, alias_set in spec.items():
        class_dirs = find_class_dirs(EXTRACT_DIR, alias_set)
        if not class_dirs:
            print(f"  [Warning] No folders found for class '{class_name}' with aliases: {sorted(alias_set)}")
            continue

        files = collect_images(class_dirs)
        if not files:
            print(f"  [Warning] No images found inside '{class_name}' folders.")
            continue

        split_map = split_files(files)
        copy_split(split_map, output_dir, class_name)
        
        print(
            f"  {class_name.ljust(10)} -> "
            f"Train: {len(split_map['train']):<5} "
            f"Valid: {len(split_map['valid']):<5} "
            f"Test: {len(split_map['test']):<5}"
        )


def main() -> None:
    try:
        extract_archive()
        build_dataset(EYE_DATA_DIR, ALIASES["eyes"], "eyes")
        build_dataset(YAWN_DATA_DIR, ALIASES["yawns"], "yawns")
        print("\nDataset preparation completed successfully!")
    except Exception as e:
        print(f"\n[Error] {e}")


if __name__ == "__main__":
    main()
