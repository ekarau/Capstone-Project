"""Multi-source YOLO dataset unifier — leakage-safe edition.

Merges 10 raw Roboflow datasets into a single, consistently labeled
YOLO dataset with 4 target classes (0:person, 1:stroller, 2:luggage, 3:box).

LEAKAGE-SAFE DESIGN:
  * Many sources contain video frames (e.g. `6Z6jTNfqkUSqM_3D_mp4-0`,
    `-10`, ...). A naive random split would leak near-duplicate frames
    across train/val/test, inflating apparent accuracy.
  * We extract a `group_key` from each image's filename (Roboflow naming
    convention strips at "_jpg.rf." / "_png.rf.") and split GROUPS, not
    individual images. All frames from one video stay in one split.
  * Augmentation operates ONLY on `train/`; val/test are untouched.

TEST-SET COMPOSITION:
  * `Elevator.yolov8` is kept as the user's "real-cabin" test scenario
    (100 % of its images go to test).
  * Every other source contributes ~5 % of its groups to test, ensuring
    test contains all four classes (person, stroller, luggage, box).
"""

from __future__ import annotations

import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Final unified class list (order = id)
TARGET_CLASSES: list[str] = ["person", "stroller", "luggage", "box"]
TARGET_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


@dataclass(frozen=True)
class SourceSpec:
    """Per-dataset configuration.

    Attributes:
        folder: Directory name under data/raw/.
        class_map: Map raw_class_id -> target_class_name (None to drop class).
        split_role: 'standard'  = group-stratified split into train/val/test.
                    'test_only' = held out 100% as test set.
        train_ratio / val_ratio / test_ratio: only used when split_role='standard'.
        max_images: Optional cap on number of images sampled.
        skip_empty_labels: If True, images with no annotations are skipped.
    """

    folder: str
    class_map: dict[int, str | None]
    split_role: str = "standard"
    train_ratio: float = 0.80
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    max_images: int | None = None
    skip_empty_labels: bool = True


# ─────────────────────────────────────────────────────────────────────
#  Source dataset specifications
#  Verified against actual data.yaml of each Roboflow export.
# ─────────────────────────────────────────────────────────────────────
SOURCES: list[SourceSpec] = [
    # Kullanıcının "asansör senaryosu" referans verisi → 100% test-only.
    SourceSpec(
        folder="Elevator.yolov8",
        class_map={0: "person"},
        split_role="test_only",
    ),
    # Person datasets — group-stratified split
    SourceSpec(
        folder="-People Counting.yolov8",
        class_map={0: "person"},
    ),
    SourceSpec(
        folder="people ditection in elevator.yolov8",
        class_map={0: "person"},
    ),
    SourceSpec(
        folder="top down view.yolov8",
        class_map={0: "person"},
        # Sadece 39 görüntü — train/val/test çok küçük olur.
        # Hepsini train'e koymak için val_ratio + test_ratio = 0.
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
    ),
    SourceSpec(
        folder="normal3.yolov8",
        class_map={0: "person"},
    ),
    # Stroller multi-class (yalnız person/stroller/luggage)
    SourceSpec(
        folder="Stroller.yolov8",
        class_map={
            0: "person",     # 0-human
            1: None,         # 1-wheelchair
            2: "luggage",    # 2-suitcase
            3: "stroller",   # 3-stroller
            4: None,         # 4-bicycle
        },
    ),
    # Luggage / suitcase datasets
    SourceSpec(
        folder="My Luggage.yolov8",
        class_map={0: "luggage"},
    ),
    SourceSpec(
        folder="luggage.yolov8",
        class_map={0: "luggage", 1: "luggage"},
    ),
    SourceSpec(
        folder="suitcase.yolov8",
        class_map={0: "luggage", 1: "luggage", 2: None, 3: None},
    ),
    # Box (155 görüntü — küçük; en az birkaç tanesi test'e ayrılır)
    SourceSpec(
        folder="box.yolov8",
        class_map={0: "box"},
        train_ratio=0.78, val_ratio=0.15, test_ratio=0.07,
    ),
    # LASTDATASET'ten dedup edilmiş yeni veri — 737 imaj, 982 person/91 stroller/
    # 130 luggage/222 box. scripts/import_lastdataset.py ile temizlendi.
    SourceSpec(
        folder="lastdataset_extra.yolov8",
        class_map={0: "person", 1: "stroller", 2: "luggage", 3: "box"},
    ),
]


@dataclass
class UnifyStats:
    sources: dict[str, dict] = field(default_factory=dict)
    train: int = 0
    val: int = 0
    test: int = 0
    dropped_empty: int = 0
    dropped_classmap: int = 0
    leaked_groups: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Dataset Unification Report\n"]
        lines.append("| Source | Train | Val | Test | Empty skipped | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for name, info in self.sources.items():
            lines.append(
                f"| {name} | {info.get('train', 0)} | {info.get('val', 0)} | "
                f"{info.get('test', 0)} | {info.get('skipped_empty', 0)} | "
                f"{info.get('notes', '')} |"
            )
        lines.append("")
        lines.append(f"**Train total**: {self.train}  ")
        lines.append(f"**Val total**: {self.val}  ")
        lines.append(f"**Test total**: {self.test}  ")
        lines.append(f"**Dropped (empty labels)**: {self.dropped_empty}  ")
        lines.append(f"**Dropped (no usable classes)**: {self.dropped_classmap}  ")
        lines.append("")
        lines.append("## Leakage check (group-prefix overlap between splits)")
        if self.leaked_groups:
            lines.append(f"⚠️ {len(self.leaked_groups)} groups found in >1 split:")
            for g in self.leaked_groups[:20]:
                lines.append(f"- `{g}`")
        else:
            lines.append("✅ No group-prefix leakage detected.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────
def _iter_split_dirs(source_root: Path) -> Iterable[tuple[str, Path, Path]]:
    """Yield (split_name, images_dir, labels_dir) for all available splits."""
    for split in ("train", "valid", "val", "test"):
        img_dir = source_root / split / "images"
        lbl_dir = source_root / split / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            yield split, img_dir, lbl_dir


# Roboflow Augmentation suffixes ("_jpg.rf.<hash>", "_png.rf.<hash>", ...)
_ROBOFLOW_RE = re.compile(r"_(jpe?g|png|bmp)\.rf\.[A-Za-z0-9]+$", re.IGNORECASE)


def _group_key(image_stem: str) -> str:
    """Extract a stable group key from an image filename stem.

    Goal: frames from the same source video / capture session share a key
    so they cannot be split across train/val/test (no near-duplicate
    leakage).

    Heuristics:
      1) Strip Roboflow's "_jpg.rf.<hash>" suffix.
      2) Strip a trailing "-<digits>" or "_<digits>" frame index, leaving
         the per-video prefix (e.g. "video123-42" -> "video123").
      3) If nothing remained, fall back to the original stem so each
         image is its own group (no aggressive grouping).
    """
    s = _ROBOFLOW_RE.sub("", image_stem)
    # Strip frame counter at the end: "_001", "-42", "_frame_99"
    s2 = re.sub(r"[_-](?:frame[_-]?)?\d{1,6}$", "", s, flags=re.IGNORECASE)
    return s2 if s2 else s


def _polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    """Convert flat [x1, y1, x2, y2, ...] polygon coords -> YOLO bbox cx,cy,w,h."""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return cx, cy, w, h


def _remap_label_file(
    label_path: Path,
    class_map: dict[int, str | None],
) -> tuple[list[str], int, int]:
    """Read a YOLO .txt and return (new_lines, kept_count, dropped_count).

    Accepts both bbox (5 fields) and polygon-segmentation (>=7 fields, odd).
    Polygon lines are converted to their axis-aligned bounding box.
    """
    if not label_path.exists():
        return [], 0, 0
    new_lines: list[str] = []
    kept, dropped = 0, 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                old_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                dropped += 1
                continue

            # Determine bbox depending on label format
            if len(coords) == 4:
                cx, cy, w, h = coords
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                conv = _polygon_to_bbox(coords)
                if conv is None:
                    dropped += 1
                    continue
                cx, cy, w, h = conv
            else:
                dropped += 1
                continue

            target_name = class_map.get(old_id)
            if target_name is None or target_name not in TARGET_ID:
                dropped += 1
                continue
            new_id = TARGET_ID[target_name]
            # Clamp normalized bbox to [0, 1]
            bbox = [max(0.0, min(1.0, v)) for v in (cx, cy, w, h)]
            # Reject degenerate bboxes
            if bbox[2] <= 0 or bbox[3] <= 0:
                dropped += 1
                continue
            new_lines.append(
                f"{new_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            )
            kept += 1
    return new_lines, kept, dropped


def _safe_copy_image(src: Path, dst_dir: Path, prefix: str) -> Path:
    """Copy image to dst_dir with a source-prefix to avoid name collisions."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_name = f"{prefix}__{src.name}"
    dst = dst_dir / new_name
    shutil.copy2(src, dst)
    return dst


def _write_label(lines: list[str], dst_dir: Path, image_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    dst = dst_dir / f"{stem}.txt"
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return dst


# ─────────────────────────────────────────────────────────────────────
#  Main API
# ─────────────────────────────────────────────────────────────────────
def _split_groups(
    groups: dict[str, list],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[set[str], set[str], set[str]]:
    """Assign each GROUP key to one split. Frames stay together → no leakage.

    The ratios are applied at the IMAGE level: groups are sorted by size
    descending and dealt out to whichever split is still under-quota
    (proportional to the running per-split totals).
    """
    total_imgs = sum(len(v) for v in groups.values())
    targets = {
        "train": train_ratio * total_imgs,
        "val": val_ratio * total_imgs,
        "test": test_ratio * total_imgs,
    }
    splits = {"train": set(), "val": set(), "test": set()}
    counts = {"train": 0, "val": 0, "test": 0}

    # Group keys sorted by size, with a deterministic shuffle for ties
    keys = list(groups.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))

    for k in keys:
        # Pick the split currently most under-target (in absolute units)
        deficits = {s: targets[s] - counts[s] for s in splits if targets[s] > 0}
        if not deficits:
            chosen = "train"
        else:
            chosen = max(deficits, key=deficits.get)
        splits[chosen].add(k)
        counts[chosen] += len(groups[k])

    return splits["train"], splits["val"], splits["test"]


def unify_datasets(
    raw_root: Path,
    out_root: Path,
    seed: int = 42,
) -> UnifyStats:
    """Walk all sources, remap labels, group-stratified split into train/val/test.

    No augmentation is applied here — `augment_train_split` is the separate
    train-only step. Val and test are kept entirely "clean" (no augmented
    duplicates of any kind).
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    rng = random.Random(seed)
    stats = UnifyStats()

    # Reset output directories
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = out_root / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # For leakage post-check across sources
    group_to_splits: dict[str, set[str]] = defaultdict(set)

    for spec in SOURCES:
        src_dir = raw_root / spec.folder
        if not src_dir.exists():
            logger.warning(f"Skip missing source: {spec.folder}")
            stats.sources[spec.folder] = {"notes": "MISSING"}
            continue

        prefix = spec.folder.replace(" ", "_").replace(".", "_")
        skipped_empty = 0
        skipped_classmap = 0

        # 1) Collect per-source images grouped by their video/session prefix
        per_group: dict[str, list[tuple[Path, list[str]]]] = defaultdict(list)
        for split, img_dir, lbl_dir in _iter_split_dirs(src_dir):
            for img_path in sorted(img_dir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                new_lines, kept, dropped = _remap_label_file(lbl_path, spec.class_map)
                if not new_lines:
                    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                        skipped_empty += 1
                        stats.dropped_empty += 1
                    else:
                        skipped_classmap += 1
                        stats.dropped_classmap += 1
                    continue
                gkey = f"{prefix}::{_group_key(img_path.stem)}"
                per_group[gkey].append((img_path, new_lines))

        # 2) Decide split membership per source
        if spec.split_role == "test_only":
            split_assign = {"train": set(), "val": set(), "test": set(per_group.keys())}
        else:
            tr, va, te = _split_groups(
                per_group,
                spec.train_ratio,
                spec.val_ratio,
                spec.test_ratio,
                rng,
            )
            split_assign = {"train": tr, "val": va, "test": te}

        # 3) Materialize files
        per_split_count = {"train": 0, "val": 0, "test": 0}
        for split_name, gkeys in split_assign.items():
            for gkey in gkeys:
                group_to_splits[gkey].add(split_name)
                for img_path, lines in per_group[gkey]:
                    dst_img = _safe_copy_image(
                        img_path, out_root / split_name / "images", prefix
                    )
                    _write_label(lines, out_root / split_name / "labels", dst_img.name)
                    per_split_count[split_name] += 1

        stats.sources[spec.folder] = {
            "train": per_split_count["train"],
            "val": per_split_count["val"],
            "test": per_split_count["test"],
            "skipped_empty": skipped_empty,
            "skipped_classmap": skipped_classmap,
            "notes": spec.split_role,
        }
        stats.train += per_split_count["train"]
        stats.val += per_split_count["val"]
        stats.test += per_split_count["test"]
        logger.info(
            f"[{spec.folder}] train={per_split_count['train']} "
            f"val={per_split_count['val']} test={per_split_count['test']} "
            f"empty={skipped_empty} classmap_drop={skipped_classmap}"
        )

    # 4) Leakage post-check — a group key showing up in >1 split is a bug
    leaked = [g for g, sset in group_to_splits.items() if len(sset) > 1]
    stats.leaked_groups = leaked
    if leaked:
        logger.warning(f"LEAKAGE: {len(leaked)} groups appear in >1 split!")
    else:
        logger.info("Leakage check passed — no group prefix in >1 split.")

    # 5) Write data.yaml + report
    # 'path' is the absolute location at preparation time. When moving the
    # dataset to a different machine (e.g. Colab), call rewrite_data_yaml()
    # to update this field — see scripts/fix_colab_yaml.py.
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "# Auto-generated by src.dataset.unify\n"
            f"path: {out_root.resolve().as_posix()}\n"
            "train: train/images\n"
            "val: val/images\n"
            "test: test/images\n\n"
            f"nc: {len(TARGET_CLASSES)}\n"
            "names:\n"
        )
        for name in TARGET_CLASSES:
            f.write(f"  - {name}\n")

    report_path = out_root / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(stats.to_markdown())

    logger.info(
        f"Done. train={stats.train} val={stats.val} test={stats.test} "
        f"empty={stats.dropped_empty} classmap_drop={stats.dropped_classmap}"
    )
    logger.info(f"data.yaml -> {yaml_path}")
    logger.info(f"report -> {report_path}")
    return stats
