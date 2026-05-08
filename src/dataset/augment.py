"""Offline class-balancing augmentation using Albumentations.

DESIGN — LEAKAGE-SAFE GUARANTEES:

  1. Operates ONLY on `<unified>/train/`. Hard-asserts that the path
     ends with `/train` so it cannot accidentally augment val or test.
  2. Each augmented copy is named `<original_stem>__aug<k>.<ext>` so
     re-runs are idempotent (existing aug files are detected and skipped
     unless `force=True`).
  3. Does NOT generate "augmented test/val" — by design, val/test stay
     completely clean. Reported metrics reflect the model's true
     generalization, not memorized augmentations of training data.
  4. Multiplier per image = max per-class multiplier across classes in
     that image. `box`-class multiplier=5 → every box image yields 4
     extra augmented copies.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from src.dataset.unify import TARGET_CLASSES, TARGET_ID
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_pipeline(aug_cfg: dict):
    """Return an Albumentations Compose with bbox support (yolo format)."""
    import albumentations as A

    geo = aug_cfg.get("geometric", {})
    photo = aug_cfg.get("photometric", {})
    occ = aug_cfg.get("occlusion", {})

    return A.Compose(
        [
            A.HorizontalFlip(p=geo.get("horizontal_flip_p", 0.5)),
            A.Affine(
                scale=(1 - geo.get("scale_limit", 0.2), 1 + geo.get("scale_limit", 0.2)),
                rotate=(-geo.get("rotate_limit_deg", 15), geo.get("rotate_limit_deg", 15)),
                translate_percent=(
                    -geo.get("translate_limit", 0.1),
                    geo.get("translate_limit", 0.1),
                ),
                p=0.7,
                fit_output=False,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=photo.get("brightness_limit", 0.25),
                contrast_limit=photo.get("contrast_limit", 0.25),
                p=0.7,
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(photo.get("hsv_h", 0.015) * 180),
                sat_shift_limit=int(photo.get("hsv_s", 0.4) * 100),
                val_shift_limit=int(photo.get("hsv_v", 0.4) * 100),
                p=0.5,
            ),
            A.GaussNoise(p=photo.get("gaussian_noise_p", 0.2)),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, occ.get("cutout_max_h_w", 0.15)),
                hole_width_range=(0.05, occ.get("cutout_max_h_w", 0.15)),
                fill=0,
                p=occ.get("random_erasing_p", 0.3),
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.3,
            min_area=20,
        ),
    )


def _read_label(path: Path) -> tuple[list[list[float]], list[int]]:
    bboxes, cls_ids = [], []
    if not path.exists():
        return bboxes, cls_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                bb = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            # Pre-clamp to strictly [0, 1] so Albumentations doesn't choke on -1e-7
            cx, cy, w, h = (max(0.0, min(1.0, v)) for v in bb)
            # Also keep bbox edges inside [0, 1]
            x_min = max(0.0, cx - w / 2)
            y_min = max(0.0, cy - h / 2)
            x_max = min(1.0, cx + w / 2)
            y_max = min(1.0, cy + h / 2)
            new_w = x_max - x_min
            new_h = y_max - y_min
            if new_w <= 0 or new_h <= 0:
                continue
            cls_ids.append(cid)
            bboxes.append([(x_min + x_max) / 2, (y_min + y_max) / 2, new_w, new_h])
    return bboxes, cls_ids


def _write_label(path: Path, bboxes: list[list[float]], cls_ids: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for cls_id, b in zip(cls_ids, bboxes):
            # Clamp to [0, 1] — Albumentations may emit -1e-7 due to float ops
            b = [max(0.0, min(1.0, float(v))) for v in b]
            # Reject degenerate bboxes after clamping
            if b[2] <= 0 or b[3] <= 0:
                continue
            f.write(
                f"{int(cls_id)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n"
            )


def augment_train_split(
    train_root: Path,
    aug_cfg: dict,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Replicate each train image based on its class multipliers.

    Args:
        train_root: Path ending in 'train' (the train/ subdir of unified data).
        aug_cfg: cfg.augmentation dict from default.yaml.
        seed: deterministic randomness.
        force: If False (default), existing __augN.* files are skipped so the
            function is idempotent. Set True to overwrite previous aug runs.

    SAFETY:
      Asserts that `train_root` ends with 'train' to prevent accidental
      augmentation of val/ or test/.
    """
    random.seed(seed)
    np.random.seed(seed)
    train_root = Path(train_root)

    # ── Safety: refuse to augment anything that is not a train split ──
    if train_root.name != "train":
        raise ValueError(
            f"augment_train_split refuses to operate on {train_root!r}: "
            "path must end with 'train' (val/test must stay clean)."
        )

    img_dir = train_root / "images"
    lbl_dir = train_root / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"train split not found at {train_root}")

    pipeline = _build_pipeline(aug_cfg)
    multipliers = aug_cfg.get("per_class_multiplier", {})
    cls_mult = {TARGET_ID[name]: int(m) for name, m in multipliers.items() if name in TARGET_ID}

    stats = {"originals": 0, "augmented_copies": 0, "skipped": 0, "already_existed": 0}

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        # Skip files that are themselves augmented copies (idempotency)
        if "__aug" in img_path.stem:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        bboxes, cls_ids = _read_label(lbl_path)
        if not bboxes:
            continue
        stats["originals"] += 1

        mult = max((cls_mult.get(c, 1) for c in cls_ids), default=1)
        if mult <= 1:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            stats["skipped"] += 1
            continue

        for k in range(1, mult):
            stem = img_path.stem
            new_img_path = img_dir / f"{stem}__aug{k}{img_path.suffix}"
            new_lbl_path = lbl_dir / f"{stem}__aug{k}.txt"
            if new_img_path.exists() and not force:
                stats["already_existed"] += 1
                continue

            try:
                out = pipeline(image=image, bboxes=bboxes, class_ids=cls_ids)
            except Exception as e:
                logger.warning(f"aug failed for {img_path.name}: {e}")
                continue
            new_bboxes = out["bboxes"]
            new_cls = out["class_ids"]
            if not new_bboxes:
                continue
            cv2.imwrite(str(new_img_path), out["image"])
            _write_label(new_lbl_path, list(new_bboxes), list(new_cls))
            stats["augmented_copies"] += 1

    logger.info(
        f"Augmented train: originals={stats['originals']} "
        f"new_copies={stats['augmented_copies']} "
        f"existed={stats['already_existed']} skipped={stats['skipped']}"
    )
    return stats
