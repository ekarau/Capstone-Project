"""Smart import of LASTDATASET into our raw data pool.

This script:
  1. Reads LASTDATASET (a 4-class person/stroller/luggage/box Roboflow export)
  2. Skips images with `aug_*` prefix (already pre-augmented — our pipeline
     does augmentation itself, double-augmentation degrades quality)
  3. Computes the "original stem" by stripping Roboflow's _jpg.rf.<hash>
  4. Skips images whose original stem already exists in our existing datas/
  5. Copies the remaining unique-and-unaugmented images into a NEW source
     directory: <datas>/lastdataset_extra.yolov8/, structured like a
     standard Roboflow YOLOv8 export so unify.py can pick it up.

Usage:
    python -m scripts.import_lastdataset \
        --src "C:/Users/karau/Desktop/LASTDATASET" \
        --existing-datas "C:/Users/karau/Desktop/Capstone/datas" \
        --out-name lastdataset_extra.yolov8

After running, add a new SourceSpec to src/dataset/unify.py:
    SourceSpec(
        folder="lastdataset_extra.yolov8",
        class_map={0:"person", 1:"stroller", 2:"luggage", 3:"box"},
    ),
Then re-run `python -m scripts.prepare_dataset`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Roboflow hash suffix: "_jpg.rf.<hex>" / "_png.rf.<hex>"
RF_RE = re.compile(r"_(jpe?g|png|bmp)\.rf\.[A-Za-z0-9]+$", re.IGNORECASE)


def original_stem(filename: str) -> str:
    """Strip Roboflow hash and 'aug_' prefix to recover the original source stem."""
    s = Path(filename).stem
    s = RF_RE.sub("", s)
    if s.startswith("aug_"):
        s = s[4:]
    return s


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """MD5 hash of file content — used as last-resort dedup fallback."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def collect_existing_stems(existing_datas: Path) -> set[str]:
    """Walk all *.yolov8/{train,valid,test}/images dirs and collect stems."""
    stems = set()
    for img_path in existing_datas.rglob("*.jpg"):
        if "/images/" in img_path.as_posix() or "\\images\\" in str(img_path):
            stems.add(original_stem(img_path.name))
    for img_path in existing_datas.rglob("*.png"):
        if "/images/" in img_path.as_posix() or "\\images\\" in str(img_path):
            stems.add(original_stem(img_path.name))
    return stems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LASTDATASET root")
    ap.add_argument("--existing-datas", required=True, help="Existing Capstone/datas dir")
    ap.add_argument(
        "--out-name",
        default="lastdataset_extra.yolov8",
        help="Folder name to create under existing-datas/",
    )
    ap.add_argument(
        "--skip-aug-prefix",
        default="aug_",
        help="Skip images whose name starts with this (default 'aug_')",
    )
    args = ap.parse_args()

    src = Path(args.src).resolve()
    existing = Path(args.existing_datas).resolve()
    out_root = existing / args.out_name

    if not src.exists():
        raise SystemExit(f"src not found: {src}")
    if not existing.exists():
        raise SystemExit(f"existing-datas not found: {existing}")
    if out_root.exists():
        print(f"WARNING: {out_root} already exists — removing.")
        shutil.rmtree(out_root)

    print(f"Source        : {src}")
    print(f"Existing data : {existing}")
    print(f"Output folder : {out_root}\n")

    # Step 1: collect stems already present in our existing data
    print("[1/3] Collecting stems from existing datas/ ...")
    existing_stems = collect_existing_stems(existing)
    print(f"      {len(existing_stems)} unique stems found.\n")

    # Step 2: walk source train/val/test and copy filtered images
    print("[2/3] Filtering and copying LASTDATASET ...")
    stats = {
        "skipped_aug": 0,
        "skipped_duplicate": 0,
        "skipped_no_label": 0,
        "kept": 0,
    }

    # Class label is preserved as-is (LASTDATASET already has 4 classes:
    # 0=person, 1=stroller, 2=luggage, 3=box — same order as ours)
    for split in ("train", "val", "test"):
        src_imgs = src / split / "images"
        src_lbls = src / split / "labels"
        if not src_imgs.is_dir():
            continue
        # Important: we ignore LASTDATASET's split structure and put
        # everything under a single 'train' (the unifier will re-split
        # using its own group-stratified logic).
        dst_imgs = out_root / "train" / "images"
        dst_lbls = out_root / "train" / "labels"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_imgs.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            # Filter 1: aug_* prefix
            if img_path.name.startswith(args.skip_aug_prefix):
                stats["skipped_aug"] += 1
                continue

            # Filter 2: duplicate stem
            stem = original_stem(img_path.name)
            if stem in existing_stems:
                stats["skipped_duplicate"] += 1
                continue

            # Filter 3: needs a non-empty label
            lbl_path = src_lbls / f"{img_path.stem}.txt"
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                stats["skipped_no_label"] += 1
                continue

            # Avoid in-batch collisions: prefix split for trace
            new_img_name = f"{split}__{img_path.name}"
            new_lbl_name = f"{split}__{img_path.stem}.txt"
            shutil.copy2(img_path, dst_imgs / new_img_name)
            shutil.copy2(lbl_path, dst_lbls / new_lbl_name)
            stats["kept"] += 1

    # Step 3: write a Roboflow-style data.yaml so unify.py recognizes it
    print("[3/3] Writing data.yaml ...\n")
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "train: ../train/images\n"
            "val: ../valid/images\n"
            "test: ../test/images\n\n"
            "nc: 4\n"
            "names: ['person', 'stroller', 'luggage', 'box']\n\n"
            "# Imported by scripts/import_lastdataset.py from LASTDATASET\n"
        )

    # Summary
    print("=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)
    print(f"  Skipped (aug_*)       : {stats['skipped_aug']}")
    print(f"  Skipped (duplicates)  : {stats['skipped_duplicate']}")
    print(f"  Skipped (no labels)   : {stats['skipped_no_label']}")
    print(f"  Kept (unique, clean)  : {stats['kept']}")
    print(f"\n  -> {out_root}")
    print("\nNext steps:")
    print("  1. Add a SourceSpec to src/dataset/unify.py:")
    print("       SourceSpec(")
    print(f'           folder="{args.out_name}",')
    print('           class_map={0:"person", 1:"stroller", 2:"luggage", 3:"box"},')
    print("       )")
    print("  2. python -m scripts.prepare_dataset --raw <datas> --out data/unified")
    print(
        "  3. python -m scripts.augment_dataset --unified data/unified --config configs/default.yaml"
    )


if __name__ == "__main__":
    main()
