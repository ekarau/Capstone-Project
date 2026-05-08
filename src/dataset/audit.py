"""Dataset audit utility — counts classes & flags broken labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def audit_yolo_dataset(root: Path) -> dict:
    """Walk a YOLO dataset and return per-split, per-class instance counts."""
    root = Path(root)
    report: dict = {"splits": {}, "issues": []}

    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        n_images = sum(1 for _ in img_dir.iterdir() if _.is_file())
        cls_counter: Counter[int] = Counter()
        empty = 0
        bad_lines = 0
        for lbl_path in lbl_dir.glob("*.txt"):
            if lbl_path.stat().st_size == 0:
                empty += 1
                continue
            with open(lbl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        bad_lines += 1
                        report["issues"].append(
                            f"{lbl_path.name}: non-bbox line ({len(parts)} fields)"
                        )
                        continue
                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        bad_lines += 1
                        continue
                    cls_counter[cls_id] += 1
        report["splits"][split] = {
            "images": n_images,
            "empty_labels": empty,
            "bad_lines": bad_lines,
            "instances_per_class": dict(cls_counter),
        }
    return report


def print_audit(report: dict, class_names: list[str] | None = None) -> None:
    print("\n=== Dataset Audit ===")
    for split, info in report["splits"].items():
        print(f"\n[{split}]")
        print(f"  images       : {info['images']}")
        print(f"  empty labels : {info['empty_labels']}")
        print(f"  bad lines    : {info['bad_lines']}")
        for cls_id, n in sorted(info["instances_per_class"].items()):
            label = (
                class_names[cls_id] if class_names and cls_id < len(class_names) else f"id={cls_id}"
            )
            print(f"  {label:>10s} : {n}")
    if report["issues"]:
        print(f"\nIssues ({len(report['issues'])}):")
        for issue in report["issues"][:20]:
            print(f"  - {issue}")
        if len(report["issues"]) > 20:
            print(f"  ... and {len(report['issues']) - 20} more")
