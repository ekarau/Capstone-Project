"""CLI entry point for unifying raw Roboflow datasets.

Usage:
    python -m scripts.prepare_dataset \
        --raw "C:/Users/karau/Desktop/Capstone/datas" \
        --out data/unified
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when running as `python scripts/prepare_dataset.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.audit import audit_yolo_dataset, print_audit
from src.dataset.unify import TARGET_CLASSES, unify_datasets


def main() -> None:
    ap = argparse.ArgumentParser(description="Unify raw Roboflow datasets.")
    ap.add_argument("--raw", required=True, help="Directory containing raw *.yolov8/ folders")
    ap.add_argument("--out", default="data/unified", help="Output unified dataset directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-only", action="store_true", help="Only audit existing --out")
    args = ap.parse_args()

    raw_root = Path(args.raw).resolve()
    out_root = Path(args.out).resolve()

    if not args.audit_only:
        if not raw_root.exists():
            raise SystemExit(f"--raw path missing: {raw_root}")
        unify_datasets(raw_root, out_root, seed=args.seed)

    print_audit(audit_yolo_dataset(out_root), class_names=TARGET_CLASSES)


if __name__ == "__main__":
    main()
