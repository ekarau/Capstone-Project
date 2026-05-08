"""CLI entry: apply offline augmentation to data/unified/train.

Usage:
    python -m scripts.augment_dataset --unified data/unified --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.audit import audit_yolo_dataset, print_audit
from src.dataset.augment import augment_train_split
from src.dataset.unify import TARGET_CLASSES
from src.utils.config_loader import ElevatorConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default="data/unified")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = ElevatorConfig.from_yaml(args.config)
    aug_cfg = cfg.augmentation.to_dict() if hasattr(cfg, "augmentation") else cfg["augmentation"]

    if not aug_cfg.get("enabled", True):
        print("Augmentation disabled in config.")
        return

    train_root = Path(args.unified) / "train"
    augment_train_split(train_root, aug_cfg, seed=cfg["random_seed"])
    print_audit(audit_yolo_dataset(args.unified), class_names=TARGET_CLASSES)


if __name__ == "__main__":
    main()
