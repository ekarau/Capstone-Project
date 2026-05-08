"""YOLOv8 training entry point.

Usage (local):
    python -m src.detection.train --data data/unified/data.yaml --preset balanced

Usage (Colab): see notebooks/02_train_colab.ipynb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_loader import ElevatorConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_preset(cfg_model: dict, preset: str | None) -> dict:
    """Apply preset overrides on top of the base model section."""
    if not preset:
        return cfg_model
    presets = cfg_model.get("presets", {})
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}; available: {list(presets)}")
    overrides = presets[preset]
    merged = dict(cfg_model)
    merged.update(overrides)
    return merged


def train(
    data_yaml: str,
    config_path: str = "configs/default.yaml",
    preset: str | None = "balanced",
    project: str = "models/runs",
    name: str = "elevator",
    resume: bool = False,
) -> Path:
    from ultralytics import YOLO

    cfg = ElevatorConfig.from_yaml(config_path)
    model_cfg = _resolve_preset(cfg.model.to_dict(), preset)

    weights = model_cfg["pretrained_weights"]
    logger.info(
        f"Training {model_cfg['variant']} | epochs={model_cfg['epochs']} "
        f"batch={model_cfg.get('batch_size', 16)} imgsz={model_cfg.get('input_size', 640)}"
    )

    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=model_cfg["epochs"],
        batch=model_cfg.get("batch_size", 16),
        imgsz=model_cfg.get("input_size", 640),
        lr0=model_cfg.get("learning_rate", 0.01),
        patience=model_cfg.get("patience", 25),
        seed=cfg["random_seed"],
        project=project,
        name=name,
        resume=resume,
        plots=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Best weights: {best}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--preset", default="balanced", choices=["fast", "balanced", "accurate"])
    ap.add_argument("--project", default="models/runs")
    ap.add_argument("--name", default="elevator")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    train(
        data_yaml=args.data,
        config_path=args.config,
        preset=args.preset,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
