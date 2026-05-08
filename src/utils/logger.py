"""Project-wide logger."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def _configure(level: int = logging.INFO, log_file: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    _configure(log_file=log_file)
    return logging.getLogger(name)
