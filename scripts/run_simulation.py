"""End-to-end energy-savings simulation for Smart Elevator CV.

Pipeline
--------

1. Load every image in ``data/sim/images/`` together with its multi-class
   ground-truth row from ``data/sim/ground_truth.csv``::

       filename, gt_person, gt_stroller, gt_luggage, gt_box, gt_is_full

   ``gt_is_full`` may be left blank — the script then derives it from the
   true cabin occupancy ratio computed from the per-class instance counts.

2. Run the trained YOLOv8 detector(s) on each image and aggregate per-class
   predictions. Hybrid mode is enabled when ``--head-weights`` is provided:
   the head model supplies the person count while the four-class model
   contributes only stroller / luggage / box.

3. Emit a per-image ACCEPT / BYPASS decision via the two-stage policy
   (PDF Algorithm 1, Andrei & Ruokokoski 2022).

4. Compare against ground truth and tabulate confusion-matrix counts plus
   per-class detection accuracy (mean absolute error of class counts).

5. Sample ``--num-calls`` synthetic hall calls. Per-call cabin states are
   drawn uniformly from the labeled images. Aggregate energy spent under:

   * **Baseline** — every call accepted (always-stop).
   * **Smart**    — accept iff classifier says not full.

   Energy per stop is taken from ``src.energy.consumption`` (Tukia 2018).

6. Write::

       results/<output>/confusion_matrix.png
       results/<output>/per_image_decisions.csv
       results/<output>/per_class_detection.csv
       results/<output>/energy_savings.csv
       results/<output>/report.md

Usage
-----

::

    python -m scripts.run_simulation \\
        --images data/sim/images \\
        --ground-truth data/sim/ground_truth.csv \\
        --weights models/weights/best.pt \\
        --rated-capacity 8 \\
        --num-calls 1000 \\
        --output results/simulation/baseline

    # Hybrid mode (with a separately trained head detector):
    python -m scripts.run_simulation \\
        --images data/sim/images \\
        --ground-truth data/sim/ground_truth.csv \\
        --weights models/weights/best.pt \\
        --head-weights models/weights/best_head.pt \\
        --rated-capacity 8 \\
        --num-calls 1000 \\
        --output results/simulation/hybrid
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from src.energy.consumption import (
    EnergyParams,
    StartProfile,
    estimate_stop_energy,
)

# ──────────────────────────────────────────────────────────────────────
#  Constants — keep aligned with notebooks/02_train.ipynb and configs/
# ──────────────────────────────────────────────────────────────────────

CLASS_NAMES = ("person", "stroller", "luggage", "box")

CLASS_AREAS_M2: dict[str, float] = {
    # ISO 8100-32:2020 §6.4 specifies Ap in [0.17, 0.22] m² depending on car
    # load; EN 81-20:2020 §5.4.2.1.1 cites 0.17 m² for the rated-mass method.
    # 0.20 m² is the mid-range value used in elevator capacity calculations
    # (Tukia et al., 2018).
    "person": 0.20,
    # EN 1888-1:2018 governs single-pushchair safety. Product survey:
    # Bugaboo Butterfly ≈ 0.22 m², UPPAbaby Vista 91×65 cm ≈ 0.60 m²;
    # population mean ≈ 0.45 m².
    "stroller": 0.45,
    # IATA Resolution 753 cabin baggage standard: 56 × 36 × 23 cm →
    # footprint 0.20 m². Used as the canonical mid-size luggage value.
    "luggage": 0.20,
    # Industry e-commerce parcel mean ≈ 46 × 41 × 15 cm → footprint
    # ≈ 0.19 m² (Red Stag Fulfillment, 2026 benchmark).
    "box": 0.20,
}

# Average mass (kg) per detected object — used by the energy simulation
# to compute a per-call cabin load instead of a single fixed average.
CLASS_WEIGHTS_KG: dict[str, float] = {
    # EN 81-20:2020 / ISO 8100-1 nominate 75 kg as the rated mass per
    # passenger; Tukia et al. (2018) use the same value.
    "person": 75.0,
    # Empty single stroller mass ≈ 8–12 kg (UPPAbaby Vista 12 kg, Bugaboo
    # Butterfly 7 kg); typical occupant child ≈ 10–12 kg. Combined mid-range
    # ≈ 20 kg.
    "stroller": 20.0,
    # Cabin baggage IATA limit ≈ 8 kg; checked baggage typically 15–23 kg.
    # Mixed elevator distribution ≈ 15 kg.
    "luggage": 15.0,
    # Average e-commerce parcel ≈ 1–3 kg (Red Stag 2026 benchmark); larger
    # logistics cartons reach 10 kg. Conservative mean ≈ 5 kg.
    "box": 5.0,
}

DEFAULT_CABIN_M2 = 2.24  # 1.4 × 1.6 m — configs/default.yaml
DEFAULT_AREA_THRESHOLD = 0.90  # area_bypass_ratio — configs/default.yaml
DEFAULT_CONF_THRESHOLD = 0.25  # tuned for higher person recall


# ──────────────────────────────────────────────────────────────────────
#  Data model
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GroundTruth:
    """Per-image labelled cabin state."""

    filename: str
    gt_person: int
    gt_stroller: int
    gt_luggage: int
    gt_box: int
    gt_is_full: bool
    gt_occupancy_ratio: float


@dataclass
class ImageDecision:
    """Per-image prediction + confusion-matrix outcome."""

    filename: str
    # Ground truth.
    gt_person: int
    gt_stroller: int
    gt_luggage: int
    gt_box: int
    gt_is_full: bool
    gt_occupancy_ratio: float
    # Predictions.
    pred_person: int
    pred_stroller: int
    pred_luggage: int
    pred_box: int
    pred_occupancy_ratio: float
    pred_is_full: bool
    # Bypass-decision outcome (TP/TN/FP/FN, image-level).
    outcome: str
    # Counting-accuracy outcome (independent of the bypass decision).
    counts_exact_match: bool  # pred counts equal GT counts for ALL classes
    count_total_error: int  # Σ |pred_c − gt_c| across classes


# ──────────────────────────────────────────────────────────────────────
#  Ground-truth loader (multi-class)
# ──────────────────────────────────────────────────────────────────────


def _occupancy_ratio_from_counts(counts: dict[str, int], cabin_m2: float) -> float:
    occ = sum(counts[c] * CLASS_AREAS_M2[c] for c in CLASS_NAMES)
    return min(occ / cabin_m2, 1.0) if cabin_m2 > 0 else 0.0


def load_ground_truth(
    csv_path: Path, *, cabin_m2: float, area_threshold: float
) -> list[GroundTruth]:
    """Read the per-image ground-truth CSV.

    The schema is ``filename, gt_person, gt_stroller, gt_luggage,
    gt_box, gt_is_full``. ``gt_is_full`` is auto-derived when blank:
    the cabin is considered full when the multi-class occupancy ratio
    reaches ``area_threshold``.
    """
    rows: list[GroundTruth] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "gt_person" not in reader.fieldnames:
            raise ValueError(
                "Ground-truth CSV must have columns: filename, gt_person, "
                "gt_stroller, gt_luggage, gt_box, gt_is_full, notes."
            )
        for row in reader:
            counts = {
                "person": int(row.get("gt_person", 0) or 0),
                "stroller": int(row.get("gt_stroller", 0) or 0),
                "luggage": int(row.get("gt_luggage", 0) or 0),
                "box": int(row.get("gt_box", 0) or 0),
            }
            gt_occ = _occupancy_ratio_from_counts(counts, cabin_m2)
            raw_full = (row.get("gt_is_full", "") or "").strip().lower()
            if raw_full in ("true", "1", "yes", "y"):
                gt_full = True
            elif raw_full in ("false", "0", "no", "n"):
                gt_full = False
            else:
                gt_full = gt_occ >= area_threshold
            rows.append(
                GroundTruth(
                    filename=row["filename"].strip(),
                    gt_person=counts["person"],
                    gt_stroller=counts["stroller"],
                    gt_luggage=counts["luggage"],
                    gt_box=counts["box"],
                    gt_is_full=gt_full,
                    gt_occupancy_ratio=gt_occ,
                )
            )
    return rows


# ──────────────────────────────────────────────────────────────────────
#  Per-image inference + decision
# ──────────────────────────────────────────────────────────────────────


def classify_outcome(gt_full: bool, pred_full: bool) -> str:
    if gt_full and pred_full:
        return "TP"
    if not gt_full and not pred_full:
        return "TN"
    if not gt_full and pred_full:
        return "FP"
    return "FN"


def _save_annotated(img_bgr, out_path: Path) -> None:
    """Save an annotated image, handling non-ASCII paths via PIL."""
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_bgr[..., ::-1]).save(out_path)  # BGR → RGB


def predict_image(
    model,
    image_path: Path,
    *,
    cabin_m2: float,
    area_threshold: float,
    conf_threshold: float,
    head_model=None,
    head_conf_threshold: float = 0.25,
    save_pred_to: Path | None = None,
    save_head_pred_to: Path | None = None,
) -> tuple[dict[str, int], float, bool]:
    """Run the detector(s) and reduce the result to per-class counts +
    occupancy ratio + bypass decision.

    Hybrid mode (``head_model`` not None):
      * person count comes from the head detector,
      * stroller / luggage / box come from the four-class model
        (its ``person`` predictions are ignored to avoid double-counting).

    When ``save_pred_to`` is provided, the four-class detector's annotated
    frame is written there. ``save_head_pred_to`` does the same for the
    head detector in hybrid mode.
    """
    counts: dict[str, int] = dict.fromkeys(CLASS_NAMES, 0)
    use_hybrid = head_model is not None

    result = model.predict(str(image_path), conf=conf_threshold, verbose=False)[0]
    if save_pred_to is not None:
        _save_annotated(result.plot(), save_pred_to)
    if result.boxes is not None:
        for box in result.boxes:
            cls_name = result.names[int(box.cls.item())]
            if cls_name not in counts:
                continue
            if use_hybrid and cls_name == "person":
                continue  # person count comes from head model
            counts[cls_name] += 1

    if use_hybrid:
        head_result = head_model.predict(str(image_path), conf=head_conf_threshold, verbose=False)[
            0
        ]
        if save_head_pred_to is not None:
            _save_annotated(head_result.plot(), save_head_pred_to)
        if head_result.boxes is not None:
            counts["person"] = len(head_result.boxes)

    occupancy = _occupancy_ratio_from_counts(counts, cabin_m2)
    pred_full = occupancy >= area_threshold
    return counts, occupancy, pred_full


# ──────────────────────────────────────────────────────────────────────
#  Synthetic call simulation
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SimulationStats:
    num_calls: int = 0
    baseline_total_j: float = 0.0
    smart_total_j: float = 0.0
    smart_bypassed: int = 0
    smart_accepted: int = 0
    mean_load_kg: float = 0.0  # average per-call cabin load
    by_outcome: dict[str, int] = field(default_factory=dict)


def _cabin_load_kg(d: ImageDecision) -> float:
    """Per-call cabin load (kg) derived from the GT counts and the
    literature-anchored mass values in CLASS_WEIGHTS_KG."""
    return (
        d.gt_person * CLASS_WEIGHTS_KG["person"]
        + d.gt_stroller * CLASS_WEIGHTS_KG["stroller"]
        + d.gt_luggage * CLASS_WEIGHTS_KG["luggage"]
        + d.gt_box * CLASS_WEIGHTS_KG["box"]
    )


def simulate_calls(
    decisions: list[ImageDecision],
    *,
    num_calls: int,
    avg_floors_per_trip: float,
    energy_params: EnergyParams,
    seed: int = 42,
) -> SimulationStats:
    """Sample ``num_calls`` synthetic hall calls and tally energy.

    Each call draws a labeled cabin state uniformly at random from the
    benchmark. The energy cost of accepting that call is computed from
    the *actual* cabin load (people + strollers + luggage + boxes), not
    from a fixed average — so heavier cabins cost proportionally more
    motor energy per stop. Both policies use the same per-call energy
    when they accept; the only difference is which calls each policy
    bypasses.
    """
    rng = random.Random(seed)
    stats = SimulationStats(num_calls=num_calls)
    floors = max(1, round(avg_floors_per_trip))

    total_load_kg = 0.0
    for _ in range(num_calls):
        d = rng.choice(decisions)
        load_kg = _cabin_load_kg(d)
        total_load_kg += load_kg
        e_stop = estimate_stop_energy(
            StartProfile(load_kg=load_kg, floors_traveled=floors, direction_up=True),
            energy_params,
        )["total_j"]
        stats.baseline_total_j += e_stop
        if d.pred_is_full:
            stats.smart_bypassed += 1
        else:
            stats.smart_total_j += e_stop
            stats.smart_accepted += 1
        stats.by_outcome[d.outcome] = stats.by_outcome.get(d.outcome, 0) + 1

    stats.mean_load_kg = total_load_kg / max(1, num_calls)
    return stats


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────


def precision_recall(decisions: list[ImageDecision]) -> dict[str, float]:
    tp = sum(1 for d in decisions if d.outcome == "TP")
    fp = sum(1 for d in decisions if d.outcome == "FP")
    fn = sum(1 for d in decisions if d.outcome == "FN")
    tn = sum(1 for d in decisions if d.outcome == "TN")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / max(1, len(decisions))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def per_class_count_metrics(decisions: list[ImageDecision]) -> dict[str, dict[str, float]]:
    """Mean absolute error and total counts per class."""
    out: dict[str, dict[str, float]] = {}
    for cls in CLASS_NAMES:
        gts = np.array([getattr(d, f"gt_{cls}") for d in decisions], dtype=float)
        preds = np.array([getattr(d, f"pred_{cls}") for d in decisions], dtype=float)
        mae = float(np.mean(np.abs(gts - preds)))
        rmse = float(np.sqrt(np.mean((gts - preds) ** 2)))
        bias = float(np.mean(preds - gts))  # positive = over-detection
        out[cls] = {
            "gt_total": int(gts.sum()),
            "pred_total": int(preds.sum()),
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
        }
    return out


# ──────────────────────────────────────────────────────────────────────
#  Reporting
# ──────────────────────────────────────────────────────────────────────


def render_confusion_matrix_png(decisions: list[ImageDecision], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for d in decisions:
        counts[d.outcome] = counts.get(d.outcome, 0) + 1

    matrix = np.array(
        [
            [counts["TN"], counts["FP"]],
            [counts["FN"], counts["TP"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["pred: not full", "pred: full"])
    ax.set_yticks([0, 1], ["GT: not full", "GT: full"])
    for (i, j), v in np.ndenumerate(matrix):
        color = "white" if v > matrix.max() / 2 else "black"
        ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=18)
    ax.set_title("Per-image bypass-decision confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_per_image_csv(decisions: list[ImageDecision], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filename",
                "gt_person",
                "gt_stroller",
                "gt_luggage",
                "gt_box",
                "gt_is_full",
                "gt_occupancy_ratio",
                "pred_person",
                "pred_stroller",
                "pred_luggage",
                "pred_box",
                "pred_is_full",
                "pred_occupancy_ratio",
                "outcome",
                "counts_exact_match",
                "count_total_error",
            ]
        )
        for d in decisions:
            w.writerow(
                [
                    d.filename,
                    d.gt_person,
                    d.gt_stroller,
                    d.gt_luggage,
                    d.gt_box,
                    d.gt_is_full,
                    f"{d.gt_occupancy_ratio:.4f}",
                    d.pred_person,
                    d.pred_stroller,
                    d.pred_luggage,
                    d.pred_box,
                    d.pred_is_full,
                    f"{d.pred_occupancy_ratio:.4f}",
                    d.outcome,
                    d.counts_exact_match,
                    d.count_total_error,
                ]
            )


def write_per_class_csv(class_metrics: dict[str, dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "gt_total", "pred_total", "mae", "rmse", "bias"])
        for cls, m in class_metrics.items():
            w.writerow(
                [
                    cls,
                    m["gt_total"],
                    m["pred_total"],
                    f"{m['mae']:.3f}",
                    f"{m['rmse']:.3f}",
                    f"{m['bias']:+.3f}",
                ]
            )


def write_energy_csv(stats: SimulationStats, out_path: Path) -> None:
    saved_j = stats.baseline_total_j - stats.smart_total_j
    saved_pct = 100.0 * saved_j / stats.baseline_total_j if stats.baseline_total_j else 0.0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["num_calls", stats.num_calls])
        w.writerow(["baseline_total_kj", f"{stats.baseline_total_j / 1000:.2f}"])
        w.writerow(["smart_total_kj", f"{stats.smart_total_j / 1000:.2f}"])
        w.writerow(["energy_saved_kj", f"{saved_j / 1000:.2f}"])
        w.writerow(["energy_saved_kwh", f"{saved_j / 3_600_000:.4f}"])
        w.writerow(["energy_saved_pct", f"{saved_pct:.2f}"])
        w.writerow(["smart_bypassed_calls", stats.smart_bypassed])
        w.writerow(["smart_accepted_calls", stats.smart_accepted])


def write_markdown_report(
    decisions: list[ImageDecision],
    stats: SimulationStats,
    metrics: dict[str, float],
    class_metrics: dict[str, dict],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    saved_kj = (stats.baseline_total_j - stats.smart_total_j) / 1000
    saved_pct = (
        100.0 * (stats.baseline_total_j - stats.smart_total_j) / stats.baseline_total_j
        if stats.baseline_total_j
        else 0.0
    )
    mode = "hybrid" if args.head_weights else "single-model"

    lines: list[str] = []
    lines.append("# Smart Elevator CV — Energy Simulation Report\n")

    lines.append("## Configuration\n")
    lines.append(f"- Mode: **{mode}**")
    lines.append(f"- Four-class weights: `{args.weights}`")
    if args.head_weights:
        lines.append(f"- Head weights: `{args.head_weights}`")
    lines.append(f"- Rated capacity: **{args.rated_capacity} persons**")
    lines.append(f"- Cabin floor area: **{args.cabin_m2:.2f} m²**")
    lines.append(f"- Confidence threshold: {args.conf_threshold:.2f}")
    lines.append(f"- Area bypass threshold: {args.area_threshold:.2f}")
    lines.append(f"- Synthetic hall calls: **{args.num_calls}**")
    lines.append(f"- Avg floors per trip: {args.avg_floors}")
    lines.append(f"- Mean per-call cabin load (dynamic): **{stats.mean_load_kg:.0f} kg**")
    lines.append("")
    lines.append("### Per-class object masses (literature-anchored)\n")
    lines.append("| Class | Mass (kg) | Source |")
    lines.append("|---|:---:|---|")
    lines.append("| person   | 75 | EN 81-20:2020 / ISO 8100-1 rated mass per passenger |")
    lines.append(
        "| stroller | 20 | Empty single stroller 7-12 kg + child 10-12 kg "
        "(EN 1888-1:2018 + product survey) |"
    )
    lines.append(
        "| luggage  | 15 | Cabin baggage IATA limit ~8 kg, "
        "checked baggage 15-23 kg; mixed mean ~15 kg |"
    )
    lines.append(
        "| box      |  5 | E-commerce parcel mean 1-3 kg; "
        "logistics carton up to 10 kg (Red Stag 2026) |"
    )
    lines.append("")

    lines.append("## Accuracy 1 — Bypass decision (image-level)\n")
    lines.append("How often the system makes the correct accept / bypass call.\n")
    lines.append("|  | Predicted: not full | Predicted: full |")
    lines.append("|---|:---:|:---:|")
    lines.append(f"| **GT: not full** | {metrics['tn']} (TN) | {metrics['fp']} (FP) |")
    lines.append(f"| **GT: full**     | {metrics['fn']} (FN) | {metrics['tp']} (TP) |")
    lines.append("")
    lines.append(f"- **Bypass accuracy**:  {metrics['accuracy']:.3f}")
    lines.append(f"- Bypass precision:    {metrics['precision']:.3f}")
    lines.append(f"- Bypass recall:       {metrics['recall']:.3f}")
    lines.append(f"- F1 score:            {metrics['f1']:.3f}")
    lines.append("")

    # Counting accuracy — independent of the bypass decision.
    n = len(decisions)
    n_exact = sum(1 for d in decisions if d.counts_exact_match)
    counting_acc = n_exact / max(1, n)
    total_count_err = sum(d.count_total_error for d in decisions)
    mean_total_err = total_count_err / max(1, n)
    lines.append("## Accuracy 2 — Counting (per-image, independent metric)\n")
    lines.append(
        "How often the per-class detection counts exactly match the ground "
        "truth. A bypass decision can be correct while counts are wrong, "
        "so this number is reported separately to verify that the model is "
        "right for the right reasons.\n"
    )
    lines.append(
        f"- **Counting accuracy** (exact-match): {counting_acc:.3f}  ({n_exact}/{n} images)"
    )
    lines.append(f"- Mean total count error per image:   {mean_total_err:.2f} objects")
    lines.append(f"- Total count error across all images: {total_count_err}")
    lines.append("")

    lines.append("## Per-class detection accuracy\n")
    lines.append("| Class | GT total | Pred total | MAE | RMSE | Bias |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for cls in CLASS_NAMES:
        m = class_metrics[cls]
        lines.append(
            f"| {cls} | {m['gt_total']} | {m['pred_total']} | "
            f"{m['mae']:.2f} | {m['rmse']:.2f} | {m['bias']:+.2f} |"
        )
    lines.append("")
    lines.append(
        "MAE / RMSE are computed over per-image counts. Bias is the mean "
        "(predicted - ground-truth) — positive values indicate over-detection."
    )
    lines.append("")

    lines.append("## Energy aggregates (synthetic day)\n")
    lines.append(f"- Baseline (always-accept): **{stats.baseline_total_j / 1000:.1f} kJ**")
    lines.append(f"- Smart (vision-gated):     **{stats.smart_total_j / 1000:.1f} kJ**")
    lines.append(
        f"- **Energy saved**: {saved_kj:.1f} kJ "
        f"({saved_kj / 3600:.3f} kWh) — **{saved_pct:.1f}%** of baseline"
    )
    lines.append(
        f"- Smart bypassed {stats.smart_bypassed} of {stats.num_calls} calls "
        f"({100 * stats.smart_bypassed / max(1, stats.num_calls):.1f}%)"
    )
    lines.append("")

    lines.append("## Per-image decisions\n")
    lines.append(
        "| filename | gt(p/s/l/b) | gt_full | gt_occ | pred(p/s/l/b) | pred_full | pred_occ | outcome |"
    )
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for d in decisions:
        gt = f"{d.gt_person}/{d.gt_stroller}/{d.gt_luggage}/{d.gt_box}"
        pr = f"{d.pred_person}/{d.pred_stroller}/{d.pred_luggage}/{d.pred_box}"
        lines.append(
            f"| {d.filename} | {gt} | {d.gt_is_full} | {d.gt_occupancy_ratio:.2f} | "
            f"{pr} | {d.pred_is_full} | {d.pred_occupancy_ratio:.2f} | {d.outcome} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument(
        "--head-weights",
        type=Path,
        default=None,
        help="Optional head-only YOLO checkpoint. When given, hybrid mode "
        "is enabled: the head detector supplies the person count and the "
        "four-class model contributes only stroller / luggage / box.",
    )
    p.add_argument("--head-conf", type=float, default=0.25)
    p.add_argument("--rated-capacity", type=int, default=8)
    p.add_argument("--cabin-m2", type=float, default=DEFAULT_CABIN_M2)
    p.add_argument("--area-threshold", type=float, default=DEFAULT_AREA_THRESHOLD)
    p.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    p.add_argument("--num-calls", type=int, default=1000)
    p.add_argument("--avg-floors", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/simulation"))
    p.add_argument(
        "--no-save-annotated",
        dest="save_annotated",
        action="store_false",
        help="Disable per-image annotated prediction snapshots "
        "(by default they are written to <output>/predictions/).",
    )
    p.set_defaults(save_annotated=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.weights.exists():
        print(f"[error] weights file not found: {args.weights}", file=sys.stderr)
        return 2

    gt_rows = load_ground_truth(
        args.ground_truth,
        cabin_m2=args.cabin_m2,
        area_threshold=args.area_threshold,
    )
    if not gt_rows:
        print("[error] ground-truth CSV has no rows.", file=sys.stderr)
        return 2

    print(f"[info] loading four-class model from {args.weights} …")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    head_model = None
    if args.head_weights is not None:
        if not args.head_weights.exists():
            print(
                f"[error] head-weights file not found: {args.head_weights}",
                file=sys.stderr,
            )
            return 2
        print(f"[info] hybrid mode — loading head model from {args.head_weights} …")
        head_model = YOLO(str(args.head_weights))

    decisions: list[ImageDecision] = []
    mode_label = "hybrid" if head_model is not None else "single-model"
    print(f"[info] scoring {len(gt_rows)} images (mode={mode_label}, conf={args.conf_threshold}) …")
    pred_dir = args.output / "predictions" if args.save_annotated else None
    head_pred_dir = (
        args.output / "predictions_head" if args.save_annotated and head_model is not None else None
    )
    for gt in gt_rows:
        img_path = args.images / gt.filename
        if not img_path.exists():
            print(f"  [warn] missing image: {img_path}")
            continue
        save_pred_to = (pred_dir / gt.filename) if pred_dir is not None else None
        save_head_pred_to = (head_pred_dir / gt.filename) if head_pred_dir is not None else None
        pred_counts, pred_occ, pred_full = predict_image(
            model,
            img_path,
            cabin_m2=args.cabin_m2,
            area_threshold=args.area_threshold,
            conf_threshold=args.conf_threshold,
            head_model=head_model,
            head_conf_threshold=args.head_conf,
            save_pred_to=save_pred_to,
            save_head_pred_to=save_head_pred_to,
        )
        outcome = classify_outcome(gt.gt_is_full, pred_full)
        gt_counts = {
            "person": gt.gt_person,
            "stroller": gt.gt_stroller,
            "luggage": gt.gt_luggage,
            "box": gt.gt_box,
        }
        count_total_error = sum(abs(pred_counts[c] - gt_counts[c]) for c in CLASS_NAMES)
        counts_exact_match = count_total_error == 0
        decisions.append(
            ImageDecision(
                filename=gt.filename,
                gt_person=gt.gt_person,
                gt_stroller=gt.gt_stroller,
                gt_luggage=gt.gt_luggage,
                gt_box=gt.gt_box,
                gt_is_full=gt.gt_is_full,
                gt_occupancy_ratio=gt.gt_occupancy_ratio,
                pred_person=pred_counts["person"],
                pred_stroller=pred_counts["stroller"],
                pred_luggage=pred_counts["luggage"],
                pred_box=pred_counts["box"],
                pred_occupancy_ratio=pred_occ,
                pred_is_full=pred_full,
                outcome=outcome,
                counts_exact_match=counts_exact_match,
                count_total_error=count_total_error,
            )
        )
        print(
            f"  {gt.filename:<48} "
            f"GT(p/s/l/b)={gt.gt_person}/{gt.gt_stroller}/"
            f"{gt.gt_luggage}/{gt.gt_box} occ={gt.gt_occupancy_ratio:.2f} "
            f"{'F' if gt.gt_is_full else 'N':<2}  "
            f"PRED(p/s/l/b)={pred_counts['person']}/{pred_counts['stroller']}/"
            f"{pred_counts['luggage']}/{pred_counts['box']} occ={pred_occ:.2f} "
            f"{'F' if pred_full else 'N'}  → {outcome}"
        )

    if not decisions:
        print("[error] no usable image-GT pairs.", file=sys.stderr)
        return 2

    metrics = precision_recall(decisions)
    class_metrics = per_class_count_metrics(decisions)
    energy_params = EnergyParams()
    stats = simulate_calls(
        decisions,
        num_calls=args.num_calls,
        avg_floors_per_trip=args.avg_floors,
        energy_params=energy_params,
        seed=args.seed,
    )

    cm_path = args.output / "confusion_matrix.png"
    csv_per_img = args.output / "per_image_decisions.csv"
    csv_per_cls = args.output / "per_class_detection.csv"
    csv_energy = args.output / "energy_savings.csv"
    md_report = args.output / "report.md"

    render_confusion_matrix_png(decisions, cm_path)
    write_per_image_csv(decisions, csv_per_img)
    write_per_class_csv(class_metrics, csv_per_cls)
    write_energy_csv(stats, csv_energy)
    write_markdown_report(decisions, stats, metrics, class_metrics, args, md_report)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mode:             {mode_label}")
    print(f"Images scored:    {len(decisions)}")
    print(
        f"Bypass accuracy:  {metrics['accuracy']:.3f}  "
        f"(TP={metrics['tp']} TN={metrics['tn']} "
        f"FP={metrics['fp']} FN={metrics['fn']}; "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f})"
    )
    n_exact = sum(1 for d in decisions if d.counts_exact_match)
    total_count_err = sum(d.count_total_error for d in decisions)
    print(
        f"Counting accuracy: {n_exact / max(1, len(decisions)):.3f}  "
        f"({n_exact}/{len(decisions)} images with exact per-class match; "
        f"total count error = {total_count_err} objects)"
    )
    print("Per-class MAE (count error):")
    for cls in CLASS_NAMES:
        m = class_metrics[cls]
        print(
            f"  {cls:<9} GT={m['gt_total']:>3}  Pred={m['pred_total']:>3}  "
            f"MAE={m['mae']:.2f}  bias={m['bias']:+.2f}"
        )
    saved_kj = (stats.baseline_total_j - stats.smart_total_j) / 1000
    saved_pct = (
        100.0 * (stats.baseline_total_j - stats.smart_total_j) / stats.baseline_total_j
        if stats.baseline_total_j
        else 0.0
    )
    print(
        f"Energy saved:     {saved_kj:.1f} kJ ({saved_kj / 3600:.3f} kWh) "
        f"= {saved_pct:.1f}% of baseline over {stats.num_calls} calls"
    )
    print(f"Reports written to: {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
