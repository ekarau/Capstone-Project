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
    "person": 0.20,  # TS EN 81-20:2020 §5.4.2.1.1
    "stroller": 0.45,  # ~90 × 50 cm single stroller
    "luggage": 0.18,  # IATA cabin / mid-size mix
    "box": 0.20,  # ~50 × 40 cm medium carton
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
    # Outcome.
    outcome: str  # 'TP' | 'TN' | 'FP' | 'FN'


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


def predict_image(
    model,
    image_path: Path,
    *,
    cabin_m2: float,
    area_threshold: float,
    conf_threshold: float,
    head_model=None,
    head_conf_threshold: float = 0.25,
) -> tuple[dict[str, int], float, bool]:
    """Run the detector(s) and reduce the result to per-class counts +
    occupancy ratio + bypass decision.

    Hybrid mode (``head_model`` not None):
      * person count comes from the head detector,
      * stroller / luggage / box come from the four-class model
        (its ``person`` predictions are ignored to avoid double-counting).
    """
    counts: dict[str, int] = dict.fromkeys(CLASS_NAMES, 0)
    use_hybrid = head_model is not None

    result = model.predict(str(image_path), conf=conf_threshold, verbose=False)[0]
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
    by_outcome: dict[str, int] = field(default_factory=dict)


def simulate_calls(
    decisions: list[ImageDecision],
    *,
    num_calls: int,
    avg_floors_per_trip: float,
    avg_passengers_per_call: float,
    energy_params: EnergyParams,
    seed: int = 42,
) -> SimulationStats:
    rng = random.Random(seed)
    stats = SimulationStats(num_calls=num_calls)

    avg_passenger_kg = avg_passengers_per_call * 75.0
    base_profile = StartProfile(
        load_kg=avg_passenger_kg,
        floors_traveled=max(1, round(avg_floors_per_trip)),
        direction_up=True,
    )
    energy_per_stop = estimate_stop_energy(base_profile, energy_params)["total_j"]

    for _ in range(num_calls):
        d = rng.choice(decisions)
        stats.baseline_total_j += energy_per_stop
        if d.pred_is_full:
            stats.smart_bypassed += 1
        else:
            stats.smart_total_j += energy_per_stop
            stats.smart_accepted += 1
        stats.by_outcome[d.outcome] = stats.by_outcome.get(d.outcome, 0) + 1
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
    lines.append(f"- Avg passengers per accepted call: {args.avg_passengers}")
    lines.append(f"- Avg floors per trip: {args.avg_floors}")
    lines.append("")

    lines.append("## Bypass-decision performance (image-level)\n")
    lines.append("|  | Predicted: not full | Predicted: full |")
    lines.append("|---|:---:|:---:|")
    lines.append(f"| **GT: not full** | {metrics['tn']} (TN) | {metrics['fp']} (FP) |")
    lines.append(f"| **GT: full**     | {metrics['fn']} (FN) | {metrics['tp']} (TP) |")
    lines.append("")
    lines.append(f"- Accuracy: **{metrics['accuracy']:.3f}**")
    lines.append(f"- Bypass precision: **{metrics['precision']:.3f}**")
    lines.append(f"- Bypass recall:    **{metrics['recall']:.3f}**")
    lines.append(f"- F1 score:         **{metrics['f1']:.3f}**")
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
    p.add_argument("--avg-passengers", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/simulation"))
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
    for gt in gt_rows:
        img_path = args.images / gt.filename
        if not img_path.exists():
            print(f"  [warn] missing image: {img_path}")
            continue
        pred_counts, pred_occ, pred_full = predict_image(
            model,
            img_path,
            cabin_m2=args.cabin_m2,
            area_threshold=args.area_threshold,
            conf_threshold=args.conf_threshold,
            head_model=head_model,
            head_conf_threshold=args.head_conf,
        )
        outcome = classify_outcome(gt.gt_is_full, pred_full)
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
        avg_passengers_per_call=args.avg_passengers,
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
        f"Bypass decision:  TP={metrics['tp']} TN={metrics['tn']} "
        f"FP={metrics['fp']} FN={metrics['fn']}  "
        f"acc={metrics['accuracy']:.3f}  P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}"
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
