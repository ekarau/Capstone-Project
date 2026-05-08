"""End-to-end energy-savings simulation for Smart Elevator CV.

Pipeline
--------

1. Load every image in `data/sim/images/` and its ground-truth row from
   `data/sim/ground_truth.csv` (`filename`, `gt_count`, `gt_is_full`).
2. Run the trained YOLOv8 detector and the class-based area estimator
   (TS EN 81-20 / ISO 8100 footprints).
3. Emit a per-image accept/bypass decision via the two-stage policy
   (PDF Algorithm 1, Andrei & Ruokokoski 2022).
4. Compare against ground truth and tabulate confusion-matrix counts.
5. Sample N synthetic hall calls with per-call cabin states drawn
   uniformly from the labeled images. Aggregate energy spent by:
   * **Baseline**: every call accepted.
   * **Smart**:    accept iff classifier says not full.
   Energy per stop is taken from `src.energy.consumption` (Tukia 2018).
6. Write `confusion_matrix.png`, `per_image_decisions.csv`,
   `energy_savings.csv` and a Markdown summary report.

Usage
-----

::

    python -m scripts.run_simulation \\
        --images data/sim/images \\
        --ground-truth data/sim/ground_truth.csv \\
        --weights models/weights/best.pt \\
        --rated-capacity 8 \\
        --num-calls 1000 \\
        --output results/simulation
"""

from __future__ import annotations

import argparse
import csv
import math
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
#  Per-image decision
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ImageDecision:
    filename: str
    gt_count: int
    gt_is_full: bool
    pred_count_total: int
    pred_count_person: int
    pred_occupancy_ratio: float
    pred_is_full: bool
    outcome: str  # 'TP' | 'TN' | 'FP' | 'FN'


def classify_outcome(gt_full: bool, pred_full: bool) -> str:
    """Confusion-matrix label for one image."""
    if gt_full and pred_full:
        return "TP"  # correct bypass — saves energy, no SLA hit
    if not gt_full and not pred_full:
        return "TN"  # correct accept — normal service
    if not gt_full and pred_full:
        return "FP"  # incorrect bypass — passenger waits, SLA violation
    return "FN"  # missed bypass — wasted stop, energy lost


def predict_image(
    model,
    image_path: Path,
    *,
    cabin_m2: float,
    area_threshold: float,
    conf_threshold: float,
    class_areas: dict[str, float],
) -> tuple[int, int, float, bool]:
    """Run the detector once and reduce the result to occupancy + decision."""
    result = model.predict(str(image_path), conf=conf_threshold, verbose=False)[0]

    occupied = 0.0
    total_count = 0
    person_count = 0
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            a = class_areas.get(cls_name, 0.0)
            if a <= 0:
                continue
            occupied += a
            total_count += 1
            if cls_name == "person":
                person_count += 1

    occupancy = min(occupied / cabin_m2, 1.0) if cabin_m2 > 0 else 0.0
    pred_full = occupancy >= area_threshold
    return total_count, person_count, occupancy, pred_full


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
    correct_bypasses: int = 0
    incorrect_bypasses: int = 0
    missed_bypass_opportunities: int = 0
    by_outcome: dict[str, int] = field(default_factory=dict)


def simulate_calls(
    decisions: list[ImageDecision],
    *,
    num_calls: int,
    avg_floors_per_trip: float,
    avg_passengers_per_call: float,
    rated_capacity: int,
    energy_params: EnergyParams,
    seed: int = 42,
) -> SimulationStats:
    """Sample `num_calls` hall calls and tally energy under the two policies.

    Each synthetic call draws an image (and therefore a cabin state) uniformly
    at random from the labeled set. The hall call costs one stop's worth of
    energy when accepted; bypassing the call saves that stop entirely.
    """
    rng = random.Random(seed)
    stats = SimulationStats(num_calls=num_calls)

    # Average passenger payload per accepted stop (kg) — used to size the
    # running-energy term realistically.
    avg_passenger_kg = energy_params.empty_car_mass_kg * 0  # avoid lint warning
    avg_passenger_kg = avg_passengers_per_call * 75.0

    base_profile = StartProfile(
        load_kg=avg_passenger_kg,
        floors_traveled=max(1, round(avg_floors_per_trip)),
        direction_up=True,
    )
    energy_per_stop = estimate_stop_energy(base_profile, energy_params)["total_j"]

    for _ in range(num_calls):
        d = rng.choice(decisions)
        # Baseline: every call leads to a stop.
        stats.baseline_total_j += energy_per_stop
        # Smart: stop iff predicted not full.
        if d.pred_is_full:
            stats.smart_bypassed += 1
        else:
            stats.smart_total_j += energy_per_stop
            stats.smart_accepted += 1
        stats.by_outcome[d.outcome] = stats.by_outcome.get(d.outcome, 0) + 1
        if d.outcome == "TP":
            stats.correct_bypasses += 1
        elif d.outcome == "FP":
            stats.incorrect_bypasses += 1
        elif d.outcome == "FN":
            stats.missed_bypass_opportunities += 1

    return stats


# ──────────────────────────────────────────────────────────────────────
#  Reporting helpers
# ──────────────────────────────────────────────────────────────────────


def render_confusion_matrix_png(decisions: list[ImageDecision], out_path: Path) -> None:
    """Save a 2 × 2 confusion-matrix figure (no third-party styling)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for d in decisions:
        counts[d.outcome] = counts.get(d.outcome, 0) + 1

    matrix = np.array(
        [
            [counts["TN"], counts["FP"]],  # GT not full
            [counts["FN"], counts["TP"]],  # GT full
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["pred: not full", "pred: full"])
    ax.set_yticks([0, 1], ["GT: not full", "GT: full"])
    for (i, j), v in np.ndenumerate(matrix):
        color = "white" if v > matrix.max() / 2 else "black"
        ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=18)
    ax.set_title("Per-image classifier confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def precision_recall(decisions: list[ImageDecision]) -> dict[str, float]:
    tp = sum(1 for d in decisions if d.outcome == "TP")
    fp = sum(1 for d in decisions if d.outcome == "FP")
    fn = sum(1 for d in decisions if d.outcome == "FN")
    tn = sum(1 for d in decisions if d.outcome == "TN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(decisions))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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


def write_per_image_csv(decisions: list[ImageDecision], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filename",
                "gt_count",
                "gt_is_full",
                "pred_count_total",
                "pred_count_person",
                "pred_occupancy_ratio",
                "pred_is_full",
                "outcome",
            ]
        )
        for d in decisions:
            w.writerow(
                [
                    d.filename,
                    d.gt_count,
                    d.gt_is_full,
                    d.pred_count_total,
                    d.pred_count_person,
                    f"{d.pred_occupancy_ratio:.4f}",
                    d.pred_is_full,
                    d.outcome,
                ]
            )


def write_energy_csv(stats: SimulationStats, out_path: Path) -> None:
    saved_j = stats.baseline_total_j - stats.smart_total_j
    saved_pct = 100.0 * saved_j / stats.baseline_total_j if stats.baseline_total_j > 0 else 0.0
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
        w.writerow(["correct_bypasses_TP", stats.correct_bypasses])
        w.writerow(["incorrect_bypasses_FP", stats.incorrect_bypasses])
        w.writerow(["missed_opportunities_FN", stats.missed_bypass_opportunities])


def write_markdown_report(
    decisions: list[ImageDecision],
    stats: SimulationStats,
    metrics: dict[str, float],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    saved_kj = (stats.baseline_total_j - stats.smart_total_j) / 1000
    saved_pct = (
        100.0 * (stats.baseline_total_j - stats.smart_total_j) / stats.baseline_total_j
        if stats.baseline_total_j > 0
        else 0.0
    )
    lines: list[str] = []
    lines.append("# Smart Elevator CV — Energy Simulation Report\n")
    lines.append("## Configuration\n")
    lines.append(f"- Weights: `{args.weights}`")
    lines.append(f"- Rated capacity: **{args.rated_capacity} persons**")
    lines.append(f"- Cabin floor area: **{args.cabin_m2:.2f} m²**")
    lines.append(f"- Confidence threshold: {args.conf_threshold:.2f}")
    lines.append(f"- Area bypass threshold: {args.area_threshold:.2f}")
    lines.append(f"- Synthetic hall calls: **{args.num_calls}**")
    lines.append(f"- Avg passengers per accepted call: {args.avg_passengers}")
    lines.append(f"- Avg floors per trip: {args.avg_floors}")
    lines.append("")
    lines.append("## Per-image classifier performance\n")
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
    lines.append("## Outcome breakdown by call\n")
    for k in ("TP", "TN", "FP", "FN"):
        lines.append(f"- {k}: {stats.by_outcome.get(k, 0)}")
    lines.append("")
    lines.append("## Per-image decisions\n")
    lines.append("| filename | gt_count | gt_full | pred_count | pred_occ | pred_full | outcome |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for d in decisions:
        lines.append(
            f"| {d.filename} | {d.gt_count} | {d.gt_is_full} | "
            f"{d.pred_count_total} | {d.pred_occupancy_ratio:.2f} | "
            f"{d.pred_is_full} | {d.outcome} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--rated-capacity", type=int, default=8)
    p.add_argument("--cabin-m2", type=float, default=DEFAULT_CABIN_M2)
    p.add_argument("--area-threshold", type=float, default=DEFAULT_AREA_THRESHOLD)
    p.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    p.add_argument("--num-calls", type=int, default=1000)
    p.add_argument(
        "--avg-floors",
        type=float,
        default=3.0,
        help="Avg floors traveled per accepted stop (Tukia model).",
    )
    p.add_argument(
        "--avg-passengers",
        type=float,
        default=4.0,
        help="Avg passengers boarding per accepted stop.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/simulation"))
    return p.parse_args()


def load_ground_truth(csv_path: Path, rated_capacity: int) -> list[dict]:
    rows: list[dict] = []
    full_threshold = math.ceil(0.85 * rated_capacity)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_count = int(row["gt_count"])
            raw = row.get("gt_is_full", "").strip().lower()
            if raw in ("true", "1", "yes", "y"):
                gt_full = True
            elif raw in ("false", "0", "no", "n"):
                gt_full = False
            else:
                # Auto-derive when the user leaves the column blank.
                gt_full = gt_count >= full_threshold
            rows.append(
                {
                    "filename": row["filename"].strip(),
                    "gt_count": gt_count,
                    "gt_is_full": gt_full,
                    "notes": row.get("notes", "").strip(),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.weights.exists():
        print(f"[error] weights file not found: {args.weights}", file=sys.stderr)
        return 2
    gt_rows = load_ground_truth(args.ground_truth, args.rated_capacity)
    if not gt_rows:
        print("[error] ground-truth CSV has no rows.", file=sys.stderr)
        return 2

    print(f"[info] loading model from {args.weights} …")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    decisions: list[ImageDecision] = []
    print(f"[info] scoring {len(gt_rows)} images at conf={args.conf_threshold} …")
    for row in gt_rows:
        img_path = args.images / row["filename"]
        if not img_path.exists():
            print(f"  [warn] missing image: {img_path}")
            continue
        total, person, occ, pred_full = predict_image(
            model,
            img_path,
            cabin_m2=args.cabin_m2,
            area_threshold=args.area_threshold,
            conf_threshold=args.conf_threshold,
            class_areas=CLASS_AREAS_M2,
        )
        outcome = classify_outcome(row["gt_is_full"], pred_full)
        decisions.append(
            ImageDecision(
                filename=row["filename"],
                gt_count=row["gt_count"],
                gt_is_full=row["gt_is_full"],
                pred_count_total=total,
                pred_count_person=person,
                pred_occupancy_ratio=occ,
                pred_is_full=pred_full,
                outcome=outcome,
            )
        )
        print(
            f"  {row['filename']:<22}  "
            f"GT={row['gt_count']:>2} {'F' if row['gt_is_full'] else 'N':<2}  "
            f"pred_n={person:>2}  occ={occ:5.2f}  "
            f"pred={'F' if pred_full else 'N'}  → {outcome}"
        )

    if not decisions:
        print("[error] no usable image-GT pairs.", file=sys.stderr)
        return 2

    metrics = precision_recall(decisions)
    energy_params = EnergyParams()  # repo defaults; configurable via configs/
    stats = simulate_calls(
        decisions,
        num_calls=args.num_calls,
        avg_floors_per_trip=args.avg_floors,
        avg_passengers_per_call=args.avg_passengers,
        rated_capacity=args.rated_capacity,
        energy_params=energy_params,
        seed=args.seed,
    )

    cm_path = args.output / "confusion_matrix.png"
    csv_per_img = args.output / "per_image_decisions.csv"
    csv_energy = args.output / "energy_savings.csv"
    md_report = args.output / "report.md"

    render_confusion_matrix_png(decisions, cm_path)
    write_per_image_csv(decisions, csv_per_img)
    write_energy_csv(stats, csv_energy)
    write_markdown_report(decisions, stats, metrics, args, md_report)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Images scored:    {len(decisions)}")
    print(
        f"Confusion matrix: TP={metrics['tp']}  TN={metrics['tn']}  "
        f"FP={metrics['fp']}  FN={metrics['fn']}"
    )
    print(
        f"Accuracy={metrics['accuracy']:.3f}  "
        f"Precision={metrics['precision']:.3f}  "
        f"Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}"
    )
    saved_kj = (stats.baseline_total_j - stats.smart_total_j) / 1000
    saved_pct = (
        100.0 * (stats.baseline_total_j - stats.smart_total_j) / stats.baseline_total_j
        if stats.baseline_total_j > 0
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
