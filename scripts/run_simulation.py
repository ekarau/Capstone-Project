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

3. Emit a per-image ACCEPT / BYPASS decision via the two-stage
   load-and-area policy.

4. Compare against ground truth and tabulate confusion-matrix counts plus
   per-class detection accuracy (mean absolute error of class counts).

5. Sample ``--num-calls`` synthetic hall calls. Per-call cabin states are
   drawn uniformly from the labeled images. Aggregate energy spent under:

   * **Baseline** — every call accepted (always-stop).
   * **Smart**    — accept iff classifier says not full.

   Energy per stop comes from ``src.energy.consumption``.

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
    "person": 0.20,
    "stroller": 0.45,
    "luggage": 0.20,
    "box": 0.20,
}

# Average mass (kg) per detected object, used by the energy simulation
# to compute a per-call cabin load instead of a single fixed average.
CLASS_WEIGHTS_KG: dict[str, float] = {
    "person": 75.0,
    "stroller": 20.0,
    "luggage": 15.0,
    "box": 5.0,
}

DEFAULT_CABIN_M2 = 2.24  # 1.4 × 1.6 m — configs/default.yaml
DEFAULT_AREA_THRESHOLD = 0.90  # area_bypass_ratio — configs/default.yaml
DEFAULT_CONF_THRESHOLD = 0.25  # tuned for higher person recall
DEFAULT_WEIGHT_RATIO = 0.80  # weight_bypass_ratio — configs/default.yaml
DEFAULT_RATED_LOAD_KG = 630.0  # max_weight_kg — configs/default.yaml


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
    gt_is_full: bool  # area-occupancy ≥ area_threshold
    gt_occupancy_ratio: float
    gt_weight_kg: float  # Σ n_c × CLASS_WEIGHTS_KG (derived from counts)
    gt_weight_full: bool  # gt_weight_kg ≥ weight_threshold_kg
    gt_should_bypass: bool  # gt_is_full OR gt_weight_full (optimal-policy ground truth)


@dataclass
class ImageDecision:
    """Per-image prediction + confusion-matrix outcome."""

    filename: str
    # Ground truth.
    gt_person: int
    gt_stroller: int
    gt_luggage: int
    gt_box: int
    gt_is_full: bool  # area-occupancy ≥ area_threshold
    gt_occupancy_ratio: float
    gt_weight_kg: float
    gt_weight_full: bool  # gt_weight_kg ≥ weight_threshold_kg
    gt_should_bypass: bool  # gt_is_full OR gt_weight_full
    # Predictions.
    pred_person: int
    pred_stroller: int
    pred_luggage: int
    pred_box: int
    pred_occupancy_ratio: float
    pred_is_full: bool  # area-bypass call from the detector
    # Bypass-decision outcome (TP/TN/FP/FN), evaluated against gt_should_bypass.
    # Records the SMART decision (weight gate first, then area gate).
    smart_bypass: bool  # gt_weight_full OR pred_is_full
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


def _cabin_load_kg_from_counts(counts: dict[str, int]) -> float:
    """Per-image cabin load (kg) from per-class GT counts."""
    return sum(counts[c] * CLASS_WEIGHTS_KG[c] for c in CLASS_NAMES)


def load_ground_truth(
    csv_path: Path,
    *,
    cabin_m2: float,
    area_threshold: float,
    weight_threshold_kg: float,
) -> list[GroundTruth]:
    """Read the per-image ground-truth CSV.

    The schema is ``filename, gt_person, gt_stroller, gt_luggage,
    gt_box, gt_is_full``. ``gt_is_full`` is auto-derived when blank:
    the cabin is considered area-full when the multi-class occupancy
    ratio reaches ``area_threshold``. The weight-bypass ground truth
    ``gt_weight_full`` is always derived from the per-class counts and
    the ``weight_threshold_kg`` value.
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
            gt_weight = _cabin_load_kg_from_counts(counts)
            raw_full = (row.get("gt_is_full", "") or "").strip().lower()
            if raw_full in ("true", "1", "yes", "y"):
                gt_full = True
            elif raw_full in ("false", "0", "no", "n"):
                gt_full = False
            else:
                gt_full = gt_occ >= area_threshold
            gt_weight_full = gt_weight >= weight_threshold_kg
            rows.append(
                GroundTruth(
                    filename=row["filename"].strip(),
                    gt_person=counts["person"],
                    gt_stroller=counts["stroller"],
                    gt_luggage=counts["luggage"],
                    gt_box=counts["box"],
                    gt_is_full=gt_full,
                    gt_occupancy_ratio=gt_occ,
                    gt_weight_kg=gt_weight,
                    gt_weight_full=gt_weight_full,
                    gt_should_bypass=gt_full or gt_weight_full,
                )
            )
    return rows


# ──────────────────────────────────────────────────────────────────────
#  Per-image inference + decision
# ──────────────────────────────────────────────────────────────────────


def classify_outcome(gt_should_bypass: bool, smart_bypass: bool) -> str:
    """Classify the smart bypass call against the optimal-policy ground truth.

    Optimal policy bypasses iff the cabin cannot accept another passenger,
    i.e. iff it is **either** area-full **or** weight-full. Smart bypass
    is the actual policy decision (weight gate first, then detector area).
    """
    if gt_should_bypass and smart_bypass:
        return "TP"
    if not gt_should_bypass and not smart_bypass:
        return "TN"
    if not gt_should_bypass and smart_bypass:
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
class CallRecord:
    """One synthetic hall call: who called from where, which cabin state was
    sampled, what each of the three policies decided, and the per-call
    energy breakdown.

    Three policies are compared:

    * ``always_accept`` — naive baseline, never bypasses.
    * ``weight_only``  — current-industry baseline; bypass iff
      ``gt_weight_kg ≥ weight_threshold_kg`` (Stage 1 of Algorithm 1).
    * ``smart``        — proposed system; bypass iff weight stage triggers
      OR detector reports area-full (Stages 1 + 2).

    ``trip_energy_kj`` is the *contextual* full-trip energy (running +
    auxiliary + door + idle) from Tukia (2018) for the cabin state at
    that distance and direction. It can be negative for light cabins
    that move upwards and reclaim energy via regenerative braking.

    ``stop_overhead_kj`` is the strictly-attributable energy cost of
    the call: the door cycle and idle-stop time the elevator would
    incur by stopping at that floor. This is what a correct bypass
    saves (≈ 920 J under our default parameters; load- and direction-
    independent).
    """

    call_id: int
    origin_floor: int
    dest_floor: int
    distance_floors: int
    direction: str  # "up" | "down"
    filename: str
    gt_person: int
    gt_stroller: int
    gt_luggage: int
    gt_box: int
    gt_weight_kg: float
    gt_is_full: bool
    gt_weight_full: bool
    gt_should_bypass: bool
    pred_is_full: bool
    weight_only_decision: str  # "accept" | "bypass"
    smart_decision: str  # "accept" | "bypass"
    outcome: str  # "TP" | "TN" | "FP" | "FN" — smart vs gt_should_bypass
    trip_energy_kj: float  # full Tukia trip energy (context)
    stop_overhead_kj: float  # door + idle at this floor (the saving target)
    cumulative_always_accept_kj: float
    cumulative_weight_only_kj: float
    cumulative_smart_kj: float


@dataclass
class SimulationStats:
    num_calls: int = 0
    # Energy attribution (overhead-only convention) for three policies.
    always_accept_total_j: float = 0.0  # naive baseline — never bypass
    weight_only_total_j: float = 0.0  # current-industry — Stage 1 only
    smart_total_j: float = 0.0  # proposed — Stage 1 + Stage 2
    # Smart-policy outcome attribution (against gt_should_bypass).
    smart_wasted_j: float = 0.0  # FN: stopped at a cabin that should bypass
    smart_saved_correctly_j: float = 0.0  # TP: correctly bypassed
    smart_lost_savings_j: float = 0.0  # energy that COULD have been saved on FN cases
    # Time attribution (per-stop time overhead; Barney 2003 / ISO 25745-2).
    always_accept_total_time_s: float = 0.0
    weight_only_total_time_s: float = 0.0
    smart_total_time_s: float = 0.0
    smart_wasted_time_s: float = 0.0
    # Decision counters per policy.
    weight_only_bypassed: int = 0
    weight_only_accepted: int = 0
    smart_bypassed: int = 0
    smart_accepted: int = 0
    # Smart vs ground-truth confusion (gt_should_bypass).
    smart_correct_bypass: int = 0  # TP
    smart_correct_accept: int = 0  # TN
    smart_wrong_bypass: int = 0  # FP — bypassed cabin that had room
    smart_wrong_accept: int = 0  # FN — accepted cabin that should have bypassed
    # Trip statistics.
    mean_load_kg: float = 0.0
    mean_distance_floors: float = 0.0
    by_outcome: dict[str, int] = field(default_factory=dict)
    # Per-call log (one row per synthetic hall call).
    call_log: list[CallRecord] = field(default_factory=list)


def simulate_calls(
    decisions: list[ImageDecision],
    *,
    num_calls: int,
    floors_count: int,
    energy_params: EnergyParams,
    seed: int = 42,
) -> SimulationStats:
    """Sample ``num_calls`` synthetic hall calls and tally per-policy energy
    and service-quality statistics for the three policies under study.

    Three policies share the same call stream:

    * **always_accept** — naive baseline: every call pays the per-stop
      overhead. This isolates "what does an unguarded elevator cost".
    * **weight_only**   — current-industry baseline: bypass iff the
      cabin's load (derived from GT counts × CLASS_WEIGHTS_KG) exceeds
      the weight threshold. This is Stage 1 of Algorithm 1, on its own.
    * **smart**         — proposed system: weight gate first, then the
      detector's area-occupancy gate. Stages 1 + 2 of Algorithm 1.

    Energy accounting follows the *overhead-only* convention: each call
    represents a request to **stop at one intermediate floor**. The
    strictly attributable energy of the call is the door cycle plus
    idle-stop time at that floor (≈ 920 J under the default Tukia 2018
    parameters, load- and direction-independent). A correct bypass at
    a floor saves exactly this overhead; the trip's running energy is
    shared with other accepted calls and is therefore not credited to
    the bypass decision.

    The full Tukia trip energy is still recorded in
    ``CallRecord.trip_energy_kj`` for context.

    The smart-policy decision is evaluated against the **optimal
    ground-truth policy** (``gt_should_bypass = gt_is_full OR
    gt_weight_full``), giving the TP / TN / FP / FN counts.
    """
    rng = random.Random(seed)
    stats = SimulationStats(num_calls=num_calls)

    # Stop-overhead energy is constant: door cycle (open + close) plus the
    # idle wait while the cabin sits at the floor. Load and direction do
    # not enter — they affect only the running energy, which is shared
    # between policies for any given trip.
    e_overhead_per_stop = (
        energy_params.power_doors_w * energy_params.door_open_close_time_s * 2
        + (energy_params.power_idle_w + energy_params.power_control_w) * energy_params.stop_time_s
    )
    # Stop-overhead time is also constant: door open + door close + idle
    # transfer (passenger boarding). Defaults give 10 s, matching the
    # 10–15 s per intermediate stop reported in Barney (2003), Strakosch
    # & Caporale (2010), and adopted by ISO 25745-2.
    t_overhead_per_stop = energy_params.door_open_close_time_s * 2 + energy_params.stop_time_s

    total_load_kg = 0.0
    total_distance = 0.0
    for call_id in range(1, num_calls + 1):
        d = rng.choice(decisions)
        load_kg = d.gt_weight_kg
        total_load_kg += load_kg

        # Sample the call's origin and destination floors.
        origin = rng.randint(1, floors_count)
        dest = rng.randint(1, floors_count)
        while dest == origin:
            dest = rng.randint(1, floors_count)
        distance_floors = abs(dest - origin)
        direction_up = dest > origin
        total_distance += distance_floors

        # Full-trip energy (Tukia 2018) — contextual only; not credited
        # to the bypass decision.
        e_trip_total = estimate_stop_energy(
            StartProfile(
                load_kg=load_kg,
                floors_traveled=distance_floors,
                direction_up=direction_up,
            ),
            energy_params,
        )["total_j"]

        # ── Policy 1 — Always-accept (naive baseline). ──────────────
        stats.always_accept_total_j += e_overhead_per_stop
        stats.always_accept_total_time_s += t_overhead_per_stop

        # ── Policy 2 — Weight-only (current-industry baseline). ─────
        if d.gt_weight_full:
            stats.weight_only_bypassed += 1
            weight_only_decision = "bypass"
        else:
            stats.weight_only_total_j += e_overhead_per_stop
            stats.weight_only_total_time_s += t_overhead_per_stop
            stats.weight_only_accepted += 1
            weight_only_decision = "accept"

        # ── Policy 3 — Smart (Algorithm 1: weight gate, then area). ─
        smart_bypass = d.smart_bypass  # gt_weight_full OR pred_is_full
        if smart_bypass:
            stats.smart_bypassed += 1
            smart_decision = "bypass"
            if d.gt_should_bypass:
                stats.smart_correct_bypass += 1
                stats.smart_saved_correctly_j += e_overhead_per_stop
            else:
                stats.smart_wrong_bypass += 1
        else:
            stats.smart_total_j += e_overhead_per_stop
            stats.smart_total_time_s += t_overhead_per_stop
            stats.smart_accepted += 1
            smart_decision = "accept"
            if d.gt_should_bypass:
                stats.smart_wrong_accept += 1
                stats.smart_wasted_j += e_overhead_per_stop
                stats.smart_lost_savings_j += e_overhead_per_stop
                stats.smart_wasted_time_s += t_overhead_per_stop
            else:
                stats.smart_correct_accept += 1

        stats.by_outcome[d.outcome] = stats.by_outcome.get(d.outcome, 0) + 1

        stats.call_log.append(
            CallRecord(
                call_id=call_id,
                origin_floor=origin,
                dest_floor=dest,
                distance_floors=distance_floors,
                direction="up" if direction_up else "down",
                filename=d.filename,
                gt_person=d.gt_person,
                gt_stroller=d.gt_stroller,
                gt_luggage=d.gt_luggage,
                gt_box=d.gt_box,
                gt_weight_kg=d.gt_weight_kg,
                gt_is_full=d.gt_is_full,
                gt_weight_full=d.gt_weight_full,
                gt_should_bypass=d.gt_should_bypass,
                pred_is_full=d.pred_is_full,
                weight_only_decision=weight_only_decision,
                smart_decision=smart_decision,
                outcome=d.outcome,
                trip_energy_kj=e_trip_total / 1000.0,
                stop_overhead_kj=e_overhead_per_stop / 1000.0,
                cumulative_always_accept_kj=stats.always_accept_total_j / 1000.0,
                cumulative_weight_only_kj=stats.weight_only_total_j / 1000.0,
                cumulative_smart_kj=stats.smart_total_j / 1000.0,
            )
        )

    stats.mean_load_kg = total_load_kg / max(1, num_calls)
    stats.mean_distance_floors = total_distance / max(1, num_calls)
    return stats


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────


def precision_recall(decisions: list[ImageDecision]) -> dict[str, float]:
    """Smart-policy bypass metrics, evaluated against ``gt_should_bypass``.

    Outcomes are stored on ``ImageDecision.outcome`` and were classified
    by :func:`classify_outcome` from ``gt_should_bypass`` and the
    smart-policy decision (``gt_weight_full OR pred_is_full``).
    """
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


def per_image_csv_columns() -> list[str]:
    return [
        "filename",
        "gt_person",
        "gt_stroller",
        "gt_luggage",
        "gt_box",
        "gt_weight_kg",
        "gt_weight_full",
        "gt_is_full",
        "gt_should_bypass",
        "gt_occupancy_ratio",
        "pred_person",
        "pred_stroller",
        "pred_luggage",
        "pred_box",
        "pred_is_full",
        "pred_occupancy_ratio",
        "smart_bypass",
        "outcome",
        "counts_exact_match",
        "count_total_error",
    ]


def per_class_count_metrics(decisions: list[ImageDecision]) -> dict[str, dict[str, float]]:
    """Per-class counting statistics.

    Reports two complementary views of counting quality:

    * Image-level error (MAE / RMSE / bias) — how far per-image predicted
      counts are from ground truth, on average. Sensitive to direction
      (over- vs under-detection) via ``bias``.
    * Object-level retrieval (TP / FN / FP, recall, precision, F1) —
      treats each detected instance as a retrieval target. For every
      (image, class) pair, ``TP = min(pred, gt)`` correctly counted
      instances, ``FN = max(0, gt - pred)`` missed instances, and
      ``FP = max(0, pred - gt)`` over-counted instances. The aggregated
      recall is the proportion of ground-truth objects that were
      correctly accounted for; precision is the proportion of predicted
      objects that were warranted.
    """
    out: dict[str, dict[str, float]] = {}
    for cls in CLASS_NAMES:
        gts = np.array([getattr(d, f"gt_{cls}") for d in decisions], dtype=float)
        preds = np.array([getattr(d, f"pred_{cls}") for d in decisions], dtype=float)
        mae = float(np.mean(np.abs(gts - preds)))
        rmse = float(np.sqrt(np.mean((gts - preds) ** 2)))
        bias = float(np.mean(preds - gts))  # positive = over-detection
        # Object-level retrieval metrics.
        tp = int(np.minimum(preds, gts).sum())
        fn = int(np.maximum(0, gts - preds).sum())
        fp = int(np.maximum(0, preds - gts).sum())
        gt_total = int(gts.sum())
        pred_total = int(preds.sum())
        recall = tp / gt_total if gt_total else 0.0
        precision = tp / pred_total if pred_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[cls] = {
            "gt_total": gt_total,
            "pred_total": pred_total,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall": recall,
            "precision": precision,
            "f1": f1,
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
        w.writerow(per_image_csv_columns())
        for d in decisions:
            w.writerow(
                [
                    d.filename,
                    d.gt_person,
                    d.gt_stroller,
                    d.gt_luggage,
                    d.gt_box,
                    f"{d.gt_weight_kg:.1f}",
                    d.gt_weight_full,
                    d.gt_is_full,
                    d.gt_should_bypass,
                    f"{d.gt_occupancy_ratio:.4f}",
                    d.pred_person,
                    d.pred_stroller,
                    d.pred_luggage,
                    d.pred_box,
                    d.pred_is_full,
                    f"{d.pred_occupancy_ratio:.4f}",
                    d.smart_bypass,
                    d.outcome,
                    d.counts_exact_match,
                    d.count_total_error,
                ]
            )


def _aggregate_object_metrics(class_metrics: dict[str, dict]) -> dict[str, float]:
    """Aggregate per-class TP/FN/FP across classes into overall recall,
    precision and F1 at the object level."""
    tp = sum(m["tp"] for m in class_metrics.values())
    fn = sum(m["fn"] for m in class_metrics.values())
    fp = sum(m["fp"] for m in class_metrics.values())
    gt_total = sum(m["gt_total"] for m in class_metrics.values())
    pred_total = sum(m["pred_total"] for m in class_metrics.values())
    recall = tp / gt_total if gt_total else 0.0
    precision = tp / pred_total if pred_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "gt_total": gt_total,
        "pred_total": pred_total,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def write_per_class_csv(class_metrics: dict[str, dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class",
                "gt_total",
                "pred_total",
                "tp",
                "fn",
                "fp",
                "recall",
                "precision",
                "f1",
                "mae",
                "rmse",
                "bias",
            ]
        )
        for cls, m in class_metrics.items():
            w.writerow(
                [
                    cls,
                    m["gt_total"],
                    m["pred_total"],
                    m["tp"],
                    m["fn"],
                    m["fp"],
                    f"{m['recall']:.3f}",
                    f"{m['precision']:.3f}",
                    f"{m['f1']:.3f}",
                    f"{m['mae']:.3f}",
                    f"{m['rmse']:.3f}",
                    f"{m['bias']:+.3f}",
                ]
            )
        # Overall aggregate row.
        agg = _aggregate_object_metrics(class_metrics)
        w.writerow(
            [
                "OVERALL",
                agg["gt_total"],
                agg["pred_total"],
                agg["tp"],
                agg["fn"],
                agg["fp"],
                f"{agg['recall']:.3f}",
                f"{agg['precision']:.3f}",
                f"{agg['f1']:.3f}",
                "",
                "",
                "",
            ]
        )


def write_call_log_csv(stats: SimulationStats, out_path: Path) -> None:
    """One row per synthetic hall call — for the timeline view in the demo."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "call_id",
                "origin_floor",
                "dest_floor",
                "distance_floors",
                "direction",
                "filename",
                "gt_person",
                "gt_stroller",
                "gt_luggage",
                "gt_box",
                "gt_weight_kg",
                "gt_weight_full",
                "gt_is_full",
                "gt_should_bypass",
                "pred_is_full",
                "weight_only_decision",
                "smart_decision",
                "outcome",
                "trip_energy_kj",
                "stop_overhead_kj",
                "cumulative_always_accept_kj",
                "cumulative_weight_only_kj",
                "cumulative_smart_kj",
            ]
        )
        for c in stats.call_log:
            w.writerow(
                [
                    c.call_id,
                    c.origin_floor,
                    c.dest_floor,
                    c.distance_floors,
                    c.direction,
                    c.filename,
                    c.gt_person,
                    c.gt_stroller,
                    c.gt_luggage,
                    c.gt_box,
                    f"{c.gt_weight_kg:.1f}",
                    c.gt_weight_full,
                    c.gt_is_full,
                    c.gt_should_bypass,
                    c.pred_is_full,
                    c.weight_only_decision,
                    c.smart_decision,
                    c.outcome,
                    f"{c.trip_energy_kj:.4f}",
                    f"{c.stop_overhead_kj:.4f}",
                    f"{c.cumulative_always_accept_kj:.4f}",
                    f"{c.cumulative_weight_only_kj:.4f}",
                    f"{c.cumulative_smart_kj:.4f}",
                ]
            )


def write_energy_csv(stats: SimulationStats, out_path: Path) -> None:
    """Write per-policy energy / time aggregates.

    Reports three policies side-by-side:
      * always_accept (naive baseline, never bypasses)
      * weight_only  (current-industry, Stage 1 only)
      * smart        (proposed, Stages 1 + 2)

    Two deltas are recorded:
      * smart_vs_always   — ceiling tasarrufu (max teorik fayda)
      * smart_vs_weight   — bizim gerçek katkımız (over current industry)
    """
    aa_j = stats.always_accept_total_j
    wo_j = stats.weight_only_total_j
    sm_j = stats.smart_total_j

    aa_t = stats.always_accept_total_time_s
    wo_t = stats.weight_only_total_time_s
    sm_t = stats.smart_total_time_s

    saved_vs_aa_j = aa_j - sm_j
    saved_vs_wo_j = wo_j - sm_j
    saved_vs_aa_pct = 100.0 * saved_vs_aa_j / aa_j if aa_j else 0.0
    saved_vs_wo_pct = 100.0 * saved_vs_wo_j / wo_j if wo_j else 0.0

    saved_vs_aa_s = aa_t - sm_t
    saved_vs_wo_s = wo_t - sm_t
    saved_vs_aa_t_pct = 100.0 * saved_vs_aa_s / aa_t if aa_t else 0.0
    saved_vs_wo_t_pct = 100.0 * saved_vs_wo_s / wo_t if wo_t else 0.0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["num_calls", stats.num_calls])
        # ── Per-policy energy totals ──
        w.writerow(["always_accept_total_kj", f"{aa_j / 1000:.2f}"])
        w.writerow(["weight_only_total_kj", f"{wo_j / 1000:.2f}"])
        w.writerow(["smart_total_kj", f"{sm_j / 1000:.2f}"])
        # ── Energy savings (smart over the two baselines) ──
        w.writerow(["energy_saved_vs_always_accept_kj", f"{saved_vs_aa_j / 1000:.2f}"])
        w.writerow(["energy_saved_vs_always_accept_kwh", f"{saved_vs_aa_j / 3_600_000:.4f}"])
        w.writerow(["energy_saved_vs_always_accept_pct", f"{saved_vs_aa_pct:.2f}"])
        w.writerow(["energy_saved_vs_weight_only_kj", f"{saved_vs_wo_j / 1000:.2f}"])
        w.writerow(["energy_saved_vs_weight_only_kwh", f"{saved_vs_wo_j / 3_600_000:.4f}"])
        w.writerow(["energy_saved_vs_weight_only_pct", f"{saved_vs_wo_pct:.2f}"])
        # ── Per-policy time totals ──
        w.writerow(["always_accept_total_time_s", f"{aa_t:.1f}"])
        w.writerow(["weight_only_total_time_s", f"{wo_t:.1f}"])
        w.writerow(["smart_total_time_s", f"{sm_t:.1f}"])
        w.writerow(["time_saved_vs_always_accept_s", f"{saved_vs_aa_s:.1f}"])
        w.writerow(["time_saved_vs_always_accept_min", f"{saved_vs_aa_s / 60:.2f}"])
        w.writerow(["time_saved_vs_always_accept_pct", f"{saved_vs_aa_t_pct:.2f}"])
        w.writerow(["time_saved_vs_weight_only_s", f"{saved_vs_wo_s:.1f}"])
        w.writerow(["time_saved_vs_weight_only_min", f"{saved_vs_wo_s / 60:.2f}"])
        w.writerow(["time_saved_vs_weight_only_pct", f"{saved_vs_wo_t_pct:.2f}"])
        # ── Per-policy bypass counts ──
        w.writerow(["weight_only_bypassed_calls", stats.weight_only_bypassed])
        w.writerow(["weight_only_accepted_calls", stats.weight_only_accepted])
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
    aa_kj = stats.always_accept_total_j / 1000
    wo_kj = stats.weight_only_total_j / 1000
    sm_kj = stats.smart_total_j / 1000
    saved_vs_aa_kj = aa_kj - sm_kj
    saved_vs_wo_kj = wo_kj - sm_kj
    saved_vs_aa_pct = 100.0 * saved_vs_aa_kj / aa_kj if aa_kj else 0.0
    saved_vs_wo_pct = 100.0 * saved_vs_wo_kj / wo_kj if wo_kj else 0.0
    mode = "hybrid" if args.head_weights else "single-model"

    lines: list[str] = []
    lines.append("# Smart Elevator CV — Energy Simulation Report\n")

    lines.append("## Configuration\n")
    lines.append(f"- Mode: **{mode}**")
    lines.append(f"- Four-class weights: `{args.weights}`")
    if args.head_weights:
        lines.append(f"- Head weights: `{args.head_weights}`")
    lines.append(f"- Rated capacity: **{args.rated_capacity} persons**")
    lines.append(f"- Rated load: **{args.rated_load_kg:.0f} kg**")
    lines.append(f"- Cabin floor area: **{args.cabin_m2:.2f} m²**")
    lines.append(f"- Confidence threshold: {args.conf_threshold:.2f}")
    lines.append(f"- Area bypass threshold (τ_A): {args.area_threshold:.2f}")
    lines.append(
        f"- Weight bypass threshold (τ_W): {args.weight_bypass_ratio:.2f} "
        f"× {args.rated_load_kg:.0f} kg = "
        f"**{args.weight_bypass_ratio * args.rated_load_kg:.0f} kg**"
    )
    lines.append(f"- Synthetic hall calls: **{args.num_calls}**")
    lines.append(f"- Building height: {args.floors_count} floors (random origin / destination)")
    lines.append(f"- Mean trip distance: {stats.mean_distance_floors:.1f} floors")
    lines.append(f"- Mean per-call cabin load (dynamic): **{stats.mean_load_kg:.0f} kg**")
    lines.append("")
    lines.append("### Per-class object masses\n")
    lines.append("| Class | Mass (kg) |")
    lines.append("|---|:---:|")
    lines.append("| person   | 75 |")
    lines.append("| stroller | 20 |")
    lines.append("| luggage  | 15 |")
    lines.append("| box      |  5 |")
    lines.append("")

    lines.append("## Accuracy 1 — Smart bypass decision (image-level)\n")
    lines.append(
        "How often the smart policy (Stage 1 weight gate **plus** Stage 2 "
        "area gate) makes the correct accept / bypass call. Ground truth "
        "is `gt_should_bypass = gt_is_full OR gt_weight_full`, i.e. the "
        "optimal policy that bypasses iff the cabin can no longer accept "
        "another passenger.\n"
    )
    lines.append("|  | Smart: accept | Smart: bypass |")
    lines.append("|---|:---:|:---:|")
    lines.append(f"| **GT: should accept** | {metrics['tn']} (TN) | {metrics['fp']} (FP) |")
    lines.append(f"| **GT: should bypass** | {metrics['fn']} (FN) | {metrics['tp']} (TP) |")
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

    # ── Object-level counting accuracy ────────────────────────────────
    agg = _aggregate_object_metrics(class_metrics)
    lines.append("## Per-class object-level counting accuracy\n")
    lines.append(
        "Treats every individual ground-truth instance as a retrieval "
        "target. For each (image, class) pair, ``TP = min(pred, gt)`` "
        "instances are correctly counted, ``FN = max(0, gt - pred)`` "
        "instances are missed (under-detection) and "
        "``FP = max(0, pred - gt)`` instances are spurious "
        "(over-detection). Recall is therefore the proportion of "
        "ground-truth objects correctly accounted for, and precision is "
        "the proportion of predicted instances that were warranted.\n"
    )
    lines.append("| Class | GT | Pred | TP | FN | FP | Recall | Precision | F1 |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cls in CLASS_NAMES:
        m = class_metrics[cls]
        lines.append(
            f"| {cls} | {m['gt_total']} | {m['pred_total']} | "
            f"{m['tp']} | {m['fn']} | {m['fp']} | "
            f"{m['recall']:.3f} | {m['precision']:.3f} | {m['f1']:.3f} |"
        )
    lines.append(
        f"| **OVERALL** | **{agg['gt_total']}** | **{agg['pred_total']}** | "
        f"**{agg['tp']}** | **{agg['fn']}** | **{agg['fp']}** | "
        f"**{agg['recall']:.3f}** | **{agg['precision']:.3f}** | "
        f"**{agg['f1']:.3f}** |"
    )
    lines.append("")
    lines.append(
        f"- **Object-level recall** (overall): **{agg['recall']:.3f}** "
        f"({agg['tp']} of {agg['gt_total']} ground-truth objects correctly counted)"
    )
    lines.append(
        f"- **Object-level precision** (overall): **{agg['precision']:.3f}** "
        f"({agg['tp']} of {agg['pred_total']} predicted instances warranted)"
    )
    lines.append(f"- **Object-level F1** (overall): **{agg['f1']:.3f}**")
    lines.append("")

    lines.append("## Per-class detection accuracy (image-level error)\n")
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

    lines.append("## Energy aggregates (stop-overhead accounting)\n")
    lines.append(
        "Each accepted call adds the per-stop overhead (door cycle + idle "
        "wait) at the called floor; bypass at a floor saves exactly this "
        "overhead. The trip's running energy is shared with other accepted "
        "calls and is therefore not credited to the bypass decision. See "
        "the call log for the per-call full Tukia (2018) trip energy.\n"
    )
    lines.append("Three policies are compared on the same 1 000-call stream:\n")
    lines.append(
        "| Policy | Bypassed | Accepted | Total energy | Δ vs always-accept | Δ vs weight-only |"
    )
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    lines.append(f"| Always-accept (naive) | 0 | {stats.num_calls} | **{aa_kj:.1f} kJ** | — | — |")
    lines.append(
        f"| Weight-only (current industry) | {stats.weight_only_bypassed} | "
        f"{stats.weight_only_accepted} | **{wo_kj:.1f} kJ** | "
        f"{aa_kj - wo_kj:.1f} kJ "
        f"({100.0 * (aa_kj - wo_kj) / aa_kj if aa_kj else 0:.1f}%) | — |"
    )
    lines.append(
        f"| **Smart (proposed)** | {stats.smart_bypassed} | "
        f"{stats.smart_accepted} | **{sm_kj:.1f} kJ** | "
        f"**{saved_vs_aa_kj:.1f} kJ ({saved_vs_aa_pct:.1f}%)** | "
        f"**{saved_vs_wo_kj:.1f} kJ ({saved_vs_wo_pct:.1f}%)** |"
    )
    lines.append("")
    lines.append(
        f"- The headline result is **smart vs weight-only**: "
        f"**{saved_vs_wo_kj:.1f} kJ ({saved_vs_wo_pct:.1f} %)** of "
        "additional savings on top of what a load-cell-only system "
        "already achieves."
    )
    lines.append(
        f"- Smart bypassed {stats.smart_bypassed} of {stats.num_calls} calls "
        f"({100 * stats.smart_bypassed / max(1, stats.num_calls):.1f}%); "
        f"of these {stats.smart_correct_bypass} were correct (TP) and "
        f"{stats.smart_wrong_bypass} were premature (FP)."
    )
    lines.append("")

    # ── Service quality (operational consequences of misclassifications) ──
    n = stats.num_calls
    # FP cases mean the passenger was *incorrectly* skipped — they were not served.
    service_rate = (n - stats.smart_wrong_bypass) / max(1, n)
    lines.append("## Service quality\n")
    lines.append(
        "Beyond raw energy, every misclassification has an operational "
        "consequence — a passenger who waits, or a stop where nobody fits. "
        "The four outcome classes are tallied below.\n"
    )
    lines.append("| Outcome | Count | Operational meaning |")
    lines.append("|---|:---:|---|")
    lines.append(
        f"| ✅ TP — correct bypass | {stats.smart_correct_bypass} | "
        f"Cabin was full, smart skipped it (saved {stats.smart_saved_correctly_j / 1000:.1f} kJ) |"
    )
    lines.append(
        f"| ✅ TN — correct accept | {stats.smart_correct_accept} | "
        f"Cabin had room, smart stopped (normal service) |"
    )
    lines.append(
        f"| ⚠ FP — wrong bypass    | {stats.smart_wrong_bypass} | "
        f"Cabin had room but smart skipped it: passenger waits |"
    )
    lines.append(
        f"| ❌ FN — wrong accept    | {stats.smart_wrong_accept} | "
        f"Cabin was full but smart stopped: wasted "
        f"{stats.smart_wasted_j / 1000:.1f} kJ on a useless stop |"
    )
    lines.append("")
    lines.append(
        f"- **Service rate** (calls served / total): "
        f"**{100 * service_rate:.1f}%**  "
        f"(baseline always reaches 100%)"
    )
    lines.append(
        f"- Wasted-stop energy (FN): **{stats.smart_wasted_j / 1000:.1f} kJ** "
        f"({100 * stats.smart_wasted_j / max(1.0, stats.smart_total_j):.1f}% of "
        f"smart total)"
    )
    lines.append(
        f"- Energy saved on correct bypasses (TP): {stats.smart_saved_correctly_j / 1000:.1f} kJ"
    )
    lines.append(
        f"- Mean trip distance: **{stats.mean_distance_floors:.1f}** floors  "
        f"(building has {args.floors_count} floors)"
    )
    lines.append("")

    # ── Time aggregates (stop-time accounting) ───────────────────────
    aa_s = stats.always_accept_total_time_s
    wo_s = stats.weight_only_total_time_s
    sm_s = stats.smart_total_time_s
    saved_vs_aa_t = aa_s - sm_s
    saved_vs_wo_t = wo_s - sm_s
    saved_vs_aa_t_pct = 100.0 * saved_vs_aa_t / aa_s if aa_s else 0.0
    saved_vs_wo_t_pct = 100.0 * saved_vs_wo_t / wo_s if wo_s else 0.0
    per_stop_s = aa_s / max(1, stats.num_calls)
    lines.append("## Time aggregates (stop-time accounting)\n")
    lines.append(
        "Each avoided stop also recovers wall-clock time. Per-stop time "
        f"overhead under our default Tukia (2018) parameters is **{per_stop_s:.1f} s "
        "= door open + door close + idle transfer**. This matches the "
        "10-15 s per intermediate stop reported in Barney (2003), "
        "Strakosch & Caporale (2010), and the time-cycle definitions of "
        "ISO 25745-2.\n"
    )
    lines.append("| Policy | Total stop-time | Δ vs always-accept | Δ vs weight-only |")
    lines.append("|---|:---:|:---:|:---:|")
    lines.append(f"| Always-accept | **{aa_s:.0f} s** ({aa_s / 60:.1f} min) | — | — |")
    lines.append(
        f"| Weight-only | **{wo_s:.0f} s** ({wo_s / 60:.1f} min) | "
        f"{aa_s - wo_s:.0f} s ({100.0 * (aa_s - wo_s) / aa_s if aa_s else 0:.1f}%) | — |"
    )
    lines.append(
        f"| **Smart** | **{sm_s:.0f} s** ({sm_s / 60:.1f} min) | "
        f"**{saved_vs_aa_t:.0f} s ({saved_vs_aa_t_pct:.1f}%)** | "
        f"**{saved_vs_wo_t:.0f} s ({saved_vs_wo_t_pct:.1f}%)** |"
    )
    lines.append("")
    lines.append(
        f"- Time wasted on FN stops: **{stats.smart_wasted_time_s:.0f} s** "
        f"(elevator opened doors at cabins that should have been bypassed)"
    )
    lines.append("")

    lines.append("## Per-image decisions\n")
    lines.append(
        "| filename | gt(p/s/l/b) | gt_kg | gt_W | gt_A | gt_BP | pred(p/s/l/b) | pred_A | smart_BP | outcome |"
    )
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for d in decisions:
        gt = f"{d.gt_person}/{d.gt_stroller}/{d.gt_luggage}/{d.gt_box}"
        pr = f"{d.pred_person}/{d.pred_stroller}/{d.pred_luggage}/{d.pred_box}"
        lines.append(
            f"| {d.filename} | {gt} | {d.gt_weight_kg:.0f} | "
            f"{'Y' if d.gt_weight_full else 'n'} | "
            f"{'Y' if d.gt_is_full else 'n'} | "
            f"{'Y' if d.gt_should_bypass else 'n'} | "
            f"{pr} | {'Y' if d.pred_is_full else 'n'} | "
            f"{'Y' if d.smart_bypass else 'n'} | {d.outcome} |"
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
    p.add_argument(
        "--rated-load-kg",
        type=float,
        default=DEFAULT_RATED_LOAD_KG,
        help="Rated cabin load (kg). Default 630 kg matches configs/default.yaml.",
    )
    p.add_argument(
        "--weight-bypass-ratio",
        type=float,
        default=DEFAULT_WEIGHT_RATIO,
        help="Stage-1 weight gate threshold as a fraction of rated load. "
        "Default 0.80, i.e. bypass when cabin load ≥ 80%% of rated.",
    )
    p.add_argument("--num-calls", type=int, default=1000)
    p.add_argument(
        "--floors-count",
        type=int,
        default=10,
        help="Number of floors in the synthetic building. Each call's "
        "origin and destination are sampled uniformly at random in "
        "[1, floors_count].",
    )
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

    weight_threshold_kg = args.weight_bypass_ratio * args.rated_load_kg

    gt_rows = load_ground_truth(
        args.ground_truth,
        cabin_m2=args.cabin_m2,
        area_threshold=args.area_threshold,
        weight_threshold_kg=weight_threshold_kg,
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
        smart_bypass = gt.gt_weight_full or pred_full
        outcome = classify_outcome(gt.gt_should_bypass, smart_bypass)
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
                gt_weight_kg=gt.gt_weight_kg,
                gt_weight_full=gt.gt_weight_full,
                gt_should_bypass=gt.gt_should_bypass,
                pred_person=pred_counts["person"],
                pred_stroller=pred_counts["stroller"],
                pred_luggage=pred_counts["luggage"],
                pred_box=pred_counts["box"],
                pred_occupancy_ratio=pred_occ,
                pred_is_full=pred_full,
                smart_bypass=smart_bypass,
                outcome=outcome,
                counts_exact_match=counts_exact_match,
                count_total_error=count_total_error,
            )
        )
        print(
            f"  {gt.filename:<48} "
            f"GT(p/s/l/b)={gt.gt_person}/{gt.gt_stroller}/"
            f"{gt.gt_luggage}/{gt.gt_box} kg={gt.gt_weight_kg:.0f} "
            f"occ={gt.gt_occupancy_ratio:.2f} "
            f"BP={'Y' if gt.gt_should_bypass else 'n'}  "
            f"PRED(p/s/l/b)={pred_counts['person']}/{pred_counts['stroller']}/"
            f"{pred_counts['luggage']}/{pred_counts['box']} occ={pred_occ:.2f} "
            f"smart_BP={'Y' if smart_bypass else 'n'}  → {outcome}"
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
        floors_count=args.floors_count,
        energy_params=energy_params,
        seed=args.seed,
    )

    cm_path = args.output / "confusion_matrix.png"
    csv_per_img = args.output / "per_image_decisions.csv"
    csv_per_cls = args.output / "per_class_detection.csv"
    csv_energy = args.output / "energy_savings.csv"
    csv_calls = args.output / "call_log.csv"
    md_report = args.output / "report.md"

    render_confusion_matrix_png(decisions, cm_path)
    write_per_image_csv(decisions, csv_per_img)
    write_per_class_csv(class_metrics, csv_per_cls)
    write_energy_csv(stats, csv_energy)
    write_call_log_csv(stats, csv_calls)
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
    print("Per-class object-level metrics:")
    for cls in CLASS_NAMES:
        m = class_metrics[cls]
        print(
            f"  {cls:<9} GT={m['gt_total']:>3}  Pred={m['pred_total']:>3}  "
            f"TP={m['tp']:>3}  FN={m['fn']:>3}  FP={m['fp']:>3}  "
            f"R={m['recall']:.3f}  P={m['precision']:.3f}  F1={m['f1']:.3f}  "
            f"MAE={m['mae']:.2f}  bias={m['bias']:+.2f}"
        )
    agg = _aggregate_object_metrics(class_metrics)
    print(
        f"  OVERALL   GT={agg['gt_total']:>3}  Pred={agg['pred_total']:>3}  "
        f"TP={agg['tp']:>3}  FN={agg['fn']:>3}  FP={agg['fp']:>3}  "
        f"R={agg['recall']:.3f}  P={agg['precision']:.3f}  F1={agg['f1']:.3f}"
    )
    aa_kj = stats.always_accept_total_j / 1000
    wo_kj = stats.weight_only_total_j / 1000
    sm_kj = stats.smart_total_j / 1000
    saved_vs_aa = aa_kj - sm_kj
    saved_vs_wo = wo_kj - sm_kj
    pct_vs_aa = 100.0 * saved_vs_aa / aa_kj if aa_kj else 0.0
    pct_vs_wo = 100.0 * saved_vs_wo / wo_kj if wo_kj else 0.0
    print()
    print(f"Energy (three policies over {stats.num_calls} calls):")
    print(f"  always_accept = {aa_kj:7.1f} kJ   (naive — never bypass)")
    print(
        f"  weight_only   = {wo_kj:7.1f} kJ   "
        f"(Stage 1 alone, current industry; bypassed {stats.weight_only_bypassed})"
    )
    print(
        f"  smart         = {sm_kj:7.1f} kJ   (Stage 1 + Stage 2; bypassed {stats.smart_bypassed})"
    )
    print(f"  Δ smart vs always_accept = {saved_vs_aa:7.1f} kJ ({pct_vs_aa:.1f}%)")
    print(f"  Δ smart vs weight_only   = {saved_vs_wo:7.1f} kJ ({pct_vs_wo:.1f}%)  ← headline")
    aa_s = stats.always_accept_total_time_s
    wo_s = stats.weight_only_total_time_s
    sm_s = stats.smart_total_time_s
    saved_t_vs_aa = aa_s - sm_s
    saved_t_vs_wo = wo_s - sm_s
    pct_t_vs_aa = 100.0 * saved_t_vs_aa / aa_s if aa_s else 0.0
    pct_t_vs_wo = 100.0 * saved_t_vs_wo / wo_s if wo_s else 0.0
    print("Time (stop-time overhead):")
    print(f"  always_accept = {aa_s:7.0f} s ({aa_s / 60:.1f} min)")
    print(f"  weight_only   = {wo_s:7.0f} s ({wo_s / 60:.1f} min)")
    print(f"  smart         = {sm_s:7.0f} s ({sm_s / 60:.1f} min)")
    print(f"  Δ smart vs always_accept = {saved_t_vs_aa:7.0f} s ({pct_t_vs_aa:.1f}%)")
    print(f"  Δ smart vs weight_only   = {saved_t_vs_wo:7.0f} s ({pct_t_vs_wo:.1f}%)")
    service_rate = (stats.num_calls - stats.smart_wrong_bypass) / max(1, stats.num_calls)
    print(
        f"Smart service quality: rate={100 * service_rate:.1f}%  "
        f"(TP={stats.smart_correct_bypass}  TN={stats.smart_correct_accept}  "
        f"FP={stats.smart_wrong_bypass}  FN={stats.smart_wrong_accept})"
    )
    print(f"  · saved on correct bypass: {stats.smart_saved_correctly_j / 1000:.1f} kJ")
    print(f"  · wasted on FN stops:      {stats.smart_wasted_j / 1000:.1f} kJ")
    print(f"  · mean trip distance:      {stats.mean_distance_floors:.1f} floors")
    print(f"Reports written to: {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
