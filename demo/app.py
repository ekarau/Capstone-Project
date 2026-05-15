"""Smart Elevator CV — interactive Streamlit demo.

Run with::

    streamlit run demo/app.py

Inputs (sidebar):
    cabin width / depth / rated load (kg)
    current cabin weight (kg)
    YOLO confidence threshold
    weight and area bypass thresholds
    class-specific footprint overrides
    weights file path

Pipeline:
    1. Stage-1 weight gate.
    2. YOLOv8 inference on the uploaded frame.
    3. Class-based area estimator with industry-standard footprints.
    4. Stage-2 area gate.
    5. Render annotated frame, per-class breakdown, gauges, and the
       final ACCEPT / BYPASS decision.

The demo loads weights once via ``@st.cache_resource`` so re-analysis
on the same model is fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from src.perception.occupancy import ClassFootprintOccupancy

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "weights" / "best.pt"
DEFAULT_HEAD_WEIGHTS = ROOT / "models" / "weights" / "best_head.pt"
SAMPLE_DIR = ROOT / "data" / "unified" / "test" / "images"

# Per-class average footprint (m²).
DEFAULT_AREAS_M2: dict[str, float] = {
    "person": 0.20,
    "stroller": 0.45,
    "luggage": 0.20,
    "box": 0.20,
}

DECISION_LABELS = {
    "ACCEPT": ("✅", "ACCEPT", "success"),
    "BYPASS_AREA": ("⚠", "BYPASS (area)", "warning"),
    "BYPASS_WEIGHT": ("🚫", "BYPASS (weight)", "error"),
}


@dataclass
class AnalysisResult:
    """Everything the UI needs after one inference run."""

    annotated_image: np.ndarray
    counts: dict[str, int]
    breakdown_m2: dict[str, float]
    occupied_m2: float
    cabin_m2: float
    occupancy_ratio: float
    weight_kg: float
    max_weight_kg: float
    weight_ratio: float
    decision: str
    decision_reason: str
    detections: list[dict] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
#  Inference + decision
# ──────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading YOLOv8 weights…")
def load_model(weights_path: str):
    """Load YOLOv8 model once and cache across reruns."""
    from ultralytics import YOLO

    return YOLO(weights_path)


def run_analysis(
    image: np.ndarray,
    model,
    *,
    cabin_m2: float,
    class_areas: dict[str, float],
    conf_threshold: float,
    weight_kg: float,
    max_weight_kg: float,
    weight_threshold: float,
    area_threshold: float,
    head_model=None,
    head_conf_threshold: float = 0.25,
) -> AnalysisResult:
    """Detect, estimate occupancy, decide accept / bypass.

    When ``head_model`` is provided the pipeline runs in **3-class + head**
    mode, mirroring ``scripts/run_simulation.py``:

      * person counts come from the dedicated head detector
      * stroller / luggage / box come from the object detector
      * any ``person`` predictions from the object detector are
        discarded to avoid double-counting
    """

    # Always run inference so we can visualize, even when weight already
    # triggers bypass — useful for the operator to see what's in the cabin.
    result = model.predict(image, conf=conf_threshold, verbose=False)[0]
    use_head_model = head_model is not None

    detections: list[dict] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            # In 3-class + head mode, drop the object detector's person
            # predictions; head_model is the canonical source for person count.
            if use_head_model and cls_name == "person":
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class": cls_name,
                    "conf": float(box.conf.item()),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

    # Aggregate per-class counts. Then delegate the footprint→ratio
    # computation to the thesis §3.5 estimator. Classes with no
    # configured footprint are filtered out by ``ClassFootprintOccupancy``.
    raw_counts: dict[str, int] = {}
    for det in detections:
        cls = det["class"]
        raw_counts[cls] = raw_counts.get(cls, 0) + 1

    # 3-class + head mode: head detector supplies the person count and we
    # expose the per-head bboxes alongside the object-detector bboxes so the
    # UI can show counts and draw them.
    head_result = None
    if use_head_model:
        head_result = head_model.predict(image, conf=head_conf_threshold, verbose=False)[0]
        head_count = 0
        if head_result.boxes is not None:
            head_count = len(head_result.boxes)
            for box in head_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class": "person",
                        "conf": float(box.conf.item()),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    }
                )
        raw_counts["person"] = head_count

    occupancy = ClassFootprintOccupancy(footprints_m2=class_areas, cabin_m2=cabin_m2).compute(
        raw_counts
    )
    counts: dict[str, int] = dict(occupancy.counts)
    breakdown: dict[str, float] = dict(occupancy.breakdown_m2)
    occupied = occupancy.occupied_m2
    occupancy_ratio = occupancy.ratio
    weight_ratio = weight_kg / max_weight_kg if max_weight_kg > 0 else 0.0

    # PDF Algorithm 1 — weight gate first, then area gate.
    if weight_ratio >= weight_threshold:
        decision = "BYPASS_WEIGHT"
        reason = (
            f"Cabin load {weight_kg:.0f} kg ≥ {weight_threshold * 100:.0f}% "
            f"of rated {max_weight_kg:.0f} kg."
        )
    elif occupancy_ratio >= area_threshold:
        decision = "BYPASS_AREA"
        reason = (
            f"Estimated floor occupancy {occupancy_ratio * 100:.1f}% "
            f"≥ {area_threshold * 100:.0f}% threshold."
        )
    else:
        decision = "ACCEPT"
        reason = (
            f"Both stages clear — load {weight_ratio * 100:.0f}%, "
            f"occupancy {occupancy_ratio * 100:.1f}%."
        )

    # Object detector annotations first, then overlay head boxes on top
    # so the operator sees every counted instance in a single frame.
    annotated_bgr = result.plot()  # BGR
    if use_head_model and head_result is not None and head_result.boxes is not None:
        import cv2  # local import keeps the demo importable without cv2 at top

        for box in head_result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_bgr,
                f"person {float(box.conf.item()):.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
    annotated = annotated_bgr[..., ::-1]  # BGR → RGB

    return AnalysisResult(
        annotated_image=annotated,
        counts=counts,
        breakdown_m2=breakdown,
        occupied_m2=occupied,
        cabin_m2=cabin_m2,
        occupancy_ratio=occupancy_ratio,
        weight_kg=weight_kg,
        max_weight_kg=max_weight_kg,
        weight_ratio=weight_ratio,
        decision=decision,
        decision_reason=reason,
        detections=detections,
    )


# ──────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────


SIM_RESULTS_ROOT = ROOT / "results" / "simulation"
SIM_IMAGES_DIR = ROOT / "data" / "sim" / "images"
SIM_GT_CSV = ROOT / "data" / "sim" / "ground_truth.csv"

OUTCOME_BADGES = {
    "TP": "✅ correct bypass",
    "TN": "✅ correct accept",
    "FP": "⚠ wrong bypass",
    "FN": "❌ missed bypass",
}


def _friendlify_per_image_df(df):
    """Turn the raw CSV columns into reader-friendly labels."""
    rename = {
        "filename": "Image",
        "gt_person": "👤 GT",
        "gt_stroller": "🚼 GT",
        "gt_luggage": "🧳 GT",
        "gt_box": "📦 GT",
        "gt_is_full": "GT full?",
        "gt_occupancy_ratio": "GT occ.",
        "pred_person": "👤 pred",
        "pred_stroller": "🚼 pred",
        "pred_luggage": "🧳 pred",
        "pred_box": "📦 pred",
        "pred_is_full": "Pred full?",
        "pred_occupancy_ratio": "Pred occ.",
        "outcome": "Decision",
        "counts_exact_match": "Counts ✓?",
        "count_total_error": "Count err.",
    }
    out = df.rename(columns=rename).copy()
    if "Decision" in out:
        out["Decision"] = out["Decision"].map(lambda v: OUTCOME_BADGES.get(v, v))
    for col in ("GT full?", "Pred full?", "Counts ✓?"):
        if col in out:
            out[col] = out[col].map(lambda v: "✅" if str(v).lower() == "true" else "❌")
    return out


def _list_run_dirs() -> list[str]:
    """Return existing simulation run sub-directory names, newest first."""
    if not SIM_RESULTS_ROOT.is_dir():
        return []
    runs = [p for p in SIM_RESULTS_ROOT.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in runs]


def _parse_energy_csv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for row in path.read_text(encoding="utf-8").splitlines()[1:]:
        if "," in row:
            k, v = row.split(",", 1)
            out[k.strip()] = v.strip()
    return out


def _run_simulation(
    run_name: str,
    weights: Path,
    head_weights: Path | None,
    head_conf: float,
    cls_conf: float,
    rated_capacity: int,
    num_calls: int,
    cabin_m2: float,
    area_threshold: float,
) -> tuple[bool, str]:
    """Invoke scripts.run_simulation as a subprocess. Returns (ok, stdout)."""
    import subprocess
    import sys as _sys

    cmd = [
        _sys.executable,
        "-m",
        "scripts.run_simulation",
        "--images",
        str(SIM_IMAGES_DIR),
        "--ground-truth",
        str(SIM_GT_CSV),
        "--weights",
        str(weights),
        "--head-conf",
        f"{head_conf:.2f}",
        "--conf-threshold",
        f"{cls_conf:.2f}",
        "--rated-capacity",
        str(rated_capacity),
        "--num-calls",
        str(num_calls),
        "--cabin-m2",
        f"{cabin_m2:.4f}",
        "--area-threshold",
        f"{area_threshold:.2f}",
        "--output",
        str(SIM_RESULTS_ROOT / run_name),
    ]
    if head_weights is not None:
        cmd.extend(["--head-weights", str(head_weights)])
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover — surfaced in UI
        return False, f"failed to launch subprocess: {exc}"
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return (result.returncode == 0), output


def _render_image_gallery(run_dir: Path) -> None:
    """Side-by-side gallery: original | object detector annotated | head detector annotated."""
    pred_dir = run_dir / "predictions"
    head_pred_dir = run_dir / "predictions_head"
    if not pred_dir.is_dir() or not SIM_IMAGES_DIR.is_dir():
        st.info(
            "No annotated predictions found in this run. Re-run the "
            "simulation (annotated frames are saved by default)."
        )
        return

    # Per-image counts for the caption.
    csv_per = run_dir / "per_image_decisions.csv"
    rows_by_name: dict[str, dict] = {}
    if csv_per.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_per)
            rows_by_name = {row["filename"]: row.to_dict() for _, row in df.iterrows()}
        except ImportError:
            pass

    images = sorted(pred_dir.iterdir())
    if not images:
        st.info("Predictions directory is empty.")
        return

    page_size = 6
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, (len(images) + page_size - 1) // page_size),
        value=1,
        step=1,
        help=f"{len(images)} images total, {page_size} per page",
    )
    start = (int(page) - 1) * page_size
    end = start + page_size

    for pred_path in images[start:end]:
        name = pred_path.name
        original = SIM_IMAGES_DIR / name
        head_pred = head_pred_dir / name if head_pred_dir.is_dir() else None
        meta = rows_by_name.get(name, {})

        st.markdown(f"**{name}**")
        cols = st.columns(3 if head_pred and head_pred.exists() else 2)
        if original.exists():
            cols[0].image(str(original), caption="Original", use_container_width=True)
        else:
            cols[0].info("Original image missing.")
        cols[1].image(str(pred_path), caption="Object detector", use_container_width=True)
        if head_pred and head_pred.exists():
            cols[2].image(str(head_pred), caption="Head detector", use_container_width=True)

        if meta:

            def _fmt(cls_emoji: str, gt_v, pr_v) -> str:
                check = "✅" if str(gt_v) == str(pr_v) else "❌"
                return f"{cls_emoji} {gt_v} → {pr_v} {check}"

            count_line = "  ·  ".join(
                [
                    _fmt("👤", meta.get("gt_person"), meta.get("pred_person")),
                    _fmt("🚼", meta.get("gt_stroller"), meta.get("pred_stroller")),
                    _fmt("🧳", meta.get("gt_luggage"), meta.get("pred_luggage")),
                    _fmt("📦", meta.get("gt_box"), meta.get("pred_box")),
                ]
            )
            decision_badge = OUTCOME_BADGES.get(meta.get("outcome", ""), meta.get("outcome", ""))
            counts_match = str(meta.get("counts_exact_match", "")).lower() == "true"
            counts_badge = "✅ all counts match" if counts_match else "❌ count error"
            st.caption(
                f"**Counts (real → detected):**  {count_line}  \n"
                f"**Decision:** {decision_badge}  ·  {counts_badge}"
            )

        st.divider()


def render_timeline_tab() -> None:
    """Per-call simulation timeline: charts and a filterable call log."""
    st.subheader("Call Timeline")
    st.markdown(
        "Per-call view of a simulation run: which floor called, what the "
        "cabin looked like, what the smart policy decided, and how the "
        "energy curves diverge between baseline and smart over time."
    )

    runs = _list_run_dirs()
    if not runs:
        st.warning("No simulation runs yet. Run one from the **Batch Simulation** tab.")
        return

    selected_run = st.selectbox(
        "Run to inspect",
        options=runs,
        index=0,
        key="timeline_run_selector",
    )
    run_dir = SIM_RESULTS_ROOT / selected_run

    call_log_path = run_dir / "call_log.csv"
    if not call_log_path.exists():
        st.warning(
            f"No call_log.csv in `{run_dir.name}`. Re-run the simulation "
            "after the latest update to generate the per-call log."
        )
        return

    try:
        import pandas as pd
    except ImportError:
        st.error("pandas is required for the timeline view. `pip install pandas`")
        return

    df = pd.read_csv(call_log_path)

    # ─── Backward-compat for older runs ──────────────────────────────
    # Earlier runs used `energy_kj` instead of `trip_energy_kj` /
    # `stop_overhead_kj`. Older 2-policy runs used
    # `cumulative_baseline_kj` and a single `decision` column. Newer
    # 3-policy runs (always-accept / weight-only / smart) use
    # `cumulative_always_accept_kj`, `cumulative_weight_only_kj`,
    # `cumulative_smart_kj` and a `smart_decision` column. Synthesize
    # whichever columns are missing so the rest of the view works on
    # any historical run.
    OVERHEAD_KJ = 0.92  # default Tukia params: door cycle + idle = 0.92 kJ
    OVERHEAD_S = 10  # default Tukia params: door cycle + idle = 10 s
    if "stop_overhead_kj" not in df.columns:
        df["stop_overhead_kj"] = OVERHEAD_KJ
    if "trip_energy_kj" not in df.columns and "energy_kj" in df.columns:
        df["trip_energy_kj"] = df["energy_kj"]
    if "cumulative_always_accept_kj" not in df.columns:
        df["cumulative_always_accept_kj"] = df.get(
            "cumulative_baseline_kj", df["call_id"] * OVERHEAD_KJ
        )
    if "cumulative_weight_only_kj" not in df.columns:
        df["cumulative_weight_only_kj"] = df["cumulative_always_accept_kj"]
    if "smart_decision" not in df.columns and "decision" in df.columns:
        df["smart_decision"] = df["decision"]
    # Old 2-policy runs predate the explicit weight_only_decision column;
    # synthesise an "always accept" stand-in so the per-call vs-weight-only
    # savings degenerate to vs always-accept (no weight gate present).
    if "weight_only_decision" not in df.columns:
        df["weight_only_decision"] = "accept"

    # ─── Summary banner ──────────────────────────────────────────────
    n_calls = len(df)
    n_bypassed = int((df["smart_decision"] == "bypass").sum())
    last = df.iloc[-1]
    aa_kj = last["cumulative_always_accept_kj"]
    wo_kj = last["cumulative_weight_only_kj"]
    sm_kj = last["cumulative_smart_kj"]
    saved_vs_aa = aa_kj - sm_kj
    saved_vs_wo = wo_kj - sm_kj
    saved_pct_vs_aa = 100.0 * saved_vs_aa / aa_kj if aa_kj else 0.0
    saved_pct_vs_wo = 100.0 * saved_vs_wo / wo_kj if wo_kj else 0.0
    # Each bypassed call saves OVERHEAD_S seconds (door + idle); same
    # constant for every call by definition of the overhead-only model.
    saved_s = n_bypassed * OVERHEAD_S

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total calls", n_calls)
    m2.metric(
        "Energy saved (vs always-accept)",
        f"{saved_vs_aa:.1f} kJ",
        f"{saved_pct_vs_aa:.1f}%",
        help=(
            "Cumulative door + idle energy avoided by smart bypasses, "
            "compared to a naive always-accept policy. Trip running "
            "energy is shared with other accepted calls and is therefore "
            "not credited to the bypass decision."
        ),
    )
    m3.metric(
        "Energy saved (vs weight-only)",
        f"{saved_vs_wo:.1f} kJ",
        f"{saved_pct_vs_wo:.1f}%",
        help=(
            "Headline result: extra savings smart adds on top of a "
            "current-industry weight-only bypass system."
        ),
    )
    m4.metric(
        "Stop-time saved",
        f"{saved_s} s",
        f"{saved_s / 60:.1f} min",
        help=(
            "Stop-time avoided by smart bypass decisions. Each unnecessary "
            "stop costs ~10 s (door open + close + idle transfer; Barney "
            "2003, Strakosch & Caporale 2010, ISO 25745-2)."
        ),
    )

    # ─── Cumulative energy chart ─────────────────────────────────────
    st.markdown("### Cumulative energy over calls")
    st.caption(
        "How quickly each of the three policies accrues stop-overhead "
        "energy across the simulated day. The gaps between curves are "
        "the savings: smart vs always-accept (max theoretical) and smart "
        "vs weight-only (extra value of vision)."
    )
    energy_df = (
        df[
            [
                "call_id",
                "cumulative_always_accept_kj",
                "cumulative_weight_only_kj",
                "cumulative_smart_kj",
            ]
        ]
        .rename(
            columns={
                "cumulative_always_accept_kj": "Always-accept",
                "cumulative_weight_only_kj": "Weight-only",
                "cumulative_smart_kj": "Smart",
            }
        )
        .set_index("call_id")
    )
    st.line_chart(energy_df, use_container_width=True)

    # ─── Call-level confusion matrix ─────────────────────────────────
    st.markdown("### How the bypass decisions break down")
    st.caption(
        "Call-level 2×2 matrix of the smart policy against the optimal-policy "
        "ground truth across all simulated hall calls. Each row is what the "
        "cabin actually allowed; each column is what the policy did."
    )
    outcome_counts = df["outcome"].value_counts()
    tp = int(outcome_counts.get("TP", 0))
    tn = int(outcome_counts.get("TN", 0))
    fp = int(outcome_counts.get("FP", 0))
    fn = int(outcome_counts.get("FN", 0))
    total = tp + tn + fp + fn
    service_rate = (tp + tn) / total if total else 0.0
    cm_df = pd.DataFrame(
        {
            "Predicted: ACCEPT": [
                f"✅ TN = {tn}",
                f"❌ FN = {fn}",
            ],
            "Predicted: BYPASS": [
                f"⚠ FP = {fp}",
                f"✅ TP = {tp}",
            ],
        },
        index=["GT: should ACCEPT", "GT: should BYPASS"],
    )
    st.table(cm_df)
    cm_m1, cm_m2, cm_m3 = st.columns(3)
    cm_m1.metric(
        "Service rate",
        f"{service_rate * 100:.1f}%",
        f"{tp + tn} of {total} calls",
        help="Share of calls where the smart policy made the right ACCEPT / BYPASS decision.",
    )
    cm_m2.metric(
        "Wrongly skipped (FP)",
        fp,
        help="Cabin had room, but smart bypassed → passenger has to wait for the next cabin.",
    )
    cm_m3.metric(
        "Wasted stops (FN)",
        fn,
        help="Cabin was full, but smart still stopped → door cycle wasted on a useless stop.",
    )

    # ─── Outcome distribution by floor ───────────────────────────────
    st.markdown("### Outcomes by call origin floor")
    st.caption(
        "How calls from each floor were handled. TP = correct bypass, "
        "TN = correct accept, FP = wrong bypass (passenger waited), "
        "FN = wrong accept (wasted stop)."
    )
    pivot = df.groupby(["origin_floor", "outcome"]).size().unstack(fill_value=0)
    for col in ("TP", "TN", "FP", "FN"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["TP", "TN", "FP", "FN"]]
    st.bar_chart(pivot, use_container_width=True)

    # ─── Call-log table (filterable) ─────────────────────────────────
    st.markdown("### Call log")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        outcome_filter = st.multiselect(
            "Filter by outcome",
            options=["TP", "TN", "FP", "FN"],
            default=["TP", "TN", "FP", "FN"],
            help=(
                "Pick which outcome classes to display. Useful for "
                "inspecting only the FN (wasted) or FP (skipped) cases."
            ),
        )
    with col_b:
        max_rows = st.slider("Max rows to show", 10, 500, 50, 10)

    view = df[df["outcome"].isin(outcome_filter)].head(max_rows).copy()

    # Per-call savings, split by which baseline they are measured against
    # so that the column sums reconcile with the two headline metrics
    # above (Energy saved vs always-accept and vs weight-only).
    #
    #   * vs always-accept: smart saves the per-stop overhead on every
    #     call where it bypasses (always-accept never bypasses, so any
    #     smart bypass is a real saving against it).
    #   * vs weight-only: smart only adds value when it bypasses a cabin
    #     that the weight-only baseline would have accepted, i.e. an
    #     area-only-full cabin. Heavy cabins that both policies bypass
    #     contribute 0 here, because weight-only would have skipped them
    #     anyway.
    is_smart_bypass = view["smart_decision"] == "bypass"
    is_weight_bypass = view["weight_only_decision"] == "bypass"
    view["Saved vs always-accept (kJ)"] = view["stop_overhead_kj"].where(is_smart_bypass, 0.0)
    view["Saved vs weight-only (kJ)"] = view["stop_overhead_kj"].where(
        is_smart_bypass & ~is_weight_bypass, 0.0
    )
    view["Time saved (s)"] = is_smart_bypass.map(lambda b: OVERHEAD_S if b else 0)

    view = view.rename(
        columns={
            "call_id": "Call",
            "origin_floor": "From",
            "dest_floor": "To",
            "distance_floors": "Floors",
            "direction": "Dir.",
            "filename": "Image",
            "smart_decision": "Decision",
            "outcome": "Outcome",
        }
    )[
        [
            "Call",
            "From",
            "To",
            "Floors",
            "Dir.",
            "Decision",
            "Outcome",
            "Saved vs always-accept (kJ)",
            "Saved vs weight-only (kJ)",
            "Time saved (s)",
            "Image",
        ]
    ]
    view["Outcome"] = view["Outcome"].map(lambda v: OUTCOME_BADGES.get(v, v))
    st.dataframe(view, use_container_width=True, hide_index=True)


def render_batch_tab(cfg: dict) -> None:
    """Run / inspect batch energy simulations.

    Cabin geometry, max load, area / weight thresholds, confidence
    threshold and per-class footprints are taken from the sidebar so
    both tabs stay consistent. Override anything via the sidebar and
    the next ``Run simulation now`` reflects it.
    """
    st.subheader("Batch Energy Simulation")
    st.markdown(
        "Run the curated ground-truth set in `data/sim/` through the "
        "trained detector(s) and inspect the bypass-decision and counting "
        "accuracies side by side. **Cabin and threshold values come from "
        "the sidebar on the left.**"
    )

    # Honest tell-tale: thesis numbers were computed with these defaults.
    THESIS_W, THESIS_D, THESIS_TAUW, THESIS_TAUA = 1.4, 1.6, 0.80, 0.90
    diverged = (
        abs(cfg["width_m"] - THESIS_W) > 1e-3
        or abs(cfg["depth_m"] - THESIS_D) > 1e-3
        or abs(cfg["weight_threshold"] - THESIS_TAUW) > 1e-3
        or abs(cfg["area_threshold"] - THESIS_TAUA) > 1e-3
    )
    if diverged:
        st.warning(
            f"Sidebar overrides the thesis defaults "
            f"(cabin {THESIS_W}x{THESIS_D} m, τ_W={THESIS_TAUW}, τ_A={THESIS_TAUA}). "
            "Numbers from a run started now will not match the published table."
        )

    # ─── Run controls ────────────────────────────────────────────────
    with st.expander("▶ Run a new simulation", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            run_name = st.text_input(
                "Output directory name",
                value="from_demo",
                help="Created under results/simulation/<name>/",
            )
            head_conf = st.slider(
                "Head detector confidence",
                0.10,
                0.90,
                0.40,
                0.05,
                help="Detection threshold for the head model only. "
                "The object detector threshold comes from the sidebar.",
            )
        with col_b:
            num_calls = st.number_input("Synthetic hall calls", 100, 5000, value=1000, step=100)
            use_head = st.checkbox("Use head detector (3-class + head mode)", value=True)

        # Rated capacity is derived from the sidebar Rated load (75 kg / passenger).
        rated_capacity = max(1, round(cfg["max_weight"] / 75.0))
        st.caption(
            f"Rated capacity (auto-derived from Rated load): "
            f"**{rated_capacity} persons**  ·  "
            f"Cabin: **{cfg['cabin_m2']:.2f} m²**  ·  "
            f"τ_A=**{cfg['area_threshold']:.2f}**  ·  "
            f"Object det. conf=**{cfg['conf_threshold']:.2f}**"
        )

        weights_path = DEFAULT_WEIGHTS
        head_path = DEFAULT_HEAD_WEIGHTS
        if not weights_path.exists():
            st.error(f"Object detector weights not found at {weights_path}.")
        elif use_head and not head_path.exists():
            st.error(f"Head detector weights not found at {head_path}.")
        elif st.button("▶ Run simulation now", type="primary"):
            with st.spinner(f"Running simulation → {run_name} (this can take a minute) …"):
                ok, output = _run_simulation(
                    run_name=run_name,
                    weights=weights_path,
                    head_weights=head_path if use_head else None,
                    head_conf=head_conf,
                    cls_conf=cfg["conf_threshold"],
                    rated_capacity=rated_capacity,
                    num_calls=num_calls,
                    cabin_m2=cfg["cabin_m2"],
                    area_threshold=cfg["area_threshold"],
                )
            if ok:
                st.success(f"Simulation finished → results/simulation/{run_name}/")
            else:
                st.error("Simulation failed. See the captured output below.")
            with st.expander("subprocess output", expanded=not ok):
                st.code(output or "(no output)")

    # ─── Run selector ────────────────────────────────────────────────
    runs = _list_run_dirs()
    if not runs:
        st.warning(
            "No simulation runs yet. Open the 'Run a new simulation' "
            "expander above and click 'Run simulation now'."
        )
        return

    selected_run = st.selectbox(
        "Run to display",
        options=runs,
        index=0,
        help="Most recently modified first.",
    )
    run_dir = SIM_RESULTS_ROOT / selected_run

    # ─── Headline metrics ────────────────────────────────────────────
    energy = _parse_energy_csv(run_dir / "energy_savings.csv")
    md_rep = run_dir / "report.md"

    # Two accuracies + energy.
    bypass_acc = "—"
    counting_acc = "—"
    if md_rep.exists():
        text = md_rep.read_text(encoding="utf-8")
        # Cheap parse: grab the bolded numbers next to the labels.
        import re

        m1 = re.search(r"Bypass accuracy[^\d]*([\d.]+)", text)
        m2 = re.search(r"Counting accuracy[^\d]*([\d.]+)", text)
        if m1:
            bypass_acc = f"{float(m1.group(1)) * 100:.1f}%"
        if m2:
            counting_acc = f"{float(m2.group(1)) * 100:.1f}%"

    st.markdown("### Headline metrics")
    st.caption(
        "Detection and decision quality on the 67-image curated set. "
        "Energy- and stop-time savings over the 1 000 synthetic hall "
        "calls are reported in the **Call Timeline** tab."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "✅ Decision accuracy",
        bypass_acc,
        help=(
            "How often the elevator controller makes the right ACCEPT / BYPASS call on each image."
        ),
    )
    m2.metric(
        "🔢 Counting accuracy",
        counting_acc,
        help=(
            "How often EVERY per-class count "
            "(people, strollers, luggage, boxes) matches the ground truth "
            "exactly. A stricter test than the decision accuracy: this "
            "verifies the model is right for the right reasons."
        ),
    )
    m3.metric(
        "⏭ Calls bypassed",
        energy.get("smart_bypassed_calls", "—"),
        f"of {energy.get('num_calls', '—')} calls",
    )

    with st.expander("What do these two accuracies mean?"):
        st.markdown(
            "We report the system's correctness on **two independent levels**:\n\n"
            "1. **Decision accuracy** — *does the controller make the "
            "right call?* The cabin is either *full* or *not full*; the "
            "controller is correct whenever it agrees with the ground "
            "truth on this binary question.\n\n"
            "2. **Counting accuracy** — *does the underlying detector "
            "see exactly the right objects?* The image is counted as "
            "correct only when the predicted person, stroller, luggage "
            "and box counts ALL match the ground truth.\n\n"
            "A high decision accuracy with a low counting accuracy means "
            "the system often arrives at the right call via wrong counts "
            "(*spurious correctness*). Reporting both prevents that "
            "shortcut from inflating the headline number."
        )

    # ─── Confusion matrix + tabular details ──────────────────────────
    cm_png = run_dir / "confusion_matrix.png"
    if cm_png.exists():
        st.markdown("### How the bypass decisions break down")
        col_cm, col_legend = st.columns([2, 1])
        with col_cm:
            st.image(str(cm_png), use_container_width=True)
        with col_legend:
            st.markdown(
                "**Outcome legend**\n\n"
                "✅ **TP** — correctly bypassed a full cabin\n\n"
                "✅ **TN** — correctly accepted a not-full cabin\n\n"
                "⚠ **FP** — bypassed when shouldn't (passenger waits)\n\n"
                "❌ **FN** — accepted a full cabin (wasted stop)"
            )

    csv_per = run_dir / "per_image_decisions.csv"
    if csv_per.exists():
        with st.expander("Per-image breakdown (full table)"):
            try:
                import pandas as pd

                df = pd.read_csv(csv_per)
                df_view = _friendlify_per_image_df(df)
                st.dataframe(df_view, use_container_width=True, hide_index=True)
            except ImportError:
                st.code(csv_per.read_text(encoding="utf-8"))

    # ─── Image gallery: original vs annotated ────────────────────────
    st.markdown("### Image gallery — original vs annotated")
    _render_image_gallery(run_dir)

    # ─── Full Markdown report ────────────────────────────────────────
    if md_rep.exists():
        with st.expander("Full Markdown report"):
            st.markdown(md_rep.read_text(encoding="utf-8"))


def render_sidebar() -> dict:
    """Render the global sidebar and return all configuration values.

    These values drive **both** the Single Frame tab and the Batch
    Simulation tab — change a slider once, both views (and the
    subprocess that scripts.run_simulation spawns) honour it.
    """
    with st.sidebar:
        st.header("Cabin geometry")
        width_m = st.number_input("Width (m)", 0.5, 3.0, value=1.4, step=0.05)
        depth_m = st.number_input("Depth (m)", 0.5, 3.0, value=1.6, step=0.05)
        cabin_m2 = width_m * depth_m
        st.caption(f"Floor area = **{cabin_m2:.2f} m²**")

        st.header("Load")
        max_weight = st.number_input("Rated load (kg)", 200.0, 2500.0, value=630.0, step=10.0)
        current_weight = st.slider(
            "Current cabin weight (kg)",
            0.0,
            float(max_weight),
            value=0.0,
            step=10.0,
        )

        st.header("Detection")
        conf_threshold = st.slider("YOLO confidence", 0.10, 0.90, 0.40, 0.05)

        st.header("Decision thresholds")
        weight_threshold = st.slider("Weight bypass τ_W", 0.50, 1.00, 0.80, 0.05)
        area_threshold = st.slider("Area bypass τ_A", 0.50, 1.00, 0.90, 0.05)

        st.header("Class footprints (m²)")
        with st.expander("Override defaults"):
            person_m2 = st.number_input("person", value=DEFAULT_AREAS_M2["person"], step=0.01)
            stroller_m2 = st.number_input("stroller", value=DEFAULT_AREAS_M2["stroller"], step=0.01)
            luggage_m2 = st.number_input("luggage", value=DEFAULT_AREAS_M2["luggage"], step=0.01)
            box_m2 = st.number_input("box", value=DEFAULT_AREAS_M2["box"], step=0.01)
        class_areas = {
            "person": person_m2,
            "stroller": stroller_m2,
            "luggage": luggage_m2,
            "box": box_m2,
        }

        st.header("Model")
        st.caption(
            f"Object detector: `{DEFAULT_WEIGHTS.relative_to(ROOT)}`  \n"
            f"Head detector:   `{DEFAULT_HEAD_WEIGHTS.relative_to(ROOT)}`"
        )

    return {
        "width_m": width_m,
        "depth_m": depth_m,
        "cabin_m2": cabin_m2,
        "max_weight": max_weight,
        "current_weight": current_weight,
        "conf_threshold": conf_threshold,
        "weight_threshold": weight_threshold,
        "area_threshold": area_threshold,
        "class_areas": class_areas,
    }


def render_single_frame_tab(cfg: dict) -> None:
    """Single-image detection + decision (the original demo screen).

    All cabin / detection / threshold parameters come from the sidebar
    via ``cfg`` so they stay consistent with the Batch Simulation tab.
    """
    cabin_m2 = cfg["cabin_m2"]
    max_weight = cfg["max_weight"]
    current_weight = cfg["current_weight"]
    conf_threshold = cfg["conf_threshold"]
    weight_threshold = cfg["weight_threshold"]
    area_threshold = cfg["area_threshold"]
    class_areas = cfg["class_areas"]

    # ─── Main: input + output ─────────────────────────────────────────
    col_input, col_output = st.columns([1, 1])

    img_array: np.ndarray | None = None

    with col_input:
        st.subheader("Input")
        upload = st.file_uploader("Upload an elevator CCTV image", type=["jpg", "jpeg", "png"])

        sample_choice = "—"
        if SAMPLE_DIR.exists():
            samples = sorted(SAMPLE_DIR.glob("*.jpg"))[:8]
            if samples:
                sample_choice = st.selectbox(
                    "…or pick a bundled test sample",
                    options=["—"] + [p.name for p in samples],
                )

        if upload is not None:
            img_array = np.array(Image.open(upload).convert("RGB"))
            st.image(img_array, caption="Uploaded frame", use_container_width=True)
        elif sample_choice != "—":
            sample_path = SAMPLE_DIR / sample_choice
            img_array = np.array(Image.open(sample_path).convert("RGB"))
            st.image(img_array, caption=sample_choice, use_container_width=True)

    with col_output:
        st.subheader("Analysis")

        if img_array is None:
            st.info("Upload an image or pick a sample on the left.")
            return

        missing: list[str] = []
        if not DEFAULT_WEIGHTS.exists():
            missing.append(str(DEFAULT_WEIGHTS))
        if not DEFAULT_HEAD_WEIGHTS.exists():
            missing.append(str(DEFAULT_HEAD_WEIGHTS))
        if missing:
            st.error("Required weights not found:\n- " + "\n- ".join(missing))
            st.caption(
                "Place the trained checkpoints under ``models/weights/`` "
                "(see notebooks/02_train.ipynb and notebooks/05_head_model_training.ipynb)."
            )
            return

        if not st.button("▶ Analyze frame", type="primary"):
            st.caption("Click the button above to run the detector.")
            return

        with st.spinner("Running detectors…"):
            model = load_model(str(DEFAULT_WEIGHTS))
            head_model = load_model(str(DEFAULT_HEAD_WEIGHTS))
            result = run_analysis(
                image=img_array,
                model=model,
                cabin_m2=cabin_m2,
                class_areas=class_areas,
                conf_threshold=conf_threshold,
                weight_kg=current_weight,
                max_weight_kg=max_weight,
                weight_threshold=weight_threshold,
                area_threshold=area_threshold,
                head_model=head_model,
            )

        st.image(
            result.annotated_image,
            caption="3-class + head detections: object detector (default colours) + head detector (red)",
            use_container_width=True,
        )

        # ── Decision badge ───────────────────────────────────────────
        emoji, label, level = DECISION_LABELS[result.decision]
        message = f"{emoji}  **{label}** — {result.decision_reason}"
        getattr(st, level)(message)

        # ── Metrics row ──────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Occupancy",
            f"{result.occupancy_ratio * 100:.1f}%",
            f"{result.occupied_m2:.2f} / {result.cabin_m2:.2f} m²",
        )
        m2.metric(
            "Weight",
            f"{result.weight_ratio * 100:.0f}%",
            f"{result.weight_kg:.0f} / {result.max_weight_kg:.0f} kg",
        )
        m3.metric("Detections", sum(result.counts.values()))

        # ── Per-class breakdown table ────────────────────────────────
        if result.counts:
            st.markdown("**Per-class breakdown**")
            rows = [
                {
                    "Class": cls,
                    "Count": result.counts[cls],
                    "Area (m²)": f"{result.breakdown_m2[cls]:.2f}",
                }
                for cls in result.counts
            ]
            st.table(rows)
        else:
            st.info("No objects detected in this frame.")

        # ── Occupancy progress bar ───────────────────────────────────
        st.progress(
            result.occupancy_ratio,
            text=f"Floor occupancy: {result.occupancy_ratio * 100:.1f}%",
        )


def main() -> None:
    st.set_page_config(
        page_title="Smart Elevator CV",
        page_icon="🛗",
        layout="wide",
    )
    st.title("🛗 Smart Elevator CV — Live Demo")
    st.markdown(
        "Upload a CCTV frame from an elevator cabin. The system detects "
        "people, strollers, luggage, and boxes; estimates floor occupancy "
        "with industry-standard footprints; and decides whether the next "
        "hall call should be **accepted** or **bypassed**."
    )

    cfg = render_sidebar()

    tab_single, tab_batch, tab_timeline = st.tabs(
        ["Single Frame", "Batch Simulation", "Call Timeline"]
    )
    with tab_single:
        render_single_frame_tab(cfg)
    with tab_batch:
        render_batch_tab(cfg)
    with tab_timeline:
        render_timeline_tab()

    # ─── Footer ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**Smart Elevator CV** · Capstone Project, Istinye University · "
        "Authors: Ege Karaurgan, Vedat Efe Gezer · "
        "Advisor: Assoc. Prof. Dr. Bahman"
    )


if __name__ == "__main__":
    main()
