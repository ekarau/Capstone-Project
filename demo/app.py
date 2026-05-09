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
    1. Stage-1 weight gate (PDF Algorithm 1, Andrei & Ruokokoski 2022).
    2. YOLOv8 inference on the uploaded frame.
    3. Class-based area estimator (TS EN 81-20 / ISO 8100 standard
       footprints).
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

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "weights" / "best.pt"
SAMPLE_DIR = ROOT / "data" / "unified" / "test" / "images"

# Per-class average footprint (m²) — literature-anchored values.
DEFAULT_AREAS_M2: dict[str, float] = {
    "person": 0.20,  # ISO 8100-32:2020 §6.4 (Ap range 0.17-0.22 m²); EN 81-20:2020
    "stroller": 0.45,  # EN 1888-1:2018 + product survey (Bugaboo 0.22 - UPPAbaby Vista 0.60)
    "luggage": 0.20,  # IATA Resolution 753 cabin baggage (56 x 36 cm = 0.20 m^2)
    "box": 0.20,  # Industry e-commerce parcel mean (Red Stag 2026 benchmark)
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
) -> AnalysisResult:
    """Detect, estimate occupancy, decide accept / bypass."""

    # Always run inference so we can visualize, even when weight already
    # triggers bypass — useful for the operator to see what's in the cabin.
    result = model.predict(image, conf=conf_threshold, verbose=False)[0]

    detections: list[dict] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class": cls_name,
                    "conf": float(box.conf.item()),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

    counts: dict[str, int] = {}
    breakdown: dict[str, float] = {}
    occupied = 0.0
    for det in detections:
        cls = det["class"]
        a = class_areas.get(cls, 0.0)
        if a <= 0:
            continue
        counts[cls] = counts.get(cls, 0) + 1
        breakdown[cls] = breakdown.get(cls, 0.0) + a
        occupied += a

    occupancy_ratio = min(occupied / cabin_m2, 1.0) if cabin_m2 > 0 else 0.0
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

    annotated = result.plot()[..., ::-1]  # BGR → RGB

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
    """Side-by-side gallery: original | 4-class annotated | head annotated."""
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
        cols[1].image(str(pred_path), caption="4-class detector", use_container_width=True)
        if head_pred and head_pred.exists():
            cols[2].image(str(head_pred), caption="Head detector", use_container_width=True)

        if meta:
            gt = (
                f"GT counts: person={meta.get('gt_person')}  "
                f"stroller={meta.get('gt_stroller')}  "
                f"luggage={meta.get('gt_luggage')}  box={meta.get('gt_box')}"
            )
            pred = (
                f"Pred counts: person={meta.get('pred_person')}  "
                f"stroller={meta.get('pred_stroller')}  "
                f"luggage={meta.get('pred_luggage')}  box={meta.get('pred_box')}"
            )
            st.caption(
                f"{gt} | {pred} | "
                f"outcome: **{meta.get('outcome')}**, "
                f"counts match: **{meta.get('counts_exact_match')}**"
            )
        st.divider()


def render_batch_tab() -> None:
    """Run / inspect batch energy simulations."""
    st.subheader("Batch Energy Simulation")
    st.markdown(
        "Run the curated ground-truth set in `data/sim/` through the "
        "trained detector(s) and inspect the bypass-decision and counting "
        "accuracies side by side."
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
            cls_conf = st.slider("Four-class conf", 0.10, 0.90, 0.40, 0.05)
            head_conf = st.slider("Head conf", 0.10, 0.90, 0.40, 0.05)
        with col_b:
            rated_capacity = st.number_input("Rated capacity", 4, 20, value=8, step=1)
            num_calls = st.number_input("Synthetic hall calls", 100, 5000, value=1000, step=100)
            use_head = st.checkbox("Use head detector (hybrid mode)", value=True)

        weights_path = ROOT / "models" / "weights" / "best.pt"
        head_path = ROOT / "models" / "weights" / "best_head.pt"
        if not weights_path.exists():
            st.error(f"Four-class weights not found at {weights_path}.")
        elif use_head and not head_path.exists():
            st.error(f"Head weights not found at {head_path}.")
        elif st.button("▶ Run simulation now", type="primary"):
            with st.spinner(f"Running simulation → {run_name} (this can take a minute) …"):
                ok, output = _run_simulation(
                    run_name=run_name,
                    weights=weights_path,
                    head_weights=head_path if use_head else None,
                    head_conf=head_conf,
                    cls_conf=cls_conf,
                    rated_capacity=rated_capacity,
                    num_calls=num_calls,
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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Bypass accuracy",
        bypass_acc,
        help="Decision-level: did the controller make the right call?",
    )
    m2.metric(
        "Counting accuracy",
        counting_acc,
        help="Detection-level: did all per-class counts match exactly?",
    )
    m3.metric(
        "Energy saved",
        f"{energy.get('energy_saved_pct', '0')} %",
        f"{float(energy.get('energy_saved_kj', 0)):.1f} kJ",
    )
    m4.metric(
        "Smart bypassed",
        energy.get("smart_bypassed_calls", "—"),
        f"of {energy.get('num_calls', '—')} calls",
    )

    # ─── Confusion matrix + tabular details ──────────────────────────
    cm_png = run_dir / "confusion_matrix.png"
    if cm_png.exists():
        st.markdown("### Bypass-decision confusion matrix")
        st.image(str(cm_png), width=420)

    csv_per = run_dir / "per_image_decisions.csv"
    if csv_per.exists():
        with st.expander("Per-image decisions table"):
            try:
                import pandas as pd

                df = pd.read_csv(csv_per)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except ImportError:
                st.code(csv_per.read_text(encoding="utf-8"))

    # ─── Image gallery: original vs annotated ────────────────────────
    st.markdown("### Image gallery — original vs annotated")
    _render_image_gallery(run_dir)

    # ─── Full Markdown report ────────────────────────────────────────
    if md_rep.exists():
        with st.expander("Full Markdown report"):
            st.markdown(md_rep.read_text(encoding="utf-8"))


def render_single_frame_tab() -> None:
    """Single-image detection + decision (the original demo screen)."""
    # ─── Sidebar ──────────────────────────────────────────────────────
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
        with st.expander("Override TS EN 81-20 defaults"):
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
        weights_path = st.text_input("Weights path", value=str(DEFAULT_WEIGHTS))

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

        if not Path(weights_path).exists():
            st.error(f"Weights not found at: `{weights_path}`")
            st.caption(
                "Train a model first (see notebooks/02_train.ipynb) or "
                "place an existing checkpoint at the path above."
            )
            return

        if not st.button("▶ Analyze frame", type="primary"):
            st.caption("Click the button above to run the detector.")
            return

        with st.spinner("Running detector…"):
            model = load_model(weights_path)
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
            )

        st.image(
            result.annotated_image,
            caption="YOLOv8 detections",
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
        "with TS EN 81-20 standard footprints; and decides whether the next "
        "hall call should be **accepted** or **bypassed**."
    )

    tab_single, tab_batch = st.tabs(["Single Frame", "Batch Simulation"])
    with tab_single:
        render_single_frame_tab()
    with tab_batch:
        render_batch_tab()

    # ─── Footer ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**Smart Elevator CV** · Capstone Project, Istinye University · "
        "Authors: Ege Karaurgan, Vedat Efe Gezer · "
        "Advisor: Assoc. Prof. Dr. Bahman · "
        "Methodology: TS EN 81-20:2020 + Andrei & Ruokokoski (2022) PDF Algorithm 1."
    )


if __name__ == "__main__":
    main()
