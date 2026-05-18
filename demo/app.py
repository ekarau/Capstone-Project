"""Smart Elevator CV — interactive Streamlit demo.

Run with::

    streamlit run demo/app.py

This dashboard runs the curated ground-truth set in ``data/sim/`` through
the trained detectors and exposes two views:

    * **Batch Simulation** — re-run the simulation, then inspect
      bypass-decision accuracy, object-level counting metrics and
      cumulative energy savings.
    * **Call Timeline** — replay individual hall calls from any saved
      run, filtered by TP / TN / FP / FN outcome.

Sidebar controls — cabin geometry, rated load, YOLO confidence and the
two decision thresholds — drive both tabs and the next
``Run simulation now`` honours them.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "weights" / "best.pt"
DEFAULT_HEAD_WEIGHTS = ROOT / "models" / "weights" / "best_head.pt"


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
    seed: int = 42,
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
        "--seed",
        str(seed),
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

    # ─── Outcome breakdown with energy attribution (Report Table 4.6) ──
    # Per-floor distribution intentionally removed: the simulator samples
    # cabins uniformly and the algorithm is floor-agnostic, so any
    # per-floor split is sample-size noise rather than an analytical
    # signal. The aggregate outcome × energy view below replaces it and
    # mirrors Table 4.6 of the thesis.
    st.markdown("### Outcome breakdown")
    st.caption(
        "Each misclassification has a distinct operational consequence "
        "beyond the raw energy figure. The service-rate cost is paid "
        "entirely in wasted stops (FN) rather than skipped passengers "
        "(FP); a wasted stop costs one door cycle, whereas a skipped "
        "passenger costs trust."
    )
    tp_kj = df.loc[df["outcome"] == "TP", "stop_overhead_kj"].sum()
    fn_kj = df.loc[df["outcome"] == "FN", "stop_overhead_kj"].sum()
    st.markdown(
        f"""
| Outcome | Count | Stop-overhead energy | Operational meaning |
|---|:---:|---|---|
| True positive (TP)  | {tp} | {tp_kj:.1f} kJ saved | Saturated cabin correctly bypassed |
| True negative (TN)  | {tn} | 0 | Non-saturated cabin correctly accepted |
| False positive (FP) | {fp} | 0 | No incoming passenger wrongly skipped |
| False negative (FN) | {fn} | {fn_kj:.1f} kJ foregone | Wasted stop at a saturated cabin |
| **Total** | **{total}** | **Service rate = {service_rate * 100:.1f} %** | — |
"""
    )

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
            seed_mode = st.radio(
                "Call-stream seed",
                ["Random 🎲", "Fixed (=42)"],
                horizontal=True,
                index=0,
                help=(
                    "Random — generate a fresh seed on every click, so the "
                    "1,000-call stream is different each run.  "
                    "Fixed — reuse seed = 42, which reproduces the thesis "
                    "numbers exactly."
                ),
            )

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
            # Resolve the seed at click time so a single random click produces
            # one call stream (further reruns of this widget alone don't
            # regenerate it; only a fresh click does).
            import random as _random

            sim_seed = _random.randint(0, 2_000_000_000) if seed_mode.startswith("Random") else 42
            with st.spinner(
                f"Running simulation → {run_name} (seed={sim_seed}) (this can take a minute) …"
            ):
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
                    seed=sim_seed,
                )
            if ok:
                st.success(
                    f"Simulation finished → results/simulation/{run_name}/  (seed={sim_seed})"
                )
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

    # Decision accuracy + object-level counting metrics + energy.
    bypass_acc = "—"
    counting_f1 = "—"
    counting_gt = counting_pred = counting_tp = counting_fn = counting_fp = "—"
    counting_recall = counting_precision = "—"
    if md_rep.exists():
        text = md_rep.read_text(encoding="utf-8")
        import re

        m1 = re.search(r"Bypass accuracy[^\d]*([\d.]+)", text)
        if m1:
            bypass_acc = f"{float(m1.group(1)) * 100:.1f}%"

        # Parse the OVERALL row of the object-level counting table.
        # Row format: "| **OVERALL** | **440** | **434** | **391** | **49** | **43** | **0.889** | **0.901** | **0.895** |"
        ov = re.search(
            r"\|\s*\*?\*?OVERALL\*?\*?\s*\|"
            r"\s*\*?\*?(\d+)\*?\*?\s*\|"  # GT
            r"\s*\*?\*?(\d+)\*?\*?\s*\|"  # Pred
            r"\s*\*?\*?(\d+)\*?\*?\s*\|"  # TP
            r"\s*\*?\*?(\d+)\*?\*?\s*\|"  # FN
            r"\s*\*?\*?(\d+)\*?\*?\s*\|"  # FP
            r"\s*\*?\*?([\d.]+)\*?\*?\s*\|"  # Recall
            r"\s*\*?\*?([\d.]+)\*?\*?\s*\|"  # Precision
            r"\s*\*?\*?([\d.]+)\*?\*?\s*\|",  # F1
            text,
        )
        if ov:
            counting_gt = int(ov.group(1))
            counting_pred = int(ov.group(2))
            counting_tp = int(ov.group(3))
            counting_fn = int(ov.group(4))
            counting_fp = int(ov.group(5))
            counting_recall = float(ov.group(6))
            counting_precision = float(ov.group(7))
            counting_f1 = float(ov.group(8))

    st.markdown("### Headline metrics")
    st.caption(
        "Detection and decision quality on the 68-image curated set. "
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
    counting_value = f"{counting_f1 * 100:.1f}%" if isinstance(counting_f1, float) else "—"
    counting_delta = (
        f"{counting_tp} of {counting_gt} objects correctly counted"
        if isinstance(counting_tp, int)
        else None
    )
    m2.metric(
        "🔢 Counting F1 (object-level)",
        counting_value,
        counting_delta,
        help=(
            "Per-object retrieval quality across all four classes. "
            "Each ground-truth instance is one retrieval target, so TP = correctly counted, "
            "FN = missed, FP = spurious detection. F1 = 2·precision·recall / (precision + recall). "
            "This replaces the older strict image-level exact-match (which failed an entire image "
            "on a single off-by-one count); the per-object view is the same metric reported "
            "in Table 4.5a of the report."
        ),
    )
    m3.metric(
        "⏭ Calls bypassed",
        energy.get("smart_bypassed_calls", "—"),
        f"of {energy.get('num_calls', '—')} calls",
    )

    # Per-object breakdown row
    if isinstance(counting_tp, int):
        st.caption(
            f"**Counting breakdown** — Ground-truth objects: {counting_gt}  ·  "
            f"Predicted: {counting_pred}  ·  ✅ TP {counting_tp}  ·  "
            f"❌ FN {counting_fn} (missed)  ·  ⚠ FP {counting_fp} (spurious)  ·  "
            f"Recall {counting_recall:.3f}  ·  Precision {counting_precision:.3f}"
        )

    with st.expander("What do these accuracies mean?"):
        st.markdown(
            "We report the system's correctness on **two independent levels**:\n\n"
            "1. **Decision accuracy** — *does the controller make the "
            "right call?* The cabin is either *full* or *not full*; the "
            "controller is correct whenever it agrees with the ground "
            "truth on this binary question.\n\n"
            "2. **Counting F1 (object-level)** — *does the underlying detector "
            "see the right number of objects?* Every ground-truth instance is treated as a "
            "retrieval target: a correct count is a true positive, a missed instance is a "
            "false negative, and a spurious detection is a false positive. F1 is the harmonic "
            "mean of recall and precision over the whole 68-image set. This is a more "
            "informative metric than the older strict image-level exact-match, which failed "
            "an entire image on any one-instance deviation in any class.\n\n"
            "A high decision accuracy with a low counting F1 means the system often "
            "arrives at the right call via wrong counts (*spurious correctness*). "
            "Reporting both prevents that shortcut from inflating the headline number."
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

    These values drive the Batch Simulation tab (and the subprocess that
    ``scripts.run_simulation`` spawns) — change a slider once and the
    next run honours it.
    """
    with st.sidebar:
        st.header("Cabin geometry")
        width_m = st.number_input("Width (m)", 0.5, 3.0, value=1.4, step=0.05)
        depth_m = st.number_input("Depth (m)", 0.5, 3.0, value=1.6, step=0.05)
        cabin_m2 = width_m * depth_m
        st.caption(f"Floor area = **{cabin_m2:.2f} m²**")

        st.header("Load")
        max_weight = st.number_input("Rated load (kg)", 200.0, 2500.0, value=630.0, step=10.0)

        st.header("Detection")
        conf_threshold = st.slider("YOLO confidence", 0.10, 0.90, 0.40, 0.05)

        st.header("Decision thresholds")
        weight_threshold = st.slider("Weight bypass τ_W", 0.50, 1.00, 0.80, 0.05)
        area_threshold = st.slider("Area bypass τ_A", 0.50, 1.00, 0.90, 0.05)

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
        "conf_threshold": conf_threshold,
        "weight_threshold": weight_threshold,
        "area_threshold": area_threshold,
    }


def main() -> None:
    st.set_page_config(
        page_title="Smart Elevator CV",
        page_icon="🛗",
        layout="wide",
    )
    st.title("🛗 Smart Elevator CV — Live Demo")
    st.markdown(
        "Run the curated ground-truth set in `data/sim/` through the "
        "trained detectors, then inspect bypass-decision accuracy, "
        "object-level counting metrics and cumulative energy savings."
    )

    cfg = render_sidebar()

    tab_batch, tab_timeline = st.tabs(["Batch Simulation", "Call Timeline"])
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
