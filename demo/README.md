# Smart Elevator CV — Streamlit Demo

Interactive web UI for the **3-class + head** detector pipeline (object detector for stroller / luggage / box + head detector for person) and the EN-81-20-grounded area model. Designed for thesis demonstrations and reproducible scenario sweeps — change cabin geometry, rated load, or detection thresholds in the sidebar and re-run the curated 68-image simulation; both views (Batch Simulation and Call Timeline) refresh against the new settings.

## What you'll see

- **Sidebar.** Cabin width / depth, rated load, YOLO confidence, weight and area bypass thresholds.
- **Batch Simulation tab.** Run the curated ground-truth set in `data/sim/` through the trained detectors, then inspect
  - call-level bypass-decision accuracy (TP / TN / FP / FN, service rate),
  - the outcome breakdown with energy attribution (mirrors Report Table 4.6),
  - object-level counting metrics (recall / precision / F1 per class),
  - cumulative energy savings curves for the three policies.
- **Call Timeline tab.** Replay individual hall calls from any saved run, filtered by outcome class (TP / TN / FP / FN) to inspect failure cases image by image.

## Quick start

```bash
# From the project root, on the `demo` branch:
pip install -e ".[demo]"          # installs streamlit on top of the base deps
streamlit run demo/app.py
```

Streamlit opens the demo at `http://localhost:8501`.

## Requirements

- Trained checkpoints at `models/weights/best.pt` (object detector) and `models/weights/best_head.pt` (head detector). The v0.2.0 release ships these separately — contact the authors.
- Python 3.10 – 3.12.
- All base dependencies plus `streamlit>=1.36`. Both are covered by the `demo` extra in `pyproject.toml`.

## Configuration cheatsheet

| Sidebar field           | Effect                                                                          | Default |
|-------------------------|---------------------------------------------------------------------------------|:---:|
| Width / Depth (m)       | Recomputes cabin floor area `A_cabin = W × D`                                   | 1.4 / 1.6 |
| Rated load (kg)         | Derives rated capacity (≈ kg / 75) and feeds it to the simulation subprocess    | 630 |
| YOLO confidence         | Detection threshold for the object detector; lower = more recall, more FPs     | 0.40 |
| Weight bypass τ_W       | Bypass when `W ≥ τ_W · W_rated`                                                 | 0.80 |
| Area bypass τ_A         | Bypass when `ρ ≥ τ_A`                                                           | 0.90 |

## Notes for the jury demo

- **Models load lazily inside the simulation subprocess** — the first `Run simulation now` therefore takes a little longer while the YOLOv8 weights are mapped into the worker process.
- **Stage-1 weight gate fires before vision.** The simulator evaluates the load-cell gate first and only invokes the detectors on calls that pass it, mirroring `scripts/run_simulation.py`.
- **No persistent storage of CCTV frames.** Runs live under `results/simulation/<run_name>/`; closing the tab does not discard them, so you can re-open the dashboard and reselect any earlier run from the dropdown.

## Reference

Karaurgan, E. & Gezer, V. E. (2026). *Smart Elevator CV: A Computer-Vision-Based Approach to Energy-Efficient Elevator Control.* Capstone Project, Istinye University. Advisor: Assoc. Prof. Dr. Bahman.
