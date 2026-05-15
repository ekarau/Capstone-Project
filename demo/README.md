# Smart Elevator CV — Streamlit Demo

Interactive web UI for the **3-class + head** detector pipeline (object detector for stroller / luggage / box + head detector for person) and the EN-81-20-grounded area model. Designed for thesis demonstrations and rapid scenario testing — change cabin geometry, current load, or detection thresholds in the sidebar and watch the bypass decision update on a single uploaded frame.

## What you'll see

- **Sidebar.** Cabin width / depth, rated load, current weight, YOLO confidence, weight and area bypass thresholds, per-class footprint overrides.
- **Main panel.** Upload an image (or pick a bundled test sample), click *Analyze*, and read off
  - the annotated frame with bounding boxes,
  - the ACCEPT / BYPASS (area) / BYPASS (weight) decision badge with a one-line justification,
  - per-class detection counts and their cumulative footprint,
  - the floor-occupancy and weight gauges.

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

| Sidebar field           | Effect                                                   | Default |
|-------------------------|----------------------------------------------------------|:---:|
| Width / Depth (m)       | Recomputes cabin floor area `A_cabin = W × D`            | 1.4 / 1.6 |
| Rated load (kg)         | Denominator of the weight-bypass ratio                   | 630 |
| Current cabin weight    | Numerator of the weight-bypass ratio                     | 0 |
| YOLO confidence         | Detection threshold; lower = more recall, more FPs       | 0.40 |
| Weight bypass τ_W       | Bypass when `W ≥ τ_W · W_rated`                          | 0.80 |
| Area bypass τ_A         | Bypass when `ρ ≥ τ_A`                                    | 0.90 |
| Per-class footprints    | Override the per-class defaults                          | see app |

## Notes for the jury demo

- **Refresh the sample dropdown** by uploading a custom CCTV image; the test set under `data/unified/test/images/` is also auto-discovered if present.
- **Models load once** (cached) — both the object detector and the head detector are kept in memory between reruns.
- **Stage-1 weight gate fires before vision.** When the weight slider exceeds `τ_W`, the detector still runs (so you can show the cabin contents) but the decision is fixed regardless of occupancy.
- **No persistent storage.** Closing the tab discards uploaded frames; nothing is logged or sent off-device.

## Reference

Karaurgan, E. & Gezer, V. E. (2026). *Smart Elevator CV: A Computer-Vision-Based Approach to Energy-Efficient Elevator Control.* Capstone Project, Istinye University. Advisor: Assoc. Prof. Dr. Bahman.
