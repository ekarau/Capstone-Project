# Smart Elevator CV

Computer-vision system that estimates how full an elevator cabin is from a single CCTV frame, then decides whether the next hall call should be served or skipped to save energy.

| | |
|---|---|
| Authors | Ege Karaurgan, Vedat Efe Gezer |
| Advisor | Assoc. Prof. Dr. Bahman |
| Institution | Istinye University, Department of Software Engineering |
| Project | Capstone (research prototype) |
| License | Proprietary, see [`LICENSE`](LICENSE) |

## Why

Most elevators only listen to the weight sensor. A cabin can already be unusable (people pressed against the door, a stroller blocking the entrance) long before the load cell reaches its limit, so the elevator keeps stopping at floors where nobody can actually fit. The result is wasted travel, longer wait times for everyone else, and avoidable energy use.

We add a second signal: **what the camera sees**. If the floor is too full, the controller bypasses the call instead of pretending there is room.

## How it works

```
CCTV frame
    │
    ├──► YOLOv8 four-class detector ──► persons, strollers, luggage, boxes
    │
    └──► YOLOv8 head detector       ──► person count (occlusion-resilient)
                │
                ▼
       per-class footprint area  (TS EN 81-20 standard footprints)
                │
                ▼
         occupancy ratio  ρ = A_occupied / A_cabin
                │
                ▼
          two-stage decision
       (weight gate + area gate)
                │
                ▼
        ACCEPT / BYPASS / BYPASS
```

### Detection

Two detectors run in parallel:

* A **four-class** YOLOv8s recognises *person*, *stroller*, *luggage*, *box*. Trained on a unified dataset of about 13 000 cabin images merged from ten public sources, with a leakage-safe split that keeps frames from the same video out of train/val/test.
* A **head-only** YOLOv8s focuses on heads in top-down camera angles. Trained on the OverHead Head Detection corpus (~6 000 images), specifically because heads stay visible in crowded cabins where bodies overlap.

In *hybrid* mode the head model supplies the person count, and the four-class model provides the non-human classes. The 4-class model's `person` predictions are deliberately discarded to avoid double-counting.

### Occupancy

We do not need pixel-perfect bounding boxes for the floor area calculation, just per-class counts:

$$
A_{\text{occupied}} = \sum_{c} n_c \cdot \bar{a}_c, \qquad \rho = \min\!\left(\frac{A_{\text{occupied}}}{A_{\text{cabin}}},\ 1\right)
$$

with class footprints anchored to published standards or industry benchmarks:

| Class    | $\bar{a}_c$ (m²) | Source |
|----------|:---:|---|
| person   | 0.20 | ISO 8100-32:2020 §6.4 specifies passenger area $A_p \in [0.17, 0.22]$ m² depending on rated load. EN 81-20:2020 §5.4.2.1.1 uses 0.17 m² for the rated-mass method. The mid-range 0.20 m² value is the conventional figure used in elevator capacity calculations (Tukia et al., 2018). |
| stroller | 0.45 | EN 1888-1:2018 governs single-pushchair safety and dimensions. Product survey: Bugaboo Butterfly 56 × 40 cm ≈ 0.22 m², UPPAbaby Vista 91 × 65 cm ≈ 0.60 m²; population mean ≈ 0.45 m². |
| luggage  | 0.20 | IATA Resolution 753 cabin-baggage standard: 56 × 36 × 23 cm → footprint 0.20 m². Adopted as the canonical mid-size value. |
| box      | 0.20 | Industry e-commerce parcel mean ≈ 46 × 41 × 15 cm → footprint ≈ 0.19 m² (Red Stag Fulfillment, 2026 benchmark). |

The model ignores object positions — two passengers standing shoulder-to-shoulder are still counted as $2 \times 0.20$ m². The constants come straight from accessibility codes, so the numbers transfer cleanly into the thesis methodology section.

### Decision

A two-stage gate, in this order:

1. If the load cell reports $W \ge \tau_W \cdot W_{\text{rated}}$, bypass on weight.
2. Else if visual occupancy $\rho \ge \tau_A$, bypass on area.
3. Otherwise, accept the call.

Defaults: $\tau_W = 0.80$, $\tau_A = 0.90$. The weight gate runs first so the cheap load reading short-circuits the more expensive vision pipeline whenever it can.

### Energy

The simulation charges every accepted hall call with the elevator energy required to actually deliver that cabin's load over the average traversal distance. The Tukia et al. (2018) model is used in **per-call dynamic mode**: instead of one fleet-average stop, each accepted call is priced by its own load and that load is built up from the labelled object counts:

$$
m_{\text{cabin}}(d) = n_{\text{person}} \bar{m}_{\text{person}} + n_{\text{stroller}} \bar{m}_{\text{stroller}} + n_{\text{luggage}} \bar{m}_{\text{luggage}} + n_{\text{box}} \bar{m}_{\text{box}}
$$

with literature-anchored per-class masses:

| Class    | $\bar{m}_c$ (kg) | Source |
|----------|:---:|---|
| person   | 75 | EN 81-20:2020 / ISO 8100-1 rated mass per passenger; same value used by Tukia et al. (2018) |
| stroller | 20 | Empty single stroller 7–12 kg (EN 1888-1:2018 + product survey) plus typical occupant child 10–12 kg |
| luggage  | 15 | IATA cabin allowance ~ 8 kg, checked baggage typically 15–23 kg; mixed elevator distribution ≈ 15 kg |
| box      |  5 | E-commerce parcel mean 1–3 kg (Red Stag Fulfillment 2026); larger logistics cartons reach ~ 10 kg |

A heavy cabin therefore costs proportionally more motor energy than a light one, so bypassing a full cabin saves substantially more joules than bypassing a near-empty one. Each accepted stop also incurs the standard door-cycle and stop-idle terms from the Tukia model. Bypass saves the entire stop.

## Results

Two configurations were measured on a curated set of 29 cabin photographs covering empty, mixed, and at-capacity scenarios. Each photo carries multi-class ground truth (`gt_person`, `gt_stroller`, `gt_luggage`, `gt_box`). 1 000 synthetic hall calls were sampled uniformly from this set.

| Configuration | Bypass acc. | Precision | Recall | F1 | Person MAE | **Energy saved** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (4-class only)   | 0.828 | 0.667 | 0.750 | 0.706 | 1.38 | 29.4 % |
| **Hybrid (+ head model)** | 0.759 | 0.533 | **1.000** | 0.696 | **1.21** | **50.1 %** |
| Δ                         | −0.07 | −0.13 | +0.25 | ≈ 0 | −0.17 | **+20.7 pp** |

The hybrid configuration trades a small precision loss for **perfect recall** (no full cabin is ever missed) and roughly doubles the energy savings. False positives in hybrid mode trace back to the head detector slightly over-counting (≈ 2–3 phantom heads per frame) on AI-rendered images that contain head-shaped artefacts. Tightening `head_conf_threshold` (default 0.25) is documented as future tuning work.

Detection metrics for the underlying models on their own validation splits:

| Model | Precision | Recall | mAP\@50 | mAP\@50–95 |
|---|:---:|:---:|:---:|:---:|
| 4-class (best_v2.pt)  | 0.953 | 0.822 | 0.877 | 0.667 |
| Head (best_head.pt)   | 0.852 | 0.692 | 0.767 | 0.519 |

## Repository layout

```
configs/      YAML for cabin geometry, thresholds, model hyper-parameters
data/         raw downloads (gitignored), unified YOLO dataset, simulation set
demo/         Streamlit UI — single-frame analysis + batch simulation viewer
notebooks/    Colab-ready notebooks: dataset audit, training, head training, demos
src/
  dataset/    audit, unification, augmentation
  detection/  YOLOv8 wrappers (training + inference)
  perception/ homography, BEV, three occupancy estimators
  energy/     Tukia 2018 power model
  control/    two-stage hall-call decision
  simulation/ baseline-vs-smart synthetic comparison
  utils/      logging, config loader, calibration helpers
scripts/      command-line entry points (dataset prep, packaging, demo, sim)
tests/        smoke tests for each module
tools/        manual calibration helpers
models/weights/  trained checkpoints (gitignored — distributed separately)
results/      generated figures, CSVs, simulation reports
```

## Quick start

Tested on Python 3.10–3.12, Windows 11 and Ubuntu 22.04.

```bash
git clone git@github.com:ekarau/Capstone-Project.git
cd Capstone-Project
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Build the unified four-class dataset from raw downloads
python -m scripts.prepare_dataset --raw data/raw --out data/unified

# Train (Colab notebooks/02_train.ipynb is much faster than CPU)
python -m src.detection.train --data data/unified/data.yaml --preset balanced

# Single-image demo
python -m scripts.demo --image path/to/cabin.jpg --weights models/weights/best.pt
```

Pretrained `best.pt` and `best_head.pt` (~22 MB each) are distributed
on request — contact the authors.

### Reproducing the energy simulation

```bash
# Baseline — four-class detector only
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --rated-capacity 8 \
    --num-calls 1000 \
    --output results/simulation/baseline

# Hybrid — adds the head detector for the person count
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --head-weights models/weights/best_head.pt \
    --rated-capacity 8 \
    --num-calls 1000 \
    --output results/simulation/hybrid
```

Both runs write a confusion matrix PNG, per-image and per-class CSVs,
and a Markdown summary report.

### Streamlit demo

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

Two tabs: *Single Frame* (upload a cabin photo, sweep the thresholds,
read off the decision) and *Batch Simulation* (auto-loads whatever
`results/simulation/` has produced).

## Limitations

* **Synthetic test set.** The 29 cabin images used in the simulation are AI-generated. They cover the full occupancy spectrum, but they are not real CCTV footage — domain gap to a deployed camera should be expected. Re-running the same protocol on real footage from a target building is the natural next step.
* **No homography.** The class-footprint occupancy model ignores where each object sits on the floor. When two passengers stand shoulder-to-shoulder, the model still adds the full 0.20 m² twice. The repository ships `FootprintOccupancy` and `BEVMaskOccupancy` (`src/perception/occupancy.py`) for the homography-based alternative — both only need four manually clicked floor corners per cabin.
* **Hybrid over-counting.** The head detector adds ~ 2–3 phantom heads per AI-generated image, dragging hybrid precision to 0.53. Lifting `head_conf_threshold` from 0.25 toward 0.40 should recover most of the lost precision; this is left as a tuning study.
* **Single-frame inference.** Multi-frame tracking (BoT-SORT, ByteTrack) would prevent the same passenger from being counted on consecutive frames if the system is later wired to a video stream.

## References

1. **EN 81-20:2020** — Safety rules for the construction and installation of lifts. European Committee for Standardization.
2. **ISO 8100-32:2020** — Lifts for the transportation of persons and goods, Part 32: Planning and selection of passenger lifts. International Organization for Standardization.
3. **EN 1888-1:2018** — Wheeled child conveyances: pushchairs and prams. European Committee for Standardization.
4. **IATA Resolution 753** — Cabin baggage standard. International Air Transport Association.
5. **Tukia, T. et al. (2018)** — High-resolution modelling of elevator power consumption. *Journal of Building Engineering*.
6. **Andrei, A. & Ruokokoski, J. (2022)** — Load- and area-based elevator group control with computer-vision occupancy sensing.
7. **Shao, S. et al. (2018)** — CrowdHuman: a benchmark for detecting human in a crowd. arXiv:1805.00123.
8. **Mohamudally, N. et al. (2015)** — Floor occupancy estimation in smart buildings.

## Citation

For academic citation, use the metadata in [`CITATION.cff`](CITATION.cff).

## License

Proprietary. Viewing and citation are permitted; reproduction, modification, redistribution, commercial use, and deployment in any safety-critical system are not. Full terms in [`LICENSE`](LICENSE).
