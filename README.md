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
    ├──► YOLOv8 three-class object detector ──► strollers, luggage, boxes
    │
    └──► YOLOv8 head detector               ──► person count (occlusion-resilient)
                │
                ▼
       per-class footprint area
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

* A **three-class** YOLOv8s object detector recognises *stroller*, *luggage*, *box*. Trained on a leakage-safe unified dataset merged from multiple public sources, with the split designed to keep frames from the same source video out of train/val/test.
* A **head-only** YOLOv8s focuses on heads in top-down camera angles. Trained on a single-class head corpus (~6 000 images), specifically because heads stay visible in crowded cabins where bodies overlap.

The two detectors are mutually exclusive in the classes they handle: the head model supplies the person count, the object detector supplies the non-human classes (stroller, luggage, box). Their outputs are concatenated into the four operational output classes without any risk of double counting.

### Occupancy

We do not need pixel-perfect bounding boxes for the floor area calculation, just per-class counts:

$$
A_{\text{occupied}} = \sum_{c} n_c \cdot \bar{a}_c, \qquad \rho = \min\!\left(\frac{A_{\text{occupied}}}{A_{\text{cabin}}},\ 1\right)
$$

with class footprints anchored to industry-standard cabin design and product dimensions:

| Class    | $\bar{a}_c$ (m²) | Notes |
|----------|:---:|---|
| person   | 0.20 | Conventional per-passenger cabin allowance used in capacity calculations. |
| stroller | 0.45 | Mid-range single pushchair (≈ 90 × 50 cm) including the occupant child. |
| luggage  | 0.20 | Mid-size cabin / check-in suitcase footprint (≈ 56 × 36 cm). |
| box      | 0.20 | Average e-commerce / logistics carton (≈ 50 × 40 cm). |

The model ignores object positions — two passengers standing shoulder-to-shoulder are still counted as $2 \times 0.20$ m². The constants are taken from common cabin-design conventions, so the numbers transfer cleanly into the thesis methodology section.

### Decision

A two-stage gate, in this order:

1. If the load cell reports $W \ge \tau_W \cdot W_{\text{rated}}$, bypass on weight.
2. Else if visual occupancy $\rho \ge \tau_A$, bypass on area.
3. Otherwise, accept the call.

Defaults: $\tau_W = 0.80$, $\tau_A = 0.90$. The weight gate runs first so the cheap load reading short-circuits the more expensive vision pipeline whenever it can.

The simulation evaluates **three policies** on the same call stream:

| Policy | Stage 1 (weight) | Stage 2 (area) | What it represents |
|---|:---:|:---:|---|
| `always_accept` | — | — | Naive baseline that never bypasses (quantifies the worst case). |
| `weight_only`   | ✓ | — | Current-industry load-bypass system on its own. |
| **`smart` (ours)** | ✓ | ✓ | Algorithm 1: weight gate, then vision area gate. |

The headline result reported in the thesis is **smart vs weight-only** — the energy and stop-time the area gate adds on top of what a load-cell-only system already saves.

### Energy

The simulation charges every accepted hall call with the elevator energy required to actually deliver that cabin's load over the average traversal distance. Each accepted call is priced by its own load and that load is built up from the labelled object counts:

$$
m_{\text{cabin}}(d) = n_{\text{person}} \bar{m}_{\text{person}} + n_{\text{stroller}} \bar{m}_{\text{stroller}} + n_{\text{luggage}} \bar{m}_{\text{luggage}} + n_{\text{box}} \bar{m}_{\text{box}}
$$

with per-class average masses:

| Class    | $\bar{m}_c$ (kg) |
|----------|:---:|
| person   | 75 |
| stroller | 20 |
| luggage  | 15 |
| box      |  5 |

A heavy cabin therefore costs proportionally more motor energy than a light one, so bypassing a full cabin saves substantially more joules than bypassing a near-empty one. Each accepted stop also incurs the standard door-cycle and stop-idle terms. Bypass saves the entire stop.

## Results

The two-detector system was measured on **68 cabin photographs** covering the empty → at-capacity spectrum. Each photo carries multi-class ground truth (`gt_person`, `gt_stroller`, `gt_luggage`, `gt_box`). 1 000 synthetic hall calls were sampled uniformly from this set, and three policies (always-accept / weight-only / smart) were evaluated on the same call stream.

### Bypass-decision quality (smart policy vs optimal-policy ground truth)

Ground truth here is `gt_should_bypass = gt_is_full OR gt_weight_full`, i.e. the optimal policy that bypasses iff the cabin can no longer accept a passenger (either area-full or weight-full).

| Configuration | Bypass acc. | Precision | Recall | F1 | Person MAE |
|---|:---:|:---:|:---:|:---:|:---:|
| **Smart (proposed)** | **0.908** | **1.000** | **0.727** | **0.842** | **0.66** |

The smart policy records **zero false positives**: no incoming passenger was ever wrongly skipped. The remaining error mode is six false negatives on crowded cabins in which the head detector failed to recover enough heads to push occupancy above τ_A.

### Energy and stop-time savings (over 1 000 synthetic hall calls)

| Policy | Bypassed | Total stop-overhead energy | Δ vs always-accept | Δ vs weight-only |
|---|:---:|:---:|:---:|:---:|
| Always-accept (naive) | 0 | 920.0 kJ | — | — |
| Weight-only (current industry) | 126 | 804.1 kJ | 115.9 kJ (12.6 %) | — |
| **Smart (proposed)** | 226 | **712.1 kJ** | **207.9 kJ (22.6 %)** | **92.0 kJ (11.4 %)** |

| Policy | Total stop-time | Δ vs weight-only |
|---|:---:|:---:|
| Always-accept | 10 000 s (166.7 min) | — |
| Weight-only | 8 740 s (145.7 min) | — |
| **Smart (proposed)** | **7 740 s** (129.0 min) | **1 000 s = 16.7 min (11.4 %)** |

The headline result is **smart vs weight-only**: the area gate adds **11.4 % extra energy / time savings on top of a current-industry load-cell-only system**. Service rate is **90.5 %** — but crucially, **0 of those 95 missed calls are passengers who were wrongly skipped**; every error is a wasted stop on an already-full cabin, which costs only a door cycle.

### Underlying detection metrics (validation splits)

| Model | Precision | Recall | mAP\@50 | mAP\@50–95 |
|---|:---:|:---:|:---:|:---:|
| 3-class object (`best.pt`) | 0.953 | 0.822 | 0.877 | 0.667 |
| Head (`best_head.pt`)      | 0.852 | 0.692 | 0.767 | 0.519 |

## Repository layout

```
configs/      YAML for cabin geometry, thresholds, model hyper-parameters
data/         raw downloads (gitignored), unified YOLO dataset, simulation set
demo/         Streamlit UI — single-frame analysis + batch simulation viewer
notebooks/    Colab-ready notebooks: dataset audit, training, head training, demos
src/
  dataset/    audit, unification, augmentation
  detection/  YOLOv8 wrappers (training + inference)
  perception/ occupancy estimator interface
  energy/     elevator power model
  control/    two-stage hall-call decision
  utils/      logging, config loader
scripts/      command-line entry points (dataset prep, packaging, simulation)
tests/        smoke tests for each module
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

# Build the unified three-class object-detector dataset from raw downloads
python -m scripts.prepare_dataset --raw data/raw --out data/unified

# Train (Colab notebooks/02_train.ipynb is much faster than CPU)
python -m src.detection.train --data data/unified/data.yaml --preset balanced
```

### Pretrained weights

Download the two trained checkpoints (~22 MB each) from the
[v0.2.0 release page](https://github.com/ekarau/Capstone-Project/releases/tag/v0.2.0)
and place them under `models/weights/`:

| File | Role | Target path |
|---|---|---|
| `best.pt` | three-class object detector (stroller, luggage, box) | `models/weights/best.pt` |
| `best_head.pt` | head detector (person count) | `models/weights/best_head.pt` |

Or download them programmatically:

```bash
mkdir -p models/weights
curl -L -o models/weights/best.pt \
  https://github.com/ekarau/Capstone-Project/releases/download/v0.2.0/best.pt
curl -L -o models/weights/best_head.pt \
  https://github.com/ekarau/Capstone-Project/releases/download/v0.2.0/best_head.pt
```

### Reproducing the energy simulation

Both commands evaluate the **three policies** (always-accept / weight-only / smart) on the same 1 000-call stream and write a confusion matrix PNG, per-image and per-class CSVs, an `energy_savings.csv` with the per-policy aggregates, a `call_log.csv` per-call timeline, and a Markdown summary report under the chosen output directory.

```bash
# Smart policy — three-class object detector + head detector
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --head-weights models/weights/best_head.pt \
    --rated-capacity 8 \
    --conf-threshold 0.40 \
    --head-conf 0.40 \
    --num-calls 1000 \
    --output results/simulation/smart_67
```

The weight-bypass threshold defaults to `0.80 × 630 kg = 504 kg` (override with `--rated-load-kg` / `--weight-bypass-ratio`).

### Streamlit demo

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

Two tabs: *Single Frame* (upload a cabin photo, sweep the thresholds,
read off the decision) and *Batch Simulation* (auto-loads whatever
`results/simulation/` has produced).

## Limitations

* **Synthetic test set.** The 68 cabin images used in the simulation are publicly-available frames augmented with AI-generated cabin photos (ChatGPT, Gemini text-to-image). They cover the full occupancy spectrum, but they are not real CCTV footage — a domain gap to a deployed camera should be expected. Real CCTV footage was not collected because no IRB / ethics-committee approval was obtained within the project scope. Re-running the same protocol on real footage from a target building is the natural next step.
* **No homography.** The class-footprint occupancy model ignores where each object sits on the floor. When two passengers stand shoulder-to-shoulder, the model still adds the full 0.20 m² twice. A position-aware alternative — for example a homography-based union of per-class disks or a birds-eye-view occupancy mask — would only require four manually clicked floor corners per cabin and is listed as future work.
* **Single-frame inference.** Multi-frame tracking (BoT-SORT, ByteTrack) would prevent the same passenger from being counted on consecutive frames if the system is later wired to a video stream. The current pipeline reads a single CCTV frame, matching the project's intended deployment.
* **No real-time benchmark.** Inference latency / FPS has not been profiled on a target edge device; "real-time" claims in the literature review are inherited from the YOLOv8 architecture and not measured for this prototype.
* **No traffic simulator.** Average Waiting Time (AWT) is not directly measured. The reported "stop-time saved" metric is the total per-stop overhead avoided across all bypassed calls, not the wait-time of an individual passenger queue. A full traffic-simulator integration (Elevate®-style, Barney 2003) is left as future work.

## Citation

For academic citation, use the metadata in [`CITATION.cff`](CITATION.cff).

## License

Proprietary. Viewing and citation are permitted; reproduction, modification, redistribution, commercial use, and deployment in any safety-critical system are not. Full terms in [`LICENSE`](LICENSE).
