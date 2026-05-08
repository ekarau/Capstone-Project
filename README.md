# Smart Elevator CV

> **Computer-vision-based occupancy sensing and bypass control for energy-efficient elevators.**
> A four-class YOLOv8 detector estimates how full an elevator cabin is from a single CCTV frame, and an EN-81-20-grounded area model decides whether incoming hall calls should be served or skipped.

| | |
|---|---|
| **Authors**       | Ege Karaurgan · Vedat Efe Gezer |
| **Advisor**       | Assoc. Prof. Dr. Bahman |
| **Institution**   | Istinye University — Department of Software Engineering |
| **Project type**  | Capstone project (research prototype) |
| **License**       | Proprietary — see [`LICENSE`](LICENSE) |

---

## What this project does

Conventional elevator controllers stop at every hall call until the **weight sensor** says the cabin is full. That decision is too late: a cabin can be physically saturated (people standing shoulder-to-shoulder, a stroller blocking the door) long before the load cell trips. Smart Elevator CV adds a second signal — **visual floor occupancy** — and skips ("bypasses") calls when the cabin no longer has usable area, even if it is below the weight limit.

The end-to-end pipeline is:

```
CCTV frame ──► YOLOv8 detector ──► per-class footprint area ──► occupancy ratio ──► bypass decision
                (person, stroller,    (TS EN 81-20 / ISO 8100      (Σ aᵢ / A_cabin)        (PDF Algorithm 1)
                 luggage, box)         standard footprints)
```

## Methodology in one minute

**Detection.** A YOLOv8-s model is fine-tuned on a unified four-class dataset (≈ 13 k training images, leakage-free split) covering persons, strollers, luggage, and boxes inside lifts.

**Occupancy estimation.** Because elevator CCTV is mounted in a top corner with a fish-eye lens, only heads and upper bodies are visible — full-body bounding boxes are physically impossible. We therefore use a **class-based footprint model**:

$$
A_\mathrm{occupied} = \sum_{c \in \mathcal{C}} n_c \cdot \bar{a}_c, \qquad \rho = \min\!\left(\frac{A_\mathrm{occupied}}{A_\mathrm{cabin}},\; 1\right)
$$

where $n_c$ is the number of detections of class $c$ and $\bar{a}_c$ is the standard floor area for that class:

| Class    | $\bar{a}_c$ (m²) | Source |
|----------|:---:|---|
| person   | 0.20 | TS EN 81-20:2020 §5.4.2.1.1 — "available car area per person" |
| stroller | 0.45 | Typical single-stroller footprint (~ 90 × 50 cm) |
| luggage  | 0.18 | IATA cabin / medium check-in mix |
| box      | 0.20 | Medium e-commerce / logistics carton (~ 50 × 40 cm) |

**Control.** A two-stage policy combines weight and area (PDF Algorithm 1, after Andrei & Ruokokoski, 2022):

1. If $W \ge \tau_W \cdot W_\mathrm{rated}$ → **bypass (weight)**.
2. Else, if $\rho \ge \tau_A$ → **bypass (area)**.
3. Otherwise → **accept**.

Defaults: $\tau_W = 0.80$, $\tau_A = 0.90$.

**Energy.** A power model after Tukia et al. (2018) translates "stops avoided" into kWh saved relative to a weight-only baseline.

## Repository layout

```
configs/                YAML configuration (cabin dimensions, thresholds, model)
data/
  raw/                  Source datasets (Roboflow downloads, untouched) — gitignored
  unified/              Merged YOLO dataset (train/val/test) — generated, gitignored
src/
  dataset/              Audit, unification, and augmentation of source datasets
  detection/            YOLOv8 training & inference wrappers
  perception/           Homography, bird's-eye view, and three occupancy estimators
  energy/               Tukia (2018) power-consumption model
  control/              PDF Algorithm 1 (weight + area bypass decision)
  simulation/           Baseline-vs-Smart synthetic comparison
  utils/                Logging, config loader, calibration helpers
notebooks/              Reproducible Colab notebooks (audit, train, demo, energy)
scripts/                Developer CLI entry points (dataset prep, packaging, demo)
tools/                  Manual calibration helpers (e.g. cabin-corner picker)
tests/                  Smoke tests for each module
models/weights/         Trained weights — gitignored (download separately)
results/                Generated figures, metrics, CSVs — partly gitignored
```

## Quick start

> Tested on Python 3.10 – 3.12, Windows 11 / Ubuntu 22.04.

```bash
# 1. Clone and create a virtual environment
git clone git@github.com:ekarau/Capstone-Project.git
cd Capstone-Project
python -m venv venv
source venv/bin/activate                # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# 2. Prepare the unified YOLO dataset (class remap + leakage-free split)
python -m scripts.prepare_dataset --raw data/raw --out data/unified

# 3. Train — Colab strongly recommended (notebooks/02_train.ipynb)
python -m src.detection.train --data data/unified/data.yaml --preset balanced

# 4. Run the end-to-end demo on a single image
python -m scripts.demo \
    --image path/to/cabin.jpg \
    --weights models/weights/best.pt
```

A pretrained checkpoint (`best.pt`, ≈ 22 MB) is distributed separately;
contact the authors for access.

### Cabin dimension override

```bash
python -m scripts.demo --image cabin.jpg \
    --cabin-width 1.4 --cabin-depth 1.6 --max-weight 630
```

Or via YAML:

```bash
python -m scripts.demo --image cabin.jpg --config configs/my_cabin.yaml
```

## Reproducing the trained model

The recommended path is the Colab notebook, which packages the dataset and code into two ZIPs and runs an end-to-end train + evaluate + occupancy-report flow on a free L4 GPU (≈ 3.7 hours for 81 epochs):

1. Locally: `python -m scripts.package_for_colab` → produces `code.zip` and `dataset.zip` under `Desktop/colab_upload/`.
2. Upload both to `Google Drive / MyDrive / Capstone /`.
3. Open `notebooks/02_train.ipynb` in Colab and run all cells.

## Results (v0.2.0, model `best_v2.pt`)

Held-out test set (367 images, sourced primarily from `Elevator.yolov8` plus sliced contributions from other corpora — never seen during training):

| Class    | Precision | Recall | mAP\@50 | mAP\@50–95 |
|----------|:---:|:---:|:---:|:---:|
| **all**      | 0.953 | 0.822 | **0.877** | 0.667 |
| person   | 0.945 | 0.626 | 0.736 | 0.451 |
| stroller | 0.966 | 0.972 | 0.982 | 0.832 |
| luggage  | 0.960 | 0.839 | 0.887 | 0.689 |
| box      | 0.940 | 0.850 | 0.904 | 0.695 |

Occupancy distribution on the test set (class-based estimator, $A_\mathrm{cabin} = 2.24\,\mathrm{m^2}$): mean **26.6 %**, median **29.0 %**, max **62.5 %**.

## Limitations and future work

- **Person recall on out-of-distribution footage drops to 0.63.** Mitigations: per-class confidence-threshold tuning, more diverse elevator footage in training, test-time augmentation.
- **The class-based footprint estimator cannot tell whether two people overlap** — it always sums $\bar{a}_c$. A more accurate alternative is the homography-based union-of-disks (`FootprintOccupancy`) or the rasterized BEV mask (`BEVMaskOccupancy`); both are implemented in `src/perception/occupancy.py` and only require the four cabin corners to be calibrated.
- **Pose estimation.** With an additional ceiling camera, YOLOv8-pose could replace bounding-box approximations with body silhouettes.
- **Tracking.** Multi-frame tracking (BoT-SORT / ByteTrack) would prevent over-counting when the same person is detected across frames.

## References

1. **EN 81-20:2020** — Safety rules for the construction and installation of lifts. European Committee for Standardization.
2. **Tukia, T. et al. (2018)** — High-resolution modeling of elevator power consumption. *Journal of Building Engineering*.
3. **Andrei, A. & Ruokokoski, J. (2022)** — Load-area-based elevator group control with computer-vision occupancy sensing.
4. **Mohamudally, N. et al. (2015)** — Floor occupancy estimation in smart buildings.

## Citation

If you reference this work in academic publications, please use the metadata in [`CITATION.cff`](CITATION.cff).

## License

This project is **proprietary**. See [`LICENSE`](LICENSE) for the full terms.
Briefly: viewing and academic citation are permitted; reproduction, modification, redistribution, commercial use, and deployment in any safety-critical system are not.
