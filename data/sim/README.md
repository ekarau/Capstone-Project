# Energy-Saving Simulation — Test Image Set

This directory hosts the curated cabin photographs and their ground-truth
labels used by `scripts/run_simulation.py`.

## Layout

```
data/sim/
├── images/                       ← put your 20–30 cabin photos here
│   ├── cabin_001.jpg
│   ├── cabin_002.jpg
│   └── ...
├── ground_truth.csv              ← labels for each image (you fill this in)
└── ground_truth_template.csv     ← copy this and start labeling
```

## How to label

1. Capture 20–30 cabin photographs covering the full occupancy spectrum:
   - ~10 **empty / lightly occupied** scenes (0–3 passengers)
   - ~10 **medium** scenes (4–6 passengers)
   - ~10 **at-or-above capacity** scenes (≥ ⌈0.85 × rated⌉ passengers)
2. Drop them into `data/sim/images/` with consistent filenames.
3. Copy the template:
   ```
   cp ground_truth_template.csv ground_truth.csv
   ```
4. For each image, edit `ground_truth.csv`:
   - `filename`: the image filename
   - `gt_person`: number of people physically present
   - `gt_stroller`: number of strollers
   - `gt_luggage`: number of luggage items
   - `gt_box`: number of boxes / cartons
   - `gt_is_full`: leave **blank** to auto-derive from the multi-class
     occupancy ratio, or write `True` / `False` to override

The script computes the true occupancy as

```
occupancy = (gt_person × 0.20) + (gt_stroller × 0.45)
          + (gt_luggage × 0.20) + (gt_box × 0.20)   [in m²]
```

and flags the cabin as full when `occupancy / cabin_area ≥ 0.90`. Footprint
values are anchored to ISO 8100-32:2020 (person), EN 1888-1:2018 (stroller),
IATA Resolution 753 (luggage), and industry e-commerce parcel benchmarks
(box) — see the project README for full citations.

## Run the simulation

```bash
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --rated-capacity 8 \
    --num-calls 1000 \
    --output results/simulation/baseline
```

Hybrid mode (with a separately trained head detector):

```bash
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --head-weights models/weights/best_head.pt \
    --rated-capacity 8 \
    --num-calls 1000 \
    --output results/simulation/hybrid
```

The script writes `confusion_matrix.png`, `per_image_decisions.csv`,
`per_class_detection.csv`, `energy_savings.csv` and `report.md` under the
chosen output directory.
