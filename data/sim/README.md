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
├── ground_truth.csv              ← (you create this) labels for each image
└── ground_truth_template.csv     ← copy/edit this file
```

## How to label

1. Capture 20–30 cabin photographs covering the full occupancy spectrum:
   - ~10 **empty / lightly occupied** scenes (0–3 passengers)
   - ~10 **medium** scenes (4–6 passengers)
   - ~10 **at-or-above capacity** scenes (≥ ⌈0.85 × rated⌉ passengers)
2. Drop them into `data/sim/images/` with consistent filenames (e.g. `cabin_001.jpg`).
3. Copy the template:
   ```
   cp ground_truth_template.csv ground_truth.csv
   ```
4. For each image, edit `ground_truth.csv`:
   - `filename`: the image filename
   - `gt_count`: number of people physically present
   - `gt_is_full`: `True` if the cabin should bypass an inbound hall call.
     Default rule: `gt_is_full = (gt_count >= ceil(0.85 × rated_capacity))`.
     For an 8-person cabin → `True` when `gt_count >= 7`.
   - `notes`: short free-text description (optional but recommended)

## Run the simulation

```bash
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --rated-capacity 8 \
    --num-calls 1000 \
    --output results/simulation
```

The script writes `confusion_matrix.png`, `energy_savings.csv`, and a Markdown
report under the chosen output directory.
