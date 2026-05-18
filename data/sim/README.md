# Energy-Saving Simulation — Test Image Set

This directory hosts the curated cabin photographs and their ground-truth
labels used by `scripts/run_simulation.py`. The current set is **68
images** spanning empty → at-capacity scenarios.

## Layout

```
data/sim/
├── images/                       ← 68 cabin photos (cabin_001.png … cabin_068.png)
├── ground_truth.csv              ← per-image multi-class labels
└── ground_truth_template.csv     ← starter template if extending the set
```

## How to label

1. Add cabin photographs covering the full occupancy spectrum.
   The current 68-image set is balanced as roughly:
   - ~25 **empty / lightly occupied** scenes (0–3 passengers, occ < 0.40)
   - ~30 **medium** scenes (4–6 passengers, 0.40 ≤ occ < 0.85)
   - ~12 **at-or-above capacity** scenes (≥ 7 persons or area-saturated)
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

The script computes two ground-truth flags per image:

```
gt_occupancy_ratio = ((gt_person × 0.20) + (gt_stroller × 0.45)
                    + (gt_luggage × 0.20) + (gt_box × 0.20)) / cabin_area
gt_is_full         = gt_occupancy_ratio ≥ area_threshold (default 0.90)

gt_weight_kg       = (gt_person × 75) + (gt_stroller × 20)
                   + (gt_luggage × 15) + (gt_box × 5)
gt_weight_full     = gt_weight_kg ≥ weight_threshold_kg (default 0.80 × 630 = 504 kg)

gt_should_bypass   = gt_is_full OR gt_weight_full   (optimal-policy ground truth)
```

`gt_should_bypass` is the reference label for the smart bypass decision.

## Run the simulation

Both commands evaluate three policies (always-accept / weight-only /
smart) on the same 1 000-call stream:

```bash
# Object detector alone (stroller / luggage / box only)
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --rated-capacity 8 \
    --conf-threshold 0.40 \
    --num-calls 1000 \
    --output results/simulation/baseline_68

# 3-class + head detector (object detector + head detector for person)
python -m scripts.run_simulation \
    --images data/sim/images \
    --ground-truth data/sim/ground_truth.csv \
    --weights models/weights/best.pt \
    --head-weights models/weights/best_head.pt \
    --rated-capacity 8 \
    --conf-threshold 0.40 \
    --head-conf 0.40 \
    --num-calls 1000 \
    --output results/simulation/hybrid_68
```

The thesis results reported in §4 were produced with the current
`best.pt` (three-class object detector) and `best_head.pt` (head
detector) on the 68-image cabin set described above. `person` is
handled exclusively by `best_head.pt` in 3-class + head mode.

The script writes `confusion_matrix.png`, `per_image_decisions.csv`,
`per_class_detection.csv`, `energy_savings.csv`, `call_log.csv` and
`report.md` under the chosen output directory. The energy CSV includes
both the smart-vs-always-accept and the smart-vs-weight-only deltas.
