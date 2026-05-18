# Smart Elevator CV — End-to-End Pipeline Reference

> Single source of truth for the project's data flow, every formula it
> uses, and every place a number is produced. Cross-references the code
> paths so you can jump straight to the implementation.

**Repository root:** `Capstone_git/`
**Companion docs:** `README.md` (high-level), `configs/default.yaml` (every constant lives here).

---

## Table of contents

1. [High-level flow](#1-high-level-flow)
2. [Stage 0 — Configuration](#stage-0--configuration)
3. [Stage 1 — Raw data → unified YOLO dataset](#stage-1--raw-data--unified-yolo-dataset)
4. [Stage 2 — Class-balanced augmentation (train-only)](#stage-2--class-balanced-augmentation-train-only)
5. [Stage 3 — Training (two YOLOv8 models)](#stage-3--training-two-yolov8-models)
6. [Stage 4 — Per-frame inference](#stage-4--per-frame-inference)
7. [Stage 5 — Occupancy ratio and cabin mass](#stage-5--occupancy-ratio-and-cabin-mass)
8. [Stage 6 — Two-stage decision (Algorithm 1)](#stage-6--two-stage-decision-algorithm-1)
9. [Stage 7 — Energy model (per-call and per-trip)](#stage-7--energy-model-per-call-and-per-trip)
10. [Stage 8 — Three-policy simulation](#stage-8--three-policy-simulation)
11. [Stage 9 — Metrics](#stage-9--metrics)
12. [Stage 10 — Outputs](#stage-10--outputs)
13. [Reference card — the five numbers to memorise](#reference-card--the-five-numbers-to-memorise)

---

## 1. High-level flow

```
                       ┌─────────────────────────────────────────┐
                       │  STAGE 0  — Configuration               │
                       │  configs/default.yaml                   │
                       │  (cabin geometry, thresholds, energy)   │
                       └────────────────────┬────────────────────┘
                                            │
            ┌───────────────────────────────┴───────────────────────────────┐
            │                                                               │
            ▼                                                               ▼
   OFFLINE  (pre-training)                                       ONLINE  (run-time)
   Stages 1–3                                                    Stages 4–6
```

Single-frame data flow:

```
[raw Roboflow datasets]
      │  Stage 1 (unify.py): class_map → group-stratified split → bbox clamp
      ▼
[data/unified/{train,val,test}] ── Stage 2 (augment.py, train only) ──┐
      │                                                               │
      ▼                                                               ▼
[YOLOv8s 3-class]  ←──── Stage 3 (train.py) ────→  [YOLOv8s head]
   best.pt                                            best_head.pt
      │                                                  │
      └───────────────────┬──────────────────────────────┘
                          │  Stage 4 (predict_image)
                          ▼
        per-class counts  n_c  (person, stroller, luggage, box)
                          │
                          │  Stage 5:  ρ = Σ n_c·ā_c / A_cabin
                          │            m_cabin = Σ n_c·m̄_c
                          ▼
        load-cell W   ──► Stage 6 (algorithm.py)  ──►  δ ∈ {accept, bypass_W, bypass_A}
                          │
                          │  Stage 7 (consumption.py): E_overhead, E_running
                          ▼
        Stage 8 (simulate_calls): 3 policies × 1 000 calls → SimulationStats
                          │
                          ▼
        Stage 9 metrics  +  Stage 10 outputs  →  thesis §4 result tables
```

---

## Stage 0 — Configuration

**File:** [`configs/default.yaml`](../configs/default.yaml)

Everything downstream reads from this file. The most load-bearing constants:

| Group | Key | Default | Notes |
|---|---|:---:|---|
| `elevator` | `width_m × depth_m` | 1.4 × 1.6 | → `A_cabin = 2.24 m²` |
| `elevator` | `max_weight_kg` | 630 | Rated load |
| `elevator` | `counterweight_K` | 0.45 | Counterweight balances 45 % of rated load |
| `elevator` | `motor_efficiency` | 0.85 | Hoisting efficiency η |
| `elevator` | `rated_speed_mps` | 1.0 | v |
| `elevator` | `acceleration_mps2` | 0.8 | a |
| `elevator` | `floor_height_m` | 3.0 | Per-floor traversal |
| `thresholds` | `weight_bypass_ratio` (τ_W) | 0.80 | Bypass when W ≥ 504 kg |
| `thresholds` | `area_bypass_ratio` (τ_A) | 0.90 | Bypass when ρ ≥ 0.90 |
| `thresholds` | `confidence_min` | 0.40 | YOLO conf |
| `thresholds` | `iou_min` | 0.45 | NMS IoU |
| `energy` | `power_doors_w` | 100 | Door motor draw |
| `energy` | `door_open_close_time_s` | 3.0 | Single direction |
| `energy` | `stop_time_s` | 4.0 | Idle while passengers board |
| `energy` | `power_idle_w / power_control_w` | 50 / 30 | Combined 80 W during motion + stop |

A scenario override is passed via `--config path/to/scenario.yaml`.

---

## Stage 1 — Raw data → unified YOLO dataset

**Code:** [`src/dataset/unify.py`](../src/dataset/unify.py)
**Entry point:** `python -m scripts.prepare_dataset --raw data/raw --out data/unified`

```
data/raw/<Roboflow exports>     (each source has its own class ids)
        │
        ▼
  ① class_map remap        e.g. Stroller.yolov8: 3 → "stroller", 2 → "luggage", 0,1,4 → None
        │                  All person labels are dropped (head model handles them).
        ▼
  ② polygon → bbox         if mask: cx = (x_min + x_max)/2, w = x_max − x_min
        │
        ▼
  ③ group_key extraction   regex strips "_jpg.rf.<hash>" and trailing "-42" frame indices.
        │                  → All frames from one video share a key.
        ▼
  ④ group-stratified split 80 / 15 / 5 by default (per source).
        │                  Elevator.yolov8 special-case: 100 % test (real-cabin scenario).
        ▼
  ⑤ bbox clamp [0, 1]     reject degenerate boxes (w ≤ 0 or h ≤ 0).
        │
        ▼
  data/unified/{train,val,test}/{images,labels}  +  data.yaml  +  REPORT.md
  TARGET_CLASSES = ["stroller", "luggage", "box"]   ← person deliberately absent
```

**Why grouping matters.** Many Roboflow exports contain successive frames
from the same video (filenames like `…_3D_mp4-0`, `…_3D_mp4-1`). A naive
random split would scatter near-duplicate frames across train / val /
test and inflate the apparent accuracy. The unifier therefore splits at
the *group* level — defined as

```
group_key = strip("_jpg.rf.<hash>") then strip trailing "-<digits>" or "_<digits>"
```

so every frame from one source video lands in a single split. A
post-split leakage check (`group_to_splits`) verifies no key appears in
more than one split.

---

## Stage 2 — Class-balanced augmentation (train-only)

**Code:** [`src/dataset/augment.py`](../src/dataset/augment.py)

Operates **only** on `<unified>/train/` (hard-asserted in code), so val
and test stay clean. Each augmented copy is named `<stem>__aug<k>.<ext>`
which makes re-runs idempotent.

Per-image multiplier:

```
multiplier(img) = max_{c ∈ classes_in(img)} per_class_multiplier[c]
                  stroller → 2,  luggage → 2,  box → 5
```

Albumentations pipeline (probabilities from `configs/default.yaml`):

| Family | Op | Range | p |
|---|---|---|:---:|
| Geometric | HorizontalFlip | — | 0.5 |
| Geometric | Affine (scale, rotate, translate) | ±25 %, ±15°, ±12 % | 0.7 |
| Photometric | RandomBrightnessContrast | ±30 % / ±30 % | 0.7 |
| Photometric | HueSaturationValue | H ±2.7°, S ±50, V ±40 | 0.5 |
| Photometric | GaussNoise | — | 0.25 |
| Occlusion | CoarseDropout | 1–3 holes, ≤ 20 % of side | 0.45 |

Bbox transformation uses `A.BboxParams(format="yolo",
min_visibility=0.3, min_area=20)` so partially-occluded boxes survive
but vanished objects are pruned.

---

## Stage 3 — Training (two YOLOv8 models)

**Code:** [`src/detection/train.py`](../src/detection/train.py), notebook `notebooks/02_train.ipynb`

Two checkpoints are produced:

| File | Role | Classes | Source corpus |
|---|---|---|---|
| `models/weights/best.pt` | Three-class object detector | stroller, luggage, box | Unified dataset (Stage 1) |
| `models/weights/best_head.pt` | Head-only detector | head (used as person count) | Separate ~6 000-image head corpus |

Both use the `balanced` preset:

```
variant = yolov8s
imgsz   = 640
epochs  = 100
batch   = 16
lr0     = 0.01
patience = 25 (early stop)
pretrained = yolov8s.pt   (transfer learning)
```

### YOLOv8 loss (three terms, Ultralytics implementation)

$$
\mathcal{L} \;=\; \lambda_{\text{box}}\,\mathcal{L}_{\text{CIoU}} \;+\; \lambda_{\text{cls}}\,\mathcal{L}_{\text{BCE}} \;+\; \lambda_{\text{dfl}}\,\mathcal{L}_{\text{DFL}}
$$

- **CIoU** — penalises localisation error with three components (overlap, centre distance, aspect ratio):

$$
\mathcal{L}_{\text{CIoU}} \;=\; 1 - \text{IoU} \;+\; \frac{\rho^{2}(b,b^{gt})}{c^{2}} \;+\; \alpha\,v
$$

  where $c$ is the diagonal of the smallest enclosing box, $\rho$ is the
  centre-to-centre distance, $v$ measures aspect-ratio inconsistency,
  and $\alpha$ is a positive trade-off coefficient.

- **DFL (Distribution Focal Loss)** — anchor-free regression: each
  bounding-box edge is modelled as a discrete distribution, DFL pulls
  the probability mass towards the two integer bins straddling the true
  offset.

- **BCE** — per-class binary cross-entropy on classification logits
  (objectness is fused with class probabilities in YOLOv8).

Non-Maximum Suppression at inference: `iou_threshold = 0.45`,
`conf_threshold = 0.40`.

### Validation metrics (from `README.md`)

| Model | Precision | Recall | mAP\@50 | mAP\@50–95 |
|---|:---:|:---:|:---:|:---:|
| 3-class (`best.pt`) | 0.953 | 0.822 | 0.877 | 0.667 |
| Head (`best_head.pt`) | 0.852 | 0.692 | 0.767 | 0.519 |

---

## Stage 4 — Per-frame inference

**Code:** [`scripts/run_simulation.py:275`](../scripts/run_simulation.py) `predict_image()`, [`src/detection/detector.py`](../src/detection/detector.py)

```
CCTV frame  (BGR np.ndarray)
   │
   ├────────────────────┬───────────────────────────────────┐
   │                    │                                   │
   ▼                    ▼                                   ▼
 YOLO 3-class       (hybrid mode = both run in parallel)  YOLO head
 → boxes:           output filter rule:                   → boxes:
   stroller         "person" class is IGNORED                each box
   luggage          when produced by the 3-class model       counts as one
   box              (avoids double counting with head)       person
   ↓                                                         ↓
   counts["stroller"] += 1                                  counts["person"] = len(boxes)
   counts["luggage"]  += 1
   counts["box"]      += 1
```

The two detectors are **mutually exclusive in the classes they
contribute** — the head model owns `person`, the object model owns the
remaining three. Outputs concatenate cleanly into the four operational
classes `(person, stroller, luggage, box)`.

---

## Stage 5 — Occupancy ratio and cabin mass

**Code:** [`src/perception/occupancy.py`](../src/perception/occupancy.py) `ClassFootprintOccupancy.compute()`

### Per-class footprint and mass

| Class | $\bar a_c$ (m²) | $\bar m_c$ (kg) | Footprint source |
|---|:---:|:---:|---|
| person | 0.20 | 75 | ISO 8100-32:2020 §6.4, EN 81-20:2020 §5.4.2.1.1 |
| stroller | 0.45 | 20 | EN 1888-1:2018 (~90 × 50 cm) |
| luggage | 0.20 | 15 | IATA Resolution 753 (56 × 36 cm) |
| box | 0.20 | 5 | Industry e-commerce parcel mean (~50 × 40 cm) |

### Formulae

Cabin area:

$$
A_{\text{cabin}} \;=\; w \times d \;=\; 1.4 \times 1.6 \;=\; 2.24\ \text{m}^{2}
$$

Visual occupancy ratio:

$$
\boxed{\;A_{\text{occupied}} \;=\; \sum_{c \in \{p,s,l,b\}} n_c \cdot \bar a_c,
\qquad
\rho \;=\; \min\!\left(\frac{A_{\text{occupied}}}{A_{\text{cabin}}},\; 1\right)\;}
$$

Cabin mass (consumed by the energy module):

$$
m_{\text{cabin}} \;=\; \sum_c n_c \cdot \bar m_c
$$

> **Position-agnostic by design.** Two passengers standing shoulder to
> shoulder still count as $2 \times 0.20 = 0.40\,\text{m}^{2}$. A
> homography-based union of per-class disks is listed as future work in
> the thesis Limitations section.

---

## Stage 6 — Two-stage decision (Algorithm 1)

**Code:** [`src/control/algorithm.py:65`](../src/control/algorithm.py) `ElevatorController.decide()`

Thresholds: $\tau_W = 0.80$, $\tau_A = 0.90$, $W_{\text{rated}} = 630\,\text{kg}$, so
the Stage 1 trip-point is $0.80 \times 630 = 504\,\text{kg}$.

```
        W (load-cell, kg)                    Frame
              │                                │
              ▼                                │
        ┌─────────────┐                        │
        │ Stage 1     │   W / W_rated ≥ τ_W ?  │
        │ Weight gate │ ──────► YES ─────────► BYPASS_BY_WEIGHT   (cheap short-circuit)
        └─────┬───────┘                        │
              │ NO                             │
              ▼                                │
        ┌─────────────────────────────────────┴────┐
        │ Stage 2  (YOLO + occupancy)              │
        │   ρ ≥ τ_A ?                              │
        │   ────────► YES ─────► BYPASS_BY_AREA    │
        │   ────────► NO ──────► ACCEPT            │
        └──────────────────────────────────────────┘
```

$$
\delta \;=\;
\begin{cases}
\text{bypass}_W & W \ge \tau_W \, W_{\text{rated}}  &(= 504\,\text{kg})\\
\text{bypass}_A & \rho \ge \tau_A                   &(= 0.90)\\
\text{accept}   & \text{otherwise.}
\end{cases}
$$

The weight gate runs **first** because the load-cell reading is free —
no need to pay YOLO inference cost when the load alone already settles
the call.

---

## Stage 7 — Energy model (per-call and per-trip)

**Code:** [`src/energy/consumption.py`](../src/energy/consumption.py)

### a) Trapezoidal velocity profile — start time

Constants $v = 1.0\ \text{m/s}$, $a = 0.8\ \text{m/s}^{2}$, so the
distance needed to reach full speed is $v^{2}/a = 1.25\ \text{m}$.

$$
t_{\text{start}}(d) \;=\;
\begin{cases}
\dfrac{d}{v} + \dfrac{v}{a} & d \ge \dfrac{v^{2}}{a} \quad\text{(trapezoidal — reaches } v\text{)}\\[6pt]
2\sqrt{\dfrac{d}{a}} & \text{otherwise (triangular)}
\end{cases}
$$

### b) Counterweight balance and potential energy

$$
\Delta m \;=\; (m_{\text{car}} + m_{\text{load}}) \;-\; (m_{\text{car}} + K \cdot m_{\text{nominal}})
\;=\; m_{\text{load}} \;-\; 0.45 \cdot 630
\;=\; m_{\text{load}} \;-\; 283.5\ \text{kg}
$$

$$
E_{\text{potential}} \;=\; \Delta m \cdot g \cdot \text{sign} \cdot d, \qquad g = 9.81\ \text{m/s}^{2}
$$

`sign = +1` going up, `−1` going down.

### c) Running energy with regenerative braking

$$
E_{\text{running}} \;=\;
\begin{cases}
E_{\text{potential}} \,/\, \eta & E_{\text{potential}} \ge 0 \quad\text{(motor lifts)}\\
E_{\text{potential}} \cdot \eta & E_{\text{potential}} < 0 \;\text{and}\; \text{regen}=\text{True}\\
0                                & E_{\text{potential}} < 0 \;\text{and}\; \text{regen}=\text{False}
\end{cases}
$$

With $\eta = 0.85$. A light cabin moving upwards can yield **negative**
$E_{\text{running}}$ (the counterweight overpowers the car and the
motor reclaims energy).

### d) Total energy of one stop

```
distance  d = floors_traveled × floor_height_m
t_start   = trapezoidal-profile time as above
E_running = (b) + (c)
```

$$
\begin{aligned}
E_{\text{aux,motion}} &= (P_{\text{idle}} + P_{\text{control}}) \cdot t_{\text{start}}
   = (50 + 30) \cdot t_{\text{start}}\ \text{J}\\
E_{\text{doors}}      &= P_{\text{doors}} \cdot t_{\text{door}} \cdot 2
   = 100 \cdot 3.0 \cdot 2 = 600\ \text{J}\\
E_{\text{stop\_idle}} &= (P_{\text{idle}} + P_{\text{control}}) \cdot t_{\text{stop}}
   = 80 \cdot 4.0 = 320\ \text{J}\\[4pt]
E_{\text{total}} &= E_{\text{running}} + E_{\text{aux,motion}} + E_{\text{doors}} + E_{\text{stop\_idle}}
\end{aligned}
$$

### e) Stop-overhead — what a bypass *actually* saves

The running energy of a trip is **shared** with all other accepted
calls on the same trip, so it cannot be credited to a single
bypass. Only the *per-floor stop overhead* is strictly attributable:

$$
\boxed{\;E_{\text{overhead}} \;=\; \underbrace{600}_{\text{doors}}
                                 \;+\; \underbrace{320}_{\text{stop idle}}
                              \;=\; 920\ \text{J / stop}\;}
$$

$$
t_{\text{overhead}} \;=\; 2 \cdot 3.0 \;+\; 4.0 \;=\; 10\ \text{s / stop}
$$

These are load- and direction-independent — every correct bypass saves
exactly $920\,\text{J}$ and $10\,\text{s}$.

### f) Stationary energy (ISO 25745-2 three-tier)

$$
E_{\text{stat}}(t) \;=\;
P_{\text{idle}} \cdot \min(t, 300)
\;+\; P_{5\text{m}} \cdot \bigl[\min(t, 1800) - 300\bigr]^{+}
\;+\; P_{30\text{m}} \cdot \bigl[t - 1800\bigr]^{+}
$$

with $P_{\text{idle}} = 50\,\text{W}$, $P_{5\text{m}} = 30\,\text{W}$,
$P_{30\text{m}} = 15\,\text{W}$, so the cabin draws less and less the
longer it sits idle.

---

## Stage 8 — Three-policy simulation

**Code:** [`scripts/run_simulation.py:418`](../scripts/run_simulation.py) `simulate_calls()`

### Driver

```
68 labelled cabin images  +  ground_truth.csv (gt_person, gt_stroller, gt_luggage, gt_box)
      │
      ▼
Per-image inference (Stages 4–6) → list[ImageDecision]
      │
      ▼
1 000 synthetic hall calls, rng(seed=42):
   • origin, dest ∈ [1, 10] uniform (re-roll if equal)
   • distance_floors = |dest − origin|
   • cabin_state = random.choice(decisions)
      │
      ▼
For each call, three policies are evaluated on the SAME stream.
```

### Policies

| Policy | Stage 1 | Stage 2 | Energy added per call |
|---|:---:|:---:|---|
| `always_accept` | — | — | $+E_{\text{overhead}}$ for **every** call (baseline) |
| `weight_only`   | ✓ | — | $0$ if $\sum n_c \bar m_c \ge 504$, else $+E_{\text{overhead}}$ |
| `smart` (ours)  | ✓ | ✓ | $0$ on either bypass, else $+E_{\text{overhead}}$ |

### Optimal-policy ground truth

$$
\text{gt\_should\_bypass} \;=\; \text{gt\_is\_full} \;\lor\; \text{gt\_weight\_full}
$$

$$
\text{gt\_weight\_full} \;=\; \Bigl(\sum_c n_c^{gt} \cdot \bar m_c \;\ge\; 504\,\text{kg}\Bigr),
\qquad
\text{gt\_is\_full} \;=\; (\rho^{gt} \;\ge\; 0.90)
$$

### Confusion accounting (smart vs ground truth)

| Outcome | Condition | Bookkeeping |
|---|---|---|
| TP | gt_bypass ∧ smart_bypass | $E_{\text{saved}} \mathrel{+}= 920\ \text{J}$ |
| TN | ¬gt_bypass ∧ ¬smart_bypass | cost 0 |
| FP | ¬gt_bypass ∧ smart_bypass | wrongly-skipped passenger — **0 in this run** |
| FN | gt_bypass ∧ ¬smart_bypass | $E_{\text{wasted}} \mathrel{+}= 920\ \text{J}$ |

---

## Stage 9 — Metrics

### Bypass decision quality

[`precision_recall()`](../scripts/run_simulation.py)

$$
P \;=\; \frac{TP}{TP+FP}, \qquad
R \;=\; \frac{TP}{TP+FN}, \qquad
F_{1} \;=\; \frac{2PR}{P+R}, \qquad
\text{Acc} \;=\; \frac{TP+TN}{n}
$$

### Counting quality — image-level

[`per_class_count_metrics()`](../scripts/run_simulation.py)

For each class $c$ across $N$ images:

$$
\text{MAE}_{c} = \frac{1}{N}\sum_{i} |n_{c,i}^{\text{pred}} - n_{c,i}^{\text{gt}}|,
\quad
\text{RMSE}_{c} = \sqrt{\frac{1}{N}\sum_{i}(n_{c,i}^{\text{pred}} - n_{c,i}^{\text{gt}})^{2}},
\quad
\text{Bias}_{c} = \frac{1}{N}\sum_{i}(n_{c,i}^{\text{pred}} - n_{c,i}^{\text{gt}})
$$

A positive bias means over-detection.

### Counting quality — object-level retrieval

Treats every individual instance as a retrieval target. For each
(image, class) pair:

$$
TP_{ic} \;=\; \min(n^{\text{pred}}, n^{\text{gt}}), \quad
FN_{ic} \;=\; \max(0,\, n^{\text{gt}} - n^{\text{pred}}), \quad
FP_{ic} \;=\; \max(0,\, n^{\text{pred}} - n^{\text{gt}})
$$

Aggregate recall / precision / F1 follow directly from these sums.

### Service rate

$$
\text{service\_rate} \;=\; \frac{TP + TN}{n}
$$

For the smart policy on this dataset: **90.5 %**. All of the missed
9.5 % are FN (wasted stops on already-full cabins) — **zero FP**
(no passenger was wrongly skipped).

### Headline numbers (from `README.md`)

| Comparison | Energy save | Time save |
|---|---|---|
| Smart vs always-accept | 207.9 kJ (22.6 %) | 16.7 min (22.6 %) |
| **Smart vs weight-only (thesis headline)** | **92.0 kJ (11.4 %)** | **16.7 min (11.4 %)** |

---

## Stage 10 — Outputs

**Code:** [`scripts/run_simulation.py:1420`](../scripts/run_simulation.py) `main()` write phase

```
results/<output>/
  ├─ confusion_matrix.png       (matplotlib heatmap, TP/TN/FP/FN)
  ├─ per_image_decisions.csv    (68 rows: GT + prediction + outcome)
  ├─ per_class_detection.csv    (MAE/RMSE/bias + TP/FN/FP per class)
  ├─ energy_savings.csv         (3 policies × energy + time + deltas)
  ├─ call_log.csv               (1 000-call timeline, cumulative kJ per policy)
  ├─ report.md                  (markdown — pastes directly into thesis §4)
  └─ predictions/, predictions_head/   (annotated per-image frames)
```

**Streamlit demo** at [`demo/app.py`](../demo/app.py) consumes these
artefacts: *Single Frame* tab (upload + live threshold sweep) and
*Batch Simulation* tab (auto-loads whatever the simulation script
produced).

---

## Reference card — the five numbers to memorise

| Symbol | Value | Meaning |
|---|:---:|---|
| $A_{\text{cabin}}$ | 2.24 m² | Cabin floor area ($1.4 \times 1.6$) |
| $\tau_W,\,\tau_A$ | 0.80, 0.90 | Stage 1 / Stage 2 thresholds |
| $W_{\text{rated}}$ | 630 kg | Rated load → Stage 1 trip-point 504 kg |
| $E_{\text{overhead}}$ | **920 J / stop** | Energy a correct bypass saves |
| $t_{\text{overhead}}$ | **10 s / stop** | Wall-clock a correct bypass saves |

---

## Appendix — direct file index

| Subsystem | File |
|---|---|
| Configuration | [`configs/default.yaml`](../configs/default.yaml) |
| Dataset unification | [`src/dataset/unify.py`](../src/dataset/unify.py) |
| Augmentation | [`src/dataset/augment.py`](../src/dataset/augment.py) |
| Dataset audit | [`src/dataset/audit.py`](../src/dataset/audit.py) |
| Training | [`src/detection/train.py`](../src/detection/train.py) |
| Inference wrapper | [`src/detection/detector.py`](../src/detection/detector.py) |
| Occupancy | [`src/perception/occupancy.py`](../src/perception/occupancy.py) |
| Decision policy | [`src/control/algorithm.py`](../src/control/algorithm.py) |
| Energy model | [`src/energy/consumption.py`](../src/energy/consumption.py) |
| End-to-end simulation | [`scripts/run_simulation.py`](../scripts/run_simulation.py) |
| Dataset prep CLI | [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py) |
| Streamlit demo | [`demo/app.py`](../demo/app.py) |
