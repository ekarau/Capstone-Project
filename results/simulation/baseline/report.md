# Smart Elevator CV — Energy Simulation Report

## Configuration

- Mode: **single-model**
- Four-class weights: `models/weights/best.pt`
- Rated capacity: **8 persons**
- Cabin floor area: **2.24 m²**
- Confidence threshold: 0.25
- Area bypass threshold: 0.90
- Synthetic hall calls: **1000**
- Avg passengers per accepted call: 4.0
- Avg floors per trip: 3.0

## Bypass-decision performance (image-level)

|  | Predicted: not full | Predicted: full |
|---|:---:|:---:|
| **GT: not full** | 18 (TN) | 3 (FP) |
| **GT: full**     | 2 (FN) | 6 (TP) |

- Accuracy: **0.828**
- Bypass precision: **0.667**
- Bypass recall:    **0.750**
- F1 score:         **0.706**

## Per-class detection accuracy

| Class | GT total | Pred total | MAE | RMSE | Bias |
|---|:---:|:---:|:---:|:---:|:---:|
| person | 162 | 122 | 1.38 | 1.72 | -1.38 |
| stroller | 22 | 34 | 0.55 | 1.02 | +0.41 |
| luggage | 43 | 47 | 0.41 | 0.74 | +0.14 |
| box | 0 | 3 | 0.10 | 0.42 | +0.10 |

MAE / RMSE are computed over per-image counts. Bias is the mean (predicted - ground-truth) — positive values indicate over-detection.

## Energy aggregates (synthetic day)

- Baseline (always-accept): **3453.9 kJ**
- Smart (vision-gated):     **2438.4 kJ**
- **Energy saved**: 1015.4 kJ (0.282 kWh) — **29.4%** of baseline
- Smart bypassed 294 of 1000 calls (29.4%)

## Per-image decisions

| filename | gt(p/s/l/b) | gt_full | gt_occ | pred(p/s/l/b) | pred_full | pred_occ | outcome |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ChatGPT Image 8 May 2026 20_43_27.png | 5/0/0/0 | False | 0.45 | 3/0/2/0 | False | 0.43 | TN |
| ChatGPT Image 8 May 2026 20_45_05.png | 5/0/0/0 | False | 0.45 | 2/3/0/0 | False | 0.78 | TN |
| ChatGPT Image 8 May 2026 20_46_24.png | 4/1/0/0 | False | 0.56 | 3/1/1/0 | False | 0.55 | TN |
| ChatGPT Image 8 May 2026 20_48_32.png | 3/0/3/0 | False | 0.51 | 2/1/4/0 | False | 0.70 | TN |
| ChatGPT Image 8 May 2026 20_49_36.png | 5/1/3/0 | False | 0.89 | 3/2/3/2 | True | 1.00 | FP |
| ChatGPT Image 8 May 2026 21_31_09.png | 5/0/2/0 | False | 0.61 | 3/0/2/0 | False | 0.43 | TN |
| ChatGPT Image 8 May 2026 21_35_56.png | 5/1/2/0 | False | 0.81 | 3/1/3/0 | False | 0.71 | TN |
| Ekran görüntüsü 2026-05-08 133244.png | 5/0/0/0 | False | 0.45 | 4/0/0/0 | False | 0.36 | TN |
| Ekran görüntüsü 2026-05-08 212029.png | 5/1/3/0 | False | 0.89 | 4/1/4/0 | False | 0.88 | TN |
| Gemini_Generated_Image_1soeth1soeth1soe.png | 3/0/0/0 | False | 0.27 | 3/0/0/0 | False | 0.27 | TN |
| Gemini_Generated_Image_3b7eda3b7eda3b7e.png | 6/2/3/0 | True | 1.00 | 4/2/3/0 | True | 1.00 | TP |
| Gemini_Generated_Image_67eq2367eq2367eq.png | 6/1/2/0 | False | 0.90 | 5/2/3/1 | True | 1.00 | FP |
| Gemini_Generated_Image_fxr0xafxr0xafxr0.png | 8/0/0/0 | False | 0.71 | 4/0/0/0 | False | 0.36 | TN |
| Gemini_Generated_Image_grpn7igrpn7igrpn.png | 5/1/2/0 | False | 0.81 | 3/2/2/0 | False | 0.83 | TN |
| Gemini_Generated_Image_ikfgexikfgexikfg.png | 6/0/0/0 | False | 0.54 | 5/0/0/0 | False | 0.45 | TN |
| Gemini_Generated_Image_iy3coniy3coniy3c.png | 7/1/2/0 | True | 0.99 | 6/1/2/0 | False | 0.90 | FN |
| Gemini_Generated_Image_iy3coniy3coniy3c2.png | 7/1/2/0 | True | 0.99 | 7/1/1/0 | True | 0.91 | TP |
| Gemini_Generated_Image_kvfbmwkvfbmwkvfb.png | 6/1/4/0 | True | 1.00 | 3/3/3/0 | True | 1.00 | TP |
| Gemini_Generated_Image_n8ni20n8ni20n8ni.png | 5/1/3/0 | False | 0.89 | 4/0/3/0 | False | 0.60 | TN |
| Gemini_Generated_Image_nrtk8anrtk8anrtk.png | 6/0/0/0 | False | 0.54 | 6/0/0/0 | False | 0.54 | TN |
| Gemini_Generated_Image_nrtk8anrtk8anrtk2.png | 6/0/0/0 | False | 0.54 | 6/0/0/0 | False | 0.54 | TN |
| Gemini_Generated_Image_on0ngwon0ngwon0n.png | 5/0/0/0 | False | 0.45 | 4/1/0/0 | False | 0.56 | TN |
| Gemini_Generated_Image_sdgsd.png | 6/3/2/0 | True | 1.00 | 4/2/0/0 | False | 0.76 | FN |
| Gemini_Generated_Image_tu8fl9tu8fl9tu8f.png | 5/2/4/0 | True | 1.00 | 3/3/4/0 | True | 1.00 | TP |
| Gemini_Generated_Image_tv0o3rtv0o3rtv0o.png | 8/1/1/0 | True | 1.00 | 7/1/2/0 | True | 0.99 | TP |
| Gemini_Generated_Image_tv0o3rtv0o3rtv0o2.png | 8/1/1/0 | True | 1.00 | 7/1/1/0 | True | 0.91 | TP |
| Gemini_Generated_Image_uxq2rsuxq2rsuxq2.png | 6/1/2/0 | False | 0.90 | 3/4/2/0 | True | 1.00 | FP |
| Gemini_Generated_Image_wysrydwysrydwysr.png | 6/1/1/0 | False | 0.82 | 6/1/1/0 | False | 0.82 | TN |
| Gemini_Generated_Image_wysrydwysrydwysr2.png | 5/1/1/0 | False | 0.73 | 5/1/1/0 | False | 0.73 | TN |