# Smart Elevator CV — Energy Simulation Report

## Configuration

- Mode: **hybrid**
- Four-class weights: `models/weights/best.pt`
- Head weights: `models/weights/best_head.pt`
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
| **GT: not full** | 14 (TN) | 7 (FP) |
| **GT: full**     | 0 (FN) | 8 (TP) |

- Accuracy: **0.759**
- Bypass precision: **0.533**
- Bypass recall:    **1.000**
- F1 score:         **0.696**

## Per-class detection accuracy

| Class | GT total | Pred total | MAE | RMSE | Bias |
|---|:---:|:---:|:---:|:---:|:---:|
| person | 162 | 195 | 1.21 | 1.61 | +1.14 |
| stroller | 22 | 34 | 0.55 | 1.02 | +0.41 |
| luggage | 43 | 47 | 0.41 | 0.74 | +0.14 |
| box | 0 | 3 | 0.10 | 0.42 | +0.10 |

MAE / RMSE are computed over per-image counts. Bias is the mean (predicted - ground-truth) — positive values indicate over-detection.

## Energy aggregates (synthetic day)

- Baseline (always-accept): **3453.9 kJ**
- Smart (vision-gated):     **1723.5 kJ**
- **Energy saved**: 1730.4 kJ (0.481 kWh) — **50.1%** of baseline
- Smart bypassed 501 of 1000 calls (50.1%)

## Per-image decisions

| filename | gt(p/s/l/b) | gt_full | gt_occ | pred(p/s/l/b) | pred_full | pred_occ | outcome |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ChatGPT Image 8 May 2026 20_43_27.png | 5/0/0/0 | False | 0.45 | 5/0/2/0 | False | 0.61 | TN |
| ChatGPT Image 8 May 2026 20_45_05.png | 5/0/0/0 | False | 0.45 | 5/3/0/0 | True | 1.00 | FP |
| ChatGPT Image 8 May 2026 20_46_24.png | 4/1/0/0 | False | 0.56 | 4/1/1/0 | False | 0.64 | TN |
| ChatGPT Image 8 May 2026 20_48_32.png | 3/0/3/0 | False | 0.51 | 3/1/4/0 | False | 0.79 | TN |
| ChatGPT Image 8 May 2026 20_49_36.png | 5/1/3/0 | False | 0.89 | 7/2/3/2 | True | 1.00 | FP |
| ChatGPT Image 8 May 2026 21_31_09.png | 5/0/2/0 | False | 0.61 | 5/0/2/0 | False | 0.61 | TN |
| ChatGPT Image 8 May 2026 21_35_56.png | 5/1/2/0 | False | 0.81 | 5/1/3/0 | False | 0.89 | TN |
| Ekran görüntüsü 2026-05-08 133244.png | 5/0/0/0 | False | 0.45 | 5/0/0/0 | False | 0.45 | TN |
| Ekran görüntüsü 2026-05-08 212029.png | 5/1/3/0 | False | 0.89 | 5/1/4/0 | True | 0.97 | FP |
| Gemini_Generated_Image_1soeth1soeth1soe.png | 3/0/0/0 | False | 0.27 | 4/0/0/0 | False | 0.36 | TN |
| Gemini_Generated_Image_3b7eda3b7eda3b7e.png | 6/2/3/0 | True | 1.00 | 7/2/3/0 | True | 1.00 | TP |
| Gemini_Generated_Image_67eq2367eq2367eq.png | 6/1/2/0 | False | 0.90 | 7/2/3/1 | True | 1.00 | FP |
| Gemini_Generated_Image_fxr0xafxr0xafxr0.png | 8/0/0/0 | False | 0.71 | 10/0/0/0 | False | 0.89 | TN |
| Gemini_Generated_Image_grpn7igrpn7igrpn.png | 5/1/2/0 | False | 0.81 | 5/2/2/0 | True | 1.00 | FP |
| Gemini_Generated_Image_ikfgexikfgexikfg.png | 6/0/0/0 | False | 0.54 | 8/0/0/0 | False | 0.71 | TN |
| Gemini_Generated_Image_iy3coniy3coniy3c.png | 7/1/2/0 | True | 0.99 | 10/1/2/0 | True | 1.00 | TP |
| Gemini_Generated_Image_iy3coniy3coniy3c2.png | 7/1/2/0 | True | 0.99 | 9/1/1/0 | True | 1.00 | TP |
| Gemini_Generated_Image_kvfbmwkvfbmwkvfb.png | 6/1/4/0 | True | 1.00 | 9/3/3/0 | True | 1.00 | TP |
| Gemini_Generated_Image_n8ni20n8ni20n8ni.png | 5/1/3/0 | False | 0.89 | 4/0/3/0 | False | 0.60 | TN |
| Gemini_Generated_Image_nrtk8anrtk8anrtk.png | 6/0/0/0 | False | 0.54 | 8/0/0/0 | False | 0.71 | TN |
| Gemini_Generated_Image_nrtk8anrtk8anrtk2.png | 6/0/0/0 | False | 0.54 | 7/0/0/0 | False | 0.62 | TN |
| Gemini_Generated_Image_on0ngwon0ngwon0n.png | 5/0/0/0 | False | 0.45 | 7/1/0/0 | False | 0.83 | TN |
| Gemini_Generated_Image_sdgsd.png | 6/3/2/0 | True | 1.00 | 6/2/0/0 | True | 0.94 | TP |
| Gemini_Generated_Image_tu8fl9tu8fl9tu8f.png | 5/2/4/0 | True | 1.00 | 7/3/4/0 | True | 1.00 | TP |
| Gemini_Generated_Image_tv0o3rtv0o3rtv0o.png | 8/1/1/0 | True | 1.00 | 11/1/2/0 | True | 1.00 | TP |
| Gemini_Generated_Image_tv0o3rtv0o3rtv0o2.png | 8/1/1/0 | True | 1.00 | 9/1/1/0 | True | 1.00 | TP |
| Gemini_Generated_Image_uxq2rsuxq2rsuxq2.png | 6/1/2/0 | False | 0.90 | 8/4/2/0 | True | 1.00 | FP |
| Gemini_Generated_Image_wysrydwysrydwysr.png | 6/1/1/0 | False | 0.82 | 9/1/1/0 | True | 1.00 | FP |
| Gemini_Generated_Image_wysrydwysrydwysr2.png | 5/1/1/0 | False | 0.73 | 6/1/1/0 | False | 0.82 | TN |