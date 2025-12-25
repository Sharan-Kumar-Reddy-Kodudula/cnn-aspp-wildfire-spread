````md
# CNN-ASPP Wildfire Spread Prediction

End-to-end deep learning pipeline for **next-day wildfire spread segmentation** using a **CNN with Atrous Spatial Pyramid Pooling (ASPP)**.  
The project emphasizes **reproducibility, clean ML engineering, and interpretability**, and is designed to mirror real-world NDWS-style workflows.

---

## 🔥 Problem Motivation

Predicting the spatial spread of wildfires is critical for early response and resource allocation.  
This project formulates next-day wildfire spread prediction as a **semantic segmentation task** over multi-channel geospatial tiles, following the structure of the **Next-Day Wildfire Spread (NDWS)** dataset.

---

## 🧠 Method Overview

- **Architecture:** CNN + Atrous Spatial Pyramid Pooling (ASPP)
- **Task:** Binary segmentation (spread vs. no-spread)
- **Training:** PyTorch Lightning + Hydra configs
- **Evaluation:** Threshold sweep, PR metrics, IoU, F1
- **Explainability:** Grad-CAM visualization on ASPP branches
- **Reproducibility:** Deterministic runs + snapshot export

---

## ⚙️ Environment Setup

```bash
conda create -n wildfire python=3.10 -y
conda activate wildfire
pip install -e ".[dev]"
````

Verify installation:

```bash
python -m cnn_aspp
pytest -q
```

---

## 🚀 Quickstart: Micro Training (Sanity Check)

Runs a small micro-dataset to validate the full training pipeline.

```bash
python -m cnn_aspp.cli.train \
  dataset.split=micro \
  task.threshold=0.05
```

Expected behavior:

* Rapid overfitting on the micro split
* Confirms correctness of data loading, model wiring, and loss computation

---

## 🌲 Full NDWS Training

> The full NDWS dataset is **not included** in this repository.
> Set `dataset.root` to the local directory containing NDWS tiles.

```bash
python -m cnn_aspp.cli.train \
  dataset.split=ndws \
  dataset.root=./cnn_aspp/data/ndws_out \
  task.threshold=0.05 \
  task.epochs=20
```

---

## 🎯 Threshold Sweep

Performs post-training threshold optimization on validation data.

```bash
python -m cnn_aspp.cli.sweep_thresh \
  dataset.root=./cnn_aspp/data/ndws_out
```

Produces:

* `lightning_logs/threshold_sweep.csv`

Example:

* Best threshold: `0.05`

---

## 📊 Evaluation

Default evaluation:

```bash
python -m cnn_aspp.cli.eval_phase7
```

Evaluation with a custom sweep file and output directory:

```bash
python -m cnn_aspp.cli.eval_phase7 \
  --sweep_csv lightning_logs/threshold_sweep.csv \
  --out_dir outputs/aspp_eval
```

Outputs:

* `eval.csv`
* `pr_curve.png`
* `best_threshold.txt`
* `Findings.md`

---

## 📦 Reproducible Snapshot Export

Exports a fully self-contained snapshot of a trained run.

```bash
python -m cnn_aspp.cli.snapshot_run \
  --version_dir tb/version_0 \
  --out_dir runs/aspp_tiny_full_ndws_v2
```

Snapshot contents:

* Exact source code
* Hydra configuration files
* Hyperparameters
* Metrics and statistics

---

## 📈 NDWS Validation Results (Example)

| Metric    | Value   |
| --------- | ------- |
| Threshold | 0.05    |
| F1        | ≈ 0.263 |
| Precision | ≈ 0.353 |
| Recall    | ≈ 0.210 |

These values are typical for NDWS-style wildfire segmentation using a lightweight CNN-ASPP model.

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM applied to the **ASPPTiny** model on the micro NDWS split.

```bash
python -m cnn_aspp.cli.xai_gradcam \
  --ckpt tb/version_2/checkpoints/epoch=1-val_iou=0.139.ckpt \
  --out_dir outputs/xai_micro \
  --data_root data/micro \
  --split val \
  --batch_size 1 \
  --num_workers 0 \
  --num_tiles 4 \
  --device cpu
```

Outputs:

* Qualitative heatmaps: `outputs/xai_micro/qualitative/`
* HTML error atlas: `outputs/xai_micro/error_atlas.html`

Observed behavior:

* Model activations concentrate around fire-affected regions
* Confirms correct feature extraction and ASPP branch integration

---

## 🗂 Repository Structure

```
cnn_aspp/
  cli/        training, sweep, eval, snapshot tools
  conf/       Hydra configuration files
  data/       NDWS dataset utilities + micro split helpers
  models/     ASPP Tiny + plain CNN baselines
  tasks/      PyTorch Lightning training modules
  utils/      reproducibility helpers
  xai/        Grad-CAM utilities
tests/        unit tests
scripts/      helper scripts
```

---

## ✅ Project Status

The full pipeline (**train → sweep → eval → snapshot → explainability**) runs cleanly and reproduces the reported validation results.

This repository is intended as a **clean reference implementation** for wildfire spread segmentation with strong ML engineering and reproducibility practices.

```
```
