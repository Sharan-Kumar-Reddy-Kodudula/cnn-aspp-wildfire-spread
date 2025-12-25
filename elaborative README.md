# cnn_aspp

Minimal **CNN + ASPP** wildfire-spread project scaffold.

The repo implements:

- data loading + validation for NDWS-style tiles  
- heuristic baselines  
- tiny-CNN sanity trainer **+ plain CNN baseline**  
- ASPP-based segmentation model (ASPPTiny)  
- full training / eval loop with threshold sweep  
- snapshot + repro-pack export for a given run  
- lightweight XAI (Grad-CAM + **per-dilation R²** diagnostics)

---

## 🧩 Project Phases & Status

| Phase | Focus                                                                          | Status                                                                                      |
| :---- | :------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------ |
| 0     | Problem spec, success criteria, risks                                          | ✅ Done (offline notes)                                                                     |
| 1     | Environment + repo skeleton (`cnn_aspp/{models,data,tasks,utils,cli,conf}`)    | ✅ Done                                                                                     |
| 2     | Data contract, validation, **micro-dataset** creation (8/4 tiles)              | ✅ Done                                                                                     |
| 3     | Heuristic baselines + Tiny-CNN **overfit sanity**                              | ✅ Heuristics done; micro overfit achieved (train F1 ≈ 0.9+), low val F1 (expected)        |
| 4     | CNN-ASPP model (ASPPTiny) + tests                                              | ✅ Model + unit tests in place                                                              |
| 5     | Training task (Lightning) + threshold sweep on micro split                     | ✅ Training & sweep scripts working (Phase-5 DoD passed)                                   |
| 6     | Losses (Tversky / Focal), dropout, augmentation configs                        | ✅ Implemented in `ASPPTiny` + training CLI variants                                       |
| 7     | Evaluation on validation / test sets + threshold tuning script                 | ✅ `sweep_thresh` + `eval_phase7` & `eval_test` implemented and run                        |
| 8     | XAI & diagnostics (Grad-CAM, ASPP branch analysis, R² correlation experiment)  | ✅ `xai_gradcam` + `xai_r2` implemented and run on micro split                             |
| 9     | Snapshot & repro-pack for a trained run (`snapshot_run`)                       | ✅ Exports metrics, config, and sample outputs for reporting                               |
| 10    | (Reserved) Stratified metrics / subsets                                        | 📝 Planned (can extend Phase 7 scripts)                                                    |
| 11    | Model card & docs (this README + `model_card.yaml`)                            | ✅ Docs in place; a new teammate can train/eval from README alone                          |

---

## ⚙️ Environment & Quick Start

### 1) Create / activate env

```bash
conda create -n wildfire python=3.10 -y
conda activate wildfire
````

### 2) Install package (editable)

From the repo root:

```bash
pip install -e ".[dev]"
```

This pulls in:

* `torch`, `pytorch-lightning`, `hydra-core`
* `numpy`, `pandas`, `rasterio`, `opencv-python`, `einops`
* `pytest`, `scikit-learn` (for R² in Phase 8)

### 3) Basic sanity check

```bash
python -m cnn_aspp
pytest -q
```

Expected output:

```text
cnn_aspp: repo skeleton OK
...                                                               [100%]
```

---

## 🗂 Repo Layout (high level)

```text
cnn_aspp/
  cnn_aspp/
    cli/            # entrypoints: train, sweep_thresh, eval_*, xai_*, snapshot_run
    conf/           # hydra configs (dataset/model/task/train)
    data/           # dataset utilities, stats & microset scripts
    models/         # tiny_cnn, ASPPTiny, plain CNN baseline, losses
    tasks/          # Lightning modules & training helpers
    utils/          # misc helpers
  conf/
    dataset/        # external dataset config (e.g., micro.yaml)
  data/
    micro/          # 8/4 micro-split for dev
  lightning_logs/   # Lightning metrics.csv runs
  tb/               # Lightning checkpoints + TensorBoard logs
  model_card.yaml   # Phase 11: model card with task/data/metrics/limitations
```

---

## 📦 Phase 2 – Data Contract & Micro-set

The NDWS-style data are stored as NPZ tiles with (at minimum):

* an input array (shape `[C,H,W]` or `[H,W]`)
* a fire mask (shape `[1,H,W]` or `[H,W]`)

### 2.1 Validate dataset & compute stats

```bash
# Class balance over full NDWS split
python -m cnn_aspp.data.class_balance \
  --root cnn_aspp/data/ndws_out \
  --out  cnn_aspp/data/class_balance.json

# Sanity check a random tile visualization
python -m cnn_aspp.data.plot_tile \
  --root cnn_aspp/data/ndws_out \
  --split train
```

Outputs:

* `cnn_aspp/data/class_balance.json` with fire pixel fractions per split
* optional PNGs for visual QA

### 2.2 Build an 8/4 micro-set

```bash
python -m cnn_aspp.data.make_microset \
  --root cnn_aspp/data/ndws_out \
  --out  data/micro \
  --train_n 8 \
  --val_n 4

echo "train count: $(find data/micro/train -type f \( -name '*.npz' -o -name '*.pt' \) | wc -l)"
echo "val   count: $(find data/micro/val   -type f \( -name '*.npz' -o -name '*.pt' \) | wc -l)"
# train count: 8
# val   count: 4
```

> ⚠️ If you see “No tiles in …/val”, make sure you are pointing at the **correct** `data/micro` directory under this repo.

---

## 🔎 Phase 3 – Heuristic Baselines

We treat “channel 3 below a threshold” as a naive fire predictor.

### 3.1 Micro-val sweep

```bash
scripts/scan_thresh.sh data/micro 3 lt 10 50 12 val
```

Example behavior (micro/val):

* Best threshold ≈ **−2.116**
* Micro/F1 ≈ **0.15**, Precision ≈ **0.08**, Recall ≈ **0.94**

### 3.2 Full-val sweep (EVTUNK subset)

```bash
scripts/scan_thresh.sh cnn_aspp/data/ndws_out 3 lt 10 50 40 val > sweep.jsonl

python - <<'PY'
import ast
best = None
for s in open("sweep.jsonl"):
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        d = ast.literal_eval(s)
        if best is None or d["F1"] > best["F1"]:
            best = d
print("BEST:", best)
PY
```

Example best:

```text
BEST: {'thresh': -2.843..., 'F1': 0.0598, 'precision': 0.0324, 'recall': 0.387, ...}
```

These serve as pre-deep-learning baselines for the model card.

---

## 🧠 Phase 3 – Tiny-CNN Sanity Trainer

The tiny CNN (`cnn_aspp.models.tiny_cnn`) is used only for **overfit sanity**.

### 3.1 Single-tile “overfit fast” (≤60 steps)

```bash
PYTORCH_MPS_DISABLE=1 \
python -m cnn_aspp.tasks.train_tiny \
  --root data/micro \
  --in_ch 12 \
  --max_steps 60 \
  --batch_size 1 \
  --lr 2e-1 \
  --mid 256 \
  --use_dice \
  --alpha 0.98 --gamma 0.5 \
  --dilate 2 \
  --single_tile \
  --skip_val \
  --accelerator cpu
```

### 3.2 Micro-set overfit with validation

```bash
PYTORCH_MPS_DISABLE=1 \
python -m cnn_aspp.tasks.train_tiny \
  --root data/micro \
  --in_ch 12 \
  --max_steps 900 \
  --batch_size 1 \
  --lr 3e-2 \
  --mid 128 \
  --use_dice \
  --alpha 0.95 --gamma 1.0 \
  --dilate 1 \
  --accelerator cpu
```

> Notes:
>
> * Do **not** pass `--pos_weight`, `--pos_prior`, or `--dilate_px` – current CLI does not expose them.
> * Losses/metrics are clamped to avoid NaNs; training continues even under extreme imbalance.

---

## 🧱 Phase 4 – CNN-ASPP Model (ASPPTiny)

The main model lives in `cnn_aspp/models/aspp_tiny.py`:

* **Stem:** 2× Conv-BN-ReLU

  * `Conv( in_ch → 64 )`
  * `Conv( 64 → 128 )`
* **ASPP:** 4 branches, dilations `{1,3,6,12}`, each `3×3 → 32`

  * concatenated → 128 channels
* **Fuse + head:**

  * `Conv(128 → 32)` → Dropout2d(p=0.1) → `Conv(32 → 32)` → `Conv(32 → 1)` logits
* **Losses integrated:**

  * `TverskyLoss` (α=0.7, β=0.3)
  * `BCEWithLogitsLoss`
  * `FocalLoss` (γ configurable)

Quick shape self-test:

```bash
python cnn_aspp/models/aspp_tiny.py
# DoD passed: forward [2,C,128,128] -> [2,1,128,128], predict() in [0,1], XAI hooks OK.
```

This verifies:

* correct spatial behavior
* `predict()` returns probabilities in `[0,1]`
* XAI feature hooks are wired.

---

## 🎓 Phase 5 – Training & Threshold Tuning

Training and evaluation are Hydra-driven through `cnn_aspp/cli/train.py` and `cnn_aspp/cli/sweep_thresh.py`.

### 5.1 Train on micro split

Basic 2-epoch sanity run (mixed precision on if GPU is available):

```bash
python -m cnn_aspp.cli.train \
  dataset.split=micro \
  task.threshold=0.05
```

Longer micro run without AMP:

```bash
python -m cnn_aspp.cli.train \
  dataset.split=micro \
  task.threshold=0.05 \
  task.epochs=10 \
  task.amp=false
```

Checkpoints & TensorBoard logs go to:

* `tb/version_*/checkpoints/*.ckpt`

### 5.2 Threshold sweep (val micro)

After training, sweep decision thresholds:

```bash
CKPT=$(ls -t tb/version_7/checkpoints/*.ckpt | head -n1)

python -m cnn_aspp.cli.sweep_thresh +ckpt="${CKPT//=/\\=}"
```

This writes a `threshold_sweep.csv` into the corresponding `lightning_logs/version_XX/` and prints the best F1/precision/recall over candidate thresholds.

### 5.3 Single-tile qualitative viz

Use the Lightning task helper to inspect predictions on one tile:

```bash
python -m cnn_aspp.tasks.viz_tile \
  --ckpt tb/version_7/checkpoints/epochepoch=019-val_IoUval/IoU=0.269.ckpt \
  --tile data/micro/train/EVTUNK_011037.npz \
  --threshold 0.05
```

This writes a PNG with input, GT, and predicted mask overlay.

---

## 🧪 Reading Metrics (any phase)

Lightning writes metrics to `lightning_logs/version_XX/metrics.csv`.

Example snippet to grab best train / val F1:

```bash
python - <<'PY'
import pandas as pd, glob, os

runs = sorted(glob.glob("lightning_logs/version_*"))
if not runs:
    raise SystemExit("No lightning_logs/version_* found")

metrics_run = max(runs, key=os.path.getmtime)
mfile = os.path.join(metrics_run, "metrics.csv")
df = pd.read_csv(mfile)
print("using run:", metrics_run)
print("columns:", list(df.columns))

def last_any(name_list):
    for name in name_list:
        if name in df.columns:
            s = df[name].dropna()
            if not s.empty:
                return float(s.iloc[-1])
    return None

val_f1     = last_any(["val_f1", "val/f1"])
val_prec   = last_any(["val_precision", "val/precision"])
val_recall = last_any(["val_recall", "val/recall"])
val_oa     = last_any(["val_oa"])

print("val_precision =", val_prec)
print("val_recall   =", val_recall)
print("val_f1       =", val_f1)
print("val_oa       =", val_oa)
PY
```

---

## 🎯 Phase 6 – Losses & Regularization Experiments

ASPPTiny exposes `compute_loss` with:

* `criterion="tversky"` (default)
* `criterion="bce"`
* `criterion="focal"` (with `focal_gamma`, `focal_alpha`)

You can run small experiments via `cnn_aspp/cli/train_aspp_tiny.py` (tiny configs under `cnn_aspp/conf/model/aspp_tiny.yaml` and `cnn_aspp/conf/train_aspp_tiny.yaml`).

Examples (all on micro split, batch size small):

```bash
# 1) Tversky, no augmentation
python -m cnn_aspp.cli.train_aspp_tiny \
  train.criterion=tversky \
  dataset.train.augment=false \
  train.batch_size=4 \
  train.num_workers=2

# 2) Tversky, with augmentation
python -m cnn_aspp.cli.train_aspp_tiny \
  train.criterion=tversky \
  dataset.train.augment=true \
  train.batch_size=4 \
  train.num_workers=2

# 3) Focal (γ=2), no augmentation
python -m cnn_aspp.cli.train_aspp_tiny \
  train.criterion=focal \
  model.focal_gamma=2.0 \
  dataset.train.augment=false \
  train.batch_size=4 \
  train.num_workers=2

# 4) Focal (γ=2), with augmentation
python -m cnn_aspp.cli.train_aspp_tiny \
  train.criterion=focal \
  model.focal_gamma=2.0 \
  dataset.train.augment=true \
  train.batch_size=4 \
  train.num_workers=2
```

Dropout is controlled via `model.dropout` (default `0.1`) in `aspp_tiny`.

---

## 🧱 Plain CNN Baseline (Phase 6 / 11 “extra”)

In addition to ASPPTiny, a **plain CNN baseline** (no ASPP branches) is provided as a simpler reference model. It reuses the stem and head structure without the dilated-convolution block.

Example training command (tiny config, micro split):

```bash
python -m cnn_aspp.cli.train_cnn_tiny \
  dataset.split=micro \
  train.criterion=tversky \
  dataset.train.augment=true \
  train.batch_size=4 \
  train.num_workers=2
```

For a full NDWS run, simply switch `dataset.split` and adjust epochs/batch size.

You can compare final validation F1 / IoU of this baseline against ASPPTiny to quantify the ASPP gain, and both are summarized in `model_card.yaml`.

---

## 📊 Phase 7 – Evaluation & Test Metrics

### 7.1 Validation sweep (Phase 7 main script)

`cnn_aspp/cli/eval_phase7.py`:

* loads a checkpoint
* runs inference on a validation split
* sweeps thresholds
* writes out `threshold_sweep.csv` and prints best metrics.

Example:

```bash
python -m cnn_aspp.cli.eval_phase7 \
  --ckpt tb/version_7/checkpoints/best.ckpt \
  --data_root data/micro \
  --split val \
  --out_dir cnn_aspp/data/ndws_out/eval_phase7 \
  --num_workers 0
```

Outputs:

* CSV with threshold vs precision/recall/F1
* printed best-threshold summary

### 7.2 Test-time evaluation

For a held-out test split you can use `cnn_aspp/cli/eval_test.py` with a fixed threshold (e.g., chosen from Phase 7 sweep):

```bash
python -m cnn_aspp.cli.eval_test \
  --ckpt tb/version_7/checkpoints/best.ckpt \
  --data_root cnn_aspp/data/ndws_out \
  --split test \
  --threshold 0.05 \
  --out_dir cnn_aspp/data/ndws_out/eval_test \
  --num_workers 2
```

This computes final test confusion stats and writes a small JSON/CSV summary.

---

## 🔍 Phase 8 – XAI & Diagnostics

Phase 8 adds qualitative and quantitative explanation for ASPPTiny.

### 8A. Grad-CAM & Error Atlas

The script `cnn_aspp/cli/xai_gradcam.py`:

* loads ASPPTiny from a Lightning checkpoint
* for each sampled tile, computes Grad-CAM maps for:

  * final conv layer
  * each ASPP branch (d=1,3,6,12)
* overlays:

  * ground-truth mask outline
  * predicted mask outline
  * Grad-CAM heatmaps
* saves a 2×4 panel PNG per tile
* writes an HTML “error atlas” for quick browsing

Example (on micro split, CPU, 4 tiles):

```bash
python -m cnn_aspp.cli.xai_gradcam \
  --ckpt tb/version_7/checkpoints/best.ckpt \
  --out_dir cnn_aspp/data/ndws_out/xai \
  --data_root data/micro \
  --split val \
  --num_tiles 12 \
  --num_workers 0
```

Output:

* PNGs: `cnn_aspp/data/ndws_out/xai/qualitative/tile_*.png`
* HTML atlas: `cnn_aspp/data/ndws_out/xai/error_atlas.html`

> For the micro/CPU run using a pre-ASPP checkpoint, the ASPP weights are randomly initialized.
> This is visible in the weak/noisy ASPP Grad-CAMs and helps sanity-check the pipeline.

### 8B. Quantitative XAI (per-dilation R² extra)

The script `cnn_aspp/cli/xai_r2.py` implements a light quantitative probe:

For each tile:

1. compute Grad-CAM for final conv → flatten to vector **y**

2. compute Grad-CAM for each ASPP branch (d=1,3,6,12) → flatten to **x_d**

3. fit a 1D linear regression

   [
   y \approx a_d x_d + b_d
   ]

4. compute **R²** for each dilation.

Then:

* write per-tile values to `xai_r2_per_tile.csv`
* aggregate mean/median/std R² per dilation into `xai_r2_summary.csv`

Example run (micro, CPU):

```bash
python -m cnn_aspp.cli.xai_r2 \
  --ckpt tb/version_7/checkpoints/best.ckpt \
  --out_dir cnn_aspp/data/ndws_out/xai \
  --data_root data/micro \
  --split val \
  --num_tiles 12 \
  --num_workers 0
```

Example summary table printed:

```text
   dilation       mean     median         std  count
0         1  -0.512602  -0.155291    0.988028      4
1         3  -0.171879  -0.107778    0.193678      4
2         6 -89.097400 -40.269397  117.362926      4
3        12  -4.809558  -4.818249    3.623761      4
```

Negative R² indicates the random ASPP branch maps (for this checkpoint) explain **less** variance in the final Grad-CAM map than a constant baseline, which is expected here.

When training a model with ASPP from scratch and re-running this phase on a larger NDWS subset, you would expect:

* R² to move closer to 0–1
* potentially higher R² for larger dilations (d=6,12), indicating that wide receptive fields align more closely with the final decision map.

---

## 📦 Phase 9 – Snapshot & Repro Pack

Phase 9 adds a “one-shot” snapshot command that bundles metrics, configs, and optional artifacts for a given training run.

`cnn_aspp/cli/snapshot_run.py`:

* finds and reads `metrics.csv` for a given Lightning version
* grabs the final (or best) val metrics
* records key Hydra configs for model/dataset/train
* optionally copies in example outputs / plots
* writes everything under a single `runs/<name>/` directory

Example:

```bash
python -m cnn_aspp.cli.snapshot_run \
  --version_dir lightning_logs/version_68 \
  --out_dir runs/aspp_tiny_v68
```

This typically writes:

```text
runs/aspp_tiny_v68/
  metrics_summary.json   # final val F1/precision/recall, OA, threshold
  metrics.csv            # (optional) copied original
  config.yaml            # merged Hydra config for this run
  notes.txt              # short text with run metadata
  # optional: example PNGs / Grad-CAM tiles
```

This snapshot directory is what you’d attach to a report or archive for future reproducibility.

---

## 📚 Phase 11 – Docs & Model Card

Phase 11 adds lightweight but complete documentation so that **a new teammate can train & evaluate the model using only this README** and the model card.

### 11.1 `model_card.yaml`

The file `model_card.yaml` summarizes:

* **Task:** binary next-day wildfire spread segmentation on NDWS
* **Data:**

  * NDWS tiling scheme, 1 km grid, daily cadence
  * 12-channel input stack (meteo, fuels, topography, history)
  * splits: train / val / test, plus micro 8/4 split
  * preprocessing and normalization assumptions
* **Architecture:**

  * ASPPTiny (stem + ASPP (d=1,3,6,12) + head)
  * **Plain CNN baseline** (same stem + head, no ASPP)
* **Training:**

  * optimizer (Adam), learning rate, batch size, epochs
  * losses (Tversky, BCE, Focal) and their key hyperparameters
  * regularization (dropout, augmentation)
  * seeds and expected hardware
* **Metrics:**

  * heuristic baseline F1 (channel-3 threshold rule)
  * Tiny-CNN micro overfit sanity F1
  * ASPPTiny val/test F1 / precision / recall / OA
  * chosen decision threshold from Phase 7
* **Explainability:**

  * qualitative Grad-CAM results (Phase 8A)
  * **per-dilation R² stats** (Phase 8B) between each ASPP branch and final conv Grad-CAM
* **Limitations & intended use:**

  * emphasizes research / exploratory use, not operational decision automation
  * lists data/model limitations and basic ethical considerations

This model card is meant to be short, machine-readable context for papers / reports.

### 11.2 README “contract”

This README provides:

* end-to-end commands for:

  * environment setup
  * data validation + micro-set creation
  * heuristic baselines
  * tiny-CNN sanity overfit
  * ASPPTiny training (and plain CNN baseline)
  * threshold sweeps, val + test evaluation
  * snapshot export (`snapshot_run`)
  * Grad-CAM and per-dilation R² experiments
* I/O contracts (shapes, normalizations) for the dataset + model
* troubleshooting notes for common gotchas (paths, CLIs, imbalanced metrics, XAI quirks)

Together, `README.md` + `model_card.yaml` satisfy the Phase 11 DoD:

> **“A new teammate can train/eval from README alone.”**

---

## 🛠 Troubleshooting

* **“No tiles in …/val”**
  Check you’re pointing at the correct `data/micro` or NDWS directory.

* **“unrecognized arguments” in CLIs**
  Each CLI exposes a specific subset of flags; inspect `python -m cnn_aspp.cli.train_aspp_tiny --help` etc.

* **Very low val F1 on micro split**
  Expected due to heavy class imbalance and tiny sample size. The micro split is for *sanity & quick iteration*, not final performance.

* **Random-looking Grad-CAMs / negative R²**
  If you use a checkpoint trained before ASPP was added, the ASPP weights are random. Train ASPPTiny from scratch and rerun Phases 7–8 for meaningful XAI.

---

Happy burning (in the strictly *simulation* sense 🔥🙂).

```

If you want, next step we can also add tiny “Phase X” labels into the section headings for Phase 3A/3B (heuristics vs tiny CNN) to make it crystal clear in the report.
```
