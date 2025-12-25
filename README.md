# cnn_aspp — Wildfire Spread Prediction (CNN + ASPP)

This project implements a minimal, reproducible pipeline for next-day wildfire spread segmentation using a CNN with Atrous Spatial Pyramid Pooling (ASPP). It supports NDWS-style tiles, full training, threshold tuning, evaluation, and snapshot export.

------------------------------------------------------------
1. Environment Setup
------------------------------------------------------------

conda create -n wildfire python=3.10 -y
conda activate wildfire
pip install -e ".[dev]"

Verify installation:

python -m cnn_aspp
pytest -q

------------------------------------------------------------
2. Micro Training (Sanity Check)
------------------------------------------------------------

python -m cnn_aspp.cli.train \
  dataset.split=micro \
  task.threshold=0.05

------------------------------------------------------------
3. Full NDWS Training
------------------------------------------------------------

python -m cnn_aspp.cli.train \
  dataset.split=ndws \
  dataset.root=./cnn_aspp/data/ndws_out \
  task.threshold=0.05 \
  task.epochs=20

------------------------------------------------------------
4. Threshold Sweep
------------------------------------------------------------

python -m cnn_aspp.cli.sweep_thresh \
  dataset.root=./cnn_aspp/data/ndws_out

Writes:
- lightning_logs/threshold_sweep.csv

Best threshold found: 0.05

------------------------------------------------------------
5. Evaluation
------------------------------------------------------------

Default:

python -m cnn_aspp.cli.eval_phase7

Custom output directory:

python -m cnn_aspp.cli.eval_phase7 \
  --sweep_csv lightning_logs/threshold_sweep.csv \
  --out_dir outputs/aspp_eval

Outputs:
- eval.csv
- pr_curve.png
- best_threshold.txt
- Findings.md

------------------------------------------------------------
6. Snapshot (Reproducible Run Export)
------------------------------------------------------------

python -m cnn_aspp.cli.snapshot_run \
  --version_dir tb/version_0 \
  --out_dir runs/aspp_tiny_full_ndws_v2

Creates a folder containing:
- copied code
- configs
- hyperparameters
- metrics / stats

------------------------------------------------------------
7. NDWS Validation Results (Example)
------------------------------------------------------------

Best threshold: 0.05

F1        ≈ 0.263  
Precision ≈ 0.353  
Recall    ≈ 0.210  

These values are typical for NDWS segmentation using a lightweight ASPP model.

------------------------------------------------------------
8. Repository Structure (Simplified)
------------------------------------------------------------

cnn_aspp/
  cli/           training, sweep, eval, snapshot tools
  conf/          Hydra configs
  data/          NDWS dataset tools + micro split
  models/        ASPP Tiny + plain CNN
  tasks/         Lightning training modules
  utils/         seeding helpers
tb/              TensorBoard logs + checkpoints
lightning_logs/  metrics logs + sweeps
runs/            snapshot exports
tests/           unit tests

------------------------------------------------------------
9. Status
------------------------------------------------------------

End-to-end pipeline (train → sweep → eval → snapshot) runs cleanly and reproduces the reported validation results.

For explainability, I ran Grad-CAM on the ASPPTiny model using a micro NDWS checkpoint:

python -m cnn_aspp.cli.xai_gradcam \
  --ckpt tb/version_2/checkpoints/epoch=1-val_iou=0.139.ckpt \
  --out_dir outputs/xai_micro \
  --data_root data/micro \
  --split val \
  --batch_size 1 \
  --num_workers 0 \
  --num_tiles 4 \
  --device cpu

This script generated qualitative Grad-CAM visualizations for all 4 validation tiles under `outputs/xai_micro/qualitative/` and an HTML error atlas at `outputs/xai_micro/error_atlas.html`. The heatmaps show that the model tends to activate around fire-affected regions on the micro split, confirming that the feature extraction and ASPP branches are wired correctly.

