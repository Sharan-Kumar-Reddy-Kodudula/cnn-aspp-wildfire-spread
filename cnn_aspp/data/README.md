# NDWS Dataset — Standardized Tiles

**Source:** Next Day Wildfire Spread (2012–2020), 18,545 tiles @ 64×64  
**CRS:** EPSG:4326  
**Tile size:** 64  

---

## 📦 Data Contract

Each sample `<tile>.npz` contains:

| Key | Shape | Dtype | Description |
|-----|--------|--------|-------------|
| **inputs** | `[C, H, W]` | `float32` | Stacked feature channels |
| **targets** | `[1, H, W]` | `{0, 1}` | Binary fire mask |
| **mask** *(optional)* | `[1, H, W]` | `{0, 1}` | Valid data mask (1 = valid) |

Each tile has a matching `<tile>.json` metadata file containing `hazard`, `t0`, `bbox`, and `crs`.

---

## 🧭 Channel Order

1. elevation  
2. NDVI  
3. erc  
4. pdsi  
5. population  
6. pr  
7. sph  
8. th  
9. tmmn  
10. tmmx  
11. vs  
12. PrevFireMask

---

## 🧩 Splits

Fire-level leakage is **already avoided upstream** during the TFRecord split.

cnn_aspp/data/ndws_out/
├── train/ # Training tiles
├── val/ # Validation tiles
└── test/ # Test tiles


## ⚙️ Normalization

Per-channel z-score normalization using statistics from `stats.json`:

x[c] = (x[c] - mean[c]) / (std[c] + 1e-6)


---

## 📊 Artifacts

| File | Purpose |
|------|----------|
| **`stats.json`** | Per-channel mean, std, min, max |
| **`class_balance.json`** | Fire vs non-fire pixel ratio per split |

---

**Maintainer:** Sharan Kumar Reddy Kodudula  
**Project:** `cnn_aspp` — ASPP-enabled CNN for Wildfire Spread Prediction  
**Last updated:** October 2025