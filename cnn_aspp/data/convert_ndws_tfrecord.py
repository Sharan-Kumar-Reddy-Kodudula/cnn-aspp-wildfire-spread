# cnn_aspp/data/convert_ndws_tfrecord.py
from __future__ import annotations
import argparse, json, pathlib, re, glob, math
from typing import Dict, Tuple, Optional, Any, List
import numpy as np

# --- robust imports for the 'tfrecord' PyPI package (handles different layouts) ---
try:
    from tfrecord.tfrecord_reader import tfrecord_loader  # older layout
except Exception:
    try:
        from tfrecord.reader import tfrecord_loader       # newer layout
    except Exception:
        try:
            from tfrecord import tfrecord_loader          # flat layout
        except Exception as e:
            raise RuntimeError(
                "Couldn't import 'tfrecord_loader' from the 'tfrecord' package.\n"
                "Install/upgrade first:\n"
                "  python -m pip install -U tfrecord 'protobuf<5'\n"
            ) from e

try:
    from tfrecord.example_pb2 import Example              # used only if loader yields raw bytes
except Exception:
    try:
        from tfrecord.proto.example_pb2 import Example
    except Exception:
        Example = None  # guarded when used

# Keys observed in your dict-style TFRecords
FEATURE_ORDER: List[str] = [
    "elevation", "NDVI", "erc", "pdsi", "population",
    "pr", "sph", "th", "tmmn", "tmmx", "vs",
    "PrevFireMask",  # include previous fire as a predictor channel
]
TARGET_KEY = "FireMask"

# Legacy candidates (for raw-bytes variants or other dumps)
KEY_CANDIDATES = {
    "inputs":    ["inputs", "x", "features"],
    "fire_mask": ["fire_mask", "y", "target", "labels", TARGET_KEY],
    "prev_mask": ["previous_fire_mask", "prev_mask", "prev_y", "PrevFireMask"],
    "uncertain": ["uncertain_mask", "invalid_mask", "unknown_mask"],
    "event_id":  ["event_id", "fire_id", "id"],
    "date":      ["date", "t0", "timestamp"],
    "region":    ["region", "state"],
    "bbox":      ["bbox"],
    "crs":       ["crs"],
}

# ---------------------------- helpers ----------------------------

def _first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _sanitize_name(s: Optional[str], default="EVTUNK") -> str:
    if not s:
        return default
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()) or default

def _as_np(val: Any, dtype) -> Optional[np.ndarray]:
    """Coerce a tfrecord value (bytes|list|np.ndarray|scalar) into np.ndarray[dtype]."""
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        return np.frombuffer(val, dtype=dtype)
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=dtype)
    if isinstance(val, np.ndarray):
        return val.astype(dtype, copy=False)
    try:
        return np.asarray(val, dtype=dtype)
    except Exception:
        return None

def _decode_str(val: Any, default=None):
    if val is None:
        return default
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8", errors="ignore")
        except Exception:
            return default
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return default
        return _decode_str(val[0], default)
    return str(val)

def _first_present_key(rec: Dict[str, Any], names) -> Optional[str]:
    for n in names:
        if n in rec:
            return n
    return None

def _first_present_value(rec: Dict[str, Any], names):
    k = _first_present_key(rec, names)
    return rec.get(k) if k is not None else None

# ---------------------- parsing: RAW BYTES path ----------------------

def _pick_feature(feat_dict, names):
    for n in names:
        if n in feat_dict:
            return feat_dict[n]
    return None

def _get_bytes_or_list_as_np_feature(feat, dtype) -> Optional[np.ndarray]:
    if feat is None:
        return None
    bl = feat.bytes_list.value
    fl = feat.float_list.value
    il = feat.int64_list.value
    if bl:
        return np.frombuffer(bl[0], dtype=dtype)
    if fl and np.issubdtype(dtype, np.floating):
        return np.asarray(fl, dtype=dtype)
    if il and np.issubdtype(dtype, np.integer):
        return np.asarray(il, dtype=dtype)
    return None

def _get_feature_str(f, names):
    feat = _pick_feature(f, names)
    if feat is None:
        return None
    bl = feat.bytes_list.value
    fl = feat.float_list.value
    il = feat.int64_list.value
    if bl:
        return bl[0]
    if il:
        return il[0]
    if fl:
        return fl[0]
    return None

def _bytes_record_to_arrays(serialized: bytes) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    if Example is None:
        raise RuntimeError("Example proto not available; update 'tfrecord' package.")
    ex = Example()
    ex.ParseFromString(serialized)
    f = ex.features.feature

    shape_json = None
    if "shape" in f and f["shape"].bytes_list.value:
        try:
            shape_json = json.loads(f["shape"].bytes_list.value[0].decode("utf-8"))
        except Exception:
            shape_json = None

    H = W = 64
    # target
    feat_y = _pick_feature(f, KEY_CANDIDATES["fire_mask"])
    if feat_y is None:
        raise ValueError(f"Missing fire_mask (tried: {KEY_CANDIDATES['fire_mask']})")
    y = _get_bytes_or_list_as_np_feature(feat_y, np.int64)
    if y is None:
        y = _get_bytes_or_list_as_np_feature(feat_y, np.int32)
    if y is None:
        raise ValueError("Could not decode fire_mask as int.")
    y = y.astype(np.int64)
    if y.size != H * W:
        raise ValueError(f"fire_mask has {y.size}, expected {H*W}")
    y = y.reshape(1, H, W)

    # inputs: stack known features if present
    x_list = []
    for name in FEATURE_ORDER:
        feat = f.get(name)
        if feat is None:
            continue
        arr = _get_bytes_or_list_as_np_feature(feat, np.float32)
        if arr is None:
            arr = _get_bytes_or_list_as_np_feature(feat, np.int64)
            if arr is not None:
                arr = arr.astype(np.float32)
        if arr is None:
            continue
        if arr.size != H * W:
            raise ValueError(f"{name} has {arr.size}, expected {H*W}")
        x_list.append(arr.reshape(H, W))
    if not x_list:
        # fallback to legacy 'inputs'
        feat_x = _pick_feature(f, KEY_CANDIDATES["inputs"])
        if feat_x is None:
            raise ValueError("No predictors found to build inputs.")
        flat = _get_bytes_or_list_as_np_feature(feat_x, np.float32)
        if flat is None:
            raise ValueError("Could not decode inputs.")
        C = flat.size // (H * W)
        x = flat.reshape(H, W, C).transpose(2, 0, 1).astype(np.float32)
    else:
        x = np.stack(x_list, axis=0).astype(np.float32)

    # mask (optional)
    feat_u = _pick_feature(f, KEY_CANDIDATES["uncertain"])
    mask = None
    if feat_u is not None:
        u = _get_bytes_or_list_as_np_feature(feat_u, np.int8)
        if u is None:
            u = _get_bytes_or_list_as_np_feature(feat_u, np.int64)
            if u is not None:
                u = u.astype(np.int8)
        if u is not None:
            if u.size != H * W:
                raise ValueError(f"uncertain has {u.size}, expected {H*W}")
            mask = (u.reshape(H, W) == 0)[None, ...]

    meta = {
        "hazard": "wildfire",
        "t0": _decode_str(_get_feature_str(f, KEY_CANDIDATES["date"]), "1970-01-01T00:00:00Z"),
        "bbox": None,
        "crs": _decode_str(_get_feature_str(f, KEY_CANDIDATES["crs"]), "EPSG:4326"),
    }
    ev = _decode_str(_get_feature_str(f, KEY_CANDIDATES["event_id"]))
    rg = _decode_str(_get_feature_str(f, KEY_CANDIDATES["region"]))
    if ev:
        meta["event_id"] = _sanitize_name(ev)
    if rg:
        meta["region"] = rg
    return x, y, mask, meta

# ---------------------- parsing: DICT path ----------------------

def _dict_record_to_arrays(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    # optional shape hint
    shape_json = None
    shape_val = rec.get("shape")
    if isinstance(shape_val, (bytes, bytearray)):
        try:
            shape_json = json.loads(shape_val.decode("utf-8"))
        except Exception:
            shape_json = None
    elif isinstance(shape_val, str):
        try:
            shape_json = json.loads(shape_val)
        except Exception:
            shape_json = None
    elif isinstance(shape_val, dict):
        shape_json = shape_val

    H = W = 64

    # target
    if TARGET_KEY not in rec:
        k = _first_present_key(rec, KEY_CANDIDATES["fire_mask"])
        if k is None:
            raise ValueError(f"Missing target '{TARGET_KEY}' and no alt target keys found.")
        y_flat = _as_np(rec[k], np.int64)
        if y_flat is None:
            y_flat = _as_np(rec[k], np.int32)
    else:
        y_flat = _as_np(rec[TARGET_KEY], np.int64)
        if y_flat is None:
            y_flat = _as_np(rec[TARGET_KEY], np.int32)

    if y_flat is None:
        raise ValueError("Could not decode fire mask.")

    if y_flat.size != H * W:
        side = int(round(math.sqrt(y_flat.size)))
        if side * side != int(y_flat.size):
            raise ValueError(f"Cannot infer H,W from target of length {y_flat.size}")
        H = W = side
    y = y_flat.astype(np.int64).reshape(1, H, W)

    # inputs: stack known predictors
    x_list = []
    for name in FEATURE_ORDER:
        if name not in rec:
            continue
        arr = _as_np(rec[name], np.float32)
        if arr is None:
            arr = _as_np(rec[name], np.int64)
            if arr is not None:
                arr = arr.astype(np.float32)
        if arr is None:
            continue
        if arr.size != H * W:
            side = int(round(math.sqrt(arr.size)))
            if side * side != int(arr.size):
                raise ValueError(f"{name} length {arr.size} doesn't match {H}x{W} and isn't square.")
            arr = arr.reshape(side, side)
            if side != H:
                raise ValueError(f"{name} tile {side}x{side} != target {H}x{W}")
        else:
            arr = arr.reshape(H, W)
        x_list.append(arr)

    if not x_list:
        feat_name_x = _first_present_key(rec, KEY_CANDIDATES["inputs"])
        if feat_name_x is None:
            raise ValueError(
                "No predictor fields found. "
                f"Expected any of {FEATURE_ORDER} or one of {KEY_CANDIDATES['inputs']}"
            )
        flat = _as_np(rec[feat_name_x], np.float32)
        if flat is None:
            raise ValueError("Could not decode legacy inputs.")
        C = flat.size // (H * W)
        x = flat.reshape(H, W, C).transpose(2, 0, 1).astype(np.float32)
    else:
        x = np.stack(x_list, axis=0).astype(np.float32)

    # uncertain/mask (optional)
    mask = None
    feat_name_u = _first_present_key(rec, KEY_CANDIDATES["uncertain"])
    if feat_name_u is not None:
        u = _as_np(rec.get(feat_name_u), np.int8)
        if u is None:
            u = _as_np(rec.get(feat_name_u), np.int64)
            if u is not None:
                u = u.astype(np.int8)
        if u is not None:
            if u.size != H * W:
                side = int(round(math.sqrt(u.size)))
                if side * side != int(u.size):
                    raise ValueError(f"uncertain length {u.size} doesn't match {H}x{W}")
                if side != H:
                    raise ValueError(f"uncertain tile {side}x{side} != target {H}x{W}")
                u = u.reshape(side, side)
            else:
                u = u.reshape(H, W)
            mask = (u == 0)[None, ...]

    meta = {
        "hazard": "wildfire",
        "t0": "1970-01-01T00:00:00Z",
        "bbox": None,
        "crs": "EPSG:4326",
    }
    return x, y, mask, meta

# ---------------------- unified dispatcher ----------------------

def _record_to_arrays(record: Any) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    if isinstance(record, (bytes, bytearray)):
        return _bytes_record_to_arrays(record)
    if isinstance(record, dict):
        return _dict_record_to_arrays(record)
    if isinstance(record, (list, tuple)) and record:
        for item in record:
            if item is not None:
                return _record_to_arrays(item)
    raise TypeError(f"Unsupported TFRecord item type: {type(record)}")

# ---------------------------- I/O ----------------------------

def write_npz(out_dir: pathlib.Path, base: str,
              inputs: np.ndarray, targets: np.ndarray,
              mask: Optional[np.ndarray], meta: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_dir / f"{base}.npz"),
                        inputs=inputs, targets=targets, **({} if mask is None else {"mask": mask}))
    with open(out_dir / f"{base}.json", "w") as f:
        json.dump({k: meta[k] for k in ("hazard", "t0", "bbox", "crs") if k in meta}, f, indent=2)

def convert_split(src_glob: str, dst_split_dir: pathlib.Path):
    idx = 0
    files = sorted(glob.glob(src_glob))
    for tfrec in files:
        for rec in tfrecord_loader(tfrec, None):
            x, y, m, meta = _record_to_arrays(rec)
            sub = meta.get("event_id", "EVTUNK")
            out_dir = dst_split_dir / sub
            base = f"{sub}_{idx:06d}"
            write_npz(out_dir, base, x, y, m, meta)
            idx += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with NDWS TFRecord shards")
    ap.add_argument("--dst", required=True, help="Output folder for standardized tiles (npz+json)")
    args = ap.parse_args()
    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    convert_split(str(src / "next_day_wildfire_spread_train_*.tfrecord"), dst / "train")
    convert_split(str(src / "next_day_wildfire_spread_test_*.tfrecord"),  dst / "test")
    convert_split(str(src / "next_day_wildfire_spread_eval_*.tfrecord"),  dst / "val")
    print("Conversion complete:", dst)

if __name__ == "__main__":
    main()
