import json, random, shutil
from pathlib import Path
import argparse

"""
Micro-set maker:
- Input: a root with {train,val,test}/... tiles (.npz or .pt).
- Output: data/micro/{train,val} with 8/4 tiles by default.
- Also writes conf/dataset/micro.yaml for quick training.
"""

def discover_tiles(root: Path, split: str):
    base = root / split
    assert base.exists(), f"Missing split folder: {base}"
    # recursive to handle nested event folders (e.g., EVTUNK)
    files = [p for p in base.rglob('*') if p.suffix in {'.npz', '.pt'}]
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Dataset root with train/val/test folders')
    ap.add_argument('--out', type=str, default='data/micro', help='Output root for micro-set')
    ap.add_argument('--train_n', type=int, default=8)
    ap.add_argument('--val_n', type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    (out / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'val').mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    train_files = discover_tiles(root, 'train')
    val_files = discover_tiles(root, 'val') or discover_tiles(root, 'test')

    if not train_files:
        raise SystemExit(f"No train tiles found under {root}/train")
    if not val_files:
        raise SystemExit(f"No val/test tiles found under {root}/val or {root}/test")

    rng.shuffle(train_files)
    rng.shuffle(val_files)

    pick_train = train_files[:args.train_n]
    pick_val = val_files[:args.val_n]

    for p in pick_train:
        shutil.copy2(p, out / 'train' / p.name)
    for p in pick_val:
        shutil.copy2(p, out / 'val' / p.name)

    print(f"Wrote micro-set to: {out}")

    # Emit a minimal Hydra dataset config
    conf_dir = Path('cnn_aspp/conf/dataset')
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / 'micro.yaml').write_text(f"""# Auto-generated micro dataset
name: micro
format: auto  # npz or pt
root: {out.as_posix()}
train_glob: train/*.npz|train/*.pt
val_glob: val/*.npz|val/*.pt
num_workers: 0
batch_size: 4
""")

if __name__ == '__main__':
    main()
