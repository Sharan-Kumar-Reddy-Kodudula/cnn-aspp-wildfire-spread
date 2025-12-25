from pathlib import Path
import argparse, numpy as np, torch

def load_tile(path: Path):
    if path.suffix == '.npz':
        d = np.load(path, allow_pickle=True)
        x = d['inputs']   # C,H,W
        y = d['targets']  # 1,H,W
        m = d['mask'] if 'mask' in d else np.ones_like(y)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long(), torch.from_numpy(m).float()
    elif path.suffix == '.pt':
        d = torch.load(path)
        x = torch.as_tensor(d['inputs']).float()
        y = torch.as_tensor(d['targets']).long()
        m = torch.as_tensor(d.get('mask', torch.ones_like(y))).float()
        return x, y, m
    else:
        raise ValueError(f"Unsupported: {path}")

def f1_from_cm(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1.item(), precision.item(), recall.item()

# --- find this function and replace the files-gathering lines ---
def eval_folder(folder: Path, ch: int, thresh: float, mode: str):
    files = []
    for ext in ('*.npz', '*.pt'):
        files.extend(list(folder.rglob(ext)))  # ← use rglob so subfolders like EVTUNK/ are seen
    if not files:
        raise SystemExit(f"No tiles in {folder}")


 

    tp=fp=fn=0.0
    for p in files:
        x, y, m = load_tile(p)
        # x: C,H,W
        d = x[ch]  # H,W
        if mode == 'gt':
            pred = (d > thresh)
        else:
            pred = (d < thresh)
        pred = pred.bool()
        yb = (y[0] > 0)
        mb = (m[0] > 0.5)
        pred = pred & mb
        yb = yb & mb

        tp += (pred & yb).sum().item()
        fp += (pred & (~yb)).sum().item()
        fn += ((~pred) & yb).sum().item()

    return f1_from_cm(torch.tensor(tp), torch.tensor(fp), torch.tensor(fn))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Dataset root with val/ or test/')
    ap.add_argument('--split', type=str, default='val')
    ap.add_argument('--channel', type=int, required=True, help='Driver channel index in inputs')
    ap.add_argument('--thresh', type=float, required=True)
    ap.add_argument('--mode', type=str, choices=['gt','lt'], default='lt', help='Compare driver > or < threshold')
    args = ap.parse_args()

    f1, p, r = eval_folder(Path(args.root)/args.split, args.channel, args.thresh, args.mode)
    print({
        'split': args.split,
        'channel': args.channel,
        'thresh': args.thresh,
        'mode': args.mode,
        'F1': round(f1, 4), 'precision': round(p, 4), 'recall': round(r, 4)
    })

if __name__ == '__main__':
    main()
