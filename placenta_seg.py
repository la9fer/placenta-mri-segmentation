"""
Placenta MRI segmentation (PASD dataset): subject-wise split, U-Net, BCE+Dice,
Dice/IoU/connected-component evaluation, OpenCV baseline, data audit, overlays.

Expected folder layout under --data_root (same as your PASD download):
    JPEGImages BTFE_image/sub001_9.jpg        SegmentationClass BTFE_mask/sub001_9.png
    JPEGImages ssh_TSE_image/...              SegmentationClass ssh_TSE_mask/...

Usage:
    python placenta_seg.py --data_root "/path/to/pasd dataset mendley"
    python placenta_seg.py --data_root ... --filter_invalid --out runs/filtered   # data-quality experiment
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEQUENCES = ["BTFE", "ssh_TSE"]
MASK_THRESH = 64  # masks are red PNGs (e.g. (128,0,0)); any channel above this counts as placenta


# ----------------------------------------------------------------------------- data
def read_mask(path):
    """Read a red-coloured mask as binary. NOTE: do NOT read as grayscale and use >127:
    pure red -> gray ~76, dark red -> ~38, so everything would become background."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return (bgr.max(axis=2) > MASK_THRESH).astype(np.uint8)


def subject_of(name):
    m = re.match(r"(sub\d+)", name)
    return m.group(1) if m else Path(name).stem.split("_")[0]


def collect_pairs(root):
    root = Path(root)
    pairs, unmatched = [], 0
    for seq in SEQUENCES:
        img_dir = root / f"JPEGImages {seq}_image"
        mask_dir = root / f"SegmentationClass {seq}_mask"
        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                pairs.append(dict(img=img_path, mask=mask_path, seq=seq, subject=subject_of(img_path.name)))
            else:
                unmatched += 1
    return pairs, unmatched


def audit_pairs(pairs):
    """Per-pair integrity flags (same checks as your main_improved.py)."""
    for p in pairs:
        img = cv2.imread(str(p["img"]), cv2.IMREAD_GRAYSCALE)
        mask = read_mask(p["mask"])
        flags = dict(unreadable=False, size_mismatch=False, empty_mask=False, bg_leak=False)
        if img is None or mask is None:
            flags["unreadable"] = True
        else:
            if img.shape != mask.shape:
                flags["size_mismatch"] = True
            elif mask.sum() == 0:
                flags["empty_mask"] = True
            else:
                leak = np.sum((mask == 1) & (img < 10)) / mask.sum()
                flags["bg_leak"] = bool(leak > 0.5)
        p["flags"] = flags
        p["valid"] = not any(flags.values())
    return pairs


def letterbox(arr, size, interp):
    """Resize longest side to `size` keeping aspect ratio, pad with zeros (as in PlaNet-S)."""
    h, w = arr.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    arr = cv2.resize(arr, (nw, nh), interpolation=interp)
    out = np.zeros((size, size), dtype=arr.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = arr
    return out


def load_pair(p, size):
    img = cv2.imread(str(p["img"]), cv2.IMREAD_GRAYSCALE)
    mask = read_mask(p["mask"])
    if img.shape != mask.shape:  # fallback so a size-mismatched pair does not crash training
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    img = letterbox(img, size, cv2.INTER_AREA)
    mask = letterbox(mask, size, cv2.INTER_NEAREST)
    return img, mask


def random_affine(img, mask, rng):
    h, w = img.shape
    ang = rng.uniform(-20, 20)
    sc = rng.uniform(0.85, 1.15)
    tx, ty = rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.1, 0.1) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:  # mild intensity jitter
        img = np.clip(img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
    return img, mask


class PlacentaDataset(Dataset):
    def __init__(self, pairs, size, augment):
        self.size, self.augment = size, augment
        self.rng = np.random.default_rng(0)
        # cache everything: the dataset is small (a few thousand 256x256 slices)
        self.items = [load_pair(p, size) for p in pairs]
        self.pairs = pairs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, mask = self.items[i]
        if self.augment:
            img, mask = random_affine(img, mask, self.rng)
        x = img.astype(np.float32) / 255.0
        # per-image z-score normalisation
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x)[None], torch.from_numpy(mask.astype(np.float32))[None]


def split_subjects(pairs, seed=42, frac=(0.7, 0.15, 0.15)):
    subs = sorted({p["subject"] for p in pairs})
    random.Random(seed).shuffle(subs)
    n = len(subs)
    n_tr = max(1, int(frac[0] * n))
    n_va = max(1, int(frac[1] * n))
    sets = dict(train=set(subs[:n_tr]), val=set(subs[n_tr:n_tr + n_va]), test=set(subs[n_tr + n_va:]))
    return {k: [p for p in pairs if p["subject"] in v] for k, v in sets.items()}, sets


# ----------------------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """4-level U-Net (3 poolings + bottleneck) with skip connections."""

    def __init__(self, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList([block(1 if i == 0 else ch[i - 1], ch[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth - 1, 0, -1)])
        self.dec = nn.ModuleList([block(ch[i], ch[i - 1]) for i in range(depth - 1, 0, -1)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.out(x)


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce, self.smooth = nn.BCEWithLogitsLoss(), smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth)
        return self.bce(logits, target) + (1 - dice).mean()


# ----------------------------------------------------------------------------- metrics
def dice_iou(pred, gt):
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    s = float(pred.sum() + gt.sum())
    dice = 1.0 if s == 0 else 2 * inter / s
    iou = 1.0 if union == 0 else inter / union
    return dice, iou


def n_components(binary):
    return cv2.connectedComponents(binary.astype(np.uint8))[0] - 1


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0])
    return np.concatenate(out)


def evaluate(probs, ds, thr=0.5):
    rows = []
    for prob, (img, gt), p in zip(probs, ds.items, ds.pairs):
        pred = (prob >= thr).astype(np.uint8)
        d, j = dice_iou(pred.astype(bool), gt.astype(bool))
        rows.append(dict(name=p["img"].stem, seq=p["seq"], subject=p["subject"], dice=d, iou=j,
                         ccc_diff=abs(n_components(pred) - n_components(gt))))
    return rows


def otsu_baseline(ds):
    """Naive classical baseline: Otsu threshold, keep the largest bright blob."""
    rows = []
    for (img, gt), p in zip(ds.items, ds.pairs):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
        pred = np.zeros_like(th)
        if n > 1:
            pred[lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = 1
        d, j = dice_iou(pred.astype(bool), gt.astype(bool))
        rows.append(dict(seq=p["seq"], dice=d, iou=j))
    return rows


def summarize(rows, key):
    v = np.array([r[key] for r in rows])
    return float(v.mean()), float(v.std())


def summary_by_seq(rows, keys=("dice", "iou")):
    out = {}
    for name, sel in [("all", rows)] + [(s, [r for r in rows if r["seq"] == s]) for s in SEQUENCES]:
        if sel:
            out[name] = {"n": len(sel), **{k: dict(zip(("mean", "std"), summarize(sel, k))) for k in keys}}
            if "ccc_diff" in sel[0]:
                out[name]["ccc_exact_match"] = float(np.mean([r["ccc_diff"] == 0 for r in sel]))
    return out


def save_overlays(rows, probs, ds, out_dir, k=4, thr=0.5):
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort([r["dice"] for r in rows])
    picks = [("worst", i) for i in order[:k]] + [("best", i) for i in order[::-1][:k]]
    for tag, i in picks:
        img, gt = ds.items[i]
        pred = (probs[i] >= thr).astype(np.uint8)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for m, col in ((gt, (0, 255, 255)), (pred, (255, 128, 0))):  # GT yellow, prediction blue-orange
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, col, 1)
        panel = np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), vis])
        cv2.putText(panel, f"{rows[i]['name']} dice={rows[i]['dice']:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / f"{tag}_{rows[i]['name']}.png"), panel)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=32, help="U-Net base channels (use 16 for a fast CPU run)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: keep only N pairs")
    ap.add_argument("--filter_invalid", action="store_true", help="drop pairs that fail the integrity audit")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    pairs, unmatched = collect_pairs(args.data_root)
    if not pairs:
        raise SystemExit(f"No image/mask pairs found under {args.data_root}. Check folder names.")
    if args.limit:
        pairs = pairs[:args.limit]
    pairs = audit_pairs(pairs)

    # ---- audit report
    flag_names = ["unreadable", "size_mismatch", "empty_mask", "bg_leak"]
    audit = {"total_pairs": len(pairs), "images_without_mask": unmatched,
             "valid": sum(p["valid"] for p in pairs), **{f: sum(p["flags"][f] for p in pairs) for f in flag_names}}
    audit["valid_pct"] = 100.0 * audit["valid"] / len(pairs)
    print("audit:", audit)
    with open(out / "audit.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "seq", "subject", "valid"] + flag_names)
        for p in pairs:
            w.writerow([p["img"].stem, p["seq"], p["subject"], p["valid"]] + [p["flags"][k] for k in flag_names])
    if audit["empty_mask"] > 0.5 * len(pairs):
        print("WARNING: most masks are empty - check mask colour/threshold (np.unique of a mask).")

    if args.filter_invalid:
        pairs = [p for p in pairs if p["valid"]]
        print(f"filtered to {len(pairs)} valid pairs")

    # ---- subject-wise split
    split, sets = split_subjects(pairs, args.seed)
    for k, v in sets.items():
        print(f"{k}: {len(v)} subjects, {len(split[k])} slices")
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"])
    json.dump({k: sorted(v) for k, v in sets.items()}, open(out / "split.json", "w"), indent=1)

    ds = {k: PlacentaDataset(split[k], args.size, augment=(k == "train")) for k in split}
    loaders = {k: DataLoader(ds[k], batch_size=args.batch, shuffle=(k == "train"), num_workers=0) for k in ds}

    model = UNet(base=args.base).to(device)
    crit = BCEDiceLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    # ---- train (select best epoch by VALIDATION Dice)
    best, bad, history = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(x)
        tl /= len(ds["train"])
        vrows = evaluate(predict_probs(model, loaders["val"], device), ds["val"])
        vd = summarize(vrows, "dice")[0]
        sched.step(1 - vd)
        history.append(dict(epoch=ep, train_loss=tl, val_dice=vd))
        print(f"epoch {ep:3d} | train loss {tl:.4f} | val dice {vd:.4f}")
        if vd > best:
            best, bad = vd, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    # ---- final test evaluation with the best checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    probs = predict_probs(model, loaders["test"], device)
    rows = evaluate(probs, ds["test"])
    base_rows = otsu_baseline(ds["test"])
    results = dict(args={k: str(v) for k, v in vars(args).items()}, audit=audit, best_val_dice=best,
                   test_unet=summary_by_seq(rows), test_otsu_baseline=summary_by_seq(base_rows),
                   history=history)
    json.dump(results, open(out / "results.json", "w"), indent=1)
    with open(out / "test_per_slice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    save_overlays(rows, probs, ds["test"], out / "overlays")

    # ---- markdown table for the README / resume
    lines = ["| Model | Sequence | n | Dice (mean±SD) | IoU (mean±SD) | CC-count exact match |", "|---|---|---|---|---|---|"]
    for name, summ in (("U-Net", results["test_unet"]), ("Otsu baseline", results["test_otsu_baseline"])):
        for seq, s in summ.items():
            cc = f"{100 * s['ccc_exact_match']:.1f}%" if "ccc_exact_match" in s else "-"
            lines.append(f"| {name} | {seq} | {s['n']} | {s['dice']['mean']:.3f}±{s['dice']['std']:.3f} | "
                         f"{s['iou']['mean']:.3f}±{s['iou']['std']:.3f} | {cc} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved everything under {out}/")


if __name__ == "__main__":
    main()
