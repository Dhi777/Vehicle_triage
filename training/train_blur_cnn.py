from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from matplotlib import cm

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from utils.cnn import TinyBlurCNN


class PlateBlurDataset(Dataset):
    def __init__(self, root: Path, split: str, input_size: Tuple[int, int] = (64, 160)):
        self.root = Path(root)
        self.split = split
        self.input_size = input_size  # (H, W)

        sharp_dir = self.root / split / "sharp"
        blurry_dir = self.root / split / "blurry"

        self.items: List[Tuple[Path, int]] = []
        self.items += [(p, 0) for p in sorted(sharp_dir.glob("*.jpg"))]
        self.items += [(p, 1) for p in sorted(blurry_dir.glob("*.jpg"))]

        if not self.items:
            raise FileNotFoundError(
                f"No images found under {self.root}/{split}/(sharp|blurry). "
                f"Expected folders:\n  {sharp_dir}\n  {blurry_dir}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.resize(gray, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA)

        
        x = gray.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5

        x = torch.from_numpy(x).unsqueeze(0)          
        y = torch.tensor([y], dtype=torch.float32)   
        return x, y


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys_true = []
    ys_prob = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)                      
        prob = torch.sigmoid(logits).squeeze(1) 

        ys_true.extend(y.squeeze(1).cpu().numpy().tolist())
        ys_prob.extend(prob.cpu().numpy().tolist())

    ys_pred = [1 if p >= 0.5 else 0 for p in ys_prob]
    acc = accuracy_score(ys_true, ys_pred)

    auc = None
    try:
        auc = roc_auc_score(ys_true, ys_prob)
    except Exception:
        auc = None

    cm = confusion_matrix(ys_true, ys_pred)
    return acc, auc, cm, ys_true, ys_prob


def best_threshold_balanced(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 19)
    best = (0.5, -1.0)

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        bal_acc = 0.5 * (tnr + tpr)

        if bal_acc > best[1]:
            best = (float(t), float(bal_acc))

    return best  
  


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="artifacts/datasets/cnn_training",
                    help="Path to cnn training dataset root (contains train/valid).")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--input_h", type=int, default=64)
    ap.add_argument("--input_w", type=int, default=160)

    ap.add_argument("--out_ckpt", type=str, default="utils/weights.pkl",
                    help="Where to save blur CNN checkpoint (dict with keys: model, input_size, blur_threshold).")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    input_size = (int(args.input_h), int(args.input_w))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Data root:", data_root.resolve())
    print("Input size (H, W):", input_size)
    print("Output checkpoint:", out_ckpt.resolve())

    train_ds = PlateBlurDataset(data_root, "train", input_size=input_size)
    valid_ds = PlateBlurDataset(data_root, "valid", input_size=input_size)

    def class_counts(ds):
        ys = [y for (_p, y) in ds.items]
        n0 = sum(1 for v in ys if v == 0)
        n1 = sum(1 for v in ys if v == 1)
        return n0, n1

    tr0, tr1 = class_counts(train_ds)
    va0, va1 = class_counts(valid_ds)
    print(f"Train counts: sharp(0)={tr0}, blurry(1)={tr1}")
    print(f"Valid counts: sharp(0)={va0}, blurry(1)={va1}")

    labels = [y for (_p, y) in train_ds.items]  # 0 or 1
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = [class_weights[y] for y in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = TinyBlurCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = -1.0
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else float("nan")

        val_acc, val_auc, cm, ys_true, ys_prob = evaluate(model, valid_loader, device)
        t_best, bal_acc = best_threshold_balanced(ys_true, ys_prob)

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_acc@0.5={val_acc:.4f} | bal_acc={bal_acc:.4f} | val_auc={val_auc}"
        )
        
        print("Confusion matrix [[TN, FP],[FN, TP]]:\n", cm)
        if bal_acc > best_acc:
            best_acc = bal_acc
            best_threshold = t_best

            torch.save(
                {
                    "model": model.state_dict(),
                    "input_size": input_size,
                    "blur_threshold": float(best_threshold),
                },
                out_ckpt,
            )
            print(f"Saved checkpoint: {out_ckpt} (best val_acc={best_acc:.4f})")

    print("Training complete.")
    print("Best val_acc:", best_acc)
    print("Best blur_threshold:", best_threshold)


if __name__ == "__main__":
    main()
