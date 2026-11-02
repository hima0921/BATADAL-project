"""
Proof of concept: Reconstruction loss of the Transformer reconstruction model on
NORMAL BATADAL data, with the CSV files passed on the command line. No Anomaly Detection,
just verifying if the model can reconstruct the data using the compressed representation
from the Encoder.

    python proof_of_concept.py "../../data/Training dataset 1.csv"
    python proof_of_concept.py file_a.csv file_b.csv --epochs 100 --out results
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from model import Transformer_Recon_Model

# ----------------------------- Config ---------------------------------------
WINDOW = 24                          # hours per window
TRAIN_FRAC, VAL_FRAC = 0.7, 0.1      # test = remaining 20%
MIN_DELTA = 1e-4                     # relative val-loss decrease that counts
BATCH_SIZE = 128
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- Loading and validation -----------------------
def load_normal_csv(path):
    """Return (feature_names, array) after checking the file is verified normal."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "ATT_FLAG" not in df.columns:
        raise SystemExit(f"{path}: no ATT_FLAG column, so the file cannot be verified "
                         f"as attack-free (is this the test set?). Rejected.")
    flags = sorted(df["ATT_FLAG"].unique().tolist())
    if flags != [0]:
        raise SystemExit(f"{path}: ATT_FLAG values are {flags}; only files labeled "
                         f"entirely 0 are accepted. -999 means unlabeled (not normal) "
                         f"and 1 is an attack. Rejected.")

    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
    steps = df["DATETIME"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise SystemExit(f"{path}: DATETIME is not strictly hourly (gaps, repeats or "
                         f"out-of-order rows). Rejected.")

    features = [c for c in df.columns if c not in ("DATETIME", "ATT_FLAG")]
    if df[features].isna().any().any():
        raise SystemExit(f"{path}: contains empty cells. Rejected.")
    return features, df[features].to_numpy(np.float32)


def make_windows(arr, window, stride=1):
    if len(arr) < window:
        raise SystemExit(f"a split has {len(arr)} rows, fewer than the window ({window}); "
                         f"use a longer file or a smaller --window")
    starts = np.arange(0, len(arr) - window + 1, stride)
    return np.stack([arr[s:s + window] for s in starts]).astype(np.float32)


def build_datasets(csv_files):
    loaded = [load_normal_csv(p) for p in csv_files]
    feature_names = loaded[0][0]
    for p, (names, _) in zip(csv_files, loaded):
        if names != feature_names:
            raise SystemExit(f"{p}: feature columns differ from {csv_files[0]}")
    arrays = [a for _, a in loaded]

    parts = {"train": [], "val": [], "test": []}
    for arr in arrays:                                   # chronological split per file
        n = len(arr)
        a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
        parts["train"].append(arr[:a])
        parts["val"].append(arr[a:b])
        parts["test"].append(arr[b:])

    train_rows = np.concatenate(parts["train"])
    keep = train_rows.std(axis=0) > 0                    # drop constant features
    dropped = [f for f, k in zip(feature_names, keep) if not k]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_rows[:, keep])

    data = {}
    for split, plist in parts.items():
        wins, src = [], []
        for i, p in enumerate(plist):
            w = make_windows(scaler.transform(p[:, keep]).astype(np.float32), WINDOW)
            wins.append(w)
            src += [i] * len(w)
        data[split] = (np.concatenate(wins), np.array(src))
    return data, int(keep.sum()), dropped


# ----------------------------- Training / evaluation ------------------------
@torch.no_grad()
def window_mse(model, arr, bs=512):
    """Per-window reconstruction MSE (eval mode)."""
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(device)
        out.append(((model(x) - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv_files", nargs="+", help="normal BATADAL CSV file(s)")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=110,
                    help="epochs without val improvement before stopping")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="poc_results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}")

    data, in_dim, dropped = build_datasets(args.csv_files)
    train_w, val_w, test_w = data["train"][0], data["val"][0], data["test"][0]
    test_src = data["test"][1]
    print(f"Features used: {in_dim} (dropped constant: {', '.join(dropped) or 'none'})")
    print(f"Windows | train {train_w.shape} | val {val_w.shape} | test {test_w.shape}")

    # Feel free to change variables.
    model = Transformer_Recon_Model(
        in_dim=in_dim,
        embedding_dim=64,
        hidden_dim=128,
        num_heads=8,
        num_encodings=4,
        num_window=WINDOW,
        num_enc_blocks=2,
        num_dec_blocks=2,
        dropout=0.0,
        activation_type="gelu").to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_w)),
                        batch_size=BATCH_SIZE, shuffle=True)

    mean_vec = train_w.mean(axis=(0, 1))                 # mean-predictor reference
    baseline = float(((test_w - mean_vec) ** 2).mean())
    print(f"Mean-predictor baseline MSE (test): {baseline:.5f}\n")

    hist = {"train": [], "val": [], "test": []}

    def log(epoch):
        tr, va, te = (window_mse(model, a).mean() for a in (train_w, val_w, test_w))
        for k, v in zip(hist, (tr, va, te)):
            hist[k].append(float(v))
        print(f"epoch {epoch:3d} | train {tr:.5f} | val {va:.5f} | test {te:.5f}")
        return float(va)

    best_val = log(0)
    best_epoch, best_state, bad = 0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for (x,) in loader:
            x = x.to(device)
            loss = criterion(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        va = log(epoch)
        if va < best_val * (1 - MIN_DELTA):
            best_val, best_epoch, bad = va, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    model.load_state_dict(best_state)

    # ---- Report (best-validation weights)
    tr_e, va_e, te_e = (window_mse(model, a) for a in (train_w, val_w, test_w))
    print(f"\nBest epoch: {best_epoch}")
    print(f"Train MSE: {hist['train'][0]:.5f} -> {tr_e.mean():.5f}")
    print(f"Val   MSE: {hist['val'][0]:.5f} -> {va_e.mean():.5f}")
    print(f"Test  MSE: {hist['test'][0]:.5f} -> {te_e.mean():.5f}")
    print(f"Mean-predictor baseline (test): {baseline:.5f}")
    print("Test MSE per input file:")
    for i, path in enumerate(args.csv_files):
        m = te_e[test_src == i]
        print(f"  {os.path.basename(path)}: {m.mean():.5f} ({len(m)} windows)")

    plt.figure(figsize=(7, 4))
    for k in hist:
        plt.plot(hist[k], label=k)
    plt.axvline(best_epoch, color="green", linestyle=":", label="best epoch")
    plt.axhline(baseline, color="gray", linestyle="--", label="mean-predictor baseline")
    plt.xlabel("epoch")
    plt.ylabel("reconstruction MSE")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out}/recon_loss.png", dpi=150)
    print(f"Saved {args.out}/recon_loss.png")


if __name__ == "__main__":
    main()