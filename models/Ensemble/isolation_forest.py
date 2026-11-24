"""
Isolation Forest + PCA detector, after Nemade et al. (2026), section 4.3:
the 36 sensors plus their 1-step differences (72 columns), standardized,
reduced to 20 principal components, then an Isolation Forest of 500 trees with
contamination 0.055. Score = -score_samples (higher = more anomalous).

Unlike the paper it is fitted on the Dataset 1 TRAIN block only (the project's
normal data), never on Dataset 2. Scores are written for exactly the hours the
XGBoost run scores, keyed by (source, row).

    python isolation_forest.py --dataset-path ../../data/processed --out-dir runs/iforest --seeds 0 1 2
"""
import os
import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "XGBoost"))
from xgb_data import load_blocks  # noqa: E402


def if_columns(feature_columns):
    """Raw sensors and their 1-step differences (the paper's 86 = 43 x 2; here 36 x 2)."""
    raw = [c for c in feature_columns
           if "_lag" not in c and "_diff" not in c and "_r" not in c
           and not c.startswith(("hour_", "dow_"))]
    return raw + [f"{c}_diff1" for c in raw]


def fit_scores(blocks, seed, n_components=20, n_estimators=500, contamination=0.055):
    """Fit on the train block; return {block name: scores} for every block."""
    cols = if_columns(blocks["train"][0].columns)
    scaler = StandardScaler().fit(blocks["train"][0][cols])
    pca = PCA(n_components=n_components, random_state=seed).fit(scaler.transform(blocks["train"][0][cols]))
    forest = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                             random_state=seed).fit(pca.transform(scaler.transform(blocks["train"][0][cols])))
    out = {name: -forest.score_samples(pca.transform(scaler.transform(feats[cols])))
           for name, (feats, _) in blocks.items()}
    return out, float(pca.explained_variance_ratio_.sum()), len(cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    blocks = load_blocks(str(args.dataset_path), time_features=False)
    for seed in args.seeds:
        scores, explained, n_cols = fit_scores(blocks, seed)
        d2_meta = blocks["d2"][1]
        rows = []
        for source in ("val", "cal"):
            meta = blocks[source][1]
            rows.append(pd.DataFrame({"source": source, "row": meta["row"], "if_score": scores[source]}))
        # every Dataset 2 hour: attack windows and the hours outside them (era check)
        rows.append(pd.DataFrame({"source": "d2", "row": d2_meta["row"], "if_score": scores["d2"]}))
        out = args.out_dir / f"seed{seed}"
        os.makedirs(out, exist_ok=True)
        pd.concat(rows).to_csv(out / "iforest_hourly_scores.csv", index=False)
        print(f"seed {seed}: {n_cols} columns, 20 components keep {explained:.1%} of the variance")


if __name__ == "__main__":
    main()
