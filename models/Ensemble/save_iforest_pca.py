"""
Fit and SAVE Suhas's IF + PCA detector (models/Ensemble/isolation_forest.py), so it can
score new data later without refitting.

It repeats fit_scores() in isolation_forest.py (lines 39-42) exactly, for each seed:
    StandardScaler -> PCA(20 components) -> IsolationForest(500 trees, contamination 0.055),
all fitted on the Dataset 1 Training block only, on the raw sensors plus their 1-step
differences (if_columns). isolation_forest.py itself is not changed.

For each seed it saves {"cols", "scaler", "pca", "forest"} with joblib.dump, then reloads
the file, rescores Calibration, Validation and Dataset 2, and compares with the saved
scores in iforest_hourly_scores.csv. It stops with an error if they differ.

Score of an hour = -forest.score_samples(pca.transform(scaler.transform(X[cols]))),
higher = more anomalous (same as isolation_forest.py).

Run from the repo root (needs models/XGBoost/xgb_data.py and models/Ensemble/isolation_forest.py):
    python models/Ensemble/save_iforest_pca.py \
        --dataset-path data/processed \
        --saved-scores results/xgboost_iforest_ensemble/iforest \
        --out-dir models/Ensemble/iforest_pca_models
"""
import argparse
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "XGBoost"))
from xgb_data import load_blocks  # noqa: E402
from isolation_forest import if_columns  # noqa: E402

# The same settings as fit_scores() in isolation_forest.py.
N_COMPONENTS = 20
N_ESTIMATORS = 500
CONTAMINATION = 0.055
TOLERANCE = 1e-12


def fit(blocks, seed):
    """Same steps as isolation_forest.fit_scores() lines 38-42."""
    train = blocks["train"][0]
    cols = if_columns(train.columns)
    scaler = StandardScaler().fit(train[cols])
    pca = PCA(n_components=N_COMPONENTS, random_state=seed).fit(scaler.transform(train[cols]))
    forest = IsolationForest(n_estimators=N_ESTIMATORS, contamination=CONTAMINATION,
                             random_state=seed).fit(pca.transform(scaler.transform(train[cols])))
    return {"cols": cols, "scaler": scaler, "pca": pca, "forest": forest}


def score(saved, feats):
    """Same score as isolation_forest.fit_scores() line 43."""
    cols = saved["cols"]
    return -saved["forest"].score_samples(saved["pca"].transform(saved["scaler"].transform(feats[cols])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=pathlib.Path, required=True)
    parser.add_argument("--saved-scores", type=pathlib.Path, required=True,
                        help="folder with seed<k>/iforest_hourly_scores.csv from isolation_forest.py")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    blocks = load_blocks(str(args.dataset_path), time_features=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
              "n_components": N_COMPONENTS, "n_estimators": N_ESTIMATORS,
              "contamination": CONTAMINATION, "seeds": {}}

    for seed in args.seeds:
        path = args.out_dir / f"seed{seed}.joblib"
        joblib.dump(fit(blocks, seed), path)

        saved = joblib.load(path)  # check the file on disk, not the object in memory
        old = pd.read_csv(args.saved_scores / f"seed{seed}" / "iforest_hourly_scores.csv")
        diffs = {}
        for source in ("cal", "val", "d2"):
            feats, meta = blocks[source]
            new = pd.Series(score(saved, feats), index=meta["row"].to_numpy())
            ref = old[old.source == source].set_index("row")["if_score"]
            if len(ref) != len(new) or not ref.index.isin(new.index).all():
                raise ValueError(f"seed {seed} {source}: saved scores cover different hours")
            diffs[source] = float(np.abs(new.loc[ref.index].to_numpy() - ref.to_numpy()).max())
        print(f"seed {seed}: {len(saved['cols'])} columns, saved {path.name}, "
              + ", ".join(f"{s} max|diff| {d:.1e}" for s, d in diffs.items()))
        if max(diffs.values()) > TOLERANCE:
            raise SystemExit(f"seed {seed}: reloaded model does not reproduce the saved scores")
        report["seeds"][str(seed)] = {"file": path.name, "max_abs_diff": diffs}

    (args.out_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    print("all seeds match; wrote", args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
