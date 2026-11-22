"""Usage: python paper_protocol_check.py <repo root>. Reproduces the XGBoost and IF+PCA protocol of Nemade et al. (2026) on Dataset 2, then re-scores it out-of-sample."""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); ROOT = sys.argv[1]  # repo root
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgb_data import build_features, NOT_FEATURES

d2 = pd.read_csv(ROOT + "/data/processed/dataset2_eval.csv")
y = (d2.ATT_FLAG == 1).to_numpy().astype(int)          # paper: 1 = attack, -999 = normal
real = (d2.attack_id > 0).to_numpy()                    # real attack windows (truth)
F = build_features(d2).iloc[23:].reset_index(drop=True)  # drop warm-up rows
y, real = y[23:], real[23:]
N = len(F); print("series", N, "attack-flag hours", y.sum(), "real-window hours", real.sum())


def best_f1_threshold(s, y):
    cands = np.linspace(s.min(), s.max(), 200)
    f1 = []
    for t in cands:
        p = s > t
        tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0)
    return cands[int(np.argmax(f1))]


def report(tag, s, yy, thr):
    p = s > thr
    tp = (p & (yy == 1)).sum(); fp = (p & (yy == 0)).sum(); fn = (~p & (yy == 1)).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    print(f"  {tag:34s} n={len(yy):5d} pos={int(yy.sum()):3d} P={pr:.3f} R={rc:.3f} F1={2*pr*rc/max(pr+rc,1e-9):.3f} "
          f"AUC={roc_auc_score(yy, s):.3f} AP={average_precision_score(yy, s):.3f}")


for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = rng.choice(np.flatnonzero(y == 0), size=5 * len(pos), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))            # 1,314 rows, chronological
    cut = int(0.8 * len(idx)); tr, te = idx[:cut], idx[cut:]
    spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                          tree_method="hist", random_state=seed).fit(F.iloc[tr], y[tr])
    s = m.predict_proba(F)[:, 1]
    thr = best_f1_threshold(s, y)                         # chosen on the SAME full series
    print(f"seed {seed}: train positives {int(y[tr].sum())}, held-out-sample positives {int(y[te].sum())}")
    report("PAPER: full series (in-sample)", s, y, thr)
    unused = np.setdiff1d(np.arange(N), tr)
    report("rows the model never trained on", s[unused], y[unused], thr)
    report("chronological 20% of sample only", s[te], y[te], thr)
    # honest: only attacks whose hours the model never saw (later-period positives)
    late = np.arange(N) > pos[int(0.8 * len(pos))]
    mask = late & ~np.isin(np.arange(N), tr)
    if y[mask].sum() > 0:
        report("unseen LATE attack period", s[mask], y[mask], thr)

# ---- Isolation Forest + PCA, paper recipe: trained on the -999 rows of Dataset 2 itself.
sens = [c for c in d2.columns if c not in NOT_FEATURES]
X = d2[sens].iloc[23:].reset_index(drop=True)
X2 = pd.concat([X, X.diff().fillna(0).add_suffix("_d1")], axis=1)
Z = StandardScaler().fit_transform(X2)
Zp = PCA(n_components=20, random_state=0).fit_transform(Z)
iso = IsolationForest(n_estimators=500, contamination=0.055, random_state=0).fit(Zp[y == 0])
s_if = -iso.score_samples(Zp)
thr = best_f1_threshold(s_if, y)
print("Isolation Forest+PCA (trained on D2 -999 rows, scored on same series)")
report("vs ATT_FLAG labels (paper)", s_if, y, thr)
rw = real.astype(int)
print("  AUC vs REAL windows (-999 attack hours counted as attacks):", round(roc_auc_score(rw, s_if), 3))
# the same IF trained only on Dataset 1, scored on Dataset 2 (what the project calls clean)
tr1 = pd.read_csv(ROOT + "/data/processed/train_scaled.csv")
X1 = tr1[sens]; X1 = pd.concat([X1, X1.diff().fillna(0).add_suffix("_d1")], axis=1)
sc = StandardScaler().fit(X1); pca = PCA(n_components=20, random_state=0).fit(sc.transform(X1))
iso1 = IsolationForest(n_estimators=500, contamination=0.055, random_state=0).fit(pca.transform(sc.transform(X1)))
s1 = -iso1.score_samples(pca.transform(sc.transform(X2)))
print("Isolation Forest+PCA trained on DATASET 1 only (project setting)")
print("  AUC vs ATT_FLAG labels:", round(roc_auc_score(y, s1), 3), "| AP:", round(average_precision_score(y, s1), 3))
print("  AUC vs REAL windows   :", round(roc_auc_score(rw, s1), 3))
