"""Walk-forward training + evaluation of the win-probability model.

Trains on models/dataset.parquet (build with `python -m bot.ml.data`), evaluates
strictly out-of-sample with expanding walk-forward folds (never shuffled — that
would leak regime), and reports:

  - calibration metrics (AUC / Brier / log-loss) for:
      logistic regression, monotonic gradient boosting, and the closed-form
      GBM analytic baseline p = Phi(lead_z)  (the model this bot used to have)
  - a reliability table for the best model
  - a first-trigger trading simulation on test windows: the current GATE stack
    vs model-threshold entries, win% and trades/day

Then refits the winner on ALL data (with isotonic calibration) and saves the
artifact to models/model.joblib for the live shadow scorer (bot/ml/model.py).

Run: python -m bot.ml.train
"""
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, MONOTONE
from .data import DATASET_PATH, MODELS_DIR

ARTIFACT_PATH = os.path.join(MODELS_DIR, "model.joblib")


def make_models():
    mono = [MONOTONE.get(f, 0) for f in FEATURES]
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000)),
        "hgb_mono": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=200, l2_regularization=1.0,
            monotonic_cst=mono, random_state=7),
    }


def gates_pass(row) -> bool:
    """The current live gate stack (minus the odds cap, which has no history):
    HA direction + AO agree + RSI side + persistence + minute>=5 + |lead|>=3bps."""
    side = 1 if row.ha_signed > 0 else -1
    if row.ao_signed == 0 or (1 if row.ao_signed > 0 else -1) != side:
        return False
    if (1 if row.rsi >= 50 else -1) != side:
        return False
    if row.lead_bps == 0 or (1 if row.lead_bps > 0 else -1) != side:
        return False
    return row.elapsed_min >= 5 and abs(row.lead_bps) >= 3


def simulate(df: pd.DataFrame, p: np.ndarray, taus, days: float):
    """First-trigger-per-window entries. Returns rows of (label, trades, /day, win%)."""
    d = df[["window_start", "elapsed_min", "up_win", "ha_signed", "ao_signed",
            "rsi", "lead_bps"]].copy()
    d["p"] = p
    d = d.sort_values(["window_start", "elapsed_min"])
    out = []

    # gate baseline
    wins = trades = 0
    for _, g in d.groupby("window_start", sort=False):
        for row in g.itertuples(index=False):
            if gates_pass(row):
                side_up = row.lead_bps > 0
                wins += int(side_up == bool(row.up_win))
                trades += 1
                break
    out.append(("gates (min5 + 3bps)", trades, trades / days,
                100 * wins / trades if trades else float("nan")))

    for tau in taus:
        wins = trades = 0
        for _, g in d.groupby("window_start", sort=False):
            for row in g.itertuples(index=False):
                conf = max(row.p, 1 - row.p)
                if conf >= tau:
                    side_up = row.p >= 0.5
                    wins += int(side_up == bool(row.up_win))
                    trades += 1
                    break
        out.append((f"model p>={tau:.2f}", trades, trades / days,
                    100 * wins / trades if trades else float("nan")))
    return out


def main():
    df = pd.read_parquet(DATASET_PATH).sort_values(["window_start", "elapsed_min"]).reset_index(drop=True)
    windows = np.sort(df.window_start.unique())
    n_win = len(windows)
    print(f"{len(df)} samples, {n_win} windows "
          f"({(windows[-1] - windows[0]) / 86_400_000:.0f} days), "
          f"base P(up)={df.up_win.mean():.3f}\n")

    # expanding walk-forward: train on first 50% of windows, then 5 test slices of 10%
    bounds = [windows[int(n_win * q)] for q in (0.5, 0.6, 0.7, 0.8, 0.9)] + [windows[-1] + 1]
    X_all = df[FEATURES].to_numpy()
    y_all = df.up_win.to_numpy()

    oos = {name: [] for name in ("logistic", "hgb_mono")}
    oos_idx = []
    for k in range(5):
        tr = df.window_start < bounds[k]
        te = (df.window_start >= bounds[k]) & (df.window_start < bounds[k + 1])
        oos_idx.append(np.where(te)[0])
        for name, mdl in make_models().items():
            cal = CalibratedClassifierCV(mdl, method="isotonic", cv=3)
            cal.fit(X_all[tr.to_numpy()], y_all[tr.to_numpy()])
            oos[name].append(cal.predict_proba(X_all[te.to_numpy()])[:, 1])
        print(f"fold {k + 1}/5 done ({te.sum()} test samples)", flush=True)

    idx = np.concatenate(oos_idx)
    y = y_all[idx]
    test_df = df.iloc[idx]
    days = test_df.window_start.nunique() / 96.0

    preds = {name: np.concatenate(chunks) for name, chunks in oos.items()}
    # analytic GBM baseline — the closed-form model this bot used to run
    preds["gbm_analytic"] = norm.cdf(test_df.lead_z.to_numpy())

    print(f"\n== out-of-sample calibration ({len(y)} samples, {days:.0f} days) ==")
    print(f"{'model':>14} {'AUC':>8} {'Brier':>8} {'LogLoss':>9}")
    for name, p in preds.items():
        p_c = np.clip(p, 1e-6, 1 - 1e-6)
        print(f"{name:>14} {roc_auc_score(y, p_c):>8.4f} "
              f"{brier_score_loss(y, p_c):>8.4f} {log_loss(y, p_c):>9.4f}")

    best = min(preds, key=lambda n: brier_score_loss(y, np.clip(preds[n], 1e-6, 1 - 1e-6)))
    print(f"\nbest by Brier: {best}")

    print("\n== reliability (best model), 10 bins ==")
    p = preds[best]
    bins = np.clip((p * 10).astype(int), 0, 9)
    print(f"{'bin':>10} {'n':>7} {'mean_p':>8} {'actual':>8}")
    for b in range(10):
        m = bins == b
        if m.sum() < 30:
            continue
        print(f"{b / 10:.1f}-{(b + 1) / 10:.1f} {m.sum():>7} {p[m].mean():>8.3f} {y[m].mean():>8.3f}")

    print("\n== first-trigger trading simulation on test windows ==")
    print(f"{'strategy':>22} {'trades':>7} {'/day':>7} {'win%':>7}")
    for label, tr_n, per_day, win in simulate(test_df, preds[best], (0.75, 0.80, 0.85, 0.90), days):
        print(f"{label:>22} {tr_n:>7} {per_day:>7.1f} {win:>7.1f}")

    # ── final artifact: refit best learner on ALL data ──────────────────────────
    if best == "gbm_analytic":
        print("\nanalytic GBM won — no learned model is worth shipping; artifact NOT saved.")
        return
    final = CalibratedClassifierCV(make_models()[best], method="isotonic", cv=3)
    final.fit(X_all, y_all)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        "model": final, "features": FEATURES, "kind": best,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df), "n_windows": n_win,
        "oos_brier": float(brier_score_loss(y, np.clip(preds[best], 1e-6, 1 - 1e-6))),
    }, ARTIFACT_PATH)
    print(f"\nartifact saved: {ARTIFACT_PATH} (kind={best})")


if __name__ == "__main__":
    main()
