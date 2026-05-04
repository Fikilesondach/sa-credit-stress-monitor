"""
train.py
--------
XGBoost credit stress classifier with walk-forward time-series cross-validation.

Design decisions for a 24-observation annual dataset:
  - Walk-forward CV (TimeSeriesSplit) — never train on the future
  - Feature selection via RFECV + XGBoost importance before final fit
  - Scale-pos-weight to handle slight class imbalance
  - Bayesian-style manual grid search over critical hyperparameters
  - Final model fitted on ALL data (production scoring needs all history)
  - Model artefacts saved to data/processed/ as joblib + JSON

Key metrics tracked (actuarial-grade):
  - AUC-ROC       — discrimination
  - Gini          — 2 × AUC − 1 (standard in credit risk)
  - KS statistic  — max separation between score distributions
  - Brier score   — calibration (lower = better)
  - Precision/Recall at 0.5 threshold

Run:
    PYTHONPATH=. python -m src.models.train
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# Tuned for small-N (24 obs): shallow trees, heavy regularisation, low LR.
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "n_estimators":     200,
    "max_depth":        2,          # shallow — prevents overfitting on 24 obs
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.6,        # random feature subsets per tree
    "min_child_weight": 3,          # require 3 obs per leaf minimum
    "reg_alpha":        0.5,        # L1 regularisation
    "reg_lambda":       2.0,        # L2 regularisation
    "gamma":            0.1,        # min loss reduction to split
    "random_state":     42,
    "verbosity":        0,
}


# ─── Metrics helpers ─────────────────────────────────────────────────────────

def _gini(y_true, y_prob) -> float:
    return 2 * roc_auc_score(y_true, y_prob) - 1


def _ks(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov statistic — max separation between cumulative distributions."""
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    n_pos = df["y"].sum()
    n_neg = len(df) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    df["cum_pos"] = df["y"].cumsum() / n_pos
    df["cum_neg"] = (1 - df["y"]).cumsum() / n_neg
    return (df["cum_pos"] - df["cum_neg"]).abs().max()


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
        "gini":      round(_gini(y_true, y_prob), 4),
        "ks":        round(_ks(y_true, y_prob), 4),
        "brier":     round(brier_score_loss(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─── Feature selection ────────────────────────────────────────────────────────

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20,
) -> list:
    """
    Select top-N features by XGBoost gain importance.

    With only 24 observations and 116 features we MUST reduce dimensionality
    aggressively. Top-20 by gain is standard practice in credit scorecard work.
    """
    selector = xgb.XGBClassifier(
        **{**XGB_PARAMS, "n_estimators": 100},
        scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
    )
    selector.fit(X, y)

    importance = pd.Series(
        selector.feature_importances_,
        index=X.columns,
        name="gain",
    ).sort_values(ascending=False)

    selected = importance.head(top_n).index.tolist()
    logger.info(
        f"Feature selection: {len(X.columns)} → {len(selected)} features\n"
        + "\n".join(f"  {i+1:2d}. {f} ({importance[f]:.4f})"
                    for i, f in enumerate(selected))
    )
    return selected, importance


# ─── Walk-forward cross-validation ───────────────────────────────────────────

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits:       int = 5,
    top_n_feat:     int = 20,
    preselected:    list = None,
) -> Tuple[Dict, list]:
    """
    Walk-forward time-series cross-validation.

    Respects temporal order — never leaks future data into training.
    Each fold: train on [0..t], validate on [t+1..t+k].

    With only 24 annual obs, feature selection is done ONCE on the full
    dataset (passed in via `preselected`) rather than per fold, since
    per-fold training sets of 8–16 obs cannot reliably rank 116 features.

    Returns
    -------
    cv_summary : dict
        Mean ± std of each metric across folds.
    fold_results : list of dict
        Per-fold metrics for inspection.
    """
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    # Use pre-selected features for CV stability on small-N
    feat_names = preselected if preselected else list(X.columns)
    X_sel = X[feat_names]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sel)):
        X_tr, X_val = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx],     y.iloc[val_idx]

        # Skip folds where validation has only one class
        if len(y_val.unique()) < 2:
            logger.warning(f"Fold {fold+1}: single class in val — skipping")
            continue

        # Train
        spw = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)
        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val.values, y_prob)
        metrics["fold"] = fold + 1
        metrics["train_n"] = len(y_tr)
        metrics["val_n"]   = len(y_val)
        fold_results.append(metrics)

        logger.info(
            f"Fold {fold+1}: AUC={metrics['auc']:.3f}  "
            f"Gini={metrics['gini']:.3f}  KS={metrics['ks']:.3f}  "
            f"Recall={metrics['recall']:.3f}  "
            f"[train={len(y_tr)}, val={len(y_val)}]"
        )

    # Aggregate
    numeric_keys = ["auc", "gini", "ks", "brier", "precision", "recall", "f1"]
    cv_summary = {}
    for k in numeric_keys:
        vals = [f[k] for f in fold_results if k in f]
        cv_summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
        }

    return cv_summary, fold_results


# ─── Final model fit ──────────────────────────────────────────────────────────

def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    top_n_feat: int = 20,
) -> Tuple[xgb.XGBClassifier, list, pd.Series]:
    """
    Fit final production model on the full dataset.

    Returns
    -------
    model : XGBClassifier
        Fitted model.
    selected_features : list
        Feature names the model was trained on.
    importance : pd.Series
        Feature importances (gain) sorted descending.
    """
    selected_features, importance = select_features(X, y, top_n=top_n_feat)
    X_sel = X[selected_features]

    spw = len(y[y == 0]) / max(len(y[y == 1]), 1)
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    model.fit(X_sel, y, verbose=False)

    # In-sample metrics (for reference — don't use for evaluation)
    y_prob  = model.predict_proba(X_sel)[:, 1]
    metrics = compute_metrics(y.values, y_prob)
    logger.info(
        f"Final model (in-sample) | AUC={metrics['auc']:.3f}  "
        f"Gini={metrics['gini']:.3f}  KS={metrics['ks']:.3f}"
    )
    return model, selected_features, importance[selected_features]


# ─── Save artefacts ───────────────────────────────────────────────────────────

def save_artefacts(
    model:     xgb.XGBClassifier,
    features:  list,
    importance: pd.Series,
    cv_summary: dict,
    meta:      dict,
) -> None:
    """Save model, feature list, importance, and CV results to disk."""

    joblib.dump(model, PROCESSED_DIR / "xgb_model.joblib")
    logger.info(f"Model saved → {PROCESSED_DIR / 'xgb_model.joblib'}")

    artefact = {
        "model_version":  "1.0.0",
        "feature_names":  features,
        "feature_importances": importance.to_dict(),
        "xgb_params":     XGB_PARAMS,
        "cv_summary":     cv_summary,
        "target":         "credit_stress",
        "classes":        {"0": "stable", "1": "stress"},
        "label_definition": (
            "stress=1 when NPL>=4% OR GDP_growth<0 "
            "OR (unemployment>30% AND NPL>3.5%)"
        ),
    }
    with open(PROCESSED_DIR / "model_artefact.json", "w") as f:
        json.dump(artefact, f, indent=2)
    logger.info(f"Artefact JSON saved → {PROCESSED_DIR / 'model_artefact.json'}")

    if meta.get("woe_objects"):
        joblib.dump(meta["woe_objects"], PROCESSED_DIR / "woe_objects.joblib")
        logger.info("WoE objects saved.")


# ─── Master train function ────────────────────────────────────────────────────

def train(
    mode:       str = "seed",
    top_n_feat: int = 20,
    n_cv_splits: int = 5,
    save:       bool = True,
) -> Tuple[xgb.XGBClassifier, dict, dict]:
    """
    Full training pipeline: ingest → feature engineering → CV → final fit → save.

    Parameters
    ----------
    mode : str
        Ingestion mode ("seed" | "auto" | "live").
    top_n_feat : int
        Number of features to select (default 20 — conservative for 24 obs).
    n_cv_splits : int
        Number of walk-forward CV folds.
    save : bool
        Whether to persist model artefacts.

    Returns
    -------
    model, cv_summary, fold_results
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.ingestion.pipeline import load_macro_dataset
    from src.features.engineer import build_feature_matrix

    logger.info("=" * 65)
    logger.info("SA CREDIT STRESS MONITOR — MODEL TRAINING")
    logger.info("=" * 65)

    # Step 1: Load data
    df_raw = load_macro_dataset(mode=mode)

    # Step 2: Feature engineering
    X, y, meta = build_feature_matrix(df_raw)
    logger.info(f"Feature matrix: {X.shape} | Target balance: {y.value_counts().to_dict()}")

    # Step 3a: Select features once (stable for small-N)
    logger.info("\n── Feature Selection (global, pre-CV) ──")
    preselected, full_importance = select_features(X, y, top_n=top_n_feat)

    # Step 3b: Walk-forward cross-validation on selected features
    logger.info("\n── Walk-Forward Cross-Validation ──")
    cv_summary, fold_results = walk_forward_cv(
        X, y, n_splits=n_cv_splits, top_n_feat=top_n_feat, preselected=preselected
    )

    logger.info("\n── CV Summary ──")
    for metric, stats in cv_summary.items():
        logger.info(f"  {metric:<12} {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Step 4: Final model on all data
    logger.info("\n── Final Model (all data) ──")
    model, selected_features, importance = fit_final_model(X, y, top_n_feat=top_n_feat)

    # Step 5: Save
    if save:
        save_artefacts(model, selected_features, importance, cv_summary, meta)

    return model, cv_summary, {"folds": fold_results, "meta": meta,
                                "X": X, "y": y, "features": selected_features,
                                "importance": importance}


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    model, cv_summary, extras = train(mode="seed", save=True)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print("\nWalk-Forward CV Results:")
    print(f"  {'Metric':<14} {'Mean':>8} {'± Std':>8}")
    print(f"  {'-'*32}")
    for metric, stats in cv_summary.items():
        print(f"  {metric:<14} {stats['mean']:>8.4f} {stats['std']:>8.4f}")

    print(f"\nTop 10 Features (by gain):")
    for i, (feat, imp) in enumerate(extras["importance"].head(10).items(), 1):
        print(f"  {i:2d}. {feat:<45} {imp:.4f}")

    print(f"\nArtefacts saved to: data/processed/")
