"""Identity-isolated XGBoost benchmark training and held-out evaluation."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.ml.benchmark import inject_anomalies
from app.ml.features import FeatureEngineer, text_matrix


def split_product_identities(df, seed=42):
    """Deduplicate identity then return disjoint 70/15/15 frames before any fitting/injection."""
    clean=df.copy()
    if clean.product_id.isna().any():
        raise ValueError("Every training product must have a non-empty identity")
    clean["product_id"]=clean.product_id.astype(str)
    if clean.product_id.eq("").any():
        raise ValueError("Every training product must have a non-empty identity")
    clean=clean.drop_duplicates("product_id",keep="last").reset_index(drop=True)
    ids = clean.product_id.unique()
    if len(ids) < 20:
        raise ValueError(f"Training requires at least 20 unique products; got {len(ids)}")
    train_ids, hold_ids = train_test_split(ids, test_size=.30, random_state=seed)
    val_ids, test_ids = train_test_split(hold_ids, test_size=.50, random_state=seed)
    partitions = [
        clean[clean.product_id.astype(str).isin(part)].reset_index(drop=True)
        for part in (train_ids, val_ids, test_ids)
    ]
    identity_sets = [set(part.product_id.astype(str)) for part in partitions]
    assert not (identity_sets[0] & identity_sets[1] or identity_sets[0] & identity_sets[2] or identity_sets[1] & identity_sets[2])
    return tuple(partitions)


def _metric_block(y, pred):
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "flagged_fraction": float(np.mean(pred)) if len(pred) else 0.0,
    }


def train(df, out_dir="models", seed=42):
    base_train, base_val, base_test = split_product_identities(df, seed)
    # Injection happens only once identities have been allocated to their permanent split.
    dtr = inject_anomalies(base_train, seed)
    dva = inject_anomalies(base_val, seed + 1)
    dte = inject_anomalies(base_test, seed + 2)
    fe = FeatureEngineer().fit(base_train)

    def matrix(part):
        return np.hstack([fe.transform(part), text_matrix(part)])

    Xtr, Xv, Xte = map(matrix, (dtr, dva, dte))
    ytr, yv, yte = [part.label.to_numpy(dtype=int) for part in (dtr, dva, dte)]
    if len(np.unique(ytr)) < 2 or len(np.unique(yv)) < 2 or len(np.unique(yte)) < 2:
        raise ValueError("Each split must contain normal and injected records; use at least 20 products with valid prices")

    positives = max(int(ytr.sum()), 1)
    negatives = max(len(ytr) - positives, 1)
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=.07, subsample=.9,
        colsample_bytree=.9, scale_pos_weight=negatives / positives,
        eval_metric="logloss", random_state=seed, n_jobs=2,
    )
    model.fit(Xtr, ytr)

    val_prob = model.predict_proba(Xv)[:, 1]
    precision, recall, thresholds = precision_recall_curve(yv, val_prob)
    eligible = np.flatnonzero(precision[:-1] >= .75)
    threshold = float(thresholds[eligible[np.argmax(recall[eligible])]]) if len(eligible) else .5

    # Test probabilities are first evaluated only after the validation threshold is fixed.
    test_prob = model.predict_proba(Xte)[:, 1]
    predictions = (test_prob >= threshold).astype(int)
    model_metrics = _metric_block(yte, predictions)
    model_metrics["pr_auc"] = float(average_precision_score(yte, test_prob))

    medians = base_train.groupby("category").price.median().to_dict()
    deviations = []
    for price, category in zip(dte.price, dte.category):
        median = medians.get(category, fe.global_median)
        deviations.append(abs((float(price) - median) / max(median, 1.0)) if pd.notna(price) else 0.0)
    rule_predictions = (np.asarray(deviations) >= .7).astype(int)
    baseline_metrics = _metric_block(yte, rule_predictions)
    metrics = {
        "threshold": threshold,
        **model_metrics,
        "rule_baseline": baseline_metrics,
        "test_products": int(len(dte)),
        "split_products": {"train": len(base_train), "validation": len(base_val), "test": len(base_test)},
        "synthetic_labels": True,
        "threshold_selection": "validation precision >= 0.75; maximum validation recall, fallback 0.5",
        "random_seed": seed,
    }
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": fe, "threshold": threshold, "version": "0.2.0", "embedding_mode": os.getenv("EMBEDDING_MODE", "lightweight"), "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")}, path / "pricing_risk.joblib")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
