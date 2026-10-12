"""Model inference plus clearly identified, offline category-rule fallback."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.features import text_matrix


class Scorer:
    def __init__(self, artifact="models/pricing_risk.joblib", reference_data="data/sample/products.jsonl"):
        self.artifact_path = Path(artifact)
        self.artifact = None
        self.artifact_mtime = None
        self.mode = "xgboost" if self.artifact else "rule_baseline"
        self.medians = {}
        self.global_median = 1.0
        self.fallback_reason = None
        self._refresh_artifact()
        if not self.artifact and Path(reference_data).exists():
            reference = pd.read_json(reference_data, lines=True)
            if "category" in reference and "price" in reference:
                self.medians = reference.groupby("category").price.median().dropna().to_dict()
                valid = pd.to_numeric(reference.price, errors="coerce").dropna()
                if len(valid): self.global_median = float(valid.median())

    def score(self, df):
        self._refresh_artifact()
        if self.artifact:
            fe = self.artifact["features"]
            try:
                emb = text_matrix(df, mode=self.artifact.get("embedding_mode"), model_name=self.artifact.get("embedding_model"))
            except Exception as exc:
                # A missing optional transformer/model cache must not block the lightweight demo.
                self.fallback_reason = f"embedding_unavailable: {type(exc).__name__}"
                self.mode = "rule_baseline_embedding_unavailable"
                emb = None
            if emb is not None:
                X = np.hstack([fe.transform(df), emb])
                probabilities = self.artifact["model"].predict_proba(X)[:, 1]
                threshold = self.artifact["threshold"]
            else:
                probabilities, threshold = self._baseline(df, fe.stats, fe.global_median)
        else:
            probabilities, threshold = self._baseline(df)

        rows = []
        for (_, row), probability in zip(df.iterrows(), probabilities):
            category = str(row.get("category") or "Unknown")
            price_value = row.get("price")
            price = float(price_value) if price_value is not None and pd.notna(price_value) else None
            if self.artifact:
                stats = self.artifact["features"].stats
                median = stats.get(category, {}).get("median", self.artifact["features"].global_median)
            else:
                median = self.medians.get(category, self.global_median)
            deviation = (price-median)/max(median, 1) if price is not None else None
            evidence = []
            if deviation is not None and deviation > 1: evidence.append(f"Price is more than twice the category median (${median:.2f}).")
            if deviation is not None and deviation < -.75: evidence.append(f"Price is below one quarter of the category median (${median:.2f}).")
            list_price = row.get("list_price")
            if list_price is not None and pd.notna(list_price) and price is not None and float(list_price) < price:
                evidence.append("List price is below the observed price; discount relationship needs review.")
            brand, description = row.get("brand"), row.get("description")
            if brand is None or pd.isna(brand) or not str(brand).strip() or description is None or pd.isna(description) or not str(description).strip():
                evidence.append("Some product attributes are incomplete.")
            if not evidence: evidence.append("Potential pricing anomaly; compare the listing with relevant products before taking action.")
            risk = int(round(float(probability)*100))
            rows.append({
                "product_id": str(row.product_id), "title": str(row.get("title") or ""), "price": price,
                "category": category, "category_deviation": deviation, "risk_probability": float(probability),
                "risk_score": risk, "risk_category": "High" if risk >= 70 else "Medium" if risk >= 40 else "Low",
                "review_priority": "Urgent" if risk >= 80 else "High" if risk >= 60 else "Normal",
                "flagged": bool(probability >= threshold), "evidence": evidence, "scoring_mode": self.mode,
                "model_version": self.artifact.get("version", "unknown") if self.mode == "xgboost" else "category-rule-v1",
            })
        return rows

    def _refresh_artifact(self):
        if not self.artifact_path.exists():
            return
        modified = self.artifact_path.stat().st_mtime_ns
        if self.artifact is None or modified != self.artifact_mtime:
            self.artifact = joblib.load(self.artifact_path)
            self.artifact_mtime = modified
            self.mode = "xgboost"

    def _baseline(self, df, feature_stats=None, global_median=None):
        medians=[]
        for row in df.to_dict("records"):
            category=str(row.get("category") or "")
            if feature_stats is not None:
                median=feature_stats.get(category,{}).get("median",global_median)
            else:
                median=self.medians.get(category,self.global_median)
            medians.append(median or 1.0)
        prices=pd.to_numeric(df.price,errors="coerce").fillna(0).to_numpy(dtype=float)
        valid=pd.to_numeric(df.price,errors="coerce").notna().to_numpy() & (prices>0)
        deviations=np.asarray([abs(p-m)/max(m,1.0) if is_valid else 0.0 for p,m,is_valid in zip(prices,medians,valid)])
        # This is a monotonic rule strength, not a calibrated probability.
        return np.clip((deviations-.35)/.9,.01,.99),(.7-.35)/.9
