import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from functools import lru_cache
import os

NUMERIC = ["log_price", "category_deviation", "category_percentile", "price_median_ratio", "discount", "missing_price", "missing_description", "missing_brand", "rating", "rating_count"]

def text_matrix(df, dimensions=32, mode=None, model_name=None):
    def as_text(value):
        if value is None or (isinstance(value, float) and np.isnan(value)): return ""
        if isinstance(value, (list, tuple)): return " ".join(map(str, value))
        return str(value)
    texts=[" ".join(as_text(row.get(k)) for k in ("title", "description", "features")) for row in df.to_dict("records")]
    mode = (mode or os.getenv("EMBEDDING_MODE", "lightweight")).lower()
    model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if mode == "transformer":
        return _transformer_matrix(tuple(texts), model_name)
    return HashingVectorizer(n_features=dimensions, alternate_sign=False, norm="l2", ngram_range=(1,2)).transform(texts).toarray().astype("float32")

@lru_cache(maxsize=2)
def _load_transformer(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def _transformer_matrix(texts, model_name):
    # Explicit opt-in may download the configured model once; default mode remains offline.
    return np.asarray(_cached_transformer_batch(tuple(texts),model_name),dtype="float32")

@lru_cache(maxsize=64)
def _cached_transformer_batch(texts,model_name):
    model=_load_transformer(model_name)
    vectors=model.encode(list(texts),batch_size=32,normalize_embeddings=True,show_progress_bar=False)
    return tuple(tuple(float(value) for value in vector) for vector in vectors)

class FeatureEngineer:
    def fit(self, df):
        prices=pd.to_numeric(df.price, errors="coerce"); self.stats={}
        for cat, group in df.assign(_p=prices).groupby("category", dropna=False):
            p=group._p.dropna()
            if len(p):
                med=float(p.median()); mad=float((p-med).abs().median())
                self.stats[str(cat)]={"median":med,"scale":max(1.4826*mad, med*.1, 1.0),"values":sorted(p.tolist())}
        self.global_median=float(prices.dropna().median()) if prices.notna().any() else 1.0
        return self
    def transform(self, df):
        out=[]
        for r in df.to_dict("records"):
            cat=r.get("category"); cat="Unknown" if cat is None or pd.isna(cat) else str(cat)
            s=self.stats.get(cat, {"median":self.global_median,"scale":max(self.global_median*.5,1.),"values":[]})
            raw=r.get("price"); valid=raw is not None and not pd.isna(raw) and float(raw)>0; p=float(raw) if valid else 0.; median=s["median"]; vals=s["values"]
            def number(key):
                value=r.get(key); return 0. if value is None or pd.isna(value) else float(value)
            description=r.get("description"); brand=r.get("brand")
            description_missing=description is None or pd.isna(description) or not str(description).strip()
            brand_missing=brand is None or pd.isna(brand) or not str(brand).strip()
            out.append([np.log1p(p) if valid else 0., (p-median)/s["scale"] if valid else 0., sum(v<=p for v in vals)/len(vals) if valid and vals else .5, p/median if valid and median>0 else 0., number("discount"), float(not valid), float(description_missing), float(brand_missing), number("rating"), number("rating_count")])
        return np.asarray(out,dtype="float32")
    def fit_transform(self, df): return self.fit(df).transform(df)
