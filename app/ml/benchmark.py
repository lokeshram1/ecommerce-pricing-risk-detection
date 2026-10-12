import numpy as np

def inject_anomalies(df, seed=42, fraction=.15):
    """Inject after identity split; copied rows retain original_price for audit."""
    out=df.copy(deep=True).reset_index(drop=True); out["original_price"]=out.price; out["label"]=0; out["injection_type"]="none"
    if not 0 <= fraction <= 1: raise ValueError("fraction must be between 0 and 1")
    eligible=out.index[out.price.notna() & (out.price>0)].to_numpy(); rng=np.random.default_rng(seed); n=min(len(eligible),int(round(len(out)*fraction)))
    if fraction > 0 and len(eligible) and n == 0: n = 1
    chosen=rng.choice(eligible,size=n,replace=False) if n else []
    for i in chosen:
        kind=("overpriced","underpriced","discount_conflict")[int(rng.integers(0,3))]; p=float(out.at[i,"price"])
        if kind=="overpriced": out.at[i,"price"]=p*float(rng.uniform(3,8))
        elif kind=="underpriced": out.at[i,"price"]=max(.01,p*float(rng.uniform(.05,.25)))
        else: out.at[i,"list_price"]=p*float(rng.uniform(.4,.8)); out.at[i,"discount"]=80.
        out.at[i,"label"]=1; out.at[i,"injection_type"]=kind
    return out
