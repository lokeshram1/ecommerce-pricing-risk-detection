"""JSONL ingestion. Source records are preserved; synthetic records are labeled explicitly."""
import json, logging, math
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)
FIELDS = ["product_id", "title", "category", "price", "list_price", "discount", "brand", "description", "features", "rating", "rating_count", "is_synthetic"]

def normalize_record(r):
    pid = r.get("product_id", r.get("asin", r.get("parent_asin")))
    if not pid: raise ValueError("missing product identifier")
    price = _number(r.get("price"))
    if price is not None and price <= 0: price = None
    lp = _number(r.get("list_price")); lp = lp if lp is not None and lp > 0 else None
    discount = _number(r.get("discount")); discount = discount if discount is not None and 0 <= discount <= 100 else None
    if discount is None and lp and price and lp >= price: discount = (lp-price)/lp*100
    rating=_number(r.get("rating")); rating=rating if rating is not None and 0<=rating<=5 else None
    rating_count=_number(r.get("rating_count")); rating_count=rating_count if rating_count is not None and rating_count>=0 else None
    return {"product_id": str(pid), "title": str(r.get("title") or ""), "category": _category(r.get("category")), "price": price, "list_price": lp, "discount": discount, "brand": str(r.get("brand") or ""), "description": _text(r.get("description")), "features": _text(r.get("features")), "rating": rating, "rating_count": rating_count, "is_synthetic": _boolean(r.get("is_synthetic", False))}

def _number(v):
    if isinstance(v, (int,float)):
        value=float(v)
        return value if math.isfinite(value) else None
    try:
        value=float(str(v).replace("$", "").replace(",", "").replace("%", "").strip()) if v not in (None, "") else None
        return value if value is None or math.isfinite(value) else None
    except (TypeError, ValueError): return None
def _text(v): return " ".join(map(str,v)) if isinstance(v,list) else str(v or "")
def _category(v): return str(v[-1] if v else "Unknown") if isinstance(v,list) else str(v or "Unknown")
def _boolean(value):
    if isinstance(value,bool): return value
    return str(value).strip().lower() in {"1","true","yes","synthetic"}

def read_jsonl(path):
    records=[]; rejected=0
    for n,line in enumerate(Path(path).open(encoding="utf-8"),1):
        try: records.append(normalize_record(json.loads(line)))
        except Exception as e: rejected += 1; log.warning("Rejected line %d: %s", n, e)
    df=pd.DataFrame(records, columns=FIELDS); before=len(df); df=df.drop_duplicates("product_id", keep="last")
    quality={"missing_price":int(df.price.isna().sum()),"missing_title":int(df.title.fillna("").eq("").sum()),"unknown_category":int(df.category.eq("Unknown").sum())}
    log.info("Ingestion: accepted=%d unique=%d duplicates=%d rejected=%d data_quality=%s",len(records),len(df),before-len(df),rejected,quality)
    return df, {"accepted":len(records),"unique":len(df),"duplicates":before-len(df),"rejected":rejected,"data_quality":quality}
