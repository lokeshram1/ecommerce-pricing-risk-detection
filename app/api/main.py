import json
from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from app.db.database import get_db
from app.db.models import Product, Score, Review
from app.ml.inference import Scorer
from app.core.config import settings
import pandas as pd

app=FastAPI(title="E-Commerce Pricing Risk Detection System",version="0.1.0",description="A human-review prioritization tool; scores are not proof of wrongdoing.")
scorer=Scorer(f"{settings.model_dir}/pricing_risk.joblib")

class ProductIn(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    product_id:str=Field(min_length=1,max_length=128); title:str=""; category:str="Unknown"; price:float|None=Field(default=None,gt=0); list_price:float|None=Field(default=None,gt=0); discount:float|None=Field(default=None,ge=0,le=100); brand:str=""; description:str=""; features:str=""; rating:float|None=Field(default=None,ge=0,le=5); rating_count:float|None=Field(default=None,ge=0); is_synthetic:bool=False
    @field_validator("features",mode="before")
    @classmethod
    def features_to_text(cls,value): return " ".join(map(str,value)) if isinstance(value,(list,tuple)) else (value or "")
class ScoreRequest(BaseModel): product:ProductIn
class BatchRequest(BaseModel): products:list[ProductIn]=Field(min_length=1,max_length=100)
class ReviewIn(BaseModel): decision:str; notes:str=""; reviewer:str="reviewer"

def asdict(p): return {k:getattr(p,k) for k in Product.__table__.columns.keys()}
def persist_score(db,p,data):
    item=Product(**p); db.merge(item); db.flush()
    s=Score(product_id=p["product_id"],risk_probability=data["risk_probability"],risk_score=data["risk_score"],risk_category=data["risk_category"],flagged=data["flagged"],category_deviation=data["category_deviation"],evidence=json.dumps(data["evidence"]),model_version=data.get("model_version","unknown"))
    db.add(s); db.flush(); data["score_id"]=s.id; return data

@app.get("/health")
def health(db:Session=Depends(get_db)):
    try: db.execute(__import__("sqlalchemy").text("SELECT 1 FROM products LIMIT 1")); scorer._refresh_artifact(); return {"status":"ok","database":"ok","model_loaded":scorer.artifact is not None,"scoring_mode":scorer.mode}
    except Exception as e: raise HTTPException(503,detail=f"Database unavailable: {e}")

@app.get("/products")
def products(offset:int=Query(0,ge=0),limit:int=Query(50,ge=1,le=200),search:str|None=None,category:str|None=None,risk_level:str|None=Query(None,pattern="^(Low|Medium|High)$"),review_status:str|None=Query(None,pattern="^(Pending|Confirmed Anomaly|Expected Pricing|Needs More Information)$"),min_risk_score:int|None=Query(None,ge=0,le=100),db:Session=Depends(get_db)):
    latest=db.query(Score.product_id.label("product_id"),func.max(Score.id).label("latest_id")).group_by(Score.product_id).subquery()
    q=db.query(Product,Score).select_from(Product).outerjoin(latest,latest.c.product_id==Product.product_id).outerjoin(Score,Score.id==latest.c.latest_id).outerjoin(Review,Review.score_id==Score.id)
    if search: q=q.filter(Product.title.ilike(f"%{search}%")|Product.product_id.ilike(f"%{search}%"))
    if category: q=q.filter(Product.category==category)
    if risk_level: q=q.filter(func.lower(Score.risk_category)==risk_level.lower())
    if review_status=="Pending": q=q.filter(Review.id.is_(None))
    elif review_status: q=q.filter(func.lower(Review.decision)==review_status.lower())
    if min_risk_score is not None: q=q.filter(Score.risk_score>=min_risk_score)
    rows=[]
    for p,s in q.order_by(Product.product_id).offset(offset).limit(limit).all():
        review=s.review.decision if s and s.review else "Pending"
        rows.append({"product":asdict(p),"score":score_dict(s) if s else None,"review_status":review})
    return {"items":rows,"limit":limit,"offset":offset}

def score_dict(s): return {"score_id":s.id,"product_id":s.product_id,"risk_probability":s.risk_probability,"risk_score":s.risk_score,"risk_category":s.risk_category,"flagged":s.flagged,"category_deviation":s.category_deviation,"evidence":json.loads(s.evidence),"model_version":s.model_version,"created_at":s.created_at}
@app.get("/products/{product_id}")
def product_detail(product_id:str,db:Session=Depends(get_db)):
    p=db.get(Product,product_id)
    if not p: raise HTTPException(404,"Product not found")
    scores=db.query(Score).filter_by(product_id=product_id).order_by(desc(Score.created_at)).all()
    return {"product":asdict(p),"latest_score":score_dict(scores[0]) if scores else None,"history":[score_dict(s) for s in scores]}

@app.post("/score")
def score_one(request:ScoreRequest,db:Session=Depends(get_db)):
    p=request.product.model_dump(); data=scorer.score(pd.DataFrame([p]))[0]; result=persist_score(db,p,data); _commit(db); return result
def _commit(db):
    try: db.commit(); return True
    except Exception as e: db.rollback(); raise HTTPException(503,f"Could not persist scoring result: {e}")

@app.post("/score/batch")
def score_batch(request:BatchRequest,db:Session=Depends(get_db)):
    ps=[x.model_dump() for x in request.products]; results=scorer.score(pd.DataFrame(ps)); out=[persist_score(db,p,d) for p,d in zip(ps,results)]; _commit(db); return {"items":out}

@app.get("/risk/summary")
def summary(db:Session=Depends(get_db)):
    rows=db.query(Score).all(); counts={k:sum(s.risk_category==k for s in rows) for k in ("Low","Medium","High")}
    return {"total_scored":len(rows),"risk_counts":counts,"flagged_percent":100*sum(s.flagged for s in rows)/len(rows) if rows else 0,"average_risk_score":sum(s.risk_score for s in rows)/len(rows) if rows else 0}
@app.get("/metrics")
def metrics():
    try:
        with open(f"{settings.model_dir}/metrics.json",encoding="utf8") as f: return json.load(f)
    except FileNotFoundError: raise HTTPException(404,"No benchmark metrics. Run python scripts/train_model.py first.")
@app.post("/reviews/{score_id}")
def review(score_id:int,request:ReviewIn,db:Session=Depends(get_db)):
    allowed={"Confirmed Anomaly","Expected Pricing","Needs More Information"}
    if request.decision not in allowed: raise HTTPException(422,f"decision must be one of: {', '.join(sorted(allowed))}")
    s=db.get(Score,score_id)
    if not s: raise HTTPException(404,"Score not found")
    if s.review: raise HTTPException(409,"A decision is already recorded for this score")
    r=Review(score_id=score_id,decision=request.decision,notes=request.notes,reviewer=request.reviewer); db.add(r); _commit(db); return {"review_id":r.id,"score_id":score_id,"decision":r.decision}
@app.get("/reviews")
def reviews(limit:int=Query(100,ge=1,le=500),db:Session=Depends(get_db)):
    return [{"id":r.id,"score_id":r.score_id,"decision":r.decision,"notes":r.notes,"reviewer":r.reviewer,"created_at":r.created_at} for r in db.query(Review).order_by(desc(Review.created_at)).limit(limit)]
