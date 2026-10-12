from scripts.generate_sample import generate
import json
from pathlib import Path
from alembic import command
from alembic.config import Config
from app.db.database import SessionLocal
from app.db.models import Product
from app.ml.train import train
from app.ml.ingestion import normalize_record
from app.ml.inference import Scorer
from app.db.models import Score
path=Path("data/sample/products.jsonl"); path.parent.mkdir(parents=True,exist_ok=True)
rows=generate(); path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
command.upgrade(Config("alembic.ini"),"head")
with SessionLocal() as db:
    for row in rows: db.merge(Product(**normalize_record(row)))
    db.commit()
metrics=train(__import__("pandas").DataFrame([normalize_record(r) for r in rows])); print("Demo ready; benchmark metrics:",metrics)
scorer=Scorer("models/pricing_risk.joblib")
with SessionLocal() as db:
    predictions=scorer.score(__import__("pandas").DataFrame([normalize_record(r) for r in rows]))
    for prediction in predictions:
        db.add(Score(product_id=prediction["product_id"],risk_probability=prediction["risk_probability"],risk_score=prediction["risk_score"],risk_category=prediction["risk_category"],flagged=prediction["flagged"],category_deviation=prediction["category_deviation"],evidence=json.dumps(prediction["evidence"]),model_version=prediction["model_version"]))
    db.commit()
print("Scored",len(predictions),"synthetic sample products")
