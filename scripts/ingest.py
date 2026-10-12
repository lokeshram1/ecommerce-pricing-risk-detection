import argparse
from app.ml.ingestion import read_jsonl
from app.db.init import init_db
from app.db.database import SessionLocal
from app.db.models import Product
p=argparse.ArgumentParser(); p.add_argument("input"); a=p.parse_args(); frame,stats=read_jsonl(a.input); init_db()
with SessionLocal() as db:
    for row in frame.to_dict("records"): db.merge(Product(**row))
    db.commit()
print(stats)
