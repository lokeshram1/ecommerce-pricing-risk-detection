from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.db.database import Base
from app.db.models import Product, Score, Review

def test_product_merge_is_idempotent_scores_are_historical():
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    Base.metadata.create_all(engine); Session=sessionmaker(engine,expire_on_commit=False)
    with Session() as db:
        original={"product_id":"duplicate-1","title":"first title","category":"Audio","brand":"Acme","description":"item","features":"wireless","is_synthetic":True}
        db.merge(Product(**original)); db.commit()
        db.merge(Product(**{**original,"title":"updated title","price":120})); db.commit()
        product=db.get(Product,"duplicate-1")
        assert product.title=="updated title"
        assert db.scalar(select(func.count()).select_from(Product))==1
        for risk in (20,70):
            db.add(Score(product_id=product.product_id,risk_probability=risk/100,risk_score=risk,risk_category="Low" if risk<40 else "High",flagged=risk>=50,category_deviation=0,evidence="[]",model_version="test"))
        db.commit()
        scores=db.scalars(select(Score).order_by(Score.id)).all()
        assert len(scores)==2
        db.add(Review(score_id=scores[-1].id,decision="Expected Pricing",notes="bundle listing",reviewer="qa")); db.commit()
        assert scores[-1].review.decision=="Expected Pricing"
    Base.metadata.drop_all(engine); engine.dispose()
