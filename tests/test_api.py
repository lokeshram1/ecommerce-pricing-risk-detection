import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.main import app, scorer
from app.db.database import Base, get_db
from app.ml.inference import Scorer

@pytest.fixture
def client(monkeypatch):
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    TestingSession=sessionmaker(bind=engine,autoflush=False,expire_on_commit=False)
    Base.metadata.create_all(engine)
    def override_db():
        db=TestingSession()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db]=override_db
    monkeypatch.setattr("app.api.main.scorer",Scorer(artifact="missing-test-model.joblib"))
    # Avoid app-global startup DB initialization; the request dependency is fully overridden.
    test_client=TestClient(app)
    yield test_client
    test_client.close()
    app.dependency_overrides.clear(); Base.metadata.drop_all(engine); engine.dispose()

PRODUCT={"product_id":"api-product-1","title":"Test headphones","category":"Headphones","price":420,"brand":"Test","description":"Test listing"}

def test_health_scoring_pagination_and_history(client):
    assert client.get("/health").json()["database"]=="ok"
    first=client.post("/score",json={"product":PRODUCT}); assert first.status_code==200
    score_id=first.json()["score_id"]
    second=client.post("/score",json={"product":{**PRODUCT,"price":430}}); assert second.status_code==200
    detail=client.get("/products/api-product-1").json()
    assert len(detail["history"])==2 and detail["latest_score"]["score_id"]==second.json()["score_id"]
    page=client.get("/products",params={"limit":1,"offset":1})
    assert page.status_code==200 and len(page.json()["items"])==0
    assert client.get("/products/no-such-product").status_code==404

def test_batch_validation_reviews_and_persistence(client):
    batch=client.post("/score/batch",json={"products":[{**PRODUCT,"product_id":f"batch-{i}"} for i in range(2)]})
    assert batch.status_code==200 and len(batch.json()["items"])==2
    assert client.post("/score/batch",json={"products":[{**PRODUCT,"product_id":f"big-{i}"} for i in range(101)]}).status_code==422
    assert client.post("/score",json={"product":{"product_id":""}}).status_code==422
    score_id=batch.json()["items"][0]["score_id"]
    assert client.post(f"/reviews/{score_id}",json={"decision":"not allowed"}).status_code==422
    review=client.post(f"/reviews/{score_id}",json={"decision":"Expected Pricing","notes":"Checked comparable bundles"})
    assert review.status_code==200
    assert client.post(f"/reviews/{score_id}",json={"decision":"Confirmed Anomaly"}).status_code==409
    assert len(client.get("/reviews").json())==1
    assert len(client.get("/products",params={"review_status":"Expected Pricing","risk_level":"High"}).json()["items"])==1
    assert client.get("/products",params={"risk_level":"Critical"}).status_code==422
    assert client.post("/score",json={"product":{**PRODUCT,"price":-5}}).status_code==422
