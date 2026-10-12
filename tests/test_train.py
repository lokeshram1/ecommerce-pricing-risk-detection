from app.ml.train import train
from app.ml.ingestion import normalize_record
from scripts.generate_sample import generate
from app.ml.inference import Scorer

def test_training_persists_test_metrics(tmp_path):
    df=__import__("pandas").DataFrame([normalize_record(r) for r in generate(240)])
    scorer=Scorer(str(tmp_path/"pricing_risk.joblib"))
    assert scorer.mode=="rule_baseline"
    metrics=train(df,str(tmp_path),seed=2)
    assert 0<=metrics["precision"]<=1 and 0<=metrics["recall"]<=1
    assert (tmp_path/"pricing_risk.joblib").exists() and metrics["synthetic_labels"] is True
    assert (tmp_path/"metrics.json").exists()
    assert metrics["rule_baseline"]["confusion_matrix"]
    prediction=scorer.score(df.iloc[:2])
    assert len(prediction)==2 and prediction[0]["scoring_mode"]=="xgboost"
