import json
import pandas as pd
from app.ml.inference import Scorer

def test_category_baseline_is_explicit_and_uses_reference_medians(tmp_path):
    reference=tmp_path/"reference.jsonl"
    reference.write_text("\n".join(json.dumps(row) for row in [
        {"category":"Audio","price":100},{"category":"Audio","price":120},
        {"category":"Storage","price":20},{"category":"Storage","price":30},
    ]))
    scorer=Scorer(tmp_path/"missing.joblib",reference)
    rows=pd.DataFrame([{"product_id":"normal","title":"","category":"Audio","price":110,"list_price":None,"brand":"x","description":"x"},{"product_id":"outlier","title":"","category":"Audio","price":1000,"list_price":None,"brand":"x","description":"x"}])
    output=scorer.score(rows)
    assert all(item["scoring_mode"]=="rule_baseline" for item in output)
    assert output[1]["risk_score"]>output[0]["risk_score"]

def test_transformer_failure_uses_labeled_rule_fallback(monkeypatch,tmp_path):
    class Features:
        stats={"Audio":{"median":100.0}}; global_median=100.0
    scorer=Scorer(tmp_path/"missing.joblib",tmp_path/"no-reference.jsonl")
    scorer.artifact={"features":Features(),"embedding_mode":"transformer","embedding_model":"missing","threshold":.5,"version":"test"}
    monkeypatch.setattr("app.ml.inference.text_matrix",lambda *args,**kwargs: (_ for _ in ()).throw(OSError("model cache unavailable")))
    row=pd.DataFrame([{"product_id":"outlier","title":"headphones","category":"Audio","price":500,"list_price":None,"brand":"x","description":"item"}])
    result=scorer.score(row)[0]
    assert result["scoring_mode"]=="rule_baseline_embedding_unavailable"
    assert scorer.fallback_reason.startswith("embedding_unavailable:")
