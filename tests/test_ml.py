import json
import pandas as pd
from app.ml.ingestion import read_jsonl,normalize_record
from app.ml.features import FeatureEngineer
from app.ml.benchmark import inject_anomalies
from app.ml.train import split_product_identities
from scripts.generate_sample import generate

def sample():
    return pd.DataFrame([normalize_record({"asin":f"x{i}","title":"test item","category":"audio","price":p,"description":"headphones","brand":"a"}) for i,p in enumerate([20,22,24,25,28,30,32,35])])
def test_ingestion_duplicate_bad_and_price_validation(tmp_path):
    f=tmp_path/"x.jsonl"; f.write_text('\n'.join([json.dumps({"asin":"a","price":"$4.50"}),json.dumps({"asin":"a","price":3}),"bad",json.dumps({"asin":"b","price":-3}),json.dumps({"asin":"c","price":"Infinity","is_synthetic":"false"})]))
    df,stats=read_jsonl(f); assert len(df)==3 and stats["duplicates"]==1 and stats["rejected"]==1 and df.iloc[1].price is None and df.iloc[2].price is None and not df.iloc[2].is_synthetic
def test_category_stats_are_fit_only_and_unseen_safe():
    df=sample(); fe=FeatureEngineer().fit(df.iloc[:4]); x=fe.transform(df.iloc[4:]); assert x.shape==(4,10) and fe.stats["audio"]["median"]==21
    unseen=df.iloc[[0]].copy(); unseen["category"]="new"; assert FeatureEngineer().fit(df).transform(unseen).shape==(1,10)
def test_benchmark_reproducible_auditable():
    df=sample(); a=inject_anomalies(df,seed=5,fraction=.5); b=inject_anomalies(df,seed=5,fraction=.5)
    assert a.equals(b) and int(a.label.sum())==4 and (a.original_price==df.price).all()
    assert set(a.injection_type)-{"none"} <= {"overpriced","underpriced","discount_conflict"}
    assert (a.loc[a.label==0,"price"]==a.loc[a.label==0,"original_price"]).all()

def test_missing_prices_and_attributes_are_finite():
    df=sample(); df.loc[0,["price","description","brand","rating","rating_count"]]=None
    fe=FeatureEngineer().fit(df)
    features=fe.transform(df)
    assert features.shape==(len(df),10) and __import__("numpy").isfinite(features).all()
    assert features[0,5]==1 and features[0,6]==1 and features[0,7]==1

def test_product_ids_split_before_preprocessing():
    df=pd.DataFrame([normalize_record(row) for row in generate(80)]); duplicate=df.iloc[[0]].copy(); duplicate["price"]=10000
    train,val,test=split_product_identities(pd.concat([df,duplicate],ignore_index=True),seed=19)
    groups=[set(part.product_id) for part in (train,val,test)]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert sum(map(len,groups))==df.product_id.nunique()
    fe=FeatureEngineer().fit(train); before=fe.stats.copy()
    changed=val.copy(); changed["price"]=changed.price*1000
    assert fe.stats==before

def test_identity_split_normalizes_mixed_id_types():
    rows=pd.DataFrame([normalize_record(row) for row in generate(80)])
    rows.loc[0,"product_id"]=123
    rows.loc[1,"product_id"]="123"
    train,val,test=split_product_identities(rows,seed=10)
    ids=[set(part.product_id.astype(str)) for part in (train,val,test)]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    assert sum(map(len,ids))==79
