import requests
from dashboard.data import load_dashboard_data

class Response:
    ok=True
    def __init__(self,body): self.body=body
    def json(self): return self.body
    def raise_for_status(self): pass

def test_dashboard_handles_empty_backend_data():
    bodies=[{"total_scored":0,"risk_counts":{"Low":0,"Medium":0,"High":0}}, {"items":[]}, {}]
    def get(*args,**kwargs): return Response(bodies.pop(0))
    summary,metrics,products,error=load_dashboard_data("http://api",get)
    assert summary["total_scored"]==0 and products==[] and error is None

def test_dashboard_reports_api_outage_without_model_or_db_access():
    def get(*args,**kwargs): raise requests.ConnectionError("offline")
    summary,metrics,products,error=load_dashboard_data("http://api",get)
    assert error and "offline" in error and not products
