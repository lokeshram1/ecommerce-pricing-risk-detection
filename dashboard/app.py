import os, requests, streamlit as st, pandas as pd, plotly.express as px
from dashboard.data import load_dashboard_data

st.set_page_config(page_title="E-Commerce Pricing Risk Monitor",page_icon="🛡️",layout="wide")
st.title("E-Commerce Pricing Risk Monitor")
st.caption("Pricing risk scores prioritize investigation. They are not proof of wrongdoing.")
api=os.getenv("API_URL","http://localhost:8000")
summary,metrics,rows,error=load_dashboard_data(api)
if error:
    st.error(f"{error}. Start the API and reload this page."); st.stop()
cols=st.columns(4)
for c,label,value in zip(cols,["Listings scored","High risk","Flagged for review","Average risk score"],[summary["total_scored"],summary["risk_counts"].get("High",0),f"{summary['flagged_percent']:.1f}%",f"{summary['average_risk_score']:.1f}"]): c.metric(label,value)
tab1,tab2,tab3=st.tabs(["Overview","Risk Explorer","Model Performance"])
with tab1:
    left,right=st.columns(2)
    counts=summary.get("risk_counts",{})
    if sum(counts.values()): left.plotly_chart(px.pie(names=list(counts),values=list(counts.values()),title="Risk distribution"),use_container_width=True)
    else: left.info("No listings have been scored yet.")
    data=pd.DataFrame([{**x["product"],**(x["score"] or {}),"review_status":x["review_status"]} for x in rows])
    if not data.empty and "category" in data and "risk_category" in data: right.plotly_chart(px.histogram(data,x="category",color="risk_category",title="Category risk mix"),use_container_width=True)
with tab2:
    category=st.selectbox("Category",["All"]+sorted({x["product"]["category"] for x in rows}))
    shown=[x for x in rows if category=="All" or x["product"]["category"]==category]
    table=pd.DataFrame([{**x["product"],**(x["score"] or {}),"review_status":x["review_status"]} for x in shown])
    if table.empty: st.info("No scored products yet. Run the demo initialization command.")
    else:
        if "risk_score" in table: table=table.sort_values("risk_score",ascending=False)
        st.dataframe(table[[c for c in ["product_id","title","category","price","risk_score","risk_category","review_status"] if c in table]],use_container_width=True,hide_index=True)
        selected=st.selectbox("Investigate product",table.product_id.tolist()); detail=requests.get(api+f"/products/{selected}",timeout=4).json()
        st.json(detail)
        if detail["latest_score"]:
            with st.form("review"):
                decision=st.selectbox("Human review decision",["Confirmed Anomaly","Expected Pricing","Needs More Information"]); notes=st.text_area("Reviewer notes")
                if st.form_submit_button("Save review"):
                    response=requests.post(api+f"/reviews/{detail['latest_score']['score_id']}",json={"decision":decision,"notes":notes},timeout=4)
                    st.success("Review saved") if response.ok else st.error(response.text)
with tab3:
    if metrics:
        st.warning("Benchmark labels are synthetic anomaly injections; metrics do not estimate real-world fraud performance.")
        st.json(metrics)
        a,b,c,d=st.columns(4)
        for col,key in zip([a,b,c,d],["precision","recall","f1","pr_auc"]): col.metric(key.upper(),f"{metrics.get(key,0):.3f}")
    else: st.info("Train the benchmark model to view metrics.")
