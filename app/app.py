import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.analytics import load_data,kpis,inventory_status,top_products,monthly_sales
from src.forecasting import chronological_model,forecast_next_days
from src.recommendations import generate_alerts,narrative

st.set_page_config(page_title="Retail Intelligence Command Center",page_icon="📦",layout="wide")
st.title("📦 Retail Store Sales & Inventory Performance Analytics")
st.caption("Decision-support dashboard • Sales intelligence • Inventory risk • Demand forecasting")
DATA="data/retail_store_inventory.csv"; DEMO="data/demo_retail_store_inventory.csv"
path=DATA if os.path.exists(DATA) else DEMO
df=load_data(path)
with st.sidebar:
    st.header("Control Panel"); st.info(f"Data source: {'Kaggle dataset' if path==DATA else 'Demo dataset'}")
    stores=sorted(df["Store ID"].unique()); cats=sorted(df["Category"].unique())
    sel_store=st.multiselect("Store",stores,default=stores); sel_cat=st.multiselect("Category",cats,default=cats)
    min_d,max_d=df["Date"].min().date(),df["Date"].max().date(); dates=st.date_input("Date range",(min_d,max_d))
    forecast_days=st.slider("Planning horizon (days)",3,30,7); safety=st.slider("Safety buffer",0.0,0.5,0.25,0.05)
f=df[df["Store ID"].isin(sel_store)&df["Category"].isin(sel_cat)].copy()
if isinstance(dates,tuple) and len(dates)==2: f=f[(f["Date"].dt.date>=dates[0])&(f["Date"].dt.date<=dates[1])]
k=kpis(f)
cols=st.columns(7)
for c,(label,val) in zip(cols,[("Revenue",f"₹{k['Revenue']:,.0f}"),("Units Sold",f"{k['Units Sold']:,}"),("Avg Price",f"₹{k['Avg Selling Price']:.2f}"),("Avg Inventory",f"{k['Avg Inventory']:,.0f}"),("Stock Cover",f"{k['Stock Cover Days']:.1f} d"),("SKUs",k["SKU Count"]),("Stores",k["Store Count"])]): c.metric(label,val)
tab1,tab2,tab3,tab4=st.tabs(["Executive Overview","Inventory Command","Forecast Lab","Business Insights"])
with tab1:
    a,b=st.columns(2); m=monthly_sales(f); a.plotly_chart(px.line(m,x="Month",y="Revenue",markers=True,title="Monthly Revenue Trend"),use_container_width=True)
    tp=top_products(f,10); b.plotly_chart(px.bar(tp.sort_values("Revenue"),x="Revenue",y="Product ID",color="Category",orientation="h",title="Top Products by Revenue"),use_container_width=True)
    s=f.groupby("Store ID",as_index=False).agg(Revenue=("Revenue","sum"),Units_Sold=("Units Sold","sum")); st.plotly_chart(px.bar(s,x="Store ID",y="Revenue",color="Store ID",title="Store Revenue Comparison"),use_container_width=True)
with tab2:
    inv=inventory_status(f,forecast_days,safety); counts=inv["Inventory_Status"].value_counts().rename_axis("Status").reset_index(name="Records"); a,b=st.columns(2)
    a.plotly_chart(px.pie(counts,names="Status",values="Records",hole=.55,title="Inventory Risk Distribution"),use_container_width=True)
    b.plotly_chart(px.scatter(inv.sample(min(3000,len(inv)),random_state=42),x="Demand Forecast",y="Inventory Level",color="Inventory_Status",hover_data=["Store ID","Product ID"],title="Inventory vs Demand"),use_container_width=True)
    alerts=generate_alerts(f,forecast_days); st.subheader("Priority Action Queue"); st.dataframe(alerts.head(25),use_container_width=True,hide_index=True)
    st.download_button("Download action queue CSV",alerts.to_csv(index=False),"inventory_action_queue.csv","text/csv")
with tab3:
    st.subheader("Demand Forecast Model")
    if len(f)>100:
        model,metrics,preds=chronological_model(f); c1,c2,c3=st.columns(3); c1.metric("MAE",f"{metrics['MAE']:.2f}"); c2.metric("RMSE",f"{metrics['RMSE']:.2f}"); c3.metric("R²",f"{metrics['R2']:.3f}")
        plot=preds.groupby("Date",as_index=False).agg(Actual=("Units Sold","sum"),Predicted=("Predicted_Units_Sold","sum")); st.plotly_chart(px.line(plot,x="Date",y=["Actual","Predicted"],title="Chronological Holdout: Actual vs Predicted"),use_container_width=True)
        store=st.selectbox("Forecast store",sorted(f["Store ID"].unique())); prods=sorted(f[f["Store ID"]==store]["Product ID"].unique()); prod=st.selectbox("Forecast product",prods)
        fc=forecast_next_days(f,store,prod,forecast_days); st.dataframe(fc,use_container_width=True,hide_index=True); st.plotly_chart(px.line(fc,x="Date",y="Predicted_Demand",markers=True,title=f"{store} / {prod} — next {forecast_days} days"),use_container_width=True)
    else: st.warning("Select a larger date range for model evaluation.")
with tab4:
    st.subheader("Management Insights"); alerts=generate_alerts(f,forecast_days)
    for _,r in alerts.head(8).iterrows(): st.write(f"**{r['Store ID']} / {r['Product ID']} — {r['Inventory_Status']}**: {narrative(r)}")
    st.markdown("### Methodology"); st.markdown("1. Clean and validate data  \n2. Engineer revenue and stock-cover features  \n3. Segment inventory risk  \n4. Train chronological gradient-boosting model  \n5. Translate forecast into replenishment actions")