import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np,pandas as pd,streamlit as st,plotly.express as px
from src.analytics import load_data,kpis,monthly_sales,top_products
from src.forecasting import chronological_model,forecast_next_days
from src.recommendations import inventory_status,generate_alerts,narrative

st.set_page_config(page_title="Retail Intelligence | Pradyuman Verma",page_icon="📦",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:1.7rem}.hero{padding:1.7rem 2rem;border-radius:20px;background:linear-gradient(135deg,#0f172a,#172554);border:1px solid #263247;color:white;margin-bottom:1.2rem}.hero h1{font-size:2.35rem;margin:0}.hero p{color:#cbd5e1;margin:.45rem 0 0}.stMetric{background:#111827;border:1px solid #263247;padding:12px;border-radius:14px}
[data-testid="stSidebar"]{border-right:1px solid #1e293b}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📦 Retail Intelligence Command Center</h1><p>Sales performance • Inventory risk • Demand forecasting • Replenishment decisions</p></div>',unsafe_allow_html=True)

@st.cache_data
def get_df():
    path="data/retail_store_inventory.csv"
    if os.path.exists(path): return load_data(path),"Supplied / Kaggle dataset"
    rng=np.random.default_rng(42); dates=pd.date_range("2025-01-01",periods=180); rows=[]; cats=["Electronics","Grocery","Clothing","Home"]; regs=["North","South","East","West"]
    for store in range(1,7):
        for prod in range(1,13):
            cat=cats[(prod-1)%4]; base=8+(prod%7)*2
            for d in dates:
                season=1+0.18*np.sin(2*np.pi*d.dayofyear/365); sold=max(0,int(rng.poisson(base*season))); price=round(80+prod*23+store*7,2)
                inv=max(0,int(rng.normal(100,25)+base*4-sold)); disc=round(float(rng.uniform(0,.2)),2)
                rows.append([d,f"S{store:02d}",f"P{prod:03d}",cat,inv,sold,max(0,int(sold*1.1)),price,disc,int(rng.random()<.2),regs[(store-1)%4]])
    return pd.DataFrame(rows,columns=["Date","Store ID","Product ID","Category","Inventory Level","Units Sold","Units Ordered","Price","Discount","Promotion","Region"]).assign(Revenue=lambda x:x["Units Sold"]*x["Price"]*(1-x["Discount"])),"Built-in demo dataset"
df,source=get_df()
with st.sidebar:
    st.markdown("## 🎛️ Control Panel"); st.caption(source)
    stores=sorted(df["Store ID"].astype(str).unique()); cats=sorted(df["Category"].unique())
    ss=st.multiselect("Stores",stores,stores); cc=st.multiselect("Categories",cats,cats); horizon=st.slider("Planning horizon",3,30,7)
f=df[df["Store ID"].astype(str).isin(ss)&df["Category"].isin(cc)].copy()
k=kpis(f); c=st.columns(6)
for col,(a,b) in zip(c,[("Revenue",f"₹{k['Revenue']:,.0f}"),("Units Sold",f"{k['Units Sold']:,}"),("Avg Inventory",f"{k['Avg Inventory']:,.0f}"),("Stock Cover",f"{k['Stock Cover Days']:.1f} d"),("Products",k["SKU Count"]),("Stores",k["Store Count"])]): col.metric(a,b)
tabs=st.tabs(["📊 Overview","📦 Inventory","🔮 Forecast","💡 Actions"])
with tabs[0]:
    a,b=st.columns(2); m=monthly_sales(f); a.plotly_chart(px.area(m,x="Month",y="Revenue",title="Revenue Trend"),use_container_width=True)
    tp=top_products(f,10); b.plotly_chart(px.bar(tp.sort_values("Revenue"),x="Revenue",y="Product ID",color="Category",orientation="h",title="Top Products by Revenue"),use_container_width=True)
    s=f.groupby("Store ID",as_index=False).agg(Revenue=("Revenue","sum")); st.plotly_chart(px.bar(s,x="Store ID",y="Revenue",color="Store ID",title="Store Revenue Comparison"),use_container_width=True)
with tabs[1]:
    inv=inventory_status(f,horizon,.25); cnt=inv["Inventory_Status"].value_counts().rename_axis("Status").reset_index(name="Count"); a,b=st.columns(2)
    a.plotly_chart(px.pie(cnt,names="Status",values="Count",hole=.58,title="Inventory Health"),use_container_width=True)
    b.plotly_chart(px.scatter(inv,x="Demand Forecast",y="Inventory Level",color="Inventory_Status",hover_data=["Store ID","Product ID"],title="Inventory vs Expected Demand"),use_container_width=True)
    st.dataframe(inv[["Store ID","Product ID","Units Sold","Inventory Level","Demand Forecast","Recommended Order","Inventory_Status"]].head(30),use_container_width=True,hide_index=True)
with tabs[2]:
    if len(f)>100:
        _,metrics,preds=chronological_model(f); a,b,c=st.columns(3); a.metric("MAE",f"{metrics['MAE']:.2f}"); b.metric("RMSE",f"{metrics['RMSE']:.2f}"); c.metric("R²",f"{metrics['R2']:.3f}")
        plot=preds.groupby("Date",as_index=False).agg(Actual=("Units Sold","sum"),Predicted=("Predicted_Units_Sold","sum")); st.plotly_chart(px.line(plot,x="Date",y=["Actual","Predicted"],title="Chronological Holdout Performance"),use_container_width=True)
        store=st.selectbox("Store",sorted(f["Store ID"].astype(str).unique())); prods=sorted(f[f["Store ID"].astype(str)==store]["Product ID"].astype(str).unique()); prod=st.selectbox("Product",prods)
        fc=forecast_next_days(f,store,prod,horizon); st.plotly_chart(px.line(fc,x="Date",y="Predicted_Demand",markers=True,title=f"Next {horizon} Days Demand Forecast"),use_container_width=True); st.dataframe(fc,use_container_width=True,hide_index=True)
    else: st.warning("Add more observations to evaluate the forecasting model.")
with tabs[3]:
    alerts=generate_alerts(f,horizon); st.subheader("Priority Action Queue")
    for _,r in alerts.head(8).iterrows():
        icon="🔴" if r["Inventory_Status"]=="CRITICAL" else ("🟠" if r["Inventory_Status"]=="REPLENISH" else ("🟡" if r["Inventory_Status"]=="OVERSTOCK" else "🟢"))
        st.markdown(f"**{icon} {r['Store ID']} / {r['Product ID']} — {r['Inventory_Status']}**  
{narrative(r)}")
    st.download_button("⬇️ Download Action Queue",alerts.to_csv(index=False),"retail_inventory_actions.csv","text/csv")
st.caption("Capstone Project • Pradyuman Verma • B.Tech CSE (Data Science) • SRMIST")
