import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from analytics import load_data, kpis, monthly_sales, top_products
from forecasting import chronological_model, forecast_next_days
from recommendations import inventory_status, generate_alerts, narrative

st.set_page_config(
    page_title="Retail Intelligence Command Center",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
.block-container{padding:1.2rem 2.5rem 2rem;max-width:1500px}
[data-testid="stSidebar"]{background:#080d18;border-right:1px solid #1e293b}
[data-testid="stSidebar"] *{font-family:Inter,system-ui,sans-serif}
.hero{position:relative;overflow:hidden;padding:2.2rem 2.4rem;border-radius:24px;
background:linear-gradient(120deg,#07111f 0%,#101d36 55%,#102a43 100%);
border:1px solid #263b57;box-shadow:0 18px 50px rgba(0,0,0,.28);margin-bottom:1.1rem}
.hero:after{content:"";position:absolute;width:220px;height:220px;right:-70px;top:-90px;border-radius:50%;
background:radial-gradient(circle,#38bdf8 0%,transparent 68%);opacity:.16}
.eyebrow{color:#38bdf8;text-transform:uppercase;letter-spacing:2px;font-size:.76rem;font-weight:800}
.hero h1{font-size:2.65rem;line-height:1.05;margin:.35rem 0 .55rem;color:#f8fafc}
.hero p{font-size:1rem;color:#b7c6d9;margin:0;max-width:760px}
.pill{display:inline-block;margin-top:1rem;padding:.35rem .7rem;border-radius:999px;
background:#0f2740;border:1px solid #234d70;color:#7dd3fc;font-size:.78rem;font-weight:700}
.section{font-size:1.05rem;font-weight:800;margin:.4rem 0 .7rem;color:#e5edf7}
.stMetric{background:linear-gradient(145deg,#101827,#0c1422);border:1px solid #243247;border-radius:16px;padding:12px 14px;box-shadow:0 8px 24px rgba(0,0,0,.12)}
.stMetric label{color:#94a3b8}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:9px 15px}
.insight{padding:14px 16px;border-radius:14px;background:#0d1726;border:1px solid #22334a;margin-bottom:9px}
.insight b{color:#f8fafc}.muted{color:#94a3b8;font-size:.88rem}
.badge{padding:4px 9px;border-radius:999px;background:#13243a;color:#7dd3fc;font-size:.72rem;font-weight:800}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Retail Decision Intelligence • Capstone 2026</div>
  <h1>📦 Retail Intelligence<br>Command Center</h1>
  <p>Turn raw retail data into decisions — understand performance, detect inventory risk, forecast demand and know what to reorder next.</p>
  <span class="pill">● LIVE ANALYTICS • FORECASTING • INVENTORY OPTIMIZATION</span>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def get_df():
    path = "data/retail_store_inventory.csv"
    if os.path.exists(path):
        return load_data(path), "Supplied / Kaggle dataset"
    rng=np.random.default_rng(42)
    dates=pd.date_range("2025-01-01",periods=180)
    rows=[]
    cats=["Electronics","Grocery","Clothing","Home"]
    regs=["North","South","East","West"]
    for store in range(1,7):
        for prod in range(1,13):
            cat=cats[(prod-1)%4]
            base=8+(prod%7)*2
            for d in dates:
                season=1+0.18*np.sin(2*np.pi*d.dayofyear/365)
                sold=max(0,int(rng.poisson(base*season)))
                price=round(80+prod*23+store*7,2)
                inv=max(0,int(rng.normal(100,25)+base*4-sold))
                disc=round(float(rng.uniform(0,.2)),2)
                rows.append([d,f"S{store:02d}",f"P{prod:03d}",cat,inv,sold,max(0,int(sold*1.1)),price,disc,int(rng.random()<.2),regs[(store-1)%4]])
    return pd.DataFrame(rows,columns=["Date","Store ID","Product ID","Category","Inventory Level","Units Sold","Units Ordered","Price","Discount","Promotion","Region"]).assign(
        Revenue=lambda x:x["Units Sold"]*x["Price"]*(1-x["Discount"])
    ), "Built-in demo dataset"

df, source = get_df()

with st.sidebar:
    st.markdown("## 🎛️ Control Center")
    st.caption(source)
    st.divider()
    stores=sorted(df["Store ID"].astype(str).unique())
    cats=sorted(df["Category"].unique())
    ss=st.multiselect("Store coverage",stores,stores)
    cc=st.multiselect("Product categories",cats,cats)
    horizon=st.slider("Forecast / planning horizon",3,30,7)
    st.divider()
    st.markdown("**Decision layer**")
    st.caption("Forecast → Risk → Replenishment")
    st.caption("Built for business review & presentation")

f=df[df["Store ID"].astype(str).isin(ss)&df["Category"].isin(cc)].copy()
k=kpis(f)
alerts=generate_alerts(f,horizon)
inv=inventory_status(f,horizon,.25)

critical=int((inv["Inventory_Status"]=="CRITICAL").sum())
replenish=int((inv["Inventory_Status"]=="REPLENISH").sum())
overstock=int((inv["Inventory_Status"]=="OVERSTOCK").sum())
total=len(inv)
healthy_pct=(int((inv["Inventory_Status"]=="HEALTHY").sum())/total*100) if total else 0
risk_pct=((critical+replenish)/total*100) if total else 0
health=max(0,min(100,round(100-risk_pct-overstock*100/max(total,1)*0.35)))

st.markdown('<div class="section">Executive Snapshot</div>',unsafe_allow_html=True)
cards=st.columns(6)
metrics=[
    ("Revenue",f"₹{k['Revenue']:,.0f}","Commercial"),
    ("Units Sold",f"{k['Units Sold']:,}","Demand"),
    ("Inventory",f"{k['Avg Inventory']:,.0f}","Stock"),
    ("Stock Cover",f"{k['Stock Cover Days']:.1f} d","Coverage"),
    ("Risk Exposure",f"{risk_pct:.1f}%","Actionable"),
    ("Inventory Health",f"{health}/100","Health Score"),
]
for col,(label,value,sub) in zip(cards,metrics):
    col.metric(label,value)
    col.caption(sub)

tab1,tab2,tab3,tab4=st.tabs([
    "📊  Executive Overview","📦  Inventory Radar","🔮  Forecast Studio","⚡  Action Center"
])

plot_template="plotly_dark"

with tab1:
    st.markdown('<div class="section">Where the business is moving</div>',unsafe_allow_html=True)
    a,b=st.columns([1.35,1])
    m=monthly_sales(f)
    fig=px.area(m,x="Month",y="Revenue",markers=True,title="Revenue Momentum")
    fig.update_layout(template=plot_template,margin=dict(l=10,r=10,t=50,b=10),hovermode="x unified")
    a.plotly_chart(fig,use_container_width=True)
    tp=top_products(f,10)
    fig2=px.bar(tp.sort_values("Revenue"),x="Revenue",y="Product ID",color="Category",orientation="h",title="Top Products Driving Revenue")
    fig2.update_layout(template=plot_template,margin=dict(l=10,r=10,t=50,b=10))
    b.plotly_chart(fig2,use_container_width=True)
    s=f.groupby("Store ID",as_index=False).agg(Revenue=("Revenue","sum"),Units=("Units Sold","sum"))
    fig3=px.bar(s,x="Store ID",y="Revenue",color="Revenue",title="Store Performance")
    fig3.update_layout(template=plot_template,showlegend=False,margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig3,use_container_width=True)
    st.markdown('<div class="section">Management Signals</div>',unsafe_allow_html=True)
    signals=[
        ("📈","Revenue engine",f"Top product {tp.iloc[0]['Product ID']} contributes ₹{tp.iloc[0]['Revenue']:,.0f} in selected data." if len(tp) else "No product data available."),
        ("📦","Inventory pressure",f"{critical} critical and {replenish} replenishment SKUs need attention."),
        ("🧊","Capital tied up",f"{overstock} SKUs are flagged as potential overstock based on demand coverage."),
    ]
    for icon,title,text in signals:
        st.markdown(f'<div class="insight"><b>{icon} {title}</b><br><span class="muted">{text}</span></div>',unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section">Inventory health radar</div>',unsafe_allow_html=True)
    cnt=inv["Inventory_Status"].value_counts().rename_axis("Status").reset_index(name="Count")
    a,b=st.columns(2)
    fig=px.pie(cnt,names="Status",values="Count",hole=.62,title="Inventory Health Distribution")
    fig.update_layout(template=plot_template,margin=dict(l=10,r=10,t=50,b=10))
    a.plotly_chart(fig,use_container_width=True)
    fig=px.scatter(inv,x="Demand Forecast",y="Inventory Level",color="Inventory_Status",size="Recommended Order",
                   hover_data=["Store ID","Product ID"],title="Demand vs Available Inventory")
    fig.update_layout(template=plot_template,margin=dict(l=10,r=10,t=50,b=10))
    b.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section">SKU-level decision table</div>',unsafe_allow_html=True)
    display=inv[["Store ID","Product ID","Units Sold","Inventory Level","Demand Forecast","Recommended Order","Inventory_Status"]].head(40).copy()
    display.columns=["Store","Product","Avg Daily Sales","Inventory","Forecast Demand","Order Qty","Status"]
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Export inventory decision queue",alerts.to_csv(index=False),"retail_inventory_actions.csv","text/csv")

with tab3:
    st.markdown('<div class="section">Demand Forecast Studio</div>',unsafe_allow_html=True)
    if len(f)>100:
        _,metrics,preds=chronological_model(f)
        a,b,c=st.columns(3)
        a.metric("MAE",f"{metrics['MAE']:.2f}")
        b.metric("RMSE",f"{metrics['RMSE']:.2f}")
        c.metric("R²",f"{metrics['R2']:.3f}")
        plot=preds.groupby("Date",as_index=False).agg(Actual=("Units Sold","sum"),Predicted=("Predicted_Units_Sold","sum"))
        fig=px.line(plot,x="Date",y=["Actual","Predicted"],title="Chronological Holdout: Actual vs Predicted",markers=False)
        fig.update_layout(template=plot_template,margin=dict(l=10,r=10,t=50,b=10),hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True)
        a,b=st.columns(2)
        store=a.selectbox("Forecast store",sorted(f["Store ID"].astype(str).unique()))
        prods=sorted(f[f["Store ID"].astype(str)==store]["Product ID"].astype(str).unique())
        prod=b.selectbox("Forecast product",prods)
        fc=forecast_next_days(f,store,prod,horizon)
        fig=px.line(fc,x="Date",y="Predicted_Demand",markers=True,title=f"{store} / {prod} — Next {horizon} Days")
        fig.update_layout(template=plot_template,margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(fc,use_container_width=True,hide_index=True)
    else:
        st.warning("Select a larger dataset window for model evaluation.")

with tab4:
    st.markdown('<div class="section">From insight to action</div>',unsafe_allow_html=True)
    st.caption("The system converts inventory signals into prioritized operational actions.")
    for _,r in alerts.head(10).iterrows():
        icon={"CRITICAL":"🔴","REPLENISH":"🟠","OVERSTOCK":"🟡","HEALTHY":"🟢"}.get(r["Inventory_Status"],"⚪")
        st.markdown(f'<div class="insight"><b>{icon} {r["Store ID"]} / {r["Product ID"]} · {r["Inventory_Status"]}</b><br><span class="muted">{narrative(r)}</span></div>',unsafe_allow_html=True)
    st.download_button("⬇️ Download complete action queue",alerts.to_csv(index=False),"retail_action_queue.csv","text/csv")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:.82rem">Retail Store Sales & Inventory Performance Analytics • Capstone 2026 • Pradyuman Verma • B.Tech CSE (Data Science), SRMIST</div>',
    unsafe_allow_html=True,
)
