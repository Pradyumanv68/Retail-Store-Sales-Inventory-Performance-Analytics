import pandas as pd
import numpy as np

DATA_URL="https://raw.githubusercontent.com/HiraAli01/ASOS-dataset-for-machine-learning/main/demand_forecasting.csv"

def load_data(path):
    df=pd.read_csv(path); df.columns=[c.strip() for c in df.columns]
    return _clean(df)

def load_remote():
    return _clean(pd.read_csv(DATA_URL))

def _clean(df):
    df=df.copy()
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    for c in ["Inventory Level","Units Sold","Units Ordered","Price","Discount","Competitor Pricing","Epidemic","Promotion","Demand"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    for c in ["Store ID","Product ID","Category","Region","Weather Condition","Seasonality"]:
        if c in df: df[c]=df[c].fillna("Unknown").astype(str)
    if "Revenue" not in df:
        df["Revenue"]=df["Units Sold"]*df["Price"]*(1-df["Discount"].clip(0,100)/100)
    if "Demand" not in df: df["Demand"]=df["Units Sold"]
    df["Demand_Gap"]=df["Demand"]-df["Units Sold"]
    df["Price_Gap"]=df["Price"]-df["Competitor Pricing"] if "Competitor Pricing" in df else 0
    return df.dropna(subset=["Date"]).copy()

def kpis(df):
    inv=float(df["Inventory Level"].mean()) if len(df) else 0
    sold=float(df["Units Sold"].sum()) if len(df) else 0
    demand=float(df["Demand"].sum()) if len(df) else 0
    return {"Revenue":df["Revenue"].sum(),"Units Sold":int(sold),"Demand":int(demand),
            "Avg Selling Price":df["Price"].mean() if len(df) else 0,"Avg Inventory":inv,
            "Stock Cover Days":inv/(demand/max(df["Date"].nunique(),1)) if demand else 0,
            "SKU Count":df["Product ID"].nunique(),"Store Count":df["Store ID"].nunique(),
            "Demand Gap":int(max(0,demand-sold))}

def monthly_sales(df):
    x=df.copy(); x["Month"]=x["Date"].dt.to_period("M").astype(str)
    return x.groupby("Month",as_index=False).agg(Revenue=("Revenue","sum"),Units_Sold=("Units Sold","sum"),Demand=("Demand","sum"))

def top_products(df,n=10):
    return df.groupby(["Product ID","Category"],as_index=False).agg(Revenue=("Revenue","sum"),Units_Sold=("Units Sold","sum"),Demand=("Demand","sum")).sort_values("Revenue",ascending=False).head(n)

def demand_by_category(df):
    return df.groupby("Category",as_index=False).agg(Demand=("Demand","mean"),Units_Sold=("Units Sold","mean"),Revenue=("Revenue","sum")).sort_values("Demand",ascending=False)

def promotion_impact(df):
    return df.groupby("Promotion",as_index=False).agg(Avg_Demand=("Demand","mean"),Avg_Units_Sold=("Units Sold","mean"),Revenue=("Revenue","sum"))
