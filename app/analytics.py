import pandas as pd
import numpy as np

def load_data(path):
    df=pd.read_csv(path); df.columns=[c.strip() for c in df.columns]
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    for c in ["Inventory Level","Units Sold","Units Ordered","Price","Discount"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    for c in ["Store ID","Product ID","Category","Region"]:
        if c in df: df[c]=df[c].fillna("Unknown").astype(str)
    if "Revenue" not in df: df["Revenue"]=df["Units Sold"]*df["Price"]*(1-df["Discount"].clip(0,1))
    return df.dropna(subset=["Date"]).copy()

def kpis(df):
    inv=float(df["Inventory Level"].mean()) if len(df) else 0; sold=float(df["Units Sold"].sum()) if len(df) else 0
    return {"Revenue":df["Revenue"].sum(),"Units Sold":int(sold),"Avg Selling Price":df["Price"].mean() if len(df) else 0,"Avg Inventory":inv,"Stock Cover Days":inv/(sold/max(df["Date"].nunique(),1)) if sold else 0,"SKU Count":df["Product ID"].nunique(),"Store Count":df["Store ID"].nunique()}

def monthly_sales(df):
    x=df.copy(); x["Month"]=x["Date"].dt.to_period("M").astype(str)
    return x.groupby("Month",as_index=False).agg(Revenue=("Revenue","sum"),Units_Sold=("Units Sold","sum"))

def top_products(df,n=10):
    return df.groupby(["Product ID","Category"],as_index=False).agg(Revenue=("Revenue","sum"),Units_Sold=("Units Sold","sum")).sort_values("Revenue",ascending=False).head(n)
