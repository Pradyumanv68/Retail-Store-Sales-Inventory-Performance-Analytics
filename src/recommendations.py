import pandas as pd, numpy as np

def inventory_status(df,forecast_days=7,safety=0.25):
    x=df.copy(); daily=x.groupby(["Store ID","Product ID"],as_index=False).agg({"Units Sold":"mean","Inventory Level":"mean","Units Ordered":"mean","Price":"mean"})
    daily["Demand Forecast"]=daily["Units Sold"]*forecast_days
    daily["Safety Stock"]=daily["Demand Forecast"]*safety
    daily["Net Position"]=daily["Inventory Level"]-daily["Demand Forecast"]
    daily["Recommended Order"]=np.ceil(np.maximum(0,daily["Demand Forecast"]+daily["Safety Stock"]-daily["Inventory Level"]))
    def status(r):
        if r["Inventory Level"]<=max(1,r["Units Sold"]*1): return "CRITICAL"
        if r["Inventory Level"]<r["Demand Forecast"]+r["Safety Stock"]: return "REPLENISH"
        if r["Inventory Level"]>max(r["Demand Forecast"]*4,50): return "OVERSTOCK"
        return "HEALTHY"
    daily["Inventory_Status"]=daily.apply(status,axis=1)
    return daily.sort_values(["Inventory_Status","Recommended Order"],ascending=[True,False])

def generate_alerts(df,forecast_days=7):
    x=inventory_status(df,forecast_days,.25)
    x["Priority"]=x["Inventory_Status"].map({"CRITICAL":1,"REPLENISH":2,"OVERSTOCK":3,"HEALTHY":4}).fillna(5)
    x["Action"]=x.apply(lambda r: "Expedite replenishment" if r["Inventory_Status"]=="CRITICAL" else ("Place replenishment order" if r["Inventory_Status"]=="REPLENISH" else ("Review markdown/promotion" if r["Inventory_Status"]=="OVERSTOCK" else "Monitor")),axis=1)
    return x.sort_values(["Priority","Recommended Order"],ascending=[True,False])

def narrative(r):
    return f"Inventory {r['Inventory Level']:.0f} vs {r['Demand Forecast']:.0f} expected units over the planning horizon. Recommended order: {r['Recommended Order']:.0f}. Action: {r['Action']}."
