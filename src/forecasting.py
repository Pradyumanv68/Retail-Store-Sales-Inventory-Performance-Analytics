import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _features(x):
    z=x.copy().sort_values("Date")
    z["dow"]=z["Date"].dt.dayofweek; z["month"]=z["Date"].dt.month; z["dayofyear"]=z["Date"].dt.dayofyear
    z["lag1"]=z["Units Sold"].shift(1); z["lag7"]=z["Units Sold"].shift(7); z["roll7"]=z["Units Sold"].rolling(7).mean()
    return z.dropna()

def chronological_model(df):
    daily=df.groupby("Date",as_index=False).agg({"Units Sold":"sum","Inventory Level":"mean","Price":"mean","Discount":"mean","Promotion":"mean"})
    z=_features(daily); feats=["dow","month","dayofyear","lag1","lag7","roll7","Inventory Level","Price","Discount","Promotion"]
    split=max(int(len(z)*.8),1); Xtr,Xte=z[feats].iloc[:split],z[feats].iloc[split:]; ytr,yte=z["Units Sold"].iloc[:split],z["Units Sold"].iloc[split:]
    model=HistGradientBoostingRegressor(max_iter=250,learning_rate=.06,max_leaf_nodes=15,random_state=42).fit(Xtr,ytr)
    pred=model.predict(Xte) if len(Xte) else model.predict(Xtr)
    actual=yte if len(yte) else ytr
    metrics={"MAE":mean_absolute_error(actual,pred),"RMSE":mean_squared_error(actual,pred)**0.5,"R2":r2_score(actual,pred) if len(actual)>1 else 0}
    out=z.iloc[split:].copy() if len(Xte) else z.iloc[:len(pred)].copy(); out["Predicted_Units_Sold"]=pred
    return model,metrics,out

def forecast_next_days(df,store,product,days=7):
    x=df[(df["Store ID"].astype(str)==str(store))&(df["Product ID"].astype(str)==str(product))].copy()
    if len(x)<14:
        base=max(x["Units Sold"].tail(7).mean() if len(x) else 0,0)
        dates=pd.date_range(df["Date"].max()+pd.Timedelta(days=1),periods=days)
        return pd.DataFrame({"Date":dates,"Predicted_Demand":np.repeat(round(base),days)})
    daily=x.groupby("Date",as_index=False).agg({"Units Sold":"sum"}); z=_features(daily)
    feats=["dow","month","dayofyear","lag1","lag7","roll7"]; split=max(int(len(z)*.8),1)
    model=HistGradientBoostingRegressor(max_iter=250,learning_rate=.06,max_leaf_nodes=12,random_state=42).fit(z[feats],z["Units Sold"])
    hist=z[["Date","Units Sold"]].copy()
    rows=[]
    for _ in range(days):
        d=hist["Date"].max()+pd.Timedelta(days=1); vals=hist["Units Sold"].tolist()
        row=pd.DataFrame([{"dow":d.dayofweek,"month":d.month,"dayofyear":d.dayofyear,"lag1":vals[-1],"lag7":vals[-7],"roll7":np.mean(vals[-7:])}])
        y=max(0,float(model.predict(row[feats])[0])); rows.append([d,y]); hist=pd.concat([hist,pd.DataFrame({"Date":[d],"Units Sold":[y]})],ignore_index=True)
    return pd.DataFrame(rows,columns=["Date","Predicted_Demand"])
