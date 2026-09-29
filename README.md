# 📦 Retail Store Sales & Inventory Performance Analytics

> **Retail Intelligence Command Center** — an end-to-end Data Science capstone for turning retail transactions and inventory signals into operational decisions.

### 🎯 Business Problem
Retailers must balance sales growth with enough stock to avoid lost sales while preventing capital from being trapped in slow-moving inventory. This project connects **sales analytics → demand forecasting → inventory risk → replenishment action** in one workflow.

### ✨ Highlights
- Executive revenue, units, inventory and stock-cover KPIs
- Product, category and store performance analysis
- Inventory health segmentation: **CRITICAL / REPLENISH / OVERSTOCK / HEALTHY**
- Chronological ML evaluation for demand forecasting
- Product-level future demand forecasting
- Recommended order quantities with safety-stock logic
- Priority action queue downloadable as CSV
- Interactive Streamlit command center with filters
- Built-in demo dataset generation so the app runs immediately

### 🧠 Analytics Architecture
```
Raw Retail Data
      ↓
Validation & Feature Engineering
      ↓
Sales + Inventory KPIs
      ↓
Demand Forecasting (Gradient Boosting)
      ↓
Inventory Risk Engine
      ↓
Replenishment Recommendations
      ↓
Interactive Dashboard
```

### 🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit · Jupyter

### ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard first checks `data/retail_store_inventory.csv`. If it is not packaged in the repository, the deployed app automatically loads the **76,000-row Retail Store Inventory and Demand Forecasting dataset** from its public source, so the live dashboard uses realistic retail observations rather than a toy dataset. The dataset includes inventory, units sold, demand, pricing, discounts, promotions, competitor pricing, weather, seasonality and epidemic indicators. citeturn0search0

### 📌 Suggested Evaluation Metrics
Forecasting is evaluated using **MAE, RMSE and R²** with a chronological holdout to reduce leakage from future observations.

### 👨‍💻 Author
**Pradyuman Verma**  
B.Tech CSE (Data Science) — SRM Institute of Science and Technology


## Dataset & analytical depth

The project uses the Retail Store Inventory and Demand Forecasting dataset, described by Kaggle as a synthetic retail dataset for inventory and demand forecasting. It contains 76,000 observations and 16 business/environmental variables. The analysis treats **Demand** as the forecasting target and connects it to inventory coverage, replenishment recommendations, promotions, pricing and competitive context. citeturn0search0

## Why this is more than a dashboard

**Observe → Diagnose → Predict → Act**

1. Observe sales, revenue, demand and inventory KPIs.
2. Diagnose category/store/inventory pressure and promotion effects.
3. Predict future demand with chronological holdout evaluation.
4. Act using safety-stock and recommended-order logic.

> Model feature importance should be interpreted as predictive contribution, not causal impact.
