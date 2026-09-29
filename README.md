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

The dashboard uses the supplied/Kaggle CSV when placed at `data/retail_store_inventory.csv`; otherwise it creates a reproducible demo dataset automatically.

### 📌 Suggested Evaluation Metrics
Forecasting is evaluated using **MAE, RMSE and R²** with a chronological holdout to reduce leakage from future observations.

### 👨‍💻 Author
**Pradyuman Verma**  
B.Tech CSE (Data Science) — SRM Institute of Science and Technology
