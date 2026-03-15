# FINSHIELD_AI
# 🛡️ FinShield AI — Real-Time Fraud Detection System

> An end-to-end AI-powered fraud detection system built with Machine Learning, RAG (Retrieval-Augmented Generation), SHAP Explainability, and a live Streamlit dashboard.

---

 📌 What is FinShield AI?

FinShield AI is a complete fraud detection pipeline that processes financial transactions in real-time, analyzes them using 7 detection engines, retrieves relevant fraud rules using a Hybrid RAG pipeline, and makes a final decision (ALLOW / REVIEW / BLOCK) using GPT-4o-mini.

Every decision is explainable — SHAP values show exactly which features caused the fraud flag.

---

## 🏗️ System Architecture

```
Real-Time Transaction Stream (Simulated)
        ↓
Data Preprocessing + Feature Engineering
        ↓
┌─────────────────────────────────────────┐
│         7 Detection Engines             │
│  • XGBoost (ML fraud probability)       │
│  • Isolation Forest (anomaly detection) │
│  • Behavior Profiler (per-user pattern) │
│  • Graph Detector (fraud ring — NetworkX│
│  • Velocity Checker (transaction bursts)│
│  • Device + IP Checker (fingerprinting) │
│  • SHAP Explainer (feature importance)  │
└─────────────────────────────────────────┘
        ↓
Risk Score Calculation (0–100)
        ↓
┌─────────────────────────────────────────┐
│         Hybrid RAG Pipeline             │
│  • Page Index (fraud rules CSV)         │
│  • Vector Search (Supabase embeddings)  │
│  • Case Memory (past fraud cases)       │
└─────────────────────────────────────────┘
        ↓
MCP Context Builder (structured evidence)
        ↓
Decision Agent — GPT-4o-mini
        ↓
ALLOW / REVIEW / BLOCK + LLM Reasoning
        ↓
┌─────────────────────────────────────────┐
│           Output Layer                  │
│  • Supabase case logging                │
│  • Streamlit live dashboard             │
│  • MLOps — monitor + drift + retrain    │
└─────────────────────────────────────────┘
```

---

## 🚀 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost + SMOTE |
| Anomaly Detection | Isolation Forest |
| Graph Detection | NetworkX |
| Explainability | SHAP |
| RAG | OpenAI Embeddings + Supabase pgvector |
| LLM Decision | GPT-4o-mini |
| API Server | FastAPI + Uvicorn |
| Database | Supabase (PostgreSQL + Vector DB) |
| Dashboard | Streamlit + Plotly |
| MLOps | Custom monitor + KS drift test + retrain pipeline |
| Dataset | Kaggle Credit Card Fraud Detection (284,807 transactions) |

---

## 📁 Project Structure

```
FinShield_AI/
├── app/
│   └── main.py                  # FastAPI server
├── data/
│   ├── creditcard.csv            # Original Kaggle dataset
│   ├── creditcard_augmented.csv  # Augmented with user/device/geo
│   └── fraud_rules.csv           # 150 fraud rules
├── models/
│   ├── xgboost_model.pkl         # Trained XGBoost model
│   ├── feature_names.json        # Saved feature names
│   ├── scaler.pkl                # StandardScaler
│   └── model_registry.json       # MLOps version history
├── src/
│   ├── preprocessing.py          # Data cleaning
│   ├── feature_engineering.py    # Feature creation
│   ├── anomaly_detector.py       # Isolation Forest
│   ├── behavior_profiler.py      # Per-user behavior
│   ├── graph_detector.py         # NetworkX fraud rings
│   ├── velocity_checker.py       # Transaction burst detection
│   ├── device_ip_checker.py      # Device fingerprinting
│   ├── risk_engine.py            # Risk scoring + SHAP
│   ├── case_logger.py            # Supabase logging
│   ├── monitor.py                # Performance monitoring
│   ├── drift_detector.py         # Concept drift detection
│   ├── retrain_pipeline.py       # Batch retraining
│   ├── train_model.py            # Model training
│   └── rag/
│       ├── rule_loader.py        # Load fraud rules CSV
│       ├── page_index.py         # Rule indexing
│       ├── retrieval_agent.py    # Rule retrieval
│       ├── vector_db_builder.py  # Build Supabase vector index
│       ├── vector_search.py      # Semantic search
│       ├── context_builder.py    # Evidence table builder
│       └── decision_agent.py     # GPT-4o-mini decision
├── logs/
│   ├── performance_log.json      # F1/precision/recall history
│   └── drift_log.json            # Drift detection history
├── dashboard.py                  # Streamlit dashboard
├── simulate_stream.py            # Real-time simulation
├── augment_dataset.py            # Dataset augmentation
├── save_features.py              # Save feature names
├── requirements.txt
└── .env                          # API keys (not in GitHub)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/FinShield-AI.git
cd FinShield-AI
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Setup environment variables
Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 5. Download dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 6. Augment dataset
```bash
python augment_dataset.py
```

### 7. Train the model
```bash
python src/train_model.py
python save_features.py
```

### 8. Build vector database
```bash
python src/rag/vector_db_builder.py
```

---

## ▶️ Running the System

Open 3 terminals:

**Terminal 1 — Start API server:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 — Simulate transactions:**
```bash
python simulate_stream.py
```

**Terminal 3 — Launch dashboard:**
```bash
python -m streamlit run dashboard.py
```

Visit `http://localhost:8501` for the dashboard.

---

## 📊 Dashboard Features

- **KPI Cards** — Total transactions, High/Medium/Low risk counts, Blocked count
- **Charts** — Risk distribution, Decision distribution, Risk over time, Fraud probability histogram
- **Detection signals** — Anomaly, behavior, graph, velocity, device counts
- **Transaction table** — Color-coded by risk level with filters
- **High risk cases** — Full investigation with SHAP + LLM reasoning side by side
- **MLOps controls** — Performance monitor, drift detector, retrain button

---

## 🧠 Why Feature Engineering?

The original Kaggle dataset has V1–V28 (PCA-anonymized), Time, and Amount columns.

Two engineered features are added:
- `high_amount_flag` — flags Amount > 2000 as suspicious
- `night_transaction_flag` — flags transactions between midnight–6am (fraud peak hours)

These simple but powerful signals improve both XGBoost accuracy and IsolationForest anomaly detection.

---

## 🔄 MLOps Pipeline

| Component | What it does |
|-----------|-------------|
| Performance Monitor | Checks F1, precision, recall after N transactions |
| Drift Detector | KS test — detects if transaction patterns have changed |
| Retrain Pipeline | Fetches new HIGH risk cases from Supabase + retrains model |
| Model Registry | Saves every model version with metrics and timestamp |

---

## 📈 Model Performance

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Logistic Regression | ~0.85 | ~0.61 | ~0.71 |
| Random Forest | ~0.92 | ~0.78 | ~0.84 |
| **XGBoost** | **~0.95** | **~0.82** | **~0.88** |

XGBoost selected as final model due to best F1 score on imbalanced fraud dataset.

---

## 🎓 About

Built as a 4th year student project demonstrating end-to-end MLOps, RAG, and explainable AI in financial fraud detection.

**Dataset:** [Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## ⚠️ Important Notes

- `creditcard.csv` is NOT included in this repo (too large + Kaggle license). Download separately.
- `.env` is NOT included. Create your own with your API keys.
- This is a simulation — not connected to real bank data.
