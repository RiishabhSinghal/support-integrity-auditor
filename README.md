# MARS — Mismatch-Aware Re-ranking System

> **Detects priority mismatches in customer support tickets using a fusion of LLM scoring, semantic clustering, resolution-time analysis, and rule-based signals — then classifies them with a fine-tuned DistilBERT model.**

---

## Problem Statement

Support agents manually assign priority levels (Low / Medium / High / Critical) to incoming tickets. This process is subjective and error-prone, leading to two critical failure modes:

| Type | Description |
|------|-------------|
| **Hidden Crisis** | A genuinely urgent ticket is under-prioritised (e.g. a fraud case marked "Low") |
| **False Alarm** | A low-urgency ticket is over-prioritised, wasting escalation bandwidth |

MARS detects these mismatches automatically and generates natural-language explanations for each flagged ticket.

---

## Architecture

```
Customer Ticket
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Preprocessing   (strip boilerplate · normalize text)   │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────── 4 Severity Signals ──────────┐
│  Signal 1 — Rule Score (10%)   keyword matching         │
│  Signal 2 — Resolution Score (20%)  RandomForest proxy  │
│  Signal 3 — Cluster Score (30%)  KMeans on embeddings   │
│  Signal 4 — LLM Score (40%)   Mistral-7B zero-shot      │
└─────────────────────────────────────────────────────────┘
      │  Weighted Fusion → inferred_severity (0–3)
      ▼
┌─────────────────────────────────────────────────────────┐
│  Mismatch Label   |delta| >= 2  →  Hidden Crisis /      │
│                                     False Alarm          │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  DistilBERT + LoRA + Metadata Head                      │
│  Fine-tuned binary classifier (Consistent / Mismatch)   │
│  Test Accuracy: 95.3%  |  Macro F1: 0.940               │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Mistral-7B Dossier  — natural language explanation      │
│  per flagged ticket                                      │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Mars_Project/
├── app_streamlit.py          # Streamlit web application
├── predict.py                # Inference engine (TicketPredictor class)
├── train_pipeline_sia.py     # End-to-end training pipeline (6 stages)
├── notebook.ipynb            # Full experimental notebook
├── requirements.txt          # Python dependencies
├── customer_support_tickets.csv  # 20,000 ticket dataset
├── mistral_scores.csv        # Pre-computed Mistral-7B severity scores
└── saved_model/              # Trained model artefacts (auto-downloaded)
    ├── model.pt
    ├── tokenizer/
    ├── channel_encoder.pkl
    ├── category_encoder.pkl
    ├── resolution_scaler.pkl
    └── threshold.json
```

---

## Installation

```bash
# 1. Clone or download the project
cd Mars_Project

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** Training and local Mistral inference require a CUDA-capable GPU.  
> The Streamlit app runs fully on CPU (model inference only).

---

## Running the App

```bash
streamlit run app_streamlit.py
```

Open **http://localhost:8501** in your browser.

The app has three tabs:

| Tab | Description |
|-----|-------------|
| 🔍 **Single Ticket** | Fill a form, get an instant verdict + AI explanation |
| 📂 **Batch CSV** | Upload a tickets CSV, score all rows, download results |
| 📊 **Dashboard** | Verdict distribution, confidence histogram, severity delta heatmap |

### Dossier Analysis Modes

Select via the **⚙️ Analysis mode** expander above the tabs:

| Mode | Description |
|------|-------------|
| `mistral_api` | Generative explanation via Mistral API *(requires `MISTRAL_API_KEY`)* |
| `local_mistral` | Runs Mistral-7B locally *(needs a GPU)* |
| `template` | Instant deterministic explanation, no LLM |

Set your API key as an environment variable or in `.streamlit/secrets.toml`:
```toml
MISTRAL_API_KEY = "your-key-here"
```

---

## Training from Scratch

```bash
# Fast run (uses rule-based proxy instead of Mistral for LLM scores)
python train_pipeline_sia.py --data customer_support_tickets.csv

# Full run with Mistral-7B scoring (needs GPU, ~4–5 hours)
python train_pipeline_sia.py --data customer_support_tickets.csv --use-llm

# Force re-run all stages even if cached files exist
python train_pipeline_sia.py --data customer_support_tickets.csv --force --epochs 8
```

### Training Pipeline Stages

| Stage | Output | Description |
|-------|--------|-------------|
| 1 | `feature_engineered.csv` | Preprocessing, embeddings, KMeans clustering |
| 2 | `mistral_scores.csv` | LLM severity scores (checkpointed every 500 rows) |
| 3 | `pseudo_labeled_dataset.csv` | Weighted signal fusion → mismatch labels |
| 4 | `train.csv / val.csv / test.csv` | Stratified 70/15/15 split + encoding |
| 5 | `saved_model/` | DistilBERT + LoRA fine-tuning |
| 6 | `sample_dossiers.json` | Mistral dossiers for 50 flagged tickets |

---

## Model Details

| Component | Detail |
|-----------|--------|
| Base model | `distilbert-base-uncased` |
| Fine-tuning | LoRA (r=8, alpha=16, target: q_lin, v_lin) |
| Extra inputs | Channel, Category, Resolution time, Priority (4 metadata scalars) |
| Optimizer | AdamW (lr=2e-4, weight_decay=0.01) |
| Scheduler | Linear warmup (10%) + linear decay |
| Epochs | 8 (best checkpoint by val Macro F1) |
| Threshold | Tuned on validation set (default: 0.62) |

### Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | **95.3%** |
| Macro F1 | **0.940** |
| Recall — Consistent | 97.98% |
| Recall — Mismatch | 88.35% |

---

## Dataset

`customer_support_tickets.csv` — 20,000 synthetic customer support tickets with fields:

`Ticket_ID`, `Ticket_Subject`, `Ticket_Description`, `Priority_Level`, `Ticket_Channel`, `Issue_Category`, `Resolution_Time_Hours`, `Customer_Email`

Priority levels: **Low · Medium · High · Critical**

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

- `transformers` + `peft` — DistilBERT fine-tuning with LoRA
- `sentence-transformers` — Semantic embeddings (all-MiniLM-L6-v2)
- `scikit-learn` — KMeans, RandomForest, preprocessing, metrics
- `streamlit` + `plotly` — Web application and charts
- `huggingface-hub` — Model download on first run
