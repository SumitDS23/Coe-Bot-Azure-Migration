# COE Analytics Chatbot

Intent-routing chatbot for Analytics at ABC.
- **INVENTORY** questions → Text-to-SQL against Enterprise Model Inventory (Excel)
- **RAG** questions → handoff to vendor's Azure OpenAI pipeline

---

## Project Structure

```
coe_prod/
├── api/
│   └── main.py          ← FastAPI server (single POST /query endpoint)
├── rag/
│   ├── router.py        ← Intent classifier (INVENTORY or RAG)
│   ├── inventory.py     ← Text-to-SQL engine (DuckDB)
│   ├── chain.py         ← RAG chain (LangChain)
│   ├── embeddings.py    ← Custom embeddings
│   └── retriever.py     ← FAISS retriever
├── ingest/
│   └── ingest.py        ← Document ingestion → builds FAISS index
├── app/
│   └── main.py          ← Streamlit UI (internal use only)
├── config/
│   └── settings.py      ← All config from .env
├── requirements.txt
├── .env.template        ← Copy to .env and fill in values
└── README.md
```

---

## Setup

### 1. Clone repo
```bash
git clone https://github.com/your-username/coe-analytics-chatbot.git
cd coe-analytics-chatbot
```

### 2. Create environment
```bash
conda create -n coe_env python=3.11 -y
conda activate coe_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
copy .env.template .env
# Fill in .env with your API keys and paths
```

### 5. Add data files (not in repo — transfer separately)
```
Enterprise_Model_Inventory.xlsx  → project root
documents/                       → project root
```

### 6. Run ingestion (builds FAISS index)
```bash
python ingest/ingest.py
```

### 7. Start API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 8. Test
```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"How many live models in Life Insurance?\", \"session_id\": \"test123\", \"conversation_history\": []}"
```

---

## API Reference

### POST /query

**Request:**
```json
{
  "question": "How many live models in Life Insurance?",
  "session_id": "user_123",
  "conversation_history": [],
  "confirm_topic_switch": false
}
```

**Response — INVENTORY:**
```json
{
  "intent": "INVENTORY",
  "answer": "There are 40 live models in Life Insurance.",
  "data": [{"Model_Name": "...", "LOB": "...", "Status": "Live"}],
  "session_id": "user_123",
  "handoff": false,
  "topic_shift": false
}
```

**Response — RAG (handoff to vendor):**
```json
{
  "intent": "RAG",
  "answer": null,
  "data": null,
  "session_id": "user_123",
  "handoff": true,
  "topic_shift": false
}
```

**Response — Topic shift detected:**
```json
{
  "intent": "INVENTORY",
  "answer": null,
  "data": null,
  "session_id": "user_123",
  "handoff": false,
  "topic_shift": true,
  "topic_shift_message": "You were exploring Life Insurance models. Did you want to switch to Mutual Funds?"
}
```

To confirm topic switch, resend with:
```json
{
  "confirm_topic_switch": true
}
```

### GET /health
```json
{"status": "ok", "model": "gemma-3-27b-it"}
```

---

## Running as Windows Service

Install nssm: https://nssm.cc/download

```bash
nssm install CoeAnalyticsAPI
# Path: C:\COE_Analytics\coe_env\Scripts\uvicorn.exe
# Args: api.main:app --host 0.0.0.0 --port 8000
# Startup dir: C:\COE_Analytics\coe_prod

nssm start CoeAnalyticsAPI
```
