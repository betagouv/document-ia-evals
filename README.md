# 📘 Evalap — Document-IA Evaluation Platform

The goal of this project is to build an **evaluation platform for document IA**, enabling experiments, benchmarking, and visualization of model / pipelines performance on document processing tasks.

---

## 🚀 Development Setup

### 1️⃣ API Server
Runs the FastAPI backend.

```bash
cd evalap
uvicorn evalap.api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API docs at:
- Swagger UI: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc

---

### 2️⃣ Runners
Runners execute evaluation jobs asynchronously.

```bash
cd evalap
PYTHONPATH="." python -m evalap.runners
```

These workers connect to the API and process queued evaluation tasks.

---

### 3️⃣ Streamlit UI
The Streamlit interface lets you browse and visualize experiments, evaluations, and datasets.

```bash
cd evalap
streamlit run evalap/ui/demo_streamlit/app.py --server.runOnSave true
```

Access the UI at:
- http://localhost:8501/

---
