# Production RAG Pipeline with CoT Faithfulness Analysis

Hybrid retrieval (dense + BM25) → cross-encoder reranking → Qwen generation,
with RAGAS evaluation CI and logit lens / DLA interpretability.


<img width="1919" height="867" alt="Screenshot 2026-04-08 202752" src="https://github.com/user-attachments/assets/8502c3cd-6ffe-4614-a53e-a3d6bef4b593" />
<img width="1919" height="794" alt="Screenshot 2026-04-08 202942" src="https://github.com/user-attachments/assets/eadbb428-8b22-4f31-9e0a-4fce3a4cf9e6" />
<img width="1448" height="814" alt="Screenshot 2026-04-08 203016" src="https://github.com/user-attachments/assets/d80b5845-923b-4aa7-b0df-1fc2ad3b2968" />
<img width="1446" height="808" alt="Screenshot 2026-04-08 203041" src="https://github.com/user-attachments/assets/1768e4b2-93bb-4685-a6f6-4b9e1c9fd1c5" />

## Project Structure

```
rag_project/
├── app/
│   ├── rag_pipeline.py        # Hybrid retrieval + reranking + Qwen generation
│   └── main.py                # FastAPI service
├── eval/
│   └── evaluate.py            # RAGAS scoring + CI threshold checks
├── interpretability/
│   └── faithfulness.py        # Logit lens + DLA on Qwen
├── .github/workflows/
│   └── eval.yml               # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Run API locally
uvicorn app.main:app --reload
# → Docs at http://localhost:8000/docs
# → POST /query, POST /interpretability, GET /health

# Run with Docker
docker-compose up --build

# Run evaluation (from repo root)
python -m eval.evaluate

# Run interpretability analysis (from repo root)
python -m interpretability.faithfulness
```

## How it works

### 1. Hybrid Retrieval
- **Dense**: `all-MiniLM-L6-v2` embeddings → FAISS IndexFlatIP (cosine similarity)
- **Sparse**: BM25Okapi via `rank_bm25`
- **Fusion**: weighted sum (α=0.6 dense, 0.4 BM25) over top-20 candidates

### 2. Reranking
- `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair
- Top-5 chunks passed to Qwen

### 3. Generation
- Qwen prompted with retrieved context + "think step by step" CoT instruction

### 4. Evaluation (RAGAS)
| Metric | Threshold |
|---|---|
| Faithfulness | 0.75 |
| Answer Relevancy | 0.70 |
| Context Precision | 0.65 |
| Context Recall | 0.60 |

CI fails automatically if any score drops below threshold.

### 5. Interpretability
- **Logit Lens**: tracks answer token probability layer-by-layer — shows when
  the model commits to context vs. parametric recall
- **DLA**: decomposes final logit into per-layer attention and MLP contributions —
  identifies which components drive or suppress the answer
- **Hallucination flag**: auto-detects layers with sharp probability dropout
