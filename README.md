# Campus RAG Assistant

Answers University of Utah students' questions about F-1 work authorization (CPT, OPT), graduate graduation deadlines and CHPC computing. Every answer cites the official page it came from, and the assistant says "I don't know" when the indexed pages don't cover the question.

The same questions come up every semester ("How many unemployment days do I get on OPT?", "How do I request a GPU on notchpeak?"), and the answers are spread across ISSS, Registrar and CHPC pages that change from year to year. A wrong answer about immigration rules costs more than no answer, so this project measures three things: whether retrieval finds the right page, whether the answer is correct, and whether the system can flag its own likely hallucinations.

## Results

> Fill this table by running the evals (see [Run the evals](#run-the-evals)). Numbers below are placeholders until then. Report the crawl date with them.

**Retrieval ablation**: gold page in top-k, 36 answerable questions (`eval/results/retrieval.md`)

| Mode | Recall@1 | Recall@5 | MRR@10 | p95 ms |
| --- | --- | --- | --- | --- |
| BM25 | – | – | – | – |
| Dense (MiniLM) | – | – | – | – |
| Hybrid (RRF) | – | – | – | – |
| Hybrid + cross-encoder rerank | – | – | – | – |

**Answers**: Qwen2.5-1.5B-Instruct, 40 questions incl. 4 the pages can't answer (`eval/results/answers.md`)

| Mode | Accuracy | Abstains when it should | Wrong when answering | Cites gold page | p95 ms |
| --- | --- | --- | --- | --- | --- |
| Dense | – | – | – | – | – |
| Hybrid + rerank | – | – | – | – | – |

**Hallucination detector**: AUROC for flagging wrong answers from model internals (`eval/results/detector.md`)

| Features | AUROC (95% CI) |
| --- | --- |
| Token log-probs (baseline) | – |
| Logit lens | – |
| Log-probs + logit lens | – |

## How it works

```
ingest/sources.yaml ──crawl──▶ corpus/chunks.jsonl  (text + URL + page title + section heading)
                                      │
question ──▶ BM25 ─┐                  │
             Dense ┴─ RRF fusion ─▶ top 20 ─▶ cross-encoder rerank ─▶ top 5 ─▶ Qwen ─▶ answer + [n] citations
                                                                                  │
                                                       logit lens over the answer tokens ─▶ hallucination score
```

1. **Crawl** (`ingest/`). Fetches the pages listed in `sources.yaml`, honouring robots.txt at one request per second. Strips navigation and footers, and splits each page at its headings, so every chunk keeps its page URL, title and section ("Post-Completion OPT > Application Window").
2. **Retrieve** (`app/retrieval.py`). BM25 with a tokenizer that keeps form names and flags intact (`I-765`, `--gres=gpu:1`, `3.5`), all-MiniLM-L6-v2 dense vectors, and reciprocal-rank fusion. Weighted fusion is still available, and now scores every candidate with both retrievers (the first version gave BM25-only candidates a dense score of 0). Then ms-marco-MiniLM cross-encoder reranking.
3. **Generate** (`app/generators.py`). Qwen2.5-1.5B-Instruct in process by default, or any OpenAI-compatible server (vLLM on a CHPC GPU node, Ollama) via `LLM_BACKEND=openai`. The prompt requires `[n]` citations and a fixed abstention sentence.
4. **Detect hallucinations** (`interpretability/detector.py`). Decodes every layer's residual stream through the unembedding for each generated token. It records how deep in the network the token becomes, and stays, the top prediction, and its late-layer probability. A logistic regression on these features is compared with a token-log-prob baseline using 5-fold cross-validated AUROC. Unlike the original logit-lens demo, it needs no knowledge of the correct answer, so it can run on live answers.

The API (`/query`) returns the answer, the cited sources with URLs, an `abstained` flag and per-stage latency. `streamlit_app.py` adds a UI with the logit-lens and DLA plots.

## Evaluation

- **Questions** (`eval/data/campus_qa.jsonl`): 40 seed questions written from the ISSS CPT and OPT pages, the Registrar's graduate graduation page and four CHPC pages, read on Oct 22, 2026. Each one has a reference answer, the gold source URL and key facts that a correct answer must contain (any-of groups, numbers matched as whole numbers). Four questions can't be answered from the pages, to test abstention. Extend it toward 150 with questions students actually ask, and check every reference answer against the live page.
- **Retrieval**: whether the gold page appears in the top k chunks (Recall@1/3/5/10, MRR@10) and latency. Needs no LLM, so it runs in CI and gates pull requests (`eval/thresholds.json`).
- **Answers**: accuracy over all questions (an abstention on an unanswerable question counts as correct), abstention rate on unanswerable questions, wrong-answer rate among answered questions, how often a correct answer cites the gold page, and p50/p95 latency.
- **RAGAS** (`eval/evaluate.py`) is still available as an optional LLM-as-judge second opinion.

## Run the evals

```bash
pip install -r requirements.txt
python -m ingest.crawl                 # build corpus/ (commit chunks.jsonl + pages.jsonl as a dated snapshot)
make test                              # unit tests, no model downloads
make eval-retrieval                    # CPU is fine, a few minutes
make eval-answers                      # GPU recommended; --lens adds detector features
make detector
```

On CHPC, edit the account in `scripts/chpc_answer_eval.slurm` (find it with `mychpc batch`) and `sbatch` it. It runs on `notchpeak-shared-short` with one GPU and keeps model weights in scratch instead of your 50 GB home.

## Serve

```bash
uvicorn app.main:app --reload          # http://localhost:8000/docs
streamlit run streamlit_app.py
docker compose up --build
```

Settings live in `app/config.py` and can be overridden with environment variables: `RETRIEVAL_MODE`, `FUSION`, `LLM_MODEL`, `LLM_BACKEND`, `LLM_BASE_URL`.

## Project structure

```
ingest/            crawler, HTML parsing, sources.yaml
corpus/            crawled snapshot (chunks.jsonl, pages.jsonl)
app/               retrieval, generators, pipeline, FastAPI service
eval/              question set, retrieval + answer evals, scoring, CI thresholds
interpretability/  hallucination detector, logit lens + DLA plots
scripts/           CHPC Slurm job
tests/             unit tests with fake embedder / reranker / LLM
```

## Limitations

- Questions about immigration status should still go to an ISSS advisor. The assistant cites pages, it doesn't give legal advice.
- The ISSS CPT page still shows 2023–24 dates, and the assistant will repeat them. The crawl date is recorded per page in `pages.jsonl` so stale answers can be traced.
- Key-fact scoring can miss correct paraphrases. Spot-check the `incorrect` rows in `answers_<mode>.jsonl`.
