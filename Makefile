.PHONY: corpus test eval-retrieval eval-answers detector serve

corpus:            ## crawl ingest/sources.yaml -> corpus/chunks.jsonl
	python -m ingest.crawl

test:              ## unit tests, no model downloads
	python -m pytest -q

eval-retrieval:    ## Recall@k / MRR ablation across bm25, dense, hybrid, hybrid_rerank (CPU is fine)
	python -m eval.retrieval_eval

eval-answers:      ## end-to-end answers for every mode + logit-lens features (GPU recommended)
	python -m eval.answer_eval --modes bm25 dense hybrid hybrid_rerank --lens

detector:          ## hallucination detector AUROC: logprob baseline vs logit lens
	python -m interpretability.detector eval/results/answers_hybrid_rerank.jsonl

serve:
	uvicorn app.main:app --reload
