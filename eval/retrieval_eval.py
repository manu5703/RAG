"""
Retrieval ablation: does the gold source page show up in the top-k chunks?
No LLM needed, so this runs in CI on every pull request.

    python -m eval.retrieval_eval                         # all four modes
    python -m eval.retrieval_eval --modes bm25 hybrid     # a subset
    python -m eval.retrieval_eval --gate                  # exit 1 if below eval/thresholds.json

Writes eval/results/retrieval.json and eval/results/retrieval.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.retrieval import MODES, build_retrieval
from app.schema import Chunk, read_chunks
from eval.scoring import load_qa, md_table, pct, recall_at_k, reciprocal_rank

logger = logging.getLogger(__name__)

KS = (1, 3, 5, 10)


def evaluate_mode(mode: str, chunks: List[Chunk], items: List[dict],
                  embedder=None, reranker=None, fusion: str = "rrf") -> Dict:
    retrieval = build_retrieval(mode, chunks, embedder=embedder, reranker=reranker,
                                fusion=fusion, alpha=settings.alpha, pool=settings.top_k_retrieve)
    gold = [i for i in items if i["answerable"] and i.get("source_url")]
    recalls = {k: [] for k in KS}
    rr, lat, per_q = [], [], []
    for item in gold:
        t0 = time.perf_counter()
        hits = retrieval.search(item["question"], max(KS))
        lat.append((time.perf_counter() - t0) * 1000)
        urls = [h.chunk.url for h in hits]
        for k in KS:
            recalls[k].append(recall_at_k(urls, item["source_url"], k))
        rr.append(reciprocal_rank(urls, item["source_url"], 10))
        per_q.append({"id": item["id"], "hit@5": recalls[5][-1], "top_urls": urls[:5]})
    n = len(gold)
    return {
        "mode": mode,
        "n_questions": n,
        **{f"recall@{k}": sum(v) / n for k, v in recalls.items()},
        "mrr@10": sum(rr) / n,
        "p50_ms": pct(lat, 50),
        "p95_ms": pct(lat, 95),
        "per_question": per_q,
        # the models are shared across modes: reuse them instead of reloading
        "_embedder": _get_embedder(retrieval),
        "_reranker": retrieval.reranker,
    }


def _get_embedder(retrieval):
    fp = retrieval.first_pass
    dense = getattr(fp, "dense", fp if hasattr(fp, "embedder") else None)
    return dense.embedder if dense is not None else None


def run(modes, corpus_path: str, qa_path: str, out_dir: Path, fusion: str,
        embedder=None, reranker=None) -> List[Dict]:
    chunks = read_chunks(corpus_path)
    items = load_qa(qa_path)
    results = []
    for mode in modes:
        logger.info("Evaluating %s over %d chunks …", mode, len(chunks))
        r = evaluate_mode(mode, chunks, items, embedder=embedder, reranker=reranker, fusion=fusion)
        embedder = r.pop("_embedder") or embedder
        reranker = r.pop("_reranker") or reranker
        results.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval.json").write_text(json.dumps(
        {"corpus": corpus_path, "n_chunks": len(chunks), "fusion": fusion, "results": results}, indent=2))
    rows = [[r["mode"], r["recall@1"], r["recall@3"], r["recall@5"], r["recall@10"], r["mrr@10"],
             r["p50_ms"], r["p95_ms"]] for r in results]
    table = md_table(["Mode", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "p50 ms", "p95 ms"], rows)
    (out_dir / "retrieval.md").write_text(
        f"Retrieval ablation on {results[0]['n_questions']} answerable questions, "
        f"{len(chunks)} chunks, fusion={fusion}.\n\n{table}\n")
    print("\n" + table)
    return results


def gate(results: List[Dict], thresholds_path: Path) -> bool:
    th = json.loads(thresholds_path.read_text())["retrieval"]
    ok = True
    for r in results:
        for metric, minimum in th.get(r["mode"], {}).items():
            status = "ok" if r[metric] >= minimum else "FAIL"
            ok &= r[metric] >= minimum
            print(f"[gate] {r['mode']:<14} {metric:<10} {r[metric]:.3f} (min {minimum})  {status}")
    return ok


def main(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--corpus", default=settings.corpus_path)
    ap.add_argument("--qa", default="eval/data/campus_qa.jsonl")
    ap.add_argument("--fusion", default=settings.fusion, choices=["rrf", "weighted"])
    ap.add_argument("--out", default="eval/results", type=Path)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--thresholds", default="eval/thresholds.json", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    results = run(args.modes, args.corpus, args.qa, args.out, args.fusion)
    if args.gate and not gate(results, args.thresholds):
        print("[gate] retrieval quality dropped below threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
