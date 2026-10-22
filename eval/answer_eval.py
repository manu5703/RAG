"""
End-to-end answer eval: accuracy, hallucination rate, citation accuracy and
per-stage latency for each retrieval mode, on the campus question set.

    python -m eval.answer_eval                                  # hybrid_rerank only
    python -m eval.answer_eval --modes dense hybrid_rerank      # compare modes
    python -m eval.answer_eval --lens                           # + logit-lens features (HF backend)

Needs an LLM: the in-process HF model by default (a GPU is much faster), or
LLM_BACKEND=openai pointing at a vLLM server. Writes, per mode:
    eval/results/answers_<mode>.jsonl   one row per question (label + detector features)
    eval/results/answers.md             summary table across modes
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.generators import build_generator
from app.rag_pipeline import RAGPipeline
from app.retrieval import MODES
from app.schema import read_chunks
from eval.scoring import is_wrong_answer, label_answer, load_qa, md_table, norm_url, pct

logger = logging.getLogger(__name__)


def logprob_features(lps: List[float]) -> Dict[str, float]:
    if not lps:
        return {"lp_mean": math.nan, "lp_min": math.nan, "lp_std": math.nan, "lp_frac_low": math.nan, "n_tokens": 0}
    a = np.asarray(lps)
    return {
        "lp_mean": float(a.mean()),
        "lp_min": float(a.min()),
        "lp_std": float(a.std()),
        "lp_frac_low": float((a < -2.0).mean()),   # share of tokens the model was unsure about
        "n_tokens": int(a.size),
    }


def run_mode(mode: str, chunks, items: List[dict], generator, embedder=None, reranker=None,
             with_lens: bool = False) -> tuple[List[dict], RAGPipeline]:
    pipe = RAGPipeline(chunks, mode=mode, generator=generator, embedder=embedder, reranker=reranker)
    lens_fn = None
    if with_lens:
        from interpretability.detector import lens_features

        if not hasattr(generator, "model"):
            raise SystemExit("--lens needs the in-process HF backend (LLM_BACKEND=hf).")
        lens_fn = lens_features

    rows = []
    for n, item in enumerate(items, 1):
        res = pipe.query(item["question"])
        gen = res.pop("_gen")
        label = label_answer(item, res["answer"], res["abstained"])
        cited_urls = [s["url"] for s in res["sources"] if s["cited"]]
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": res["answer"],
            "label": label,
            "wrong": is_wrong_answer(label),
            "abstained": res["abstained"],
            "cited_gold": bool(item.get("source_url")) and
                          any(norm_url(u) == norm_url(item["source_url"]) for u in cited_urls),
            "retrieved_gold": bool(item.get("source_url")) and
                              any(norm_url(s["url"]) == norm_url(item["source_url"]) for s in res["sources"]),
            "latency_ms": res["latency_ms"],
            **{f"t_{k}": v for k, v in res["timings_ms"].items()},
            **logprob_features(gen.token_logprobs),
        }
        if lens_fn is not None and gen.answer_ids:
            row.update(lens_fn(generator.model, gen.prompt_ids, gen.answer_ids))
        rows.append(row)
        logger.info("[%s %d/%d] %-12s %s", mode, n, len(items), label, item["id"])
    return rows, pipe


def summarize(mode: str, rows: List[dict]) -> Dict:
    ans = [r for r in rows if r["label"] in ("correct", "incorrect", "missed")]
    una = [r for r in rows if r["label"] in ("abstained_ok", "hallucinated")]
    answered = [r for r in rows if not r["abstained"]]
    correct = [r for r in ans if r["label"] == "correct"]
    return {
        "mode": mode,
        "accuracy": (len(correct) + sum(r["label"] == "abstained_ok" for r in una)) / len(rows),
        "answerable_acc": len(correct) / len(ans) if ans else math.nan,
        "abstain_ok_rate": sum(r["label"] == "abstained_ok" for r in una) / len(una) if una else math.nan,
        "wrong_when_answering": sum(r["wrong"] for r in answered) / len(answered) if answered else math.nan,
        "citation_acc": sum(r["cited_gold"] for r in correct) / len(correct) if correct else math.nan,
        "p50_ms": pct([r["latency_ms"] for r in rows], 50),
        "p95_ms": pct([r["latency_ms"] for r in rows], 95),
        "n": len(rows),
    }


def main(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["hybrid_rerank"], choices=MODES)
    ap.add_argument("--corpus", default=settings.corpus_path)
    ap.add_argument("--qa", default="eval/data/campus_qa.jsonl")
    ap.add_argument("--out", default="eval/results", type=Path)
    ap.add_argument("--lens", action="store_true", help="also compute logit-lens features for the detector")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    chunks, items = read_chunks(args.corpus), load_qa(args.qa)
    generator = build_generator()
    embedder = reranker = None
    summaries = []
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        rows, pipe = run_mode(mode, chunks, items, generator, embedder, reranker, with_lens=args.lens)
        embedder, reranker = pipe._embedder, pipe._reranker
        with (args.out / f"answers_{mode}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summaries.append(summarize(mode, rows))

    table = md_table(
        ["Mode", "Accuracy", "Answerable acc.", "Abstains when it should", "Wrong when answering",
         "Cites gold page", "p50 ms", "p95 ms"],
        [[s["mode"], s["accuracy"], s["answerable_acc"], s["abstain_ok_rate"], s["wrong_when_answering"],
          s["citation_acc"], s["p50_ms"], s["p95_ms"]] for s in summaries],
    )
    model = getattr(generator, "model_name", "?")
    (args.out / "answers.md").write_text(f"Answer eval: {summaries[0]['n']} questions, LLM={model}.\n\n{table}\n")
    (args.out / "answers.json").write_text(json.dumps({"llm": model, "summaries": summaries}, indent=2))
    print("\n" + table)


if __name__ == "__main__":
    main()
