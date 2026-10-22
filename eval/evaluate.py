"""
Optional RAGAS evaluation (LLM-as-judge) on the campus question set.
Exits 1 if any score drops below its threshold.

The primary evals are eval.retrieval_eval (runs in CI) and eval.answer_eval
(key-fact scoring, no judge model needed). Use this one for a second opinion.

Usage:
    OPENAI_API_KEY=sk-... python -m eval.evaluate
"""

import json
import logging
import sys
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings
from app.rag_pipeline import RAGPipeline
from app.schema import read_chunks
from eval.scoring import load_qa

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── CI thresholds ────────────────────────────────────────────────────────────

THRESHOLDS = {
    "faithfulness":      0.75,
    "answer_relevancy":  0.70,
    "context_precision": 0.65,
    "context_recall":    0.60,
}

# ─── Dataset ──────────────────────────────────────────────────────────────────
# The same campus question set the other evals use; RAGAS scores the
# answerable questions (it has no notion of a correct abstention).

QA_PATH = "eval/data/campus_qa.jsonl"


# ─── Evaluation runner ────────────────────────────────────────────────────────

def run_eval() -> bool:
    """Run the full evaluation. Returns True if all thresholds pass."""
    pipeline = RAGPipeline(chunks=read_chunks(settings.corpus_path))
    golden = [i for i in load_qa(QA_PATH) if i["answerable"]]

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        result = pipeline.query(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["context_used"])
        ground_truths.append(item["answer"])
        logger.info("Q: %s → A: %s", item["question"], result["answer"][:80])

    dataset = Dataset.from_dict(
        {
            "question":    questions,
            "answer":      answers,
            "contexts":    contexts,
            "ground_truth": ground_truths,
        }
    )

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # ── Report ────────────────────────────────────────────────────
    print("\n── RAGAS Scores ─────────────────────────────────────────")
    failed = False
    for metric, threshold in THRESHOLDS.items():
        score  = results[metric]
        status = "✓" if score >= threshold else "✗ FAIL"
        print(f"  {metric:<25} {score:.3f}  (min {threshold})  {status}")
        if score < threshold:
            failed = True

    # ── Persist for CI artifact ───────────────────────────────────
    scores_dir = Path("eval/results")
    scores_dir.mkdir(exist_ok=True)
    scores_path = scores_dir / "scores.json"
    scores_path.write_text(
        json.dumps({k: round(results[k], 4) for k in THRESHOLDS}, indent=2)
    )
    logger.info("Scores written to %s", scores_path)

    return not failed


if __name__ == "__main__":
    passed = run_eval()
    if not passed:
        print("\n[CI] Evaluation failed — scores below threshold.")
        sys.exit(1)
    print("\n[CI] All scores passed.")
