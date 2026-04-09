"""
RAGAS Evaluation — runs on a golden Q&A dataset and fails CI if any score
drops below its threshold.

Usage:
    python -m eval.evaluate          # from repo root
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

from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── CI thresholds ────────────────────────────────────────────────────────────

THRESHOLDS = {
    "faithfulness":      0.75,
    "answer_relevancy":  0.70,
    "context_precision": 0.65,
    "context_recall":    0.60,
}

# ─── Golden dataset ───────────────────────────────────────────────────────────
# Expand this with GPT-4-generated pairs over your real corpus.

GOLDEN = [
    {
        "question":       "Where is the Eiffel Tower?",
        "ground_truth":   "The Eiffel Tower is in Paris, France.",
        "reference_context": "The Eiffel Tower is located in Paris, France.",
    },
    {
        "question":       "What is RAG?",
        "ground_truth":   "RAG combines retrieval systems with generative language models.",
        "reference_context": "RAG combines retrieval systems with generative language models.",
    },
    {
        "question":       "What is Python used for?",
        "ground_truth":   "Python is widely used for data science and ML development.",
        "reference_context": "Python is widely used for data science and ML development.",
    },
    {
        "question":       "What is machine learning?",
        "ground_truth":   "Machine learning is a subset of artificial intelligence.",
        "reference_context": "Machine learning is a subset of artificial intelligence.",
    },
    {
        "question":       "What do Docker containers do?",
        "ground_truth":   "Docker containers package code and dependencies together.",
        "reference_context": "Docker containers package code and dependencies together.",
    },
]

CHUNKS = [g["reference_context"] for g in GOLDEN]


# ─── Evaluation runner ────────────────────────────────────────────────────────

def run_eval() -> bool:
    """Run the full evaluation. Returns True if all thresholds pass."""
    pipeline = RAGPipeline(chunks=CHUNKS)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in GOLDEN:
        result = pipeline.query(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["context_used"])
        ground_truths.append(item["ground_truth"])
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
    scores_dir = Path("eval")
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
