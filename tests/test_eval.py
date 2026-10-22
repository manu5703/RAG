import json
from pathlib import Path

import numpy as np
import pytest

from app.generators import cited_indices, format_context, is_abstention
from app.rag_pipeline import RAGPipeline
from app.schema import write_chunks
from eval import retrieval_eval
from eval.answer_eval import run_mode, summarize
from eval.scoring import (
    facts_present,
    label_answer,
    load_qa,
    norm_url,
    reciprocal_rank,
    recall_at_k,
    term_present,
)
from interpretability.detector import cv_auroc, summarize_lens
from tests.conftest import EchoGenerator


def test_numbers_match_whole_numbers_only():
    assert term_present("3", "about 3 to 5 months")
    assert not term_present("3", "within 30 days")
    assert term_present("3.5", "a GPA of 3.5 or higher")
    assert not term_present("5", "3.5 GPA")
    assert term_present("august 28", "Apply by August 28, 2026.")


def test_facts_present_any_of_groups():
    groups = [["12", "twelve"], ["opt"]]
    assert facts_present(groups, "Twelve months of full-time CPT removes OPT eligibility.")
    assert not facts_present(groups, "Twelve months is a long time.")


def test_labels():
    a = {"answerable": True, "must_include": [["90"]]}
    u = {"answerable": False, "must_include": []}
    assert label_answer(a, "Up to 90 days.", False) == "correct"
    assert label_answer(a, "Up to 60 days.", False) == "incorrect"
    assert label_answer(a, "I don't know based on the provided sources.", True) == "missed"
    assert label_answer(u, "It costs $470.", False) == "hallucinated"
    assert label_answer(u, "I don't know based on the provided sources.", True) == "abstained_ok"


def test_url_normalisation_and_ranking_metrics():
    assert norm_url("https://www.chpc.utah.edu/docs/index.php") == norm_url("http://chpc.utah.edu/docs/")
    urls = ["https://a/x", "https://b/y/index.php", "https://c/z"]
    assert recall_at_k(urls, "https://b/y/", 1) == 0.0
    assert recall_at_k(urls, "https://b/y/", 2) == 1.0
    assert reciprocal_rank(urls, "https://c/z") == pytest.approx(1 / 3)
    assert reciprocal_rank(urls, "https://d") == 0.0


def test_abstention_and_citation_parsing(corpus):
    assert is_abstention("I don't know based on the provided sources.")
    assert is_abstention("The sources do not specify the fee.")
    assert not is_abstention("Full-time CPT is 21 to 40 hours per week [1].")
    assert cited_indices("See [2] and [1][2], not [9].", 5) == [2, 1]
    ctx = format_context(corpus[:2])
    assert ctx.startswith("[1] (Curricular Practical Training > Policies)")


def test_seed_question_set_is_well_formed():
    items = load_qa(Path(__file__).resolve().parents[1] / "eval/data/campus_qa.jsonl")
    assert len(items) >= 40
    for i in items:
        assert i["question"].strip() and i["category"]
        if i["answerable"]:
            assert i["source_url"].startswith("https://") and i["must_include"]
            # the reference answer itself must pass its own key-fact check
            assert facts_present(i["must_include"], i["answer"]), i["id"]
    assert sum(not i["answerable"] for i in items) >= 4


def test_pipeline_returns_cited_sources(corpus, fake_embedder, fake_reranker):
    pipe = RAGPipeline(corpus, mode="hybrid_rerank", generator=EchoGenerator(),
                       embedder=fake_embedder, reranker=fake_reranker)
    res = pipe.query("days of unemployment on OPT")
    assert "90 days" in res["answer"] and not res["abstained"]
    assert res["sources"][0]["cited"] and res["sources"][0]["url"].endswith("/post-opt/")
    assert set(res["timings_ms"]) == {"retrieve_ms", "rerank_ms", "generate_ms"}
    pipe.rebuild(corpus[:3])  # re-index without reloading models
    assert len(pipe.chunks) == 3


def test_answer_eval_end_to_end(corpus, fake_embedder, fake_reranker):
    items = [
        {"id": "a", "category": "opt", "answerable": True, "question": "days of unemployment on OPT",
         "answer": "", "must_include": [["90"]], "source_url": "https://ic.utah.edu/post-opt/"},
        {"id": "b", "category": "none", "answerable": False, "question": "OPT fee in dollars",
         "answer": "", "must_include": [], "source_url": ""},
    ]
    gen = EchoGenerator(abstain_on={"OPT fee in dollars"})
    rows, _ = run_mode("hybrid_rerank", corpus, items, gen, fake_embedder, fake_reranker)
    assert [r["label"] for r in rows] == ["correct", "abstained_ok"]
    assert rows[0]["cited_gold"] and rows[0]["lp_mean"] < 0
    s = summarize("hybrid_rerank", rows)
    assert s["accuracy"] == 1.0 and s["citation_acc"] == 1.0


def test_retrieval_eval_writes_ablation_table(tmp_path, corpus, fake_embedder, fake_reranker, monkeypatch):
    write_chunks(corpus, tmp_path / "chunks.jsonl")
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(x) for x in [
        {"id": "1", "category": "opt", "answerable": True, "question": "unemployment days OPT",
         "answer": "90", "must_include": [["90"]], "source_url": "https://ic.utah.edu/post-opt"},
        {"id": "2", "category": "chpc", "answerable": True, "question": "request a GPU gres",
         "answer": "--gres=gpu", "must_include": [["--gres=gpu"]],
         "source_url": "https://chpc.utah.edu/documentation/guides/gpus-accelerators.php"},
    ]))
    results = retrieval_eval.run(["bm25", "hybrid_rerank"], str(tmp_path / "chunks.jsonl"), str(qa),
                                 tmp_path / "out", "rrf", embedder=fake_embedder, reranker=fake_reranker)
    assert [r["recall@5"] for r in results] == [1.0, 1.0]
    assert "| hybrid_rerank |" in (tmp_path / "out" / "retrieval.md").read_text()
    th = tmp_path / "th.json"
    th.write_text(json.dumps({"retrieval": {"bm25": {"recall@5": 0.9}}}))
    assert retrieval_eval.gate(results, th)
    th.write_text(json.dumps({"retrieval": {"bm25": {"recall@5": 1.1}}}))
    assert not retrieval_eval.gate(results, th)


def test_lens_summary_commit_depth():
    # 5 layers x 2 tokens. Token 0 is top-1 from layer 1 on; token 1 only at the last layer.
    top1 = np.array([[0, 0], [1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool)
    prob = np.full((5, 2), 0.5)
    f = summarize_lens(top1, prob)
    assert f["lens_commit_mean"] == pytest.approx((0.25 + 1.0) / 2)
    assert f["lens_commit_max"] == 1.0
    assert f["lens_late_frac"] == 0.5


def test_detector_cv_auroc_on_separable_features():
    rng = np.random.default_rng(0)
    rows = [{"wrong": i % 2 == 1, "lp_mean": (-2.0 if i % 2 else -0.2) + rng.normal(0, 0.1)} for i in range(40)]
    res = cv_auroc(rows, ["lp_mean"])
    assert res["auroc"] > 0.95 and res["ci95"][0] > 0.8
    tiny = cv_auroc(rows[:3], ["lp_mean"])
    assert "note" in tiny
