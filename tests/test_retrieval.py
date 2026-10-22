import numpy as np
import pytest

from app.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    build_retrieval,
    rerank,
    tokenize,
)
from app.schema import Chunk, Hit


def test_tokenize_keeps_form_names_and_flags():
    toks = tokenize("File Form I-765; use --gres=gpu:1 with a 3.5 GPA")
    assert "i-765" in toks and "765" in toks
    assert "gres=gpu:1" in toks and "gres" in toks
    assert "3.5" in toks
    assert "the" not in tokenize("the a of")


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_each_mode_finds_the_obvious_chunk(mode, corpus, fake_embedder, fake_reranker):
    r = build_retrieval(mode, corpus, embedder=fake_embedder, reranker=fake_reranker, pool=6)
    hits = r.search("how many days of unemployment on OPT", k=3)
    assert hits[0].chunk.section == "Unemployment"
    assert len(hits) == 3


def test_unknown_mode_rejected(corpus):
    with pytest.raises(ValueError):
        build_retrieval("magic", corpus)


def test_empty_corpus_rejected(fake_embedder):
    with pytest.raises(ValueError):
        BM25Retriever().build([])
    with pytest.raises(ValueError):
        DenseRetriever(fake_embedder).build([])


def test_weighted_fusion_scores_bm25_only_candidates_with_their_real_dense_score(corpus, fake_embedder):
    """
    Regression test for the original fusion: a chunk that BM25 ranked highly but
    that fell outside the dense top-k got a dense score of 0, so it was unfairly
    demoted. Now every chunk is scored by both retrievers.
    """
    hybrid = HybridRetriever(DenseRetriever(fake_embedder), BM25Retriever(), fusion="weighted", alpha=0.6)
    hybrid.build(corpus)
    q = "GPU gres partition"
    d = hybrid.dense.scores(q)
    b = hybrid.bm25.scores(q)
    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    expected = 0.6 * norm(d) + 0.4 * norm(b)
    hits = hybrid.search(q, k=len(corpus))
    got = {h.chunk.id: h.score for h in hits}
    for i, c in enumerate(corpus):
        assert got[c.id] == pytest.approx(float(expected[i]), abs=1e-6)


def test_rrf_rewards_agreement_between_retrievers(corpus, fake_embedder):
    hybrid = HybridRetriever(DenseRetriever(fake_embedder), BM25Retriever(), fusion="rrf").build(corpus)
    hits = hybrid.search("home directory 50 GB backed up", k=6)
    assert hits[0].chunk.section == "Home"
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_rerank_reorders_by_reranker_score(corpus, fake_reranker):
    hits = [Hit(c, 0.0) for c in corpus]
    out = rerank(fake_reranker, "Fall 2026 graduate degree deadline", hits, k=2)
    assert out[0].chunk.section == "Deadlines"
    assert len(out) == 2


def test_timings_are_recorded(corpus, fake_embedder, fake_reranker):
    r = build_retrieval("hybrid_rerank", corpus, embedder=fake_embedder, reranker=fake_reranker, pool=4)
    t = {}
    r.search("OPT unemployment", k=2, timings=t)
    assert set(t) == {"retrieve_ms", "rerank_ms"}


def test_header_is_indexed():
    c = Chunk(text="Up to 90 days.", title="Post-Completion OPT", section="Unemployment")
    assert c.indexed_text.startswith("Post-Completion OPT > Unemployment")
    r = BM25Retriever().build([c, Chunk(text="unrelated text about storage")])
    assert r.search("unemployment", 1)[0].chunk is c
