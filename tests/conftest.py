"""
Test doubles so the suite runs in seconds with no model downloads.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.generators import GenResult  # noqa: E402
from app.retrieval import tokenize  # noqa: E402
from app.schema import Chunk  # noqa: E402


class FakeEmbedder:
    """Hashed bag-of-words: texts sharing words get similar vectors."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            out[i, 0] += 1e-3  # never an all-zero vector
        return out


class FakeReranker:
    """Scores by word overlap with the query."""

    def score(self, query, texts):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) for t in texts], dtype="float32")


class EchoGenerator:
    """Answers with the first source and cites it, or abstains if told to."""

    model_name = "echo"

    def __init__(self, abstain_on=()):
        self.abstain_on = set(abstain_on)

    def generate(self, question, chunks):
        if question in self.abstain_on or not chunks:
            return GenResult(text="I don't know based on the provided sources.", model="echo",
                             token_logprobs=[-0.1, -3.0])
        return GenResult(text=f"{chunks[0].text} [1]", model="echo", token_logprobs=[-0.1, -0.2, -0.3])


@pytest.fixture
def corpus():
    return [
        Chunk(text="Full-time CPT is 21 to 40 hours per week. Part-time CPT is 1 to 20 hours.",
              url="https://ic.utah.edu/cpt/index.php", title="Curricular Practical Training", section="Policies"),
        Chunk(text="You may have up to 90 days of unemployment during post-completion OPT.",
              url="https://ic.utah.edu/post-opt/", title="Post-Completion OPT", section="Unemployment"),
        Chunk(text="File the I-765 no earlier than 90 days before the program end date.",
              url="https://ic.utah.edu/post-opt/", title="Post-Completion OPT", section="Application Window"),
        Chunk(text="Request GPUs with #SBATCH --gres=gpu:1 and a GPU partition.",
              url="https://chpc.utah.edu/documentation/guides/gpus-accelerators.php", title="GPUs", section="Slurm"),
        Chunk(text="Home directories are 50 GB by default and are not backed up.",
              url="https://chpc.utah.edu/resources/storage_services.php", title="Storage", section="Home"),
        Chunk(text="The Fall 2026 deadline to apply for a graduate degree is August 28, 2026.",
              url="https://registrar.utah.edu/handbook/graduategraduation.php", title="Apply for Graduate Degree",
              section="Deadlines"),
    ]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_reranker():
    return FakeReranker()
