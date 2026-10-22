"""
Answer generators. Two backends behind one interface:

  HFGenerator             runs a Hugging Face chat model in-process (default Qwen2.5-1.5B-Instruct).
                          Returns per-token log-probs, and the model itself is reused by the
                          hallucination detector for logit-lens features.
  OpenAICompatGenerator   calls any OpenAI-compatible /v1/chat/completions endpoint: a vLLM
                          server on a CHPC GPU node, Ollama, or a hosted API.

Both use the same grounded prompt: answer only from the numbered sources,
cite them as [n], and abstain with a fixed sentence when the answer isn't there.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.schema import Chunk

logger = logging.getLogger(__name__)

ABSTAIN = "I don't know based on the provided sources."

SYSTEM_PROMPT = (
    "You answer questions from University of Utah students using ONLY the numbered "
    "sources provided. Be brief and specific: give the rule, number, date or command "
    "the student needs. Cite the sources you used like [1] or [2][3]. If the sources "
    f"do not contain the answer, reply exactly: {ABSTAIN}"
)


def format_context(chunks: List[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        head = f" ({c.header})" if c.header else ""
        blocks.append(f"[{i}]{head}\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: List[Chunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]


_ABSTAIN_RE = re.compile(r"i (do not|don't|cannot|can't) (know|find|answer)|not (contain|mention|provide|specif)", re.I)


def is_abstention(text: str) -> bool:
    return ABSTAIN.lower() in text.lower() or bool(_ABSTAIN_RE.search(text[:200]))


def cited_indices(text: str, n_sources: int) -> List[int]:
    """1-based source numbers cited in the answer, in order of first mention."""
    seen: List[int] = []
    for m in re.finditer(r"\[(\d+)\]", text):
        i = int(m.group(1))
        if 1 <= i <= n_sources and i not in seen:
            seen.append(i)
    return seen


@dataclass
class GenResult:
    text: str
    model: str
    token_logprobs: List[float] = field(default_factory=list)
    prompt_ids: Optional[list] = None   # HF backend only: needed for logit-lens features
    answer_ids: Optional[list] = None

    @property
    def mean_logprob(self) -> float:
        return sum(self.token_logprobs) / len(self.token_logprobs) if self.token_logprobs else float("nan")

    @property
    def min_logprob(self) -> float:
        return min(self.token_logprobs) if self.token_logprobs else float("nan")


class HFGenerator:
    """Lazy-loading Hugging Face chat model (loads on first generate)."""

    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.0):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto") if torch.cuda.is_available() \
            else dict(torch_dtype=torch.float32)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def generate(self, question: str, chunks: List[Chunk]) -> GenResult:
        import torch

        text = self.tokenizer.apply_chat_template(
            build_messages(question, chunks), tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt")
        inputs = {k: enc[k].to(self.model.device) for k in ("input_ids", "attention_mask") if k in enc}
        do_sample = self.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=None, top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        answer_ids = out.sequences[0][prompt_len:]
        logprobs = []
        for step, tok in enumerate(answer_ids):
            if step >= len(out.scores):
                break
            lp = torch.log_softmax(out.scores[step][0].float(), dim=-1)[tok].item()
            logprobs.append(lp)
        # drop trailing special tokens (eos) from the answer
        keep = [i for i, t in enumerate(answer_ids.tolist()) if t not in self.tokenizer.all_special_ids]
        answer_ids_kept = [answer_ids[i].item() for i in keep]
        return GenResult(
            text=self.tokenizer.decode(answer_ids_kept, skip_special_tokens=True).strip(),
            model=self.model_name,
            token_logprobs=[logprobs[i] for i in keep if i < len(logprobs)],
            prompt_ids=inputs["input_ids"][0].tolist(),
            answer_ids=answer_ids_kept,
        )


class OpenAICompatGenerator:
    """
    Any OpenAI-compatible chat endpoint. On CHPC, for example:
        vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
        LLM_BACKEND=openai LLM_BASE_URL=http://<node>:8001/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    """

    def __init__(self, base_url: str, model_name: str, api_key: str = "",
                 max_new_tokens: int = 256, temperature: float = 0.0, timeout_s: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s

    def generate(self, question: str, chunks: List[Chunk]) -> GenResult:
        import requests

        payload = {
            "model": self.model_name,
            "messages": build_messages(question, chunks),
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "logprobs": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_s)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        lps = [t["logprob"] for t in content if t.get("logprob") is not None and math.isfinite(t["logprob"])]
        return GenResult(text=choice["message"]["content"].strip(), model=self.model_name, token_logprobs=lps)


def build_generator():
    from app.config import settings

    if settings.llm_backend == "openai":
        return OpenAICompatGenerator(settings.llm_base_url, settings.llm_model, settings.llm_api_key,
                                     settings.max_new_tokens, settings.temperature)
    return HFGenerator(settings.llm_model, settings.max_new_tokens, settings.temperature)
