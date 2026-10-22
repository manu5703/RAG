"""
Hallucination detector: can we tell, from the model's internals alone, that
an answer is wrong before a student acts on it?

The old `detect_context_dropout` needed the correct answer token up front,
so it couldn't run on a live answer. This version only uses what the model
produced:

  Baseline (logprob)   mean / min / std of the generated tokens' log-probs and
                       the share of tokens below -2 nats.
  Logit lens (lens)    for every generated token, decode each layer's residual
                       stream through the final norm + unembedding and record
                       (a) the depth at which the emitted token becomes and stays
                       the top-1 prediction ("commit depth") and (b) its
                       probability through the last third of layers. Grounded
                       copying from the sources tends to commit early; tokens the
                       model makes up tend to commit late.

Both feature sets are scored the same way: logistic regression, 5-fold
stratified cross-validation, AUROC with a bootstrap 95% CI, on answered
questions where label wrong=True means incorrect or hallucinated.

    python -m eval.answer_eval --lens                     # produces the features
    python -m interpretability.detector eval/results/answers_hybrid_rerank.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

LOGPROB_FEATURES = ["lp_mean", "lp_min", "lp_std", "lp_frac_low"]
LENS_FEATURES = ["lens_commit_mean", "lens_commit_max", "lens_late_frac", "lens_late_prob_mean", "lens_final_prob_mean"]


# ─── Feature extraction (needs torch + the HF model) ──────────────────────────

def lens_probs(model, prompt_ids: Sequence[int], answer_ids: Sequence[int]):
    """
    Returns (top1, prob): arrays of shape (n_layers + 1, n_answer_tokens).
    top1[l, t]  = whether layer l's logit-lens prediction for answer token t is that token
    prob[l, t]  = logit-lens probability of answer token t at layer l
    """
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(answer_ids)], device=device)
    n_prompt, n_ans = len(prompt_ids), len(answer_ids)
    # the token at position p is predicted from the hidden state at p - 1
    pos = torch.arange(n_prompt - 1, n_prompt - 1 + n_ans, device=device)
    target = torch.tensor(list(answer_ids), device=device)

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        norm = model.model.norm
        top1, prob = [], []
        for hs in out.hidden_states:
            h = hs[0, pos]
            # the last hidden state is already normalised by the model
            if hs is not out.hidden_states[-1]:
                h = norm(h)
            logits = model.lm_head(h).float()
            p = torch.softmax(logits, dim=-1)
            prob.append(p[torch.arange(n_ans), target].cpu().numpy())
            top1.append((logits.argmax(-1) == target).cpu().numpy())
    return np.stack(top1), np.stack(prob)


def summarize_lens(top1: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """Turn per-layer, per-token logit-lens arrays into fixed-size features."""
    n_layers = top1.shape[0]
    commit = []
    for t in range(top1.shape[1]):
        col = top1[:, t]
        # first layer from which the token stays top-1 through the last layer
        depth = n_layers
        for layer in range(n_layers - 1, -1, -1):
            if col[layer]:
                depth = layer
            else:
                break
        commit.append(depth / max(n_layers - 1, 1))
    commit = np.asarray(commit)
    late = prob[int(n_layers * 2 / 3):]
    return {
        "lens_commit_mean": float(commit.mean()),
        "lens_commit_max": float(commit.max()),
        "lens_late_frac": float((commit > 0.9).mean()),
        "lens_late_prob_mean": float(late.mean()),
        "lens_final_prob_mean": float(prob[-1].mean()),
    }


def lens_features(model, prompt_ids, answer_ids) -> Dict[str, float]:
    if not answer_ids:
        return {k: math.nan for k in LENS_FEATURES}
    return summarize_lens(*lens_probs(model, prompt_ids, answer_ids))


# ─── Scoring (numpy + scikit-learn only) ──────────────────────────────────────

def _bootstrap_auc(y: np.ndarray, s: np.ndarray, n: int = 2000, seed: int = 0):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def cv_auroc(rows: List[dict], features: List[str], folds: int = 5, seed: int = 0) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([[r.get(f, np.nan) for f in features] for r in rows], dtype=float)
    y = np.array([int(r["wrong"]) for r in rows])
    keep = ~np.isnan(X).any(axis=1)
    X, y = X[keep], y[keep]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    folds = min(folds, n_pos, n_neg)
    if folds < 2:
        return {"features": features, "n": int(len(y)), "n_wrong": n_pos, "auroc": math.nan,
                "ci95": (math.nan, math.nan), "note": "need at least 2 right and 2 wrong answers"}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    scores = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                               method="predict_proba")[:, 1]
    return {"features": features, "n": int(len(y)), "n_wrong": n_pos,
            "auroc": float(roc_auc_score(y, scores)), "ci95": _bootstrap_auc(y, scores), "folds": folds}


def evaluate_detector(rows: List[dict]) -> List[Dict]:
    answered = [r for r in rows if not r["abstained"]]
    sets = {"logprob (baseline)": LOGPROB_FEATURES, "logit lens": LENS_FEATURES,
            "logprob + logit lens": LOGPROB_FEATURES + LENS_FEATURES}
    out = []
    if not answered:
        return out
    for name, feats in sets.items():
        if not all(f in answered[0] for f in feats):   # lens features only exist with --lens
            continue
        res = cv_auroc(answered, feats)
        res["name"] = name
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", type=Path, help="eval/results/answers_<mode>.jsonl from `eval.answer_eval --lens`")
    ap.add_argument("--out", type=Path, default=Path("eval/results/detector.md"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.answers.read_text().splitlines() if l.strip()]
    results = evaluate_detector(rows)
    lines = [f"Hallucination detector on {args.answers.name}: answered questions only, "
             f"label = wrong answer. Logistic regression, stratified CV.\n",
             "| Features | n | wrong | AUROC | 95% CI |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        lo, hi = r["ci95"]
        lines.append(f"| {r['name']} | {r['n']} | {r['n_wrong']} | {r['auroc']:.3f} | {lo:.3f}–{hi:.3f} |")
        if "note" in r:
            lines.append(f"\nNote ({r['name']}): {r['note']}")
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
