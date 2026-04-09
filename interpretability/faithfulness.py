"""
Faithfulness Interpretability — Logit Lens + Direct Logit Attribution (DLA).

Logit Lens
----------
Projects each layer's residual-stream hidden state through the final layer-norm
and unembedding matrix to read out the probability of `target_token` at every
depth. This shows *when* the model "commits" to the answer.

Direct Logit Attribution (DLA)
-------------------------------
Decomposes the final logit into additive contributions from each layer's
attention block and MLP block. High positive DLA → that component is writing
the answer token; negative DLA → it is suppressing it.

Correct pre-norm architecture (Qwen1.5 / Llama-style):
  h_mid = h_in + attn(input_layernorm(h_in))
  h_out = h_mid + mlp(post_attention_layernorm(h_mid))

Both blocks add to the residual stream, so their contributions are
independently projected onto the unembedding direction of `target_token`.
"""

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen1.5-0.5B-Chat"


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(model_name: str = _DEFAULT_MODEL):
    """Load tokenizer + model (float32 for interpretability precision)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Always use float32 on CPU for interpretability precision;
    # avoid device_map="auto" which can attempt unsupported disk offload.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_hidden_states=True,
        output_attentions=False,   # we re-run attn manually for DLA
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return tokenizer, model


# ─── Logit Lens ───────────────────────────────────────────────────────────────

def logit_lens(
    tokenizer,
    model,
    prompt: str,
    target_token: str,
) -> Dict:
    """
    For every transformer layer, project its hidden state through the final
    layer-norm + unembedding matrix and record P(target_token).
    """
    inputs    = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
    hidden_states = outputs.hidden_states
    lm_head       = model.lm_head

    # Final layer-norm (Qwen1.5 stores it at model.model.norm)
    ln_f = getattr(model.model, "norm", None)

    probs_per_layer = []
    for hs in hidden_states:
        h = hs[0, -1, :]           # last-token position
        if ln_f is not None:
            h = ln_f(h.unsqueeze(0)).squeeze(0)
        logits = lm_head(h)
        prob   = torch.softmax(logits, dim=-1)[target_id].item()
        probs_per_layer.append(prob)

    return {
        "target_token":    target_token,
        "probs_per_layer": probs_per_layer,
        "n_layers":        len(probs_per_layer),
    }


# ─── Direct Logit Attribution ─────────────────────────────────────────────────

def direct_logit_attribution(
    tokenizer,
    model,
    prompt: str,
    target_token: str,
) -> Dict:
    """
    Decomposes the final logit of `target_token` into per-layer attention and
    MLP contributions by projecting each block's residual-stream output onto
    the unembedding row of the target token.

    Uses the correct pre-norm residual bookkeeping:
      attn_contrib = attn(input_layernorm(h_in))
      h_mid        = h_in + attn_contrib
      mlp_contrib  = mlp(post_attention_layernorm(h_mid))
    """
    inputs    = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states   # (n_layers+1,) of (1, seq_len, hidden)
    W_U           = model.lm_head.weight    # (vocab_size, hidden)
    target_dir    = W_U[target_id]          # (hidden,) — unembedding direction

    layers    = model.model.layers
    dla_attn  = []
    dla_mlp   = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            h_in = hidden_states[layer_idx]   # (1, seq_len, hidden)

            # ── Attention contribution ────────────────────────────
            # Pre-norm: apply input_layernorm before self_attn
            normed_attn = layer.input_layernorm(h_in)
            attn_out    = layer.self_attn(
                hidden_states=normed_attn,
                attention_mask=None,
            )[0]                               # (1, seq_len, hidden)
            attn_contrib = attn_out[0, -1, :] # (hidden,)
            dla_attn.append(torch.dot(attn_contrib, target_dir).item())

            # ── MLP contribution ──────────────────────────────────
            # MLP input is the post-attention residual stream
            h_mid        = h_in + attn_out    # (1, seq_len, hidden)
            normed_mlp   = layer.post_attention_layernorm(h_mid)
            mlp_out      = layer.mlp(normed_mlp)  # (1, seq_len, hidden)
            mlp_contrib  = mlp_out[0, -1, :]  # (hidden,)
            dla_mlp.append(torch.dot(mlp_contrib, target_dir).item())

    return {
        "target_token":  target_token,
        "dla_attention": dla_attn,
        "dla_mlp":       dla_mlp,
        "n_layers":      len(layers),
    }


# ─── Hallucination detector ───────────────────────────────────────────────────

def detect_context_dropout(logit_lens_result: Dict, threshold: float = 0.05) -> Dict:
    """
    Flags layers where P(target_token) drops sharply — a sign that the model
    abandons retrieved context and falls back to parametric memory.
    """
    probs  = logit_lens_result["probs_per_layer"]
    diffs  = [probs[i] - probs[i - 1] for i in range(1, len(probs))]
    dropout_layers = [i + 1 for i, d in enumerate(diffs) if d < -threshold]
    peak_layer     = int(np.argmax(probs))

    return {
        "peak_grounding_layer": peak_layer,
        "dropout_layers":       dropout_layers,
        "likely_hallucination": len(dropout_layers) > 2,
    }


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_logit_lens(result: Dict, save_path: str = "logit_lens.png") -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(result["probs_per_layer"], marker="o", linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel(f"P('{result['target_token']}')")
    plt.title("Logit Lens — Answer Token Probability Across Layers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Logit-lens plot saved → %s", save_path)


def plot_dla(result: Dict, save_path: str = "dla.png") -> None:
    n = result["n_layers"]
    x = np.arange(n)
    w = 0.4

    plt.figure(figsize=(12, 4))
    plt.bar(x - w / 2, result["dla_attention"], w, label="Attention", color="steelblue")
    plt.bar(x + w / 2, result["dla_mlp"],       w, label="MLP",       color="salmon")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Layer")
    plt.ylabel("Direct Logit Attribution")
    plt.title(f"DLA — Contribution to '{result['target_token']}'")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("DLA plot saved → %s", save_path)


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tokenizer, model = load_model()

    prompt = (
        "Context:\n[1] The Eiffel Tower is located in Paris, France.\n\n"
        "Question: Where is the Eiffel Tower?\n\nAnswer:"
    )
    target = " Paris"

    print("\n── Logit Lens ──────────────────────────────────────────")
    ll = logit_lens(tokenizer, model, prompt, target)
    plot_logit_lens(ll)
    diag = detect_context_dropout(ll)
    print(f"  Peak grounding layer : {diag['peak_grounding_layer']}")
    print(f"  Dropout layers       : {diag['dropout_layers']}")
    print(f"  Likely hallucination : {diag['likely_hallucination']}")

    print("\n── Direct Logit Attribution ────────────────────────────")
    dla = direct_logit_attribution(tokenizer, model, prompt, target)
    plot_dla(dla)
    print(f"  Top attention layer  : Layer {int(np.argmax(dla['dla_attention']))}")
    print(f"  Top MLP layer        : Layer {int(np.argmax(dla['dla_mlp']))}")
