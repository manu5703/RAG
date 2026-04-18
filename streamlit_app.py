"""
Run: streamlit run streamlit_app.py
"""

import io
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from app.rag_pipeline import RAGPipeline
from app.document_loader import load_pdf_chunks
from app.config import settings
from interpretability.faithfulness import (
    logit_lens,
    direct_logit_attribution,
    detect_context_dropout,
)

# Page config 

st.set_page_config(
    page_title="RAG + Faithfulness",
    page_icon="🔍",
    layout="wide",
)

# Load models once — index is rebuilt separately when PDF changes 

SAMPLE_CHUNKS = [
    "The Eiffel Tower is located in Paris, France.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is widely used for data science and ML development.",
    "RAG combines retrieval systems with generative language models.",
    "Docker containers package code and dependencies together.",
]


@st.cache_resource(show_spinner="Loading models — this takes a minute on first run…")
def load_pipeline():
    return RAGPipeline(chunks=SAMPLE_CHUNKS)


pipeline = load_pipeline()

# Sidebar — PDF upload 
with st.sidebar:
    st.title("Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf",
        help="Drag and drop or click to browse. The index rebuilds automatically.",
    )

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        # Only re-parse and re-index if this is a new file
        if st.session_state.get("loaded_file_key") != file_key:
            with st.spinner(f"Parsing {uploaded_file.name}…"):
                chunks, truncated = load_pdf_chunks(io.BytesIO(uploaded_file.read()))

            if not chunks:
                st.error("Could not extract text from this PDF.")
            else:
                with st.spinner(f"Indexing {len(chunks)} chunks…"):
                    pipeline.index.build(chunks)

                st.session_state["loaded_file_key"] = file_key
                st.session_state["chunk_count"]     = len(chunks)
                st.session_state["doc_name"]        = uploaded_file.name
                st.session_state["truncated"]       = truncated
                st.success(f"Indexed **{len(chunks)} chunks** from {uploaded_file.name}")
                if truncated:
                    st.warning(
                        "Document was truncated at 300,000 characters to fit in memory. "
                        "Questions about content beyond that point won't be answered. "
                        "To process more, raise `MAX_CHARS` in `app/document_loader.py`."
                    )

        else:
            st.success(
                f"**{st.session_state['doc_name']}** "
                f"({st.session_state['chunk_count']} chunks)"
            )
            if st.session_state.get("truncated"):
                st.warning("Document was truncated at 300,000 characters.")

    else:
        # No PDF uploaded — use the sample corpus
        if st.session_state.get("loaded_file_key"):
            # User removed the file — reset to sample chunks
            with st.spinner("Resetting to sample corpus…"):
                pipeline.index.build(SAMPLE_CHUNKS)
            st.session_state.pop("loaded_file_key", None)
            st.session_state.pop("chunk_count", None)
            st.session_state.pop("doc_name", None)

        st.info("No PDF uploaded — using the built-in sample corpus.")

    st.divider()
    st.caption(f"**LLM:** `{settings.llm_model}`")
    st.caption(f"**Embedder:** `{settings.embed_model}`")
    st.caption(f"**Reranker:** `{settings.rerank_model}`")
    st.divider()
    st.caption(f"Top-K retrieve: **{settings.top_k_retrieve}**")
    st.caption(f"Top-K rerank: **{settings.top_k_rerank}**")
    st.caption(f"Dense weight α: **{settings.alpha}**")

# Tabs 

tab_query, tab_interp = st.tabs(["💬 Query", "🔬 Interpretability"])


# TAB 1 — Query

with tab_query:
    doc_label = (
        f"**{st.session_state['doc_name']}**"
        if st.session_state.get("doc_name")
        else "the sample corpus"
    )
    st.header("Ask a Question")
    st.caption(f"Answering from {doc_label}")

    question = st.text_input(
        "Question",
        placeholder="Type your question here…",
        label_visibility="collapsed",
    )

    if st.button("Submit", type="primary", disabled=not question.strip()):
        llm_loaded = getattr(pipeline.llm, "_model", None) is not None
        spinner_msg = "Retrieving and generating…" if llm_loaded else "Loading Qwen on first query — takes ~30 s…"
        with st.spinner(spinner_msg):
            result = pipeline.query(question)

        st.success(f"**Answer:** {result['answer']}")
        st.caption(f"Latency: {result['latency_ms']} ms")

        # Build the interpretability prompt from this result and save to session
        context_block = "\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(result["context_used"])
        )
        auto_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        first_word = result["answer"].split()[0] if result["answer"].strip() else ""
        auto_target = f" {first_word}" if first_word else ""

        st.session_state["auto_prompt"]  = auto_prompt
        st.session_state["auto_target"]  = auto_target

        st.subheader("Context used")
        for i, chunk in enumerate(result["context_used"], 1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)

        st.info("Switch to the **Interpretability** tab to analyse this answer.")


# TAB 2 — Interpretability

with tab_interp:
    st.header("Faithfulness Interpretability")

    # ── Auto-populate from last query if available ────────────────
    has_auto = bool(st.session_state.get("auto_prompt"))
    if has_auto:
        st.success(
            "Auto-filled from your last query. "
            "Edit the fields below or click **Run Analysis** to proceed."
        )

    default_prompt = st.session_state.get(
        "auto_prompt",
        "Context:\n[1] The Eiffel Tower is located in Paris, France.\n\n"
        "Question: Where is the Eiffel Tower?\n\nAnswer:",
    )
    default_target = st.session_state.get("auto_target", " Paris")

    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area(
            "Prompt (auto-filled from last query)",
            height=200,
            value=default_prompt,
        )
    with col2:
        target_token = st.text_input(
            "Target token (first word of answer)",
            value=default_target,
            help="Include the leading space — tokenizers treat ' Paris' and 'Paris' differently.",
        )
        run_btn = st.button(
            "Run Analysis",
            type="primary",
            disabled=not (prompt.strip() and target_token.strip()),
        )

    if run_btn:
        tokenizer = pipeline.llm.tokenizer
        model     = pipeline.llm.model

        with st.spinner("Running logit lens…"):
            ll_result  = logit_lens(tokenizer, model, prompt, target_token)

        with st.spinner("Running DLA…"):
            dla_result = direct_logit_attribution(tokenizer, model, prompt, target_token)

        diagnosis = detect_context_dropout(ll_result)

        # Diagnosis banner 
        st.divider()
        d1, d2, d3 = st.columns(3)
        d1.metric("Peak grounding layer", diagnosis["peak_grounding_layer"])
        d2.metric("Dropout layers",       len(diagnosis["dropout_layers"]))
        flag = diagnosis["likely_hallucination"]
        d3.metric("Likely hallucination", "Yes ⚠️" if flag else "No ✓")

        if flag:
            st.warning(
                f"Sharp probability drops at layers {diagnosis['dropout_layers']}. "
                "The model may be abandoning retrieved context and falling back "
                "to parametric memory."
            )

        st.divider()

        # Logit Lens 
        st.subheader("Logit Lens — P(target token) per layer")
        st.caption(
            "A peak early and sustained → context-grounded. "
            "A late peak or frequent drops → parametric recall."
        )

        probs  = ll_result["probs_per_layer"]
        layers = list(range(len(probs)))
        peak   = diagnosis["peak_grounding_layer"]

        fig_ll = go.Figure()
        fig_ll.add_trace(go.Scatter(
            x=layers, y=probs,
            mode="lines+markers",
            line=dict(color="#4C9BE8", width=2),
            marker=dict(size=6),
            name=f"P('{target_token}')",
        ))
        fig_ll.add_trace(go.Scatter(
            x=[peak], y=[probs[peak]],
            mode="markers",
            marker=dict(size=14, color="gold", symbol="star",
                        line=dict(color="black", width=1)),
            name=f"Peak (layer {peak})",
        ))
        for dl in diagnosis["dropout_layers"]:
            fig_ll.add_vrect(
                x0=dl - 0.5, x1=dl + 0.5,
                fillcolor="red", opacity=0.15, line_width=0,
                annotation_text="drop", annotation_position="top left",
            )
        fig_ll.update_layout(
            xaxis_title="Layer",
            yaxis_title=f"P('{target_token}')",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_ll, use_container_width=True)

        # DLA 
        st.subheader("Direct Logit Attribution — per-layer contribution")
        st.caption(
            "Positive bars → component writes the answer. "
            "Negative bars → suppresses it. "
            "Attention copies from context; early MLPs inject parametric memory."
        )

        n      = dla_result["n_layers"]
        x      = list(range(n))
        d_attn = dla_result["dla_attention"]
        d_mlp  = dla_result["dla_mlp"]

        fig_dla = go.Figure()
        fig_dla.add_trace(go.Bar(
            x=x, y=d_attn, name="Attention",
            marker_color="#4C9BE8", opacity=0.85,
        ))
        fig_dla.add_trace(go.Bar(
            x=x, y=d_mlp, name="MLP",
            marker_color="#E8734C", opacity=0.85,
        ))
        fig_dla.add_hline(y=0, line_color="black", line_width=0.8)

        top_attn = int(np.argmax(d_attn))
        top_mlp  = int(np.argmax(d_mlp))
        fig_dla.add_annotation(
            x=top_attn, y=d_attn[top_attn],
            text=f"Top attn\nL{top_attn}", showarrow=True,
            arrowhead=2, ax=0, ay=-30, font=dict(color="#4C9BE8"),
        )
        fig_dla.add_annotation(
            x=top_mlp, y=d_mlp[top_mlp],
            text=f"Top MLP\nL{top_mlp}", showarrow=True,
            arrowhead=2, ax=0, ay=-30, font=dict(color="#E8734C"),
        )
        fig_dla.update_layout(
            barmode="group",
            xaxis_title="Layer",
            yaxis_title="Direct Logit Attribution",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_dla, use_container_width=True)

        # Raw numbers
        with st.expander("Raw numbers"):
            df = pd.DataFrame({
                "Layer":         x,
                "P(target)":     [round(p, 5) for p in probs[1:]],
                "DLA Attention": [round(v, 4) for v in d_attn],
                "DLA MLP":       [round(v, 4) for v in d_mlp],
            })
            st.dataframe(df, use_container_width=True)
