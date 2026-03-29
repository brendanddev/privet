
import os
import streamlit as st
import plotly.graph_objects as go
from core.rag_debugger import RAGDebugger
from utils.config import load_config
from utils.logger import setup_logger

logger = setup_logger()


@st.cache_resource
def load_debugger():
    """Load and cache the RAGDebugger instance so it connects to ChromaDB once.

    Reads chroma_path and collection_name from config so custom values
    are respected instead of using RAGDebugger's hardcoded defaults.
    """
    config = load_config()
    return RAGDebugger(
        chroma_path=config.get("chroma_path", "./chroma_db"),
        collection_name=config.get("collection_name", "documents"),
    )


def render_chunk_distribution(debugger: RAGDebugger):
    """
    Render a bar chart of chunk size distribution using RAGDebugger data.

    Tiny chunks under 200 chars are usually noise like headers or page numbers.
    Ideal chunks are 200-1000 chars.
    Oversized chunks over 1000 chars reduce retrieval precision because the
    model gets too much unfocused context at once.

    Args:
        debugger (RAGDebugger): Active debugger instance connected to ChromaDB
    """
    st.subheader("Chunk Size Distribution")
    st.caption("Shows how your documents are split into chunks. Ideal chunks are 200-1000 chars.")

    results = debugger.collection.get(include=["documents"])
    lengths = [len(doc) for doc in results["documents"]]

    buckets = {"0-200": 0, "200-500": 0, "500-1000": 0, "1000+": 0}
    for l in lengths:
        if l < 200:
            buckets["0-200"] += 1
        elif l < 500:
            buckets["200-500"] += 1
        elif l < 1000:
            buckets["500-1000"] += 1
        else:
            buckets["1000+"] += 1

    # Red for problematic, green for ideal, yellow for too large
    colors = {
        "0-200": "#e74c3c",
        "200-500": "#2ecc71",
        "500-1000": "#2ecc71",
        "1000+": "#f39c12"
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(buckets.keys()),
            y=list(buckets.values()),
            marker_color=[colors[b] for b in buckets.keys()],
            text=list(buckets.values()),
            textposition="auto"
        )
    ])

    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    )

    st.plotly_chart(fig, width='stretch')


def render_index_health(debugger: RAGDebugger):
    """
    Analyze the collection using RAGDebugger and flag quality issues.

    Checks three things:
    - Tiny chunks: under 200 chars are usually noise that hurts retrieval quality
    - Oversized chunks: over 1000 chars give the model too much unfocused context
    - Average chunk size: compared against chunk_size * 4 (approx chars per token)

    Args:
        debugger (RAGDebugger): Active debugger instance connected to ChromaDB
    """
    st.subheader("Index Health")
    st.caption("Flags potential issues with how your documents are chunked and indexed.")

    results = debugger.collection.get(include=["documents"])
    lengths = [len(doc) for doc in results["documents"]]
    total = len(lengths)

    if total == 0:
        st.warning("No chunks found in the index.")
        return

    config = load_config()
    chunk_size = config.get("chunk_size", 256)
    avg_char_threshold = chunk_size * 4

    tiny = sum(1 for l in lengths if l < 200)
    oversized = sum(1 for l in lengths if l > avg_char_threshold)
    avg = int(sum(lengths) / total)

    tiny_pct = (tiny / total) * 100
    if tiny_pct > 20:
        st.error(f"{tiny} tiny chunks ({tiny_pct:.0f}%) — likely noise or headers. Consider cleaning your source documents.")
    elif tiny_pct > 10:
        st.warning(f"{tiny} small chunks ({tiny_pct:.0f}%) — some noise detected.")
    else:
        st.success(f"Tiny chunks: {tiny} ({tiny_pct:.0f}%) — looks clean.")

    oversized_pct = (oversized / total) * 100
    if oversized_pct > 50:
        st.warning(f"{oversized} large chunks ({oversized_pct:.0f}%) — chunks may be too large for precise retrieval.")
    else:
        st.success(f"Oversized chunks: {oversized} ({oversized_pct:.0f}%) — within acceptable range.")

    if avg < 200:
        st.error(f"Avg chunk size: {avg} chars — too small overall.")
    elif avg > avg_char_threshold:
        st.warning(f"Avg chunk size: {avg} chars — consider tuning chunk size down (threshold: {avg_char_threshold} chars for chunk_size={chunk_size}).")
    else:
        st.success(f"Avg chunk size: {avg} chars — within ideal range.")

    # Summary metrics
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", total)
    with col2:
        st.metric("Avg Size", f"{avg} chars")
    with col3:
        st.metric("Tiny Chunks", tiny)


def render_retrieval_confidence(sources):
    """
    Render a visual confidence meter for each retrieved source chunk.

    The confidence score is cosine similarity between 0 and 1:
    - Above 0.7: strong match — chunk is closely related to the query
    - 0.4 to 0.7: moderate match — relevant but not precise
    - Below 0.4: weak match — model may produce inaccurate answers from
      irrelevant context, sometimes called hallucination

    Args:
        sources (list): List of source dicts from the last query result
    """
    st.subheader("Retrieval Confidence")
    st.caption("How closely each retrieved chunk matched your query. Above 0.7 is strong, below 0.4 may lead to inaccurate answers.")

    if not sources:
        st.caption("No query yet — ask a question to see retrieval confidence.")
        return

    for i, source in enumerate(sources):
        score = source.get("score")
        if score is None:
            continue

        if score >= 0.7:
            label = "Strong"
            color = "normal"
        elif score >= 0.4:
            label = "Moderate"
            color = "off"
        else:
            label = "Weak"
            color = "inverse"

        st.caption(f"[{i+1}] {source['file']} — Page {source['page']} — {label} ({score})")
        st.progress(max(0.0, min(float(score), 1.0)))


def render_sidebar(engine, get_collection_stats, audit_log=None):
    """
    Render the full debug sidebar panel.

    Uses RAGDebugger as the single source of truth for collection data.
    Separated from app.py to keep the main UI focused on chat logic.

    Args:
        engine (RAGEngine): The active RAG engine instance
        get_collection_stats (callable): Returns total chunk count and unique filenames
        audit_log: PrivacyAuditLog instance, or None to skip audit logging
    """
    debugger = load_debugger()

    with st.sidebar:
        st.title("Debug Panel")
        st.divider()

        # Engine Stats
        st.subheader("Engine Stats")
        stats = engine.get_stats()
        st.metric("Startup Time", f"{stats['startup_time']}s")
        st.metric("Last Query Time", f"{stats['last_query_time']}s" if stats['last_query_time'] else "No queries yet")
        st.caption(f"**LLM:** {stats['llm_model']}")
        st.caption(f"**Embed:** {stats['embed_model']}")

        st.divider()

        render_model_switcher(engine)

        st.divider()

        # Collection Stats + Document Management
        st.subheader("Collection Stats")
        total_chunks, unique_files = get_collection_stats()
        st.metric("Total Chunks", total_chunks)
        st.metric("Documents Loaded", len(unique_files))

        if unique_files:
            with st.expander("Manage documents"):
                for f in unique_files:
                    col1, col2 = st.columns([3, 1])
                    exists_on_disk = os.path.exists(os.path.join("docs", f))
                    with col1:
                        if exists_on_disk:
                            st.caption(f"{f}")
                        else:
                            st.caption(f"{f} — not in docs folder")
                    with col2:
                        if st.button("Remove", key=f"remove_{f}"):
                            success = engine.remove_document(f)
                            if success:
                                file_path = os.path.join("docs", f)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                if audit_log is not None:
                                    audit_log.log_document_access(
                                        filename=f,
                                        action="REMOVED",
                                        chunk_count=0
                                    )
                                st.success(f"Removed {f}")
                                logger.info(f"Document removed via UI: {f}")
                                st.rerun()
                            else:
                                st.error(f"Could not remove {f}")

        st.divider()

        # Chunk Distribution Chart
        render_chunk_distribution(debugger)

        st.divider()

        # Index Health
        render_index_health(debugger)

        st.divider()

        # Retrieval Confidence
        sources = st.session_state.get("last_sources", [])
        render_retrieval_confidence(sources)

def render_model_switcher(engine):
    """
    Render model selection dropdowns for LLM and embedding model.

    Lists all models currently pulled in Ollama and allows switching both the LLM and embedding model
    without restarting the app. Only available when provider is ollama.

    Switching models reinitializes the engine settings but does not affect the existing ChromaDB index.

    Args:
        engine (RAGEngine): The active RAG engine instance
    """
    from utils.config import load_config

    st.subheader("Model Switcher")
    st.caption("Switch models without restarting. Changes apply immediately.")

    try:
        provider = load_config().get("provider", "ollama")
        if provider != "ollama":
            st.caption("Model switcher only available for Ollama provider.")
            return

        import ollama

        # Get all models pulled in Ollama
        models = ollama.list()
        model_names = [m.model for m in models.models]

        if not model_names:
            st.warning("No models found. Pull a model with: ollama pull <model>")
            return

        # LLM selector
        current_llm = engine.llm_model
        selected_llm = st.selectbox(
            "LLM Model",
            options=model_names,
            index=model_names.index(current_llm) if current_llm in model_names else 0,
            help="Used to generate answers"
        )

        # Embedding model selector
        current_embed = engine.embed_model_name
        selected_embed = st.selectbox(
            "Embedding Model",
            options=model_names,
            index=model_names.index(current_embed) if current_embed in model_names else 0,
            help="Used to convert text to vectors for retrieval. Must match the model used to build the index."
        )

        # Only show apply button if something changed
        if selected_llm != current_llm or selected_embed != current_embed:
            st.warning("Changing the embedding model requires re-indexing your documents.")
            if st.button("Apply"):
                with st.spinner("Switching models..."):
                    engine.switch_models(selected_llm, selected_embed)
                st.success(f"Switched to {selected_llm} / {selected_embed}")
                st.rerun()

    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}")