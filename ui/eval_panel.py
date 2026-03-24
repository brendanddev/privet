
"""
ui/eval_panel.py 

Evaluation scores sidebar panel for Streamlit.

Shows live quality metrics for the last query, trend charts, and
a low-quality query log for debugging retrieval failures.
"""

import streamlit as st
from utils.rag_evaluator import RAGEvaluator, EvalResult


# Colour thresholds for score badges
def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#22c55e"
    elif score >= 0.5:
        return "#f59e0b"
    else:
        return "#ef4444"


def _score_badge(label: str, score: float) -> str:
    color = _score_color(score)
    pct = round(score * 100)
    return (
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:0.8rem;color:#94a3b8'>{label}</span>"
        f"<span style='float:right;font-weight:600;color:{color}'>{pct}%</span>"
        f"</div>"
        f"<div style='background:#1e293b;border-radius:4px;height:6px;margin-bottom:10px'>"
        f"<div style='background:{color};width:{pct}%;height:6px;border-radius:4px'></div>"
        f"</div>"
    )


def render_eval_panel(
    evaluator: RAGEvaluator,
    last_result: EvalResult | None = None
):
    """
    Render the evaluation panel in st.sidebar.

    Args:
        evaluator:   The RAGEvaluator instance
        last_result: The EvalResult from the most recent query, or None
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Answer Quality")

    # --- Last query scores ---
    if last_result is not None:
        composite = last_result.composite_score
        color = _score_color(composite)

        st.sidebar.markdown(
            f"**Overall:** <span style='color:{color};font-weight:700;font-size:1.1rem'>"
            f"{round(composite * 100)}%</span> "
            f"<span style='color:#64748b;font-size:0.8rem'>({last_result.tier} eval)</span>",
            unsafe_allow_html=True
        )

        st.sidebar.markdown(
            _score_badge("Faithfulness", last_result.faithfulness) +
            _score_badge("Answer Relevance", last_result.answer_relevance) +
            _score_badge("Context Precision", last_result.context_precision) +
            _score_badge("Source Coverage", last_result.source_coverage),
            unsafe_allow_html=True
        )

        # Warning flags
        if last_result.faithfulness < 0.5:
            st.sidebar.warning("⚠️ Low faithfulness — answer may contain hallucinations.")
        if last_result.answer_relevance < 0.5:
            st.sidebar.warning("⚠️ Low relevance — answer may not address the question.")
        if last_result.context_precision < 0.4:
            st.sidebar.info("ℹ️ Low context precision — retrieval may be returning noisy chunks.")

        # Faithfulness claim detail
        if last_result.faithfulness_detail and last_result.faithfulness_detail != "[]":
            import json
            try:
                claims = json.loads(last_result.faithfulness_detail)
                if claims:
                    with st.sidebar.expander("🔍 Faithfulness detail", expanded=False):
                        for c in claims:
                            icon = "✓" if c["supported"] else "✗"
                            color = "#22c55e" if c["supported"] else "#ef4444"
                            st.markdown(
                                f"<span style='color:{color}'>{icon}</span> "
                                f"<small>{c['claim'][:80]}{'...' if len(c['claim'])>80 else ''}</small>"
                                f"<br><small style='color:#64748b'>entailment: {c['entailment']}</small>",
                                unsafe_allow_html=True
                            )
            except Exception:
                pass

    else:
        st.sidebar.caption("No queries scored yet.")

    # --- Summary stats ---
    summary = evaluator.get_summary()
    if summary and (summary.get("total_queries") or 0) > 0:
        with st.sidebar.expander("📈 Overall stats", expanded=False):
            total = summary.get("total_queries") or 0
            avg = summary.get("avg_composite", 0) or 0
            low = summary.get("low_quality_count", 0) or 0

            st.markdown(f"**Queries scored:** {total}")
            st.markdown(
                f"**Avg quality:** "
                f"<span style='color:{_score_color(avg)}'>{round(avg*100)}%</span>",
                unsafe_allow_html=True
            )
            if low > 0:
                st.markdown(
                    f"**Low quality (<50%):** "
                    f"<span style='color:#ef4444'>{low}</span>",
                    unsafe_allow_html=True
                )

            # Metric breakdown
            st.markdown("---")
            metrics = [
                ("Faithfulness",   summary.get("avg_faithfulness", 0) or 0),
                ("Relevance",      summary.get("avg_relevance", 0) or 0),
                ("Ctx Precision",  summary.get("avg_precision", 0) or 0),
            ]
            for name, val in metrics:
                st.markdown(
                    _score_badge(name, val),
                    unsafe_allow_html=True
                )

    # --- Trend chart ---
    trend = evaluator.get_trend(days=14)
    if len(trend) >= 2:
        with st.sidebar.expander("📉 Quality trend (14 days)", expanded=False):
            import pandas as pd
            df = pd.DataFrame(trend)
            df["day"] = pd.to_datetime(df["day"])
            df = df.set_index("day")

            chart_data = df[["avg_composite", "avg_faithfulness", "avg_relevance"]].rename(
                columns={
                    "avg_composite": "Composite",
                    "avg_faithfulness": "Faithfulness",
                    "avg_relevance": "Relevance"
                }
            )
            st.line_chart(chart_data, height=150)

    # --- Low quality queries ---
    low_quality = evaluator.get_low_quality(threshold=0.5, n=5)
    if low_quality:
        with st.sidebar.expander(
            f"🔴 Low quality queries ({len(low_quality)})", expanded=False
        ):
            for row in low_quality:
                score = row.get("composite_score", 0)
                q = row.get("question", "")[:60]
                st.markdown(
                    f"<small style='color:#ef4444'>{round(score*100)}%</small> "
                    f"<small>{q}{'...' if len(row.get('question',''))>60 else ''}</small>",
                    unsafe_allow_html=True
                )
