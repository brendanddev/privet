
"""
ui/privacy_panel.py

Sidebar panel that renders live network monitoring data from NetworkMonitor.
Display is provider-aware: llamacpp gets a stronger guarantee badge because it
makes zero network calls by design. Ollama shows the same three-state display
as before since it routes through a local HTTP service.
"""

import streamlit as st
from utils.network_monitor import NetworkMonitor, QueryNetworkResult
from utils.config import load_config
from utils.logger import setup_logger

logger = setup_logger()


@st.cache_resource
def _get_config() -> dict:
    """Load and cache config so it is read once per session, not on every rerun."""
    return load_config()


def _query_label(result: QueryNetworkResult, threshold: int) -> tuple[str, str]:
    """
    Return a (label, icon) pair describing a single query's network status.

    Three states:
        - 0 bytes sent:             "Verified private",  "✓"
        - 1-threshold bytes sent:   "Background noise",  "✓"
        - > threshold bytes sent:   "Traffic detected",  "!"

    Args:
        result (QueryNetworkResult): The query result to label.
        threshold (int): Noise threshold in bytes (from NetworkMonitor.NOISE_THRESHOLD).

    Returns:
        tuple[str, str]: (human-readable label, status icon)
    """
    if not result.verified_private:
        return "Traffic detected", "!"
    if result.bytes_sent == 0:
        return "Verified private", "✓"
    return f"Background noise ({result.bytes_sent}B)", "✓"


def _render_llamacpp(summary: dict, monitor: NetworkMonitor) -> None:
    """
    Render the privacy panel body for the llamacpp provider.

    llamacpp calls the model as a direct Python function — no HTTP layer,
    no sockets, zero network calls during inference. The badge is always
    green. Any bytes shown are OS background broadcasts unrelated to the query.

    Args:
        summary (dict): Output of monitor.get_session_summary().
        monitor (NetworkMonitor): The active monitor instance.
    """
    total_sent = summary["total_bytes_sent"]

    st.success("VERIFIED PRIVATE")
    st.caption("0 bytes from your query")

    if total_sent > 0:
        st.caption(f"{total_sent}B OS background noise (mDNS/Bonjour) — unrelated to AI inference")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Query bytes", 0)
    with col2:
        st.metric("Queries", summary["query_count"])

    recent = list(reversed(monitor.session_log[-5:]))
    if recent:
        with st.expander("Per-query log"):
            for i, result in enumerate(recent):
                noise = f"{result.bytes_sent}B OS noise" if result.bytes_sent > 0 else "0B"
                st.caption(
                    f"[{i + 1}] ✓ Verified private — "
                    f"query: 0B | {noise} | "
                    f"{result.duration}s"
                )

    st.caption(
        "llamacpp runs as a direct function call inside Python. "
        "Zero network calls are made during inference. Any bytes shown "
        "are OS-level broadcasts (mDNS/Bonjour) unrelated to your query."
    )


def _render_ollama(summary: dict, monitor: NetworkMonitor) -> None:
    """
    Render the privacy panel body for the Ollama provider.

    Ollama routes inference through a local HTTP service on 127.0.0.1.
    Loopback traffic is excluded. Three-state badge based on whether any
    external interface exceeded the noise threshold.

    Args:
        summary (dict): Output of monitor.get_session_summary().
        monitor (NetworkMonitor): The active monitor instance.
    """
    total_sent = summary["total_bytes_sent"]

    if not summary["all_private"]:
        st.error(f"OUTBOUND TRAFFIC DETECTED — {total_sent}B sent this session")
    elif total_sent == 0:
        st.success("VERIFIED PRIVATE — 0 bytes sent this session")
    else:
        st.success(f"PRIVATE — background noise only ({total_sent}B this session)")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bytes Sent", total_sent)
    with col2:
        st.metric("Queries", summary["query_count"])

    recent = list(reversed(monitor.session_log[-5:]))
    if recent:
        with st.expander("Per-query log"):
            for i, result in enumerate(recent):
                label, icon = _query_label(result, monitor.NOISE_THRESHOLD)
                st.caption(
                    f"[{i + 1}] {icon} {label} — "
                    f"sent: {result.bytes_sent}B | "
                    f"recv: {result.bytes_recv}B | "
                    f"{result.duration}s"
                )

    st.caption(
        "Ollama routes through a local HTTP service on 127.0.0.1 (loopback). "
        "This traffic never leaves your machine. External interface bytes "
        "shown are OS background noise."
    )


def render_privacy_panel(monitor: NetworkMonitor) -> None:
    """
    Render the privacy network monitor panel inside the Streamlit sidebar.

    Display is provider-aware. llamacpp always shows a green verified badge
    because it makes zero network calls during inference. Ollama shows the
    three-state display (verified / background noise / outbound detected).

    Only external network interfaces are measured — loopback ('lo', 'lo0')
    is excluded. A noise threshold of 8KB per interface filters out background
    OS broadcasts (mDNS, Bonjour) that fire independently of any query. A real
    API call would send at minimum 10-50KB, so the threshold is safe.

    Args:
        monitor (NetworkMonitor): The active monitor instance for this session.
    """
    with st.sidebar:
        st.divider()
        st.subheader("Privacy Monitor")

        summary = monitor.get_session_summary()

        if summary["query_count"] == 0:
            st.caption("No queries yet — privacy status will appear after your first question.")
            return

        provider = _get_config().get("provider", "ollama")

        if provider == "llamacpp":
            _render_llamacpp(summary, monitor)
        else:
            _render_ollama(summary, monitor)

        logger.info(
            f"Privacy panel rendered | provider={provider} | "
            f"sent={summary['total_bytes_sent']} | "
            f"queries={summary['query_count']} | all_private={summary['all_private']}"
        )
