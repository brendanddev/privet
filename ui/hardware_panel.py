
"""
ui/hardware_panel.py

Hardware info sidebar panel for Streamlit.

Shows CPU, RAM usage, GPU backend, and hardware tier badge.
Warns when available memory drops below 15% of total.
"""

import streamlit as st
from utils.hardware import HardwareProfile


_TIER_COLORS = {
    "high":     "#22c55e",
    "standard": "#3b82f6",
    "low":      "#f59e0b",
    "minimal":  "#ef4444",
}


def render_hardware_panel(profile: HardwareProfile):
    """
    Render the hardware info panel in st.sidebar.

    Shows CPU, RAM, GPU, and tier badge. Always visible, no expander.
    Emits a low-memory warning when available RAM is under 15%.

    Args:
        profile: HardwareProfile from HardwareProfiler.profile()
    """
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Hardware")

    # CPU
    cpu = profile.cpu_brand[:30] if len(profile.cpu_brand) > 30 else profile.cpu_brand
    st.sidebar.caption(f"**CPU:** {cpu}")

    # RAM
    st.sidebar.caption(
        f"**RAM:** {profile.used_ram_gb} / {profile.total_ram_gb} GB used "
        f"({profile.ram_percent}%)"
    )

    # GPU
    if profile.gpu.backend == "none":
        gpu_line = "CPU only"
    else:
        gpu_line = f"{profile.gpu.backend.upper()} — {profile.gpu.name}"
    st.sidebar.caption(f"**GPU:** {gpu_line}")

    # Tier badge
    color = _TIER_COLORS.get(profile.tier, "#3b82f6")
    st.sidebar.markdown(
        f"**Tier:** <span style='background:{color};color:#fff;"
        f"padding:2px 8px;border-radius:4px;font-size:0.8rem;"
        f"font-weight:600'>{profile.tier_label}</span>",
        unsafe_allow_html=True,
    )

    # Low memory warning
    if profile.available_ram_gb < (profile.total_ram_gb * 0.15):
        st.sidebar.warning("⚠️ Low memory — consider closing other apps")

