"""Streamlit evaluation dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from evaluation.runner import EvaluationRunner, MODES
except ImportError:
    print("Install streamlit and plotly: pip install streamlit plotly")
    raise


def main() -> None:
    st.set_page_config(page_title="ArchonAI Evaluation", layout="wide")
    st.title("ArchonAI Evaluation Dashboard")

    if "results" not in st.session_state:
        with st.spinner("Running evaluation..."):
            runner = EvaluationRunner()
            st.session_state.results = runner.run()
            st.session_state.runner = runner

    results = st.session_state.results
    by_mode = {}
    for r in results:
        by_mode.setdefault(r["mode"], []).append(r)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Evaluations", len(results))
    with col2:
        st.metric("Modes", len(MODES))
    with col3:
        avg_bleu = sum(r["bleu"] for r in results) / len(results)
        st.metric("Avg BLEU", f"{avg_bleu:.3f}")
    with col4:
        avg_lat = sum(r["latency_ms"] for r in results) / len(results)
        st.metric("Avg Latency", f"{avg_lat:.0f}ms")

    st.subheader("Metrics by Mode")
    modes = list(by_mode.keys())
    fig = go.Figure()
    for metric, color in [("bleu", "#6366f1"), ("codebleu", "#22d3ee"), ("hallucination_rate", "#f43f5e")]:
        fig.add_trace(go.Bar(
            name=metric,
            x=modes,
            y=[sum(r[metric] for r in by_mode[m]) / len(by_mode[m]) for m in modes],
        ))
    fig.update_layout(barmode="group", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latency Distribution")
    fig2 = px.box(
        [{**r, "mode_label": r["mode"]} for r in results],
        x="mode_label", y="latency_ms",
        color="mode_label",
        template="plotly_dark",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if st.button("Export Reports"):
        runner = st.session_state.runner
        md = runner.generate_markdown_report()
        csv = runner.generate_csv_report()
        st.success(f"Saved: {md}, {csv}")


if __name__ == "__main__":
    main()

# ArchonAI platform module - architecture reference
