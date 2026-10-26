"""
Exam Proctoring Telemetry & Infraction Analytics Engine.
Renders interactive Plotly charts for candidate suspicion timelines,
violation distributions, and academic integrity gauges.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go


def suspicion_timeline_chart(timeline_data: List[float]) -> go.Figure:
    """
    Renders rolling suspicion index trend over the test window.
    """
    fig = go.Figure()

    # Suspicion index curve
    fig.add_trace(
        go.Scatter(
            y=timeline_data,
            mode="lines",
            name="Suspicion Index (%)",
            line=dict(color="#00F5D4", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 245, 212, 0.12)",
        )
    )

    # Threshold horizontal alert lines
    fig.add_hline(
        y=40,
        line_dash="dot",
        line_color="#FEE440",
        annotation_text="Warning Threshold (40%)",
        annotation_position="top right",
        annotation_font_color="#FEE440"
    )
    fig.add_hline(
        y=75,
        line_dash="dash",
        line_color="#FF0055",
        annotation_text="Critical Flag Threshold (75%)",
        annotation_position="top right",
        annotation_font_color="#FF0055"
    )

    fig.update_layout(
        title="Candidate Suspicion Index Timeline",
        title_font=dict(size=18, color="#00F5D4"),
        xaxis_title="Monitoring Steps (Frames)",
        yaxis_title="Suspicion Score (0-100)",
        yaxis=dict(range=[0, 105]),
        paper_bgcolor="#0A0F1D",
        plot_bgcolor="#070B14",
        font=dict(color="#9FB3C8"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig


def violation_distribution_pie(infractions: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates donut chart summarizing categorical distribution of flagged infractions.
    """
    if not infractions:
        fig = go.Figure()
        fig.add_annotation(
            text="No Infractions Detected — 100% Clean Exam Record",
            showarrow=False,
            font=dict(size=16, color="#00F5D4"),
        )
        fig.update_layout(
            paper_bgcolor="#0A0F1D",
            plot_bgcolor="#070B14",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    type_counts: Dict[str, int] = {}
    for inf in infractions:
        v_type = inf.get("type", "UNKNOWN")
        type_counts[v_type] = type_counts.get(v_type, 0) + 1

    labels = list(type_counts.keys())
    values = list(type_counts.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=["#00F5D4", "#FEE440", "#FF0055", "#9B5DE5", "#00BBF9"]),
                textinfo="label+percent",
            )
        ]
    )

    fig.update_layout(
        title="Infraction Type Breakdown",
        title_font=dict(size=18, color="#00F5D4"),
        paper_bgcolor="#0A0F1D",
        plot_bgcolor="#070B14",
        font=dict(color="#FFFFFF"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def academic_integrity_gauge(suspicion_score: float) -> go.Figure:
    """
    Renders academic integrity health gauge.
    100 - suspicion_score represents overall candidate integrity.
    """
    integrity_score = max(0.0, min(100.0, 100.0 - suspicion_score))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=integrity_score,
            title={"text": "Candidate Academic Integrity Score", "font": {"color": "#00F5D4", "size": 18}},
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
                "bar": {"color": "#00F5D4" if integrity_score > 60 else "#FF0055"},
                "bgcolor": "#101626",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 0, 85, 0.3)"},
                    {"range": [50, 75], "color": "rgba(254, 228, 64, 0.3)"},
                    {"range": [75, 100], "color": "rgba(0, 245, 212, 0.3)"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#0A0F1D",
        font=dict(color="#9FB3C8"),
        height=260,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig
