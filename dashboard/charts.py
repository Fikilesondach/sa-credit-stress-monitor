"""
charts.py
---------
Plotly chart components for the SA Credit Stress Monitor dashboard.

Each function takes clean Python data (dicts/lists) and returns a
go.Figure ready to pass to st.plotly_chart().

Colour palette:
  SA green  #007A4D   stable regime
  SA gold   #FFB612   watch regime
  Orange    #E8620A   elevated stress
  SA red    #DE3831   acute stress / actual stress
  Navy      #002395   lines, titles, axes
  Light bg  #F4F6FA   card backgrounds
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ── Palette ───────────────────────────────────────────────────────────────────
GREEN  = "#007A4D"
GOLD   = "#FFB612"
ORANGE = "#E8620A"
RED    = "#DE3831"
NAVY   = "#002395"
GREY   = "#6B7280"
LIGHT  = "#F4F6FA"
WHITE  = "#FFFFFF"

REGIME_COLOURS = {
    "Stable":           GREEN,
    "Watch":            GOLD,
    "Elevated Stress":  ORANGE,
    "Acute Stress":     RED,
}


def regime_gauge(prob: float, regime_label: str, regime_colour: str) -> go.Figure:
    """
    Circular gauge showing current stress probability.
    The needle colour and background match the regime.
    """
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = round(prob * 100, 1),
        number = {
            "suffix": "%",
            "font": {"size": 36, "color": regime_colour},
            "valueformat": ".1f",
        },
        gauge  = {
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": GREY,
                "ticksuffix": "%",
                "tickfont": {"size": 10},
                "nticks": 6,
            },
            "bar":        {"color": regime_colour, "thickness": 0.28},
            "bgcolor":    WHITE,
            "borderwidth": 2,
            "bordercolor": GREY,
            "steps": [
                {"range": [0,  30], "color": "#D4EDDA"},
                {"range": [30, 50], "color": "#FFF3CD"},
                {"range": [50, 70], "color": "#FFE0CC"},
                {"range": [70, 100], "color": "#F8D7DA"},
            ],
            "threshold": {
                "line": {"color": NAVY, "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        domain = {"x": [0.05, 0.95], "y": [0.15, 0.95]},
        title  = {
            "text": f"<b>{regime_label}</b>",
            "font": {"size": 15, "color": regime_colour},
            "align": "center",
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor=WHITE,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def historical_chart(data: list) -> go.Figure:
    """
    Dual-layer time series:
      - Shaded bands for actual stress periods (red)
      - Line + markers for model stress probability
      - Annotated historical events
    """
    df = pd.DataFrame(data)

    fig = go.Figure()

    # Shade actual stress years
    for _, row in df[df["actual_stress"] == 1].iterrows():
        fig.add_vrect(
            x0=row["year"] - 0.45, x1=row["year"] + 0.45,
            fillcolor=RED, opacity=0.12,
            layer="below", line_width=0,
        )

    # Stable zone band
    fig.add_hrect(y0=0, y1=0.30, fillcolor=GREEN, opacity=0.04, layer="below", line_width=0)
    fig.add_hrect(y0=0.30, y1=0.50, fillcolor=GOLD, opacity=0.04, layer="below", line_width=0)
    fig.add_hrect(y0=0.50, y1=0.70, fillcolor=ORANGE, opacity=0.04, layer="below", line_width=0)
    fig.add_hrect(y0=0.70, y1=1.0, fillcolor=RED, opacity=0.04, layer="below", line_width=0)

    # Threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY,
                  line_width=1.5, opacity=0.7,
                  annotation_text="Threshold (50%)", annotation_position="top right",
                  annotation_font_color=GREY)

    # Probability line
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["stress_prob"],
        mode="lines+markers",
        name="Model stress probability",
        line=dict(color=NAVY, width=2.5),
        marker=dict(
            size=9,
            color=[RED if p >= 0.5 else GREEN for p in df["stress_prob"]],
            line=dict(color=WHITE, width=1.5),
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Stress probability: %{y:.1%}<br>"
            "<extra></extra>"
        ),
    ))

    # Annotate key events
    events = {
        2002: "Rand crisis",
        2009: "GFC",
        2016: "SOE crisis",
        2020: "COVID-19",
        2023: "Load shedding",
    }
    for yr, label in events.items():
        row = df[df["year"] == yr]
        if row.empty:
            continue
        prob = row["stress_prob"].values[0]
        fig.add_annotation(
            x=yr, y=prob,
            text=f"<b>{label}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor=RED,
            arrowwidth=1.5,
            ay=-45 if prob > 0.5 else 45,
            ax=0,
            font=dict(size=10, color=RED),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=RED,
            borderwidth=1,
        )

    fig.update_layout(
        title=dict(
            text="<b>SA Credit Stress Probability — 2000 to 2023</b><br>"
                 "<sup>Red bands = actual stress years  ·  Threshold = 50%</sup>",
            font=dict(size=16, color=NAVY),
            x=0,
        ),
        xaxis=dict(title="Year", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Stress Probability",
            tickformat=".0%",
            range=[-0.05, 1.05],
            showgrid=True,
            gridcolor="#E5E7EB",
        ),
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=90, b=50, l=60, r=30),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Inter, sans-serif", color=GREY),
    )
    return fig


def feature_importance_chart(features: list, top_n: int = 12) -> go.Figure:
    """
    Horizontal bar chart of global SHAP feature importance (top N).
    Bars coloured by magnitude tier.
    """
    df = pd.DataFrame(features).head(top_n).sort_values("importance")

    # Clean feature names for display
    def fmt(name: str) -> str:
        return (name
                .replace("_pct_gdp", " (% GDP)")
                .replace("_yoy_chg", " YoY Δ")
                .replace("_roll2m", " (2yr avg)")
                .replace("_roll3m", " (3yr avg)")
                .replace("_lag1", " [t-1]")
                .replace("_lag2", " [t-2]")
                .replace("_woe", " [WoE]")
                .replace("_", " ")
                .title())

    labels = [fmt(f) for f in df["feature"]]
    max_v  = df["importance"].max()
    colours = [
        RED if v >= 0.6 * max_v else ORANGE if v >= 0.3 * max_v else NAVY
        for v in df["importance"]
    ]

    fig = go.Figure(go.Bar(
        x=df["importance"], y=labels,
        orientation="h",
        marker=dict(color=colours, line=dict(color=WHITE, width=0.5)),
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
        textfont=dict(size=10, color=GREY),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.6f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Global Feature Importance</b><br><sup>Mean |SHAP value| across 2000–2023</sup>",
            font=dict(size=15, color=NAVY), x=0,
        ),
        xaxis=dict(title="Mean |SHAP Value|", showgrid=True, gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(showgrid=False),
        height=420,
        margin=dict(t=80, b=40, l=200, r=80),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Inter, sans-serif", color=GREY),
    )
    return fig


def shap_waterfall(shap_values: dict, top_n: int = 12) -> go.Figure:
    """
    Waterfall chart showing per-feature SHAP contribution for a single prediction.
    Red = pushes toward stress, Green = pushes toward stable.
    """
    # Sort by absolute SHAP, take top N
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sorted_items = sorted(sorted_items, key=lambda x: x[1])  # ascending for waterfall

    def fmt(name: str) -> str:
        return (name.replace("_pct_gdp", " (% GDP)")
                    .replace("_yoy_chg", " YoY Δ")
                    .replace("_roll2m", " (2yr avg)")
                    .replace("_lag1", " [t-1]")
                    .replace("_woe", " [WoE]")
                    .replace("_", " ").title())

    labels = [fmt(k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colours = [RED if v > 0 else GREEN for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colours, line=dict(color=WHITE, width=0.5)),
        text=[f"{'+' if v > 0 else ''}{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.6f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=NAVY, line_width=1.5)

    fig.update_layout(
        title=dict(
            text="<b>Prediction Explained — SHAP Waterfall</b><br>"
                 "<sup>🔴 Pushes toward stress  ·  🟢 Pushes toward stable</sup>",
            font=dict(size=15, color=NAVY), x=0,
        ),
        xaxis=dict(title="SHAP Value (impact on stress probability)", showgrid=True,
                   gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(showgrid=False),
        height=420,
        margin=dict(t=80, b=40, l=220, r=80),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Inter, sans-serif", color=GREY),
    )
    return fig


def confusion_strip(data: list) -> go.Figure:
    """
    Mini strip chart: each year as a dot, coloured by correct/incorrect prediction.
    Gives an immediate visual model accuracy read at a glance.
    """
    df = pd.DataFrame(data)
    df["predicted"] = (df["stress_prob"] >= 0.5).astype(int)
    df["correct"]   = (df["predicted"] == df["actual_stress"])
    df["outcome"]   = df.apply(lambda r: (
        "True Positive"  if r["predicted"] == 1 and r["actual_stress"] == 1 else
        "True Negative"  if r["predicted"] == 0 and r["actual_stress"] == 0 else
        "False Positive" if r["predicted"] == 1 and r["actual_stress"] == 0 else
        "False Negative"
    ), axis=1)

    colour_map = {
        "True Positive":  RED,
        "True Negative":  GREEN,
        "False Positive": ORANGE,
        "False Negative": GOLD,
    }

    fig = go.Figure()
    for outcome, colour in colour_map.items():
        sub = df[df["outcome"] == outcome]
        fig.add_trace(go.Scatter(
            x=sub["year"],
            y=[0.5] * len(sub),
            mode="markers+text",
            name=f"{outcome} ({len(sub)})",
            marker=dict(color=colour, size=20, symbol="circle",
                        line=dict(color=WHITE, width=2)),
            text=sub["year"].astype(str).str[-2:],
            textposition="middle center",
            textfont=dict(size=9, color=WHITE),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"Outcome: {outcome}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="<b>Model Accuracy by Year</b><br>"
                 "<sup>🔴 True Positive  🟢 True Negative  🟠 False Positive  🟡 False Negative</sup>",
            font=dict(size=15, color=NAVY), x=0,
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(visible=False),
        height=180,
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5,
                    font=dict(size=11)),
        margin=dict(t=80, b=60, l=30, r=30),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Inter, sans-serif"),
    )
    return fig
