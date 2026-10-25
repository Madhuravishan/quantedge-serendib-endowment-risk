"""
Static Interactive HTML Figure Exporter & Web Asset Synchronizer
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
Team: 'Pandas Predictors' | QuantEdge 1.0 National Challenge

Generates standalone interactive Plotly visualizations exported to docs/assets/charts/
and mirrors official PDF deliverables into docs/report/ for GitHub Pages deployment.
"""

import sys
from pathlib import Path
import shutil
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from src.config import (
    MANDATE_SIZE_USD,
    TARGET_WEIGHTS,
    SERENDIB_CONSERVATIVE_WEIGHTS,
    CLASSIC_60_40_WEIGHTS,
    STRESS_PERIODS,
    HYPOTHETICAL_STAGFLATION_SHOCKS,
)
from src.data_loader import load_market_dataset
from src.portfolio_opt import (
    compute_portfolio_return_series,
    compute_drawdown_series,
    calculate_portfolio_kpis,
    optimize_markowitz_portfolio,
    compute_efficient_frontier,
    simulate_random_portfolios,
)
from src.evt_engine import (
    fit_gpd_pot,
    compute_mean_excess_function,
    compute_evt_risk_metrics,
)
from src.stress_test import evaluate_all_stress_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.export_static_showcase")


def create_cumulative_wealth_figure(
    daily_returns: pd.DataFrame,
    weights: dict = SERENDIB_CONSERVATIVE_WEIGHTS,
    aum: float = MANDATE_SIZE_USD,
) -> go.Figure:
    """Generate interactive 2-panel Plotly figure for Cumulative Wealth and Underwater Drawdown."""
    port_ret = compute_portfolio_return_series(daily_returns, weights)
    b_6040_ret = compute_portfolio_return_series(daily_returns, CLASSIC_60_40_WEIGHTS)

    wealth_dollars = (1.0 + port_ret).cumprod() * aum
    wealth_60_40_dollars = (1.0 + b_6040_ret).cumprod() * aum

    _, dd_series = compute_drawdown_series(port_ret)
    _, dd_60_40 = compute_drawdown_series(b_6040_ret)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            "<b>Cumulative Endowment Wealth ($ USD)</b>",
            "<b>Underwater Drawdown Profile (%)</b>"
        ],
    )

    # Panel 1: Wealth Compounding
    fig.add_trace(
        go.Scatter(
            x=wealth_dollars.index,
            y=wealth_dollars.values,
            name="Serendib Conservative ($10M AUM)",
            line=dict(color="#38bdf8", width=2.5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Portfolio Wealth:</b> $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=wealth_60_40_dollars.index,
            y=wealth_60_40_dollars.values,
            name="Classic 60/40 Benchmark",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>60/40 Wealth:</b> $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Panel 2: Drawdowns
    fig.add_trace(
        go.Scatter(
            x=dd_series.index,
            y=dd_series.values * 100.0,
            name="Serendib Drawdown",
            fill="tozeroy",
            line=dict(color="#f87171", width=1.5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_60_40.index,
            y=dd_60_40.values * 100.0,
            name="60/40 Benchmark Drawdown",
            line=dict(color="#94a3b8", width=1, dash="dot"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>60/40 Drawdown:</b> %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    # Highlight 15% board drawdown threshold
    fig.add_hline(
        y=-15.0,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.8,
        annotation_text="15.0% Board Limit / Circuit Breaker",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#f59e0b"),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=580,
        margin=dict(t=60, b=40, l=60, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
            font=dict(size=11, color="#f8fafc"),
        ),
        paper_bgcolor="rgba(15,23,42,0.95)",
        plot_bgcolor="rgba(30,41,59,0.7)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    )
    fig.update_yaxes(title_text="Portfolio Value ($ USD)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)

    return fig


def create_efficient_frontier_figure(returns_df: pd.DataFrame) -> go.Figure:
    """Generate interactive Markowitz Efficient Frontier with simulated cloud and anchor points."""
    frontier_df = compute_efficient_frontier(returns_df, n_points=60)
    sim_df = simulate_random_portfolios(returns_df, n_portfolios=2000)

    # Compute key benchmark locations
    _, max_s_w = optimize_markowitz_portfolio(returns_df, target="max_sharpe")
    _, min_v_w = optimize_markowitz_portfolio(returns_df, target="min_volatility")
    kpi_max_s = calculate_portfolio_kpis(compute_portfolio_return_series(returns_df, max_s_w), max_s_w, "Max Sharpe")
    kpi_min_v = calculate_portfolio_kpis(compute_portfolio_return_series(returns_df, min_v_w), min_v_w, "Min Vol")
    kpi_60_40 = calculate_portfolio_kpis(compute_portfolio_return_series(returns_df, CLASSIC_60_40_WEIGHTS), CLASSIC_60_40_WEIGHTS, "60/40")
    kpi_serendib = calculate_portfolio_kpis(compute_portfolio_return_series(returns_df, SERENDIB_CONSERVATIVE_WEIGHTS), SERENDIB_CONSERVATIVE_WEIGHTS, "Serendib")

    fig = go.Figure()

    # Monte Carlo Cloud
    fig.add_trace(
        go.Scatter(
            x=sim_df["Volatility"] * 100.0,
            y=sim_df["Annual_Return"] * 100.0,
            mode="markers",
            marker=dict(
                size=4,
                color=sim_df["Sharpe_Ratio"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="Sharpe Ratio", font=dict(color="#f8fafc"))),
                opacity=0.35,
            ),
            name="Simulated Portfolios (N=2,000)",
            hovertemplate="<b>Vol:</b> %{x:.2f}%<br><b>Return:</b> %{y:.2f}%<br><b>Sharpe:</b> %{marker.color:.3f}<extra></extra>",
        )
    )

    # Efficient Frontier Curve
    fig.add_trace(
        go.Scatter(
            x=frontier_df["Volatility"] * 100.0,
            y=frontier_df["Target_Return"] * 100.0,
            mode="lines",
            line=dict(color="#38bdf8", width=3.5),
            name="Efficient Frontier (Markowitz)",
            hovertemplate="<b>Frontier Vol:</b> %{x:.2f}%<br><b>Target Return:</b> %{y:.2f}%<extra></extra>",
        )
    )

    # Anchor points
    fig.add_trace(
        go.Scatter(
            x=[kpi_max_s.annual_volatility * 100.0],
            y=[kpi_max_s.annual_return * 100.0],
            mode="markers+text",
            marker=dict(color="#f59e0b", size=14, symbol="star", line=dict(color="#ffffff", width=1)),
            text=["Max Sharpe (14.82% / 15.66%)"],
            textposition="top center",
            name="Tangency Portfolio (Max Sharpe)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[kpi_min_v.annual_volatility * 100.0],
            y=[kpi_min_v.annual_return * 100.0],
            mode="markers+text",
            marker=dict(color="#10b981", size=12, symbol="diamond", line=dict(color="#ffffff", width=1)),
            text=["Min Variance (3.76% / 5.08%)"],
            textposition="bottom right",
            name="Global Minimum Variance",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[kpi_60_40.annual_volatility * 100.0],
            y=[kpi_60_40.annual_return * 100.0],
            mode="markers+text",
            marker=dict(color="#94a3b8", size=12, symbol="square", line=dict(color="#ffffff", width=1)),
            text=["Classic 60/40 (8.61% / 11.71%)"],
            textposition="middle right",
            name="Classic 60/40 Benchmark",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[kpi_serendib.annual_volatility * 100.0],
            y=[kpi_serendib.annual_return * 100.0],
            mode="markers+text",
            marker=dict(color="#ef4444", size=16, symbol="circle", line=dict(color="#ffffff", width=2)),
            text=["<b>Serendib Conservative (8.25% / 11.31%)</b>"],
            textposition="bottom left",
            name="Serendib Conservative Mandate",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=540,
        title=dict(
            text="<b>Markowitz Modern Portfolio Theory & The Efficient Frontier</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=14, color="#f8fafc"),
        ),
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Expected Return (%)",
        margin=dict(t=60, b=50, l=60, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
            font=dict(size=10, color="#f8fafc"),
        ),
        paper_bgcolor="rgba(15,23,42,0.95)",
        plot_bgcolor="rgba(30,41,59,0.7)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    )

    return fig


def create_evt_mean_excess_figure(port_ret: pd.Series) -> go.Figure:
    """Generate interactive Mean Excess Function / Mean Residual Life plot for EVT-POT threshold justification."""
    losses = -port_ret.values
    fit_res, _ = fit_gpd_pot(port_ret, threshold_percentile=0.95)
    mef_data = compute_mean_excess_function(losses, min_percentile=0.80, max_percentile=0.99, n_thresholds=40)

    fig = go.Figure()

    # Empirical MEF line
    fig.add_trace(
        go.Scatter(
            x=mef_data["Threshold_u"] * 100.0,
            y=mef_data["Mean_Excess"] * 100.0,
            mode="lines+markers",
            marker=dict(color="#38bdf8", size=6),
            line=dict(color="#38bdf8", width=2),
            name="Empirical Mean Excess e(u)",
            hovertemplate="<b>Threshold u:</b> %{x:.2f}%<br><b>Mean Excess:</b> %{y:.3f}%<extra></extra>",
        )
    )

    # Threshold indicator
    u_pct = fit_res.threshold_u * 100.0
    fig.add_vline(
        x=u_pct,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"Selected Threshold u* = {u_pct:.2f}% (95th Pct)",
        annotation_position="top left",
        annotation_font=dict(color="#f59e0b", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        title=dict(
            text="<b>EVT-POT Diagnostic: Mean Excess Function / Mean Residual Life Plot</b><br><sup>Pickands-Balkema-de Haan (1975) GPD Threshold Justification</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=13, color="#f8fafc"),
        ),
        xaxis_title="Threshold u (% Daily Loss)",
        yaxis_title="Mean Excess e(u) = E[L - u | L > u] (%)",
        margin=dict(t=70, b=50, l=60, r=40),
        legend=dict(
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
            font=dict(size=10, color="#f8fafc"),
        ),
        paper_bgcolor="rgba(15,23,42,0.95)",
        plot_bgcolor="rgba(30,41,59,0.7)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    )

    return fig


def create_stress_scenarios_figure(
    daily_returns: pd.DataFrame,
    weights: dict = SERENDIB_CONSERVATIVE_WEIGHTS,
    aum: float = MANDATE_SIZE_USD,
) -> go.Figure:
    """Generate comparative bar chart of USD dollar crisis losses on $10M corpus."""
    stress_recs, stress_df = evaluate_all_stress_scenarios(daily_returns, weights, aum=aum)

    scenarios = [r["Crisis Scenario"] for r in stress_recs]
    serendib_losses = [abs(r["raw_dollar_impact"]) for r in stress_recs]
    b_6040_losses = [abs(float(r["Classic 60/40 Dollar Loss"].replace("$", "").replace(",", ""))) for r in stress_recs]
    saved = [r["raw_capital_preserved"] for r in stress_recs]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=serendib_losses,
            name="Serendib Conservative ($10M AUM)",
            marker=dict(color="#38bdf8", line=dict(color="#0284c7", width=1.5)),
            text=[f"${x:,.0f}" for x in serendib_losses],
            textposition="auto",
            hovertemplate="<b>Scenario:</b> %{x}<br><b>Serendib Loss:</b> $%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=b_6040_losses,
            name="Classic 60/40 Benchmark",
            marker=dict(color="#94a3b8", line=dict(color="#64748b", width=1.5)),
            text=[f"${x:,.0f}" for x in b_6040_losses],
            textposition="auto",
            hovertemplate="<b>Scenario:</b> %{x}<br><b>60/40 Loss:</b> $%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=500,
        title=dict(
            text="<b>Crisis Stress Testing Dollar Impact on USD 10,000,000 AUM</b><br><sup>Empirical Historical Replays vs. 1970s Geopolitical Stagflation Shock</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=13, color="#f8fafc"),
        ),
        yaxis_title="Crisis Dollar Loss ($ USD)",
        xaxis_title="",
        yaxis=dict(tickprefix="$"),
        margin=dict(t=70, b=60, l=70, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
            font=dict(size=11, color="#f8fafc"),
        ),
        paper_bgcolor="rgba(15,23,42,0.95)",
        plot_bgcolor="rgba(30,41,59,0.7)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    )

    return fig


def export_all_showcase_assets(
    docs_dir: Path = Path("docs"),
    report_dir: Path = Path("report"),
) -> None:
    """Build all interactive HTML charts and synchronize PDF deliverables into docs/."""
    charts_dir = docs_dir / "assets" / "charts"
    docs_report_dir = docs_dir / "report"

    charts_dir.mkdir(parents=True, exist_ok=True)
    docs_report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Ingesting market data to generate interactive showcase charts...")
    market_data = load_market_dataset()
    serendib_ret = compute_portfolio_return_series(market_data.log_returns, SERENDIB_CONSERVATIVE_WEIGHTS)

    # 1. Cumulative Wealth Chart
    logger.info("Exporting cumulative_wealth.html...")
    fig_wealth = create_cumulative_wealth_figure(market_data.simple_returns)
    fig_wealth.write_html(str(charts_dir / "cumulative_wealth.html"), include_plotlyjs="cdn", full_html=True)

    # 2. Efficient Frontier Chart
    logger.info("Exporting efficient_frontier.html...")
    fig_frontier = create_efficient_frontier_figure(market_data.simple_returns)
    fig_frontier.write_html(str(charts_dir / "efficient_frontier.html"), include_plotlyjs="cdn", full_html=True)

    # 3. EVT Mean Excess Function Chart
    logger.info("Exporting evt_mean_excess.html...")
    fig_mef = create_evt_mean_excess_figure(serendib_ret)
    fig_mef.write_html(str(charts_dir / "evt_mean_excess.html"), include_plotlyjs="cdn", full_html=True)

    # 4. Stress Scenarios Chart
    logger.info("Exporting stress_scenarios.html...")
    fig_stress = create_stress_scenarios_figure(market_data.simple_returns)
    fig_stress.write_html(str(charts_dir / "stress_scenarios.html"), include_plotlyjs="cdn", full_html=True)

    # 5. Mirror PDF Reports to docs/report/
    logger.info("Mirroring official PDF deliverables into docs/report/...")
    tear_sheet_src = report_dir / "tear_sheet.pdf"
    investment_memo_src = report_dir / "investment_memo.pdf"

    if tear_sheet_src.exists():
        shutil.copy2(tear_sheet_src, docs_report_dir / "tear_sheet.pdf")
        logger.info(f"Copied {tear_sheet_src} -> {docs_report_dir / 'tear_sheet.pdf'}")
    else:
        logger.warning(f"Source tear sheet {tear_sheet_src} not found.")

    if investment_memo_src.exists():
        shutil.copy2(investment_memo_src, docs_report_dir / "investment_memo.pdf")
        logger.info(f"Copied {investment_memo_src} -> {docs_report_dir / 'investment_memo.pdf'}")
    else:
        logger.warning(f"Source investment memo {investment_memo_src} not found.")

    logger.info("Static interactive showcase assets successfully exported to docs/!")


if __name__ == "__main__":
    export_all_showcase_assets()
