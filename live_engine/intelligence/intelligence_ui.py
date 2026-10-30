"""
Market Intelligence Center UI Component
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 3)

Renders the institutional Market Intelligence Center dashboard:
1. 🌐 Executive Market Summary
2. ⚡ ETF Trend Intelligence & Factor Breakdown
3. 📰 News Sentiment & Macro Event Telemetry
4. 🏛️ Macroeconomic Intelligence & Yield Curve Regime
5. 🚨 Statistical Anomaly Detection Alerts
6. 🎯 Systematic Opportunity Screener
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from live_engine.core.data_validator import DataFreshness, ProviderStatus
from live_engine.core.api_manager import LiveAPIManager
from live_engine.core.indicators import TECHNICAL_DISCLAIMER
from live_engine.intelligence.trend_analyzer import (
    analyze_universe_trends,
    get_trend_breadth,
    TrendInsight,
)
from live_engine.intelligence.news_analyzer import (
    analyze_news_feed,
    AnalyzedArticle,
)
from live_engine.intelligence.economic_analyzer import (
    analyze_all_economic_data,
    SeriesAnalysis,
    YieldCurveAnalysis,
)
from live_engine.intelligence.anomaly_detector import (
    scan_universe_anomalies,
    AnomalyAlert,
)
from live_engine.intelligence.opportunity_screener import (
    ScreeningCriteria,
    run_universe_screening,
    SCREENER_DISCLAIMER,
)
from live_engine.intelligence.market_summary import (
    compile_market_intelligence_summary,
    ExecutiveMarketSummary,
)

ETF_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]


def render_regime_badge(regime_text: str) -> str:
    """Render visual macro regime banner badge."""
    if "Risk-On" in regime_text:
        return '<span style="background:rgba(16,185,129,0.25); color:#34d399; border:1px solid rgba(16,185,129,0.5); padding:6px 14px; border-radius:8px; font-weight:800; font-size:1.05rem;">🟢 RISK-ON / EXPANSIONARY</span>'
    elif "Risk-Off" in regime_text:
        return '<span style="background:rgba(239,68,68,0.25); color:#f87171; border:1px solid rgba(239,68,68,0.5); padding:6px 14px; border-radius:8px; font-weight:800; font-size:1.05rem;">🔴 RISK-OFF / DEFENSIVE</span>'
    else:
        return '<span style="background:rgba(245,158,11,0.25); color:#fbbf24; border:1px solid rgba(245,158,11,0.5); padding:6px 14px; border-radius:8px; font-weight:800; font-size:1.05rem;">🟡 TRANSITIONAL / NEUTRAL REGIME</span>'


def render_sentiment_badge(label: str) -> str:
    """Render sentiment label badge."""
    if label == "Positive":
        return '<span style="background:rgba(16,185,129,0.2); color:#34d399; border:1px solid rgba(16,185,129,0.4); padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.78rem;">🟢 POSITIVE</span>'
    elif label == "Negative":
        return '<span style="background:rgba(239,68,68,0.2); color:#f87171; border:1px solid rgba(239,68,68,0.4); padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.78rem;">🔴 NEGATIVE</span>'
    else:
        return '<span style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid rgba(148,163,184,0.3); padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.78rem;">⚪ NEUTRAL</span>'


def render_severity_badge(severity: str) -> str:
    """Render anomaly alert severity badge."""
    if severity == "High":
        return '<span style="background:rgba(239,68,68,0.25); color:#f87171; border:1px solid rgba(239,68,68,0.5); padding:2px 8px; border-radius:4px; font-weight:800; font-size:0.78rem;">🚨 HIGH SEVERITY</span>'
    elif severity == "Medium":
        return '<span style="background:rgba(245,158,11,0.25); color:#fbbf24; border:1px solid rgba(245,158,11,0.5); padding:2px 8px; border-radius:4px; font-weight:800; font-size:0.78rem;">⚠️ MEDIUM</span>'
    else:
        return '<span style="background:rgba(56,189,248,0.2); color:#38bdf8; border:1px solid rgba(56,189,248,0.4); padding:2px 8px; border-radius:4px; font-weight:800; font-size:0.78rem;">ℹ️ LOW</span>'


def render_market_intelligence_tab():
    """Main Streamlit renderer for the Market Intelligence Center."""
    st.subheader("🧠 Market Intelligence Center")
    st.markdown(
        "Autonomous institutional market intelligence engine. "
        "Synthesizes live cross-asset trends, sentiment extraction, macroeconomic yield curve diagnostics, "
        "statistical anomaly alerts, and research opportunity screening."
    )

    st.info(
        "🔒 **Fiduciary Sandbox Notice:** This intelligence engine generates automated empirical insights, "
        "sentiment scores, and screening filters. It operates strictly as an independent analytics layer and "
        "does NOT alter the endowment's core Markowitz portfolio, VaR thresholds, or regulatory risk models."
    )

    # Initialize intelligence state if missing
    if "intel_history_cache" not in st.session_state:
        st.session_state["intel_history_cache"] = {}
    if "intel_news_cache" not in st.session_state:
        st.session_state["intel_news_cache"] = []
    if "intel_fred_cache" not in st.session_state:
        st.session_state["intel_fred_cache"] = {}
    if "intel_last_run" not in st.session_state:
        st.session_state["intel_last_run"] = None

    # Control Bar
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        st.markdown(
            f"<div style='font-size:0.85rem; color:#94a3b8; margin-top:8px;'>"
            f"<b>Intelligence Telemetry Status:</b> "
            f"{'Last calculated at ' + st.session_state['intel_last_run'] if st.session_state['intel_last_run'] else 'Initial state (Click to run analysis)'}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_ctrl2:
        run_analysis = st.button("⚡ Run Full Market Analysis", type="primary", use_container_width=True)

    # Load/fetch data if run clicked or first time
    if run_analysis or not st.session_state["intel_history_cache"]:
        with st.spinner("Executing intelligence models across 7 ETFs, FRED macro series, and news feeds..."):
            finnhub_key = st.session_state.get("live_finnhub_key", "")
            av_key = st.session_state.get("live_alphavantage_key", "")
            fred_key = st.session_state.get("live_fred_key", "")

            # 1. Gather historical data for 7 ETFs
            hist_dict = {}
            for sym in ETF_UNIVERSE:
                res = LiveAPIManager.fetch_etf_daily_series(
                    symbol=sym,
                    alphavantage_key=av_key,
                    outputsize="compact",
                    use_cache=True,
                )
                if res.status == ProviderStatus.CONNECTED and res.dataframe is not None:
                    hist_dict[sym] = res.dataframe

            # 2. Gather FRED series
            fred_dict = {}
            for sid in ["DTB3", "DGS10", "FEDFUNDS", "CPIAUCSL", "UNRATE"]:
                res_f = LiveAPIManager.fetch_economic_indicator(
                    series_id=sid,
                    fred_key=fred_key,
                    limit=60,
                    use_cache=True,
                )
                if res_f.status == ProviderStatus.CONNECTED and res_f.dataframe is not None:
                    fred_dict[sid] = res_f.dataframe

            # 3. Gather news
            news_res = LiveAPIManager.fetch_news(
                category="general",
                finnhub_key=finnhub_key,
                limit=20,
                use_cache=True,
            )
            raw_articles = news_res.metrics.get("articles", []) if news_res.metrics else []

            st.session_state["intel_history_cache"] = hist_dict
            st.session_state["intel_fred_cache"] = fred_dict
            st.session_state["intel_news_cache"] = raw_articles
            st.session_state["intel_last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run Intelligence Analyses on cached data
    hist_data = st.session_state["intel_history_cache"]
    fred_data = st.session_state["intel_fred_cache"]
    news_data = st.session_state["intel_news_cache"]

    # 1. Trend Analysis
    trend_insights = analyze_universe_trends(hist_data)
    breadth_info = get_trend_breadth(trend_insights)

    # 2. News Analysis
    news_analysis = analyze_news_feed(news_data)

    # 3. Economic Analysis
    economic_analysis = analyze_all_economic_data(fred_data)

    # 4. Anomaly Detection
    anomaly_analysis = scan_universe_anomalies(hist_data)

    # 5. Default Screening
    screening_analysis = run_universe_screening(trend_insights)

    # 6. Consolidated Executive Summary
    exec_summary = compile_market_intelligence_summary(
        trend_insights=trend_insights,
        news_analysis=news_analysis,
        economic_analysis=economic_analysis,
        anomaly_analysis=anomaly_analysis,
        screening_analysis=screening_analysis,
        data_freshness=DataFreshness.CACHED if not run_analysis else DataFreshness.FRESH,
    )

    st.markdown("<hr style='margin:16px 0; border:0; border-top:1px solid rgba(148,163,184,0.15);'>", unsafe_allow_html=True)

    # =========================================================================
    # Navigation Tabs for the 6 Intelligence Subsections
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌐 Executive Overview",
        "⚡ ETF Trend Intelligence",
        "📰 News Sentiment & Events",
        "🏛️ Macroeconomic Intelligence",
        "🚨 Anomaly Detection Alerts",
        "🎯 Opportunity Screener",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("#### Institutional Market Regime Assessment")

        # Top Macro Stance Banner Card
        st.markdown(
            f"""<div style="background:linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95)); border:1px solid rgba(148,163,184,0.2); border-radius:12px; padding:20px 24px; margin-bottom:20px; box-shadow:0 6px 16px rgba(0,0,0,0.3);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                    <div>
                        <div style="font-size:0.80rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;">SYNTHESIZED MACRO STANCE</div>
                        <div style="margin-top:6px;">{render_regime_badge(exec_summary.macro_regime)}</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:0.80rem; color:#94a3b8;">Analysis As Of: <b>{exec_summary.timestamp}</b></span><br>
                        <span style="font-size:0.75rem; color:#64748b;">Telemetry Freshness: <b>{exec_summary.data_freshness.value}</b></span>
                    </div>
                </div>
                <div style="font-size:0.95rem; color:#e2e8f0; margin-top:12px; line-height:1.5;">
                    {exec_summary.macro_regime_rationale}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # 4 Core Intelligence KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">ETF Market Breadth</div>
                    <div class="metric-value">{exec_summary.bullish_assets_count}/{exec_summary.total_assets_evaluated} Bullish</div>
                    <div class="metric-sub">{breadth_info.get('market_breadth_label', 'Market Breadth')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            sent_score = exec_summary.news_sentiment_score
            sent_color = "#34d399" if sent_score >= 0.1 else "#f87171" if sent_score <= -0.1 else "#38bdf8"
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">News Sentiment Stance</div>
                    <div class="metric-value" style="color:{sent_color};">{sent_score:+.2f}</div>
                    <div class="metric-sub">{exec_summary.sentiment_stance}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c3:
            yc_col = "#f87171" if "Inverted" in exec_summary.yield_curve_regime else "#34d399"
            yc_str = f"{exec_summary.yield_curve_spread_bps:+.0f} bps" if exec_summary.yield_curve_spread_bps is not None else "N/A"
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Yield Curve (10Y - 3M)</div>
                    <div class="metric-value" style="color:{yc_col}; font-size:1.55rem;">{exec_summary.yield_curve_regime}</div>
                    <div class="metric-sub">Spread: {yc_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c4:
            anom_col = "#f87171" if exec_summary.high_severity_anomalies_count > 0 else "#34d399"
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Active Market Anomalies</div>
                    <div class="metric-value" style="color:{anom_col};">{exec_summary.active_anomalies_count} Detected</div>
                    <div class="metric-sub">{exec_summary.high_severity_anomalies_count} High Severity Shocks</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Key Intelligence Takeaways
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📌 Key Macroeconomic & Market Takeaways")
        for item in exec_summary.key_takeaways:
            st.markdown(f"- {item}")

    # -------------------------------------------------------------------------
    # TAB 2: ETF TREND INTELLIGENCE
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("#### Cross-Asset Trend Intelligence & Factor Attribution")
        st.markdown(
            "Empirical multi-factor breakdown across all 7 endowment ETF assets. "
            "Evaluates moving average geometry, momentum, RSI, and volatility."
        )

        # Multi-Asset Summary Table
        summary_rows = [t.to_dict() for t in trend_insights.values()]
        df_trends = pd.DataFrame(summary_rows)
        st.dataframe(df_trends, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Deep Factor Attribution by Asset")

        selected_asset = st.selectbox(
            "Select Asset for Factor-Level Breakdown",
            options=ETF_UNIVERSE,
            key="intel_asset_selector",
        )

        insight: TrendInsight = trend_insights.get(selected_asset)
        if insight:
            st.markdown(
                f"""<div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.15); border-radius:10px; padding:18px 20px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <div>
                            <span style="font-weight:800; font-size:1.15rem; color:#f8fafc;">{insight.symbol} — {insight.name}</span><br>
                            <span style="font-size:0.85rem; color:#94a3b8;">Classification: <b>{insight.classification}</b> | Trend Strength: <b>{insight.strength_score:.1f}/100</b></span>
                        </div>
                        <div>
                            <span class="{'status-badge-green' if insight.classification=='Bullish' else 'status-badge-red' if insight.classification=='Bearish' else 'status-badge-yellow'}">
                                {insight.classification.upper()}
                            </span>
                        </div>
                    </div>
                    <div style="font-weight:700; font-size:0.88rem; color:#38bdf8; margin-bottom:8px;">Contributing Diagnostic Factors:</div>
                """,
                unsafe_allow_html=True,
            )

            for factor in insight.contributing_factors:
                st.markdown(f"- {factor}")

            st.markdown("</div>", unsafe_allow_html=True)

        st.warning(f"⚠️ {TECHNICAL_DISCLAIMER}")

    # -------------------------------------------------------------------------
    # TAB 3: NEWS SENTIMENT INTELLIGENCE
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("#### Real-Time Financial News Sentiment & Event Extraction")
        st.markdown(
            "Rule-based institutional sentiment analysis inspired by the Loughran-McDonald lexicon. "
            "Analyzes raw headline and summary feeds from Finnhub to identify macro catalysts."
        )

        # Sentiment Distribution Metrics
        ns1, ns2, ns3, ns4 = st.columns(4)
        with ns1:
            st.metric("Total Articles Analyzed", news_analysis.get("total_articles", 0))
        with ns2:
            pos_c = news_analysis.get("positive_count", 0)
            pos_r = news_analysis.get("positive_ratio", 0.0)
            st.metric("Positive Sentiment", f"{pos_c} articles", f"{pos_r * 100:.0f}%")
        with ns3:
            neg_c = news_analysis.get("negative_count", 0)
            neg_r = news_analysis.get("negative_ratio", 0.0)
            st.metric("Negative Sentiment", f"{neg_c} articles", f"-{neg_r * 100:.0f}%")
        with ns4:
            st.metric("Neutral / Factual", f"{news_analysis.get('neutral_count', 0)} articles")

        # High Impact Events Section
        high_imp: List[AnalyzedArticle] = news_analysis.get("high_impact_events", [])
        if high_imp:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### ⚡ High-Impact Macroeconomic News Events")
            for h in high_imp[:3]:
                st.markdown(
                    f"""<div style="background:rgba(239,68,68,0.1); border-left:4px solid #ef4444; padding:10px 16px; margin-bottom:8px; border-radius:0 6px 6px 0;">
                        <div style="font-weight:700; color:#f8fafc;"><a href="{h.url}" target="_blank" style="color:#f87171; text-decoration:none;">{h.headline}</a></div>
                        <div style="font-size:0.80rem; color:#94a3b8; margin-top:2px;">Events: <b>{', '.join(h.macro_events)}</b> | Source: {h.source} | Published: {h.published_time}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Full Analyzed Articles Feed
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📰 Analyzed Article Stream")
        articles_list: List[AnalyzedArticle] = news_analysis.get("articles", [])
        if articles_list:
            for art in articles_list[:12]:
                with st.container():
                    st.markdown(
                        f"""<div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.15); border-radius:8px; padding:14px 18px; margin-bottom:12px;">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                <div style="font-weight:700; font-size:1.0rem; color:#f8fafc; line-height:1.4;">
                                    <a href="{art.url}" target="_blank" style="color:#38bdf8; text-decoration:none;">{art.headline}</a>
                                </div>
                                <div style="margin-left:12px; white-space:nowrap;">
                                    {render_sentiment_badge(art.sentiment_label)}
                                </div>
                            </div>
                            <div style="font-size:0.85rem; color:#cbd5e1; margin-top:6px;">{art.summary[:240] + '...' if len(art.summary) > 240 else art.summary}</div>
                            <div style="font-size:0.75rem; color:#64748b; margin-top:8px;">
                                <b>Score:</b> {art.sentiment_score:+.2f} &nbsp;|&nbsp; 
                                <b>Source:</b> {art.source} &nbsp;|&nbsp; 
                                <b>Published:</b> {art.published_time} &nbsp;|&nbsp; 
                                <b>Related:</b> {', '.join(art.related_etfs) if art.related_etfs else 'Broad Market'}
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No live financial news articles currently loaded. Configure Finnhub API key to stream live feeds.")

    # -------------------------------------------------------------------------
    # TAB 4: MACROECONOMIC INTELLIGENCE
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("#### Macroeconomic Intelligence & Yield Curve Diagnostics")
        st.markdown(
            "Synthesizes Federal Reserve (FRED) observations. "
            "Strictly bifurcates **Empirical Observable Facts** from **Institutional Economic Interpretations**."
        )

        # Yield Curve Diagnostic Card
        yc_data: YieldCurveAnalysis = economic_analysis.get("yield_curve")
        if yc_data:
            st.markdown(
                f"""<div style="background:rgba(30,41,59,0.6); border:1px solid rgba(148,163,184,0.15); border-radius:10px; padding:16px 20px; margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:800; font-size:1.1rem; color:#38bdf8;">10Y - 3M Sovereign Yield Curve Spread (T10Y3M)</span>
                        <span class="{'status-badge-red' if 'Inverted' in yc_data.status else 'status-badge-green'}">
                            {yc_data.status.upper()}
                        </span>
                    </div>
                    <div style="font-size:0.88rem; color:#f8fafc; margin-bottom:6px;">
                        <b>Observable Fact:</b> {yc_data.observable_fact}
                    </div>
                    <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.5;">
                        <b>Institutional Interpretation:</b> {yc_data.economic_interpretation}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Side-by-Side Table: Observable Facts vs Institutional Interpretations
        st.markdown("##### 📊 Macro Indicator Factual Audit vs Economic Meaning")
        series_map: Dict[str, SeriesAnalysis] = economic_analysis.get("series_analyses", {})
        if series_map:
            macro_table_rows = [s.to_dict() for s in series_map.values()]
            df_macro_audit = pd.DataFrame(macro_table_rows)
            st.dataframe(df_macro_audit, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 5: ANOMALY DETECTION ALERTS
    # -------------------------------------------------------------------------
    with tab5:
        st.markdown("#### Statistical Market Anomaly Detection")
        st.markdown(
            "Automated detection of empirical market dislocations using rolling z-scores ($Z = \\frac{x - \\mu}{\\sigma}$). "
            "Triggers alerts on return shocks ($|Z| \\ge 2.0\\sigma$), volume surges ($>2.0\\times$), volatility spikes, and moving average dislocations."
        )

        anom_total = anomaly_analysis.get("total_anomalies", 0)
        high_cnt = anomaly_analysis.get("high_severity_count", 0)
        med_cnt = anomaly_analysis.get("medium_severity_count", 0)

        st.markdown(
            f"""<div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.15); border-radius:8px; padding:12px 18px; margin-bottom:16px;">
                <b>Anomaly Detection Engine Summary:</b> {anom_total} active statistical alerts detected 
                ({high_cnt} High Severity, {med_cnt} Medium Severity) across the 7-asset ETF universe.
            </div>""",
            unsafe_allow_html=True,
        )

        alerts_list: List[AnomalyAlert] = anomaly_analysis.get("alerts", [])
        if alerts_list:
            for alert in alerts_list:
                st.markdown(
                    f"""<div style="background:rgba(30,41,59,0.6); border:1px solid rgba(148,163,184,0.2); border-radius:8px; padding:14px 18px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-weight:800; font-size:1.05rem; color:#f8fafc;">
                                {alert.asset} — {alert.anomaly_type}
                            </span>
                            <div>{render_severity_badge(alert.severity)}</div>
                        </div>
                        <div style="font-size:0.86rem; color:#cbd5e1; margin-bottom:6px;">
                            {alert.narrative}
                        </div>
                        <div style="font-size:0.75rem; color:#94a3b8;">
                            <b>Trigger Condition:</b> {alert.threshold_description} &nbsp;|&nbsp; 
                            <b>Z-Score:</b> {alert.z_score:+.2f}σ &nbsp;|&nbsp; 
                            <b>Date:</b> {alert.date}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """<div style="padding:24px; text-align:center; color:#34d399; background:rgba(16,185,129,0.1); border-radius:10px; border:1px solid rgba(16,185,129,0.3);">
                    <b>Zero Statistical Anomalies Detected</b><br>
                    All 7 ETFs currently trade within normal 2-sigma return distributions and baseline volume parameters.
                </div>""",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # TAB 6: OPPORTUNITY SCREENER
    # -------------------------------------------------------------------------
    with tab6:
        st.markdown("#### Systematic Research Opportunity Screener")
        st.markdown(
            "Configurable institutional research screener evaluating the 7 endowment assets against "
            "quantitative momentum, trend, and volatility constraints."
        )

        with st.expander("⚙️ Configure Research Screening Filters", expanded=True):
            sc_c1, sc_c2 = st.columns(2)
            with sc_c1:
                req_5d = st.checkbox("Require Positive 5-Day Momentum (> 0%)", value=True)
                req_sma20 = st.checkbox("Require Price > 20-Day SMA", value=True)
                req_sma50 = st.checkbox("Require Price > 50-Day SMA", value=False)
            with sc_c2:
                max_vol = st.slider("Max Annualized Volatility Ceiling (%)", 10.0, 45.0, 25.0, 1.0)
                min_rsi = st.slider("Min RSI (14)", 20.0, 50.0, 40.0, 5.0)
                max_rsi = st.slider("Max RSI (14)", 50.0, 85.0, 70.0, 5.0)

            custom_criteria = ScreeningCriteria(
                require_positive_5d_momentum=req_5d,
                require_price_above_sma20=req_sma20,
                require_price_above_sma50=req_sma50,
                max_rolling_volatility_pct=max_vol,
                min_rsi_14=min_rsi,
                max_rsi_14=max_rsi,
            )

        # Re-run screening with user criteria
        custom_screening = run_universe_screening(trend_insights, custom_criteria)

        pass_c = custom_screening["passed_count"]
        fail_c = custom_screening["failed_count"]

        # Results Summary Row
        sr1, sr2 = st.columns(2)
        with sr1:
            st.markdown(
                f"""<div class="metric-card" style="border-left:4px solid #10b981;">
                    <div class="metric-label">Assets Passing Screening Criteria</div>
                    <div class="metric-value" style="color:#34d399;">{pass_c} / {len(ETF_UNIVERSE)}</div>
                    <div class="metric-sub">{', '.join([a.symbol for a in custom_screening['passed_assets']]) if pass_c > 0 else 'None'}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with sr2:
            st.markdown(
                f"""<div class="metric-card" style="border-left:4px solid #ef4444;">
                    <div class="metric-label">Assets Failing Criteria</div>
                    <div class="metric-value" style="color:#f87171;">{fail_c} / {len(ETF_UNIVERSE)}</div>
                    <div class="metric-sub">{', '.join([a.symbol for a in custom_screening['failed_assets']]) if fail_c > 0 else 'None'}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📋 Granular Screening Audit Matrix")
        st.dataframe(custom_screening["audit_df"], use_container_width=True, hide_index=True)

        st.warning(f"⚠️ {SCREENER_DISCLAIMER}")
