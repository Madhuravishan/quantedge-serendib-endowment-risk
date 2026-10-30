"""
Live Market Data UI Component (Phase 2 Expanded Architecture)
QuantEdge 1.0 - Serendib University Endowment Risk Engine

Subsections:
1. 🏛️ Market Overview
2. 📈 ETF Historical Performance
3. ⚡ Market Trends & Technical Indicators
4. 🌐 Economic Indicators
5. 📰 Financial News
6. 📄 SEC Filings
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from live_engine.core.data_validator import ProviderStatus, DataFreshness, LiveResult
from live_engine.core.api_manager import LiveAPIManager
from live_engine.core.indicators import compute_technical_indicators, TECHNICAL_DISCLAIMER
from live_engine.collectors.fred_collector import FRED_SERIES_CATALOG
from live_engine.collectors.sec_collector import SEC_ENTITY_CATALOG


ETF_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]
ETF_DESCRIPTIONS = {
    "SPY": "SPDR S&P 500 ETF Trust (US Large-Cap Core Equity)",
    "EFA": "iShares MSCI EAFE ETF (Developed International Markets)",
    "EEM": "iShares MSCI Emerging Markets ETF (Emerging Markets Equity)",
    "AGG": "iShares Core U.S. Aggregate Bond ETF (US Investment Grade Fixed Income)",
    "GLD": "SPDR Gold Shares (Precious Metals / Real Asset Hedge)",
    "XLE": "Energy Select Sector SPDR Fund (Natural Resources & Energy)",
    "XLK": "Technology Select Sector SPDR Fund (US Secular Tech Growth)",
}


def render_status_badge(status: ProviderStatus) -> str:
    """Render institutional HTML status badge."""
    if status == ProviderStatus.CONNECTED:
        return '<span style="background:rgba(16,185,129,0.2); color:#34d399; border:1px solid rgba(16,185,129,0.4); padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.80rem;">🟢 CONNECTED</span>'
    elif status == ProviderStatus.RATE_LIMITED:
        return '<span style="background:rgba(245,158,11,0.2); color:#fbbf24; border:1px solid rgba(245,158,11,0.4); padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.80rem;">🟡 RATE LIMITED</span>'
    elif status == ProviderStatus.FAILED:
        return '<span style="background:rgba(239,68,68,0.2); color:#f87171; border:1px solid rgba(239,68,68,0.4); padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.80rem;">🔴 FAILED</span>'
    else:
        return '<span style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid rgba(148,163,184,0.3); padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.80rem;">⚪ NOT CONFIGURED</span>'


def render_freshness_badge(freshness: DataFreshness) -> str:
    """Render institutional HTML data freshness badge."""
    if freshness == DataFreshness.FRESH:
        return '<span style="background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.35); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem;">⚡ FRESH</span>'
    elif freshness == DataFreshness.CACHED:
        return '<span style="background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid rgba(56,189,248,0.35); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem;">💾 CACHED</span>'
    elif freshness == DataFreshness.DELAYED:
        return '<span style="background:rgba(245,158,11,0.15); color:#f59e0b; border:1px solid rgba(245,158,11,0.35); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem;">⏳ DELAYED</span>'
    else:
        return '<span style="background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.35); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem;">❌ UNAVAILABLE</span>'


def render_trend_badge(classification: str) -> str:
    """Render visual trend classification badge."""
    if classification == "Bullish":
        return '<span style="background:rgba(16,185,129,0.25); color:#34d399; border:1px solid rgba(16,185,129,0.5); padding:4px 12px; border-radius:6px; font-weight:800; font-size:0.90rem;">🟢 BULLISH TREND</span>'
    elif classification == "Bearish":
        return '<span style="background:rgba(239,68,68,0.25); color:#f87171; border:1px solid rgba(239,68,68,0.5); padding:4px 12px; border-radius:6px; font-weight:800; font-size:0.90rem;">🔴 BEARISH TREND</span>'
    else:
        return '<span style="background:rgba(148,163,184,0.2); color:#cbd5e1; border:1px solid rgba(148,163,184,0.4); padding:4px 12px; border-radius:6px; font-weight:800; font-size:0.90rem;">⚪ NEUTRAL / CONSOLIDATION</span>'


def render_live_market_data_tab():
    """Main entry point for the 🌐 Live Market Data dashboard section."""
    st.subheader("🌐 Live Market Data & Technical Telemetry")
    st.markdown(
        "Institutional live external market monitoring interface. "
        "Provides real-time quotes, technical indicators, economic data, news feeds, and SEC filings. "
        "Operates as an independent telemetry layer, completely decoupled from the core $10,000,000 endowment portfolio calculations."
    )

    st.info(
        "🔒 **Phase 2 Fiduciary Sandbox:** All external queries are performed independently with failure isolation. "
        "API keys are held in temporary browser session state and never committed, logged, or saved to disk. "
        "The core endowment risk engine, historical datasets, and portfolio optimization models remain strictly untouched."
    )

    # Initialize session state variables
    if "live_finnhub_key" not in st.session_state:
        st.session_state["live_finnhub_key"] = os.environ.get("FINNHUB_API_KEY", "")
    if "live_alphavantage_key" not in st.session_state:
        st.session_state["live_alphavantage_key"] = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if "live_fred_key" not in st.session_state:
        st.session_state["live_fred_key"] = os.environ.get("FRED_API_KEY", "")
    if "live_sec_ua" not in st.session_state:
        st.session_state["live_sec_ua"] = os.environ.get(
            "SEC_USER_AGENT",
            "SerendibEndowmentRiskEngine/1.0 (endowment-audit@serendib.ac.lk)",
        )
    if "selected_live_etf" not in st.session_state:
        st.session_state["selected_live_etf"] = "SPY"
    if "selected_fred_series" not in st.session_state:
        st.session_state["selected_fred_series"] = "DTB3"
    if "selected_sec_cik" not in st.session_state:
        st.session_state["selected_sec_cik"] = "0000884394"

    # =========================================================================
    # 0. API Credentials Expander
    # =========================================================================
    with st.expander("🔑 API Configuration & Session Credentials (Click to Expand / Manage)", expanded=False):
        st.markdown(
            "<span style='font-size:0.85rem; color:#94a3b8;'>"
            "Enter your personal API keys below. Credentials remain in ephemeral session memory only. "
            "Leave blank to test 'Not Configured' status or fallback mode."
            "</span>",
            unsafe_allow_html=True,
        )

        c_cfg1, c_cfg2 = st.columns(2)
        with c_cfg1:
            st.session_state["live_finnhub_key"] = st.text_input(
                "Finnhub API Key (Quotes & News)",
                value=st.session_state["live_finnhub_key"],
                type="password",
                placeholder="Enter personal Finnhub token",
                help="Free token from https://finnhub.io. Free tier allows 60 calls/minute.",
            )
            st.session_state["live_fred_key"] = st.text_input(
                "FRED API Key (Federal Reserve Economic Data)",
                value=st.session_state["live_fred_key"],
                type="password",
                placeholder="Enter 32-character FRED hex key",
                help="Free key from https://fred.stlouisfed.org.",
            )
        with c_cfg2:
            st.session_state["live_alphavantage_key"] = st.text_input(
                "Alpha Vantage API Key (Daily Historical Series)",
                value=st.session_state["live_alphavantage_key"],
                type="password",
                placeholder="Enter Alpha Vantage key",
                help="Free key from https://www.alphavantage.co (5 calls/min).",
            )
            st.session_state["live_sec_ua"] = st.text_input(
                "SEC EDGAR User-Agent (No API Key Required)",
                value=st.session_state["live_sec_ua"],
                type="default",
                help="SEC policy requires a non-generic User-Agent containing contact email.",
            )

        col_clr, col_msg = st.columns([1, 4])
        with col_clr:
            if st.button("🧹 Clear All Keys"):
                st.session_state["live_finnhub_key"] = ""
                st.session_state["live_alphavantage_key"] = ""
                st.session_state["live_fred_key"] = ""
                st.rerun()
        with col_msg:
            st.caption("Keys are wiped when the browser session terminates or upon clicking 'Clear All Keys'.")

    st.markdown("<hr style='margin:16px 0; border:0; border-top:1px solid rgba(148,163,184,0.15);'>", unsafe_allow_html=True)

    # =========================================================================
    # 6 Dedicated Subsections
    # =========================================================================
    sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs([
        "🏛️ Market Overview",
        "📈 ETF Historical Performance",
        "⚡ Market Trends & Indicators",
        "🌐 Economic Indicators",
        "📰 Financial News",
        "📄 SEC Filings",
    ])

    # -------------------------------------------------------------------------
    # SUBSECTION 1: MARKET OVERVIEW
    # -------------------------------------------------------------------------
    with sub1:
        st.markdown("#### 1. ETF Universe Market Overview")
        st.markdown("Select any asset from the endowment's 7-ETF universe to view latest quote telemetry.")

        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_etf = st.selectbox(
                "Select Endowment Asset Ticker",
                options=ETF_UNIVERSE,
                index=ETF_UNIVERSE.index(st.session_state["selected_live_etf"]),
                format_func=lambda s: f"{s} — {ETF_DESCRIPTIONS.get(s, '')}",
                key="etf_selector_sub1",
            )
            st.session_state["selected_live_etf"] = selected_etf
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh_quote = st.button("🔄 Refresh Quote", key="btn_refresh_quote", use_container_width=True)

        # Retrieve quote
        with st.spinner(f"Querying Finnhub for {selected_etf} market quote..."):
            quote_res = LiveAPIManager.fetch_etf_quote(
                symbol=selected_etf,
                finnhub_key=st.session_state["live_finnhub_key"],
                use_cache=not refresh_quote,
            )

        # Display Top Metric Cards
        if quote_res.status == ProviderStatus.CONNECTED and quote_res.metrics:
            m = quote_res.metrics
            curr_p = m.get("current_price")
            chg = m.get("change", 0.0)
            chg_pct = m.get("percent_change", 0.0)
            chg_str = f"{chg:+.2f} ({chg_pct:+.2f}%)" if curr_p else "N/A"

            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">{selected_etf} Current Price</div>
                        <div class="metric-value">${curr_p:.2f}</div>
                        <div class="metric-sub" style="color:{'#34d399' if chg >= 0 else '#f87171'}; font-weight:600;">{chg_str}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with kpi_cols[1]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Day High / Low</div>
                        <div class="metric-value" style="font-size:1.4rem;">${m.get('high_price', 0):.2f} / ${m.get('low_price', 0):.2f}</div>
                        <div class="metric-sub">Session Trading Range</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with kpi_cols[2]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Previous Close</div>
                        <div class="metric-value">${m.get('previous_close', 0):.2f}</div>
                        <div class="metric-sub">Open: ${m.get('open_price', 0):.2f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with kpi_cols[3]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Telemetry Status</div>
                        <div style="margin-top:6px; margin-bottom:4px;">{render_status_badge(quote_res.status)}</div>
                        <div class="metric-sub">{render_freshness_badge(quote_res.freshness)} | Code: {quote_res.http_status or '200'}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                st.markdown("##### Detailed Quote Fields")
                if quote_res.dataframe is not None:
                    st.dataframe(quote_res.dataframe, use_container_width=True, hide_index=True)
            with col_d2:
                st.markdown("##### Audit & Metadata Telemetry")
                st.markdown(f"""
                - **Asset Symbol:** `{selected_etf}` ({ETF_DESCRIPTIONS.get(selected_etf, '')})
                - **Data Provider:** `{quote_res.provider}` ({quote_res.data_source})
                - **Market Observation Time:** `{quote_res.market_timestamp or 'N/A'}`
                - **Application Retrieval Time:** `{quote_res.retrieval_timestamp}`
                - **Data Freshness:** `{quote_res.freshness.value}`
                """)
        else:
            st.markdown(
                f"""<div style="padding:24px; text-align:center; color:#94a3b8; background:rgba(30,41,59,0.5); border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                    <div style="margin-bottom:8px;">{render_status_badge(quote_res.status)} {render_freshness_badge(quote_res.freshness)}</div>
                    <div style="font-size:1.0rem; color:#f8fafc; font-weight:600;">{selected_etf} Quote Unavailable</div>
                    <div style="font-size:0.85rem; margin-top:6px;">{quote_res.error_message or 'Configure your Finnhub API Key above to retrieve live quotes.'}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # SUBSECTION 2: ETF HISTORICAL PERFORMANCE
    # -------------------------------------------------------------------------
    with sub2:
        st.markdown(f"#### 2. Historical Price Performance: {selected_etf}")
        st.markdown("Daily price observations retrieved via Alpha Vantage API.")

        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            time_filter = st.radio(
                "Select Historical Lookback Range",
                options=["1M", "3M", "6M", "1Y", "5Y", "All Available"],
                index=3,
                horizontal=True,
            )
        with col_h2:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh_hist = st.button("🔄 Refresh Historical", key="btn_refresh_hist", use_container_width=True)

        outputsize_param = "full" if time_filter in ("5Y", "All Available") else "compact"

        with st.spinner(f"Retrieving daily historical time series for {selected_etf} from Alpha Vantage..."):
            hist_res = LiveAPIManager.fetch_etf_daily_series(
                symbol=selected_etf,
                alphavantage_key=st.session_state["live_alphavantage_key"],
                outputsize=outputsize_param,
                use_cache=not refresh_hist,
            )

        if hist_res.status == ProviderStatus.CONNECTED and hist_res.dataframe is not None and not hist_res.dataframe.empty:
            df_hist = hist_res.dataframe.copy()
            df_hist["Date"] = pd.to_datetime(df_hist["Date"])
            df_hist = df_hist.sort_values(by="Date", ascending=True).reset_index(drop=True)

            # Filter by date range
            latest_date = df_hist["Date"].max()
            if time_filter == "1M":
                cutoff = latest_date - timedelta(days=30)
            elif time_filter == "3M":
                cutoff = latest_date - timedelta(days=90)
            elif time_filter == "6M":
                cutoff = latest_date - timedelta(days=180)
            elif time_filter == "1Y":
                cutoff = latest_date - timedelta(days=365)
            elif time_filter == "5Y":
                cutoff = latest_date - timedelta(days=365 * 5)
            else:
                cutoff = df_hist["Date"].min()

            df_filtered = df_hist[df_hist["Date"] >= cutoff].copy()
            if df_filtered.empty:
                df_filtered = df_hist.copy()

            # Plotly historical performance chart with subplots for volume
            fig_perf = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.75, 0.25],
                subplot_titles=(f"{selected_etf} Daily Price Trajectory ($ USD)", "Daily Trading Volume"),
            )

            fig_perf.add_trace(
                go.Scatter(
                    x=df_filtered["Date"],
                    y=df_filtered["Close"],
                    mode="lines",
                    name=f"{selected_etf} Close",
                    line=dict(color="#38bdf8", width=2.5),
                    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig_perf.add_trace(
                go.Bar(
                    x=df_filtered["Date"],
                    y=df_filtered["Volume"],
                    name="Volume",
                    marker=dict(color="rgba(148, 163, 184, 0.4)"),
                    hovertemplate="<b>Volume:</b> %{y:,.0f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            fig_perf.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=40, r=40, t=50, b=30),
                height=480,
                showlegend=False,
            )
            fig_perf.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_perf.update_yaxes(title_text="Volume", row=2, col=1)

            st.plotly_chart(fig_perf, use_container_width=True)

            # Metadata header
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; align-items:center; font-size:0.85rem; color:#94a3b8; margin-bottom:12px;'>"
                f"<span><b>Source:</b> {hist_res.data_source}</span>"
                f"<span><b>Observations Displayed:</b> {len(df_filtered)} / {len(df_hist)} total</span>"
                f"<span><b>Retrieved:</b> {hist_res.retrieval_timestamp} | {render_freshness_badge(hist_res.freshness)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.expander(f"📋 View Tabular Historical Observations ({len(df_filtered)} rows)"):
                df_display = df_filtered.sort_values(by="Date", ascending=False).copy()
                df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                f"""<div style="padding:24px; text-align:center; color:#94a3b8; background:rgba(30,41,59,0.5); border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                    <div style="margin-bottom:8px;">{render_status_badge(hist_res.status)} {render_freshness_badge(hist_res.freshness)}</div>
                    <div style="font-size:1.0rem; color:#f8fafc; font-weight:600;">{selected_etf} Historical Data Unavailable</div>
                    <div style="font-size:0.85rem; margin-top:6px;">{hist_res.error_message or 'Configure your Alpha Vantage API Key above to retrieve historical observations.'}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # SUBSECTION 3: MARKET TRENDS & TECHNICAL INDICATORS
    # -------------------------------------------------------------------------
    with sub3:
        st.markdown(f"#### 3. Market Trends & Technical Indicators: {selected_etf}")
        st.markdown("Empirical trend analytics computed from retrieved daily price observations.")

        if hist_res.status == ProviderStatus.CONNECTED and hist_res.dataframe is not None and not hist_res.dataframe.empty:
            df_enriched, t_metrics = compute_technical_indicators(hist_res.dataframe)

            # Indicator Summary Metrics Row
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Market Trend Classification</div>
                        <div style="margin-top:6px; margin-bottom:4px;">{render_trend_badge(t_metrics['trend_classification'])}</div>
                        <div class="metric-sub">Direction: {t_metrics['recent_direction']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with t_col2:
                rsi_val = t_metrics.get("rsi_14_numeric")
                rsi_str = t_metrics["rsi_14"]
                rsi_color = "#f87171" if (rsi_val and rsi_val > 70) else "#34d399" if (rsi_val and rsi_val < 30) else "#38bdf8"
                rsi_cond = "Overbought (>70)" if (rsi_val and rsi_val > 70) else "Oversold (<30)" if (rsi_val and rsi_val < 30) else "Neutral Range (30-70)"
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">RSI (14 Periods)</div>
                        <div class="metric-value" style="color:{rsi_color};">{rsi_str}</div>
                        <div class="metric-sub">{rsi_cond}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with t_col3:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Rolling Volatility (20-Day)</div>
                        <div class="metric-value">{t_metrics['rolling_vol_20']}</div>
                        <div class="metric-sub">Annualized 252-day basis</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with t_col4:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Daily Price Return</div>
                        <div class="metric-value" style="color:{'#34d399' if (t_metrics['daily_return_pct'] or 0) >= 0 else '#f87171'};">
                            {f"{t_metrics['daily_return_pct']:+.2f}%" if t_metrics['daily_return_pct'] is not None else 'N/A'}
                        </div>
                        <div class="metric-sub">Latest Session Move</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Moving Average Comparison Table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Moving Average Signals")
            ma_df = pd.DataFrame([
                {
                    "Indicator": "20-Day Simple Moving Average (SMA 20)",
                    "Indicator Value": t_metrics["sma_20"],
                    "Current Price vs SMA": f"${t_metrics['latest_close']:.2f}" if t_metrics['latest_close'] else "N/A",
                    "Position": "Above SMA 20 (Bullish)" if (t_metrics['latest_close'] and t_metrics['sma_20_numeric'] and t_metrics['latest_close'] > t_metrics['sma_20_numeric']) else "Below SMA 20 (Bearish)" if t_metrics['sma_20_numeric'] else "Insufficient data",
                },
                {
                    "Indicator": "50-Day Simple Moving Average (SMA 50)",
                    "Indicator Value": t_metrics["sma_50"],
                    "Current Price vs SMA": f"${t_metrics['latest_close']:.2f}" if t_metrics['latest_close'] else "N/A",
                    "Position": "Above SMA 50 (Bullish)" if (t_metrics['latest_close'] and t_metrics['sma_50_numeric'] and t_metrics['latest_close'] > t_metrics['sma_50_numeric']) else "Below SMA 50 (Bearish)" if t_metrics['sma_50_numeric'] else "Insufficient data",
                },
                {
                    "Indicator": "200-Day Simple Moving Average (SMA 200)",
                    "Indicator Value": t_metrics["sma_200"],
                    "Current Price vs SMA": f"${t_metrics['latest_close']:.2f}" if t_metrics['latest_close'] else "N/A",
                    "Position": "Above SMA 200 (Long-Term Bullish)" if (t_metrics['latest_close'] and t_metrics['sma_200_numeric'] and t_metrics['latest_close'] > t_metrics['sma_200_numeric']) else "Below SMA 200 (Long-Term Bearish)" if t_metrics['sma_200_numeric'] else "Insufficient data",
                },
            ])
            st.dataframe(ma_df, use_container_width=True, hide_index=True)

            # Interactive Plotly Overlay Chart: Price + SMA 20 + SMA 50 + RSI Subplot
            fig_tech = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.10,
                row_heights=[0.70, 0.30],
                subplot_titles=(f"{selected_etf} Price & Moving Average Overlays", "RSI (14) Momentum Oscillator"),
            )

            fig_tech.add_trace(
                go.Scatter(x=df_enriched["Date"], y=df_enriched["Close"], mode="lines", name="Close Price", line=dict(color="#f8fafc", width=2)),
                row=1, col=1,
            )
            if "SMA_20" in df_enriched.columns and df_enriched["SMA_20"].notna().any():
                fig_tech.add_trace(
                    go.Scatter(x=df_enriched["Date"], y=df_enriched["SMA_20"], mode="lines", name="SMA 20", line=dict(color="#38bdf8", width=1.5)),
                    row=1, col=1,
                )
            if "SMA_50" in df_enriched.columns and df_enriched["SMA_50"].notna().any():
                fig_tech.add_trace(
                    go.Scatter(x=df_enriched["Date"], y=df_enriched["SMA_50"], mode="lines", name="SMA 50", line=dict(color="#f59e0b", width=1.5)),
                    row=1, col=1,
                )

            # RSI trace
            if "RSI_14" in df_enriched.columns and df_enriched["RSI_14"].notna().any():
                fig_tech.add_trace(
                    go.Scatter(x=df_enriched["Date"], y=df_enriched["RSI_14"], mode="lines", name="RSI (14)", line=dict(color="#c084fc", width=1.5)),
                    row=2, col=1,
                )
                # Overbought (70) and Oversold (30) reference lines
                fig_tech.add_hline(y=70, line_dash="dash", line_color="rgba(248,113,113,0.6)", row=2, col=1)
                fig_tech.add_hline(y=30, line_dash="dash", line_color="rgba(52,211,153,0.6)", row=2, col=1)

            fig_tech.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=40, r=40, t=50, b=30),
                height=480,
            )
            fig_tech.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

            st.plotly_chart(fig_tech, use_container_width=True)

            # Institutional Disclaimer
            st.warning(f"⚠️ {TECHNICAL_DISCLAIMER}")
        else:
            st.markdown(
                f"""<div style="padding:24px; text-align:center; color:#94a3b8; background:rgba(30,41,59,0.5); border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                    <div style="font-size:1.0rem; color:#f8fafc; font-weight:600;">Technical Indicators Require Price Observations</div>
                    <div style="font-size:0.85rem; margin-top:6px;">Please retrieve historical observations for {selected_etf} in the <b>ETF Historical Performance</b> tab to compute technical indicators.</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # SUBSECTION 4: ECONOMIC INDICATORS (FRED)
    # -------------------------------------------------------------------------
    with sub4:
        st.markdown("#### 4. Macroeconomic Indicators & Federal Reserve Series")
        st.markdown("Direct economic telemetry from Federal Reserve Economic Data (FRED).")

        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            series_options = list(FRED_SERIES_CATALOG.keys())
            chosen_series = st.selectbox(
                "Select Macroeconomic Indicator Series",
                options=series_options,
                index=series_options.index(st.session_state["selected_fred_series"]),
                format_func=lambda s: f"{s}: {FRED_SERIES_CATALOG[s]['name']} ({FRED_SERIES_CATALOG[s]['frequency']})",
                key="fred_selector_sub4",
            )
            st.session_state["selected_fred_series"] = chosen_series
        with col_f2:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh_fred = st.button("🔄 Refresh Series", key="btn_refresh_fred", use_container_width=True)

        meta_fred = FRED_SERIES_CATALOG[chosen_series]

        # Indicator Description Card
        st.markdown(
            f"""<div style="background:rgba(30,41,59,0.6); border:1px solid rgba(148,163,184,0.15); border-radius:8px; padding:12px 18px; margin-bottom:16px;">
                <div style="font-weight:700; color:#38bdf8; font-size:1.05rem;">{chosen_series} — {meta_fred['name']}</div>
                <div style="font-size:0.85rem; color:#cbd5e1; margin-top:4px;">{meta_fred['description']}</div>
                <div style="font-size:0.78rem; color:#94a3b8; margin-top:4px;">
                    <b>Category:</b> {meta_fred['category']} &nbsp;|&nbsp; 
                    <b>Frequency:</b> {meta_fred['frequency']} &nbsp;|&nbsp; 
                    <b>Units:</b> {meta_fred['units']}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.spinner(f"Retrieving {chosen_series} observations from FRED..."):
            fred_res = LiveAPIManager.fetch_economic_indicator(
                series_id=chosen_series,
                fred_key=st.session_state["live_fred_key"],
                limit=60,
                use_cache=not refresh_fred,
            )

        if fred_res.status == ProviderStatus.CONNECTED and fred_res.dataframe is not None and not fred_res.dataframe.empty:
            df_fred = fred_res.dataframe.copy()
            val_col = [c for c in df_fred.columns if c.startswith("Value (")][0]
            df_fred_clean = df_fred.dropna(subset=[val_col]).copy()
            df_fred_clean["Date"] = pd.to_datetime(df_fred_clean["Date"])
            df_fred_clean = df_fred_clean.sort_values(by="Date", ascending=True).reset_index(drop=True)

            latest_val = fred_res.metrics.get("latest_value")

            # KPI metrics
            e_cols = st.columns(3)
            with e_cols[0]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Latest Observation</div>
                        <div class="metric-value">{f"{latest_val:.2f}" if latest_val is not None else 'N/A'} {meta_fred['units']}</div>
                        <div class="metric-sub">Market Date: {fred_res.market_timestamp}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with e_cols[1]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Reporting Frequency</div>
                        <div class="metric-value" style="font-size:1.5rem;">{meta_fred['frequency']}</div>
                        <div class="metric-sub">{len(df_fred_clean)} valid observations</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with e_cols[2]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Telemetry Status</div>
                        <div style="margin-top:6px; margin-bottom:4px;">{render_status_badge(fred_res.status)}</div>
                        <div class="metric-sub">{render_freshness_badge(fred_res.freshness)} | Retrieved: {fred_res.retrieval_timestamp}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Interactive Plotly Chart
            fig_fred = go.Figure()
            fig_fred.add_trace(
                go.Scatter(
                    x=df_fred_clean["Date"],
                    y=df_fred_clean[val_col],
                    mode="lines+markers",
                    name=meta_fred["name"],
                    line=dict(color="#10b981", width=2.5),
                    marker=dict(size=5),
                    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y:.2f}<extra></extra>",
                )
            )
            fig_fred.update_layout(
                title=f"{chosen_series} — {meta_fred['name']} ({meta_fred['frequency']})",
                xaxis_title="Observation Date",
                yaxis_title=meta_fred["units"],
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=40, r=40, t=50, b=40),
                height=400,
            )
            st.plotly_chart(fig_fred, use_container_width=True)

            with st.expander(f"📋 View Tabular FRED Observations ({len(df_fred)} records)"):
                st.dataframe(df_fred, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                f"""<div style="padding:24px; text-align:center; color:#94a3b8; background:rgba(30,41,59,0.5); border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                    <div style="margin-bottom:8px;">{render_status_badge(fred_res.status)} {render_freshness_badge(fred_res.freshness)}</div>
                    <div style="font-size:1.0rem; color:#f8fafc; font-weight:600;">{chosen_series} Series Unavailable</div>
                    <div style="font-size:0.85rem; margin-top:6px;">{fred_res.error_message or 'Configure your FRED API Key above to retrieve economic series.'}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # SUBSECTION 5: FINANCIAL NEWS
    # -------------------------------------------------------------------------
    with sub5:
        st.markdown("#### 5. Financial Market News Telemetry")
        st.markdown("Curated market headlines and company news feeds via Finnhub.")

        col_n1, col_n2 = st.columns([3, 1])
        with col_n1:
            news_scope = st.radio(
                "News Scope",
                options=["General Market Headlines", f"Selected ETF: {selected_etf}"],
                horizontal=True,
            )
        with col_n2:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh_news = st.button("🔄 Refresh News", key="btn_refresh_news", use_container_width=True)

        target_sym = selected_etf if news_scope != "General Market Headlines" else None

        with st.spinner("Fetching latest financial news from Finnhub..."):
            news_res = LiveAPIManager.fetch_news(
                category="general",
                symbol=target_sym,
                finnhub_key=st.session_state["live_finnhub_key"],
                limit=15,
                use_cache=not refresh_news,
            )

        if news_res.status == ProviderStatus.CONNECTED and news_res.metrics.get("articles"):
            articles = news_res.metrics["articles"]
            st.markdown(
                f"<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:12px;'>"
                f"Displaying <b>{len(articles)} articles</b> from Finnhub &nbsp;|&nbsp; "
                f"Retrieved: `{news_res.retrieval_timestamp}` &nbsp;|&nbsp; "
                f"{render_freshness_badge(news_res.freshness)}"
                f"</div>",
                unsafe_allow_html=True,
            )

            for art in articles:
                with st.container():
                    st.markdown(
                        f"""<div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.15); border-radius:8px; padding:14px 18px; margin-bottom:12px;">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                <div style="font-weight:700; font-size:1.02rem; color:#f8fafc; line-height:1.4;">
                                    <a href="{art['url']}" target="_blank" style="color:#38bdf8; text-decoration:none;">{art['headline']}</a>
                                </div>
                                <span style="font-size:0.75rem; color:#94a3b8; background:rgba(15,23,42,0.8); padding:2px 8px; border-radius:4px; margin-left:12px; white-space:nowrap;">
                                    {art['source']}
                                </span>
                            </div>
                            <div style="font-size:0.85rem; color:#cbd5e1; margin-top:6px; line-height:1.5;">
                                {art['summary'][:280] + '...' if len(art['summary']) > 280 else art['summary']}
                            </div>
                            <div style="font-size:0.76rem; color:#64748b; margin-top:8px;">
                                <b>Published:</b> {art['datetime']} &nbsp;|&nbsp; 
                                <b>Related:</b> {art['related']} &nbsp;|&nbsp; 
                                <a href="{art['url']}" target="_blank" style="color:#38bdf8; text-decoration:underline;">Read Full Article at Source ↗</a>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f"""<div style="padding:24px; text-align:center; color:#94a3b8; background:rgba(30,41,59,0.5); border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                    <div style="margin-bottom:8px;">{render_status_badge(news_res.status)} {render_freshness_badge(news_res.freshness)}</div>
                    <div style="font-size:1.0rem; color:#f8fafc; font-weight:600;">Financial News Feed Unavailable</div>
                    <div style="font-size:0.85rem; margin-top:6px;">{news_res.error_message or 'Configure your Finnhub API Key above to stream live financial news.'}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # SUBSECTION 6: SEC FILINGS
    # -------------------------------------------------------------------------
    with sub6:
        st.markdown("#### 6. SEC EDGAR Corporate Regulatory Submissions")
        st.markdown("Real-time filings retrieved directly from the SEC public submissions API.")

        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            sec_keys = list(SEC_ENTITY_CATALOG.keys())
            chosen_cik = st.selectbox(
                "Select Filer Entity",
                options=sec_keys,
                index=sec_keys.index(st.session_state["selected_sec_cik"]),
                format_func=lambda c: f"{SEC_ENTITY_CATALOG[c]['name']} ({SEC_ENTITY_CATALOG[c]['ticker']}) — CIK {c}",
                key="sec_selector_sub6",
            )
            st.session_state["selected_sec_cik"] = chosen_cik
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh_sec = st.button("🔄 Refresh Filings", key="btn_refresh_sec", use_container_width=True)

        entity_meta = SEC_ENTITY_CATALOG[chosen_cik]

        st.markdown(
            f"""<div style="background:rgba(30,41,59,0.6); border:1px solid rgba(148,163,184,0.15); border-radius:8px; padding:12px 18px; margin-bottom:16px;">
                <div style="font-weight:700; color:#38bdf8; font-size:1.05rem;">{entity_meta['name']} (CIK {chosen_cik})</div>
                <div style="font-size:0.85rem; color:#cbd5e1; margin-top:4px;"><b>Entity Type:</b> {entity_meta['type']} &nbsp;|&nbsp; <b>Role:</b> {entity_meta['role']}</div>
                <div style="font-size:0.78rem; color:#94a3b8; margin-top:4px;">
                    SEC fair-access compliance active under User-Agent: <code>{st.session_state['live_sec_ua']}</code>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.spinner(f"Querying SEC EDGAR for CIK {chosen_cik} filings..."):
            sec_res = LiveAPIManager.fetch_sec_submissions(
                cik=chosen_cik,
                user_agent=st.session_state["live_sec_ua"],
                limit=15,
                use_cache=not refresh_sec,
            )

        if sec_res.status == ProviderStatus.CONNECTED and sec_res.dataframe is not None and not sec_res.dataframe.empty:
            df_sec = sec_res.dataframe

            st.markdown(
                f"<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:12px;'>"
                f"Retrieved <b>{len(df_sec)} recent filings</b> &nbsp;|&nbsp; "
                f"Status: {render_status_badge(sec_res.status)} &nbsp;|&nbsp; "
                f"{render_freshness_badge(sec_res.freshness)} &nbsp;|&nbsp; "
                f"Retrieved: `{sec_res.retrieval_timestamp}`"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Render filings table
            st.dataframe(
                df_sec,
                column_config={
                    "Document Link": st.column_config.LinkColumn(
                        "Document Link",
                        help="Direct link to official SEC EDGAR archive document",
                        validate=r"^https?://.*",
                        display_text="View Filing ↗",
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown(
                f"""<div style="padding:24px; text-align:center; color:#94a3b8; background:rgba(30,41,59,0.5); border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                    <div style="margin-bottom:8px;">{render_status_badge(sec_res.status)} {render_freshness_badge(sec_res.freshness)}</div>
                    <div style="font-size:1.0rem; color:#f8fafc; font-weight:600;">SEC Filings Telemetry Unavailable</div>
                    <div style="font-size:0.85rem; margin-top:6px;">{sec_res.error_message or 'Ensure a valid SEC User-Agent is configured above.'}</div>
                </div>""",
                unsafe_allow_html=True,
            )
