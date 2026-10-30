"""
Macroeconomic Indicator Intelligence Module
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 3)

Analyzes Federal Reserve Economic Data (FRED) series:
- 3-Month T-Bill (DTB3)
- 10-Year Treasury Yield (DGS10)
- Federal Funds Rate (FEDFUNDS)
- Consumer Price Index (CPIAUCSL)
- Unemployment Rate (UNRATE)

Generates:
- Trajectory and rate-of-change analysis
- Yield curve slope & inversion diagnostics (10Y minus 3M spread)
- Clear separation between Observable Facts and Institutional Interpretations
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

# FRED Series Catalog (Catalog metadata for economic intelligence)
FRED_SERIES_CATALOG: Dict[str, Dict[str, str]] = {
    "DTB3": {
        "name": "3-Month Treasury Bill Secondary Market Rate",
        "description": "Short-term sovereign borrowing yield. Primary institutional proxy for the risk-free rate ($R_f$) and cash equivalent yield.",
        "frequency": "Daily",
        "units": "Percent (%)",
        "category": "Interest Rates",
    },
    "DGS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "description": "Benchmark long-term US sovereign risk-free yield. Foundation for asset pricing models, equity hurdle rates, and discount cash flows.",
        "frequency": "Daily",
        "units": "Percent (%)",
        "category": "Interest Rates",
    },
    "FEDFUNDS": {
        "name": "Federal Funds Effective Rate",
        "description": "Volume-weighted median rate at which commercial banks lend reserve balances to each other overnight. Key indicator of Federal Reserve monetary policy stance.",
        "frequency": "Monthly",
        "units": "Percent (%)",
        "category": "Monetary Policy",
    },
    "CPIAUCSL": {
        "name": "Consumer Price Index for All Urban Consumers",
        "description": "Headline inflation measure tracking price changes for a basket of consumer goods. Direct benchmark for the endowment's 2.5% inflation hurdle.",
        "frequency": "Monthly",
        "units": "Index (1982-1984=100)",
        "category": "Inflation & Prices",
    },
    "UNRATE": {
        "name": "Civilian Unemployment Rate",
        "description": "Percentage of the civilian labor force actively seeking employment. Core macroeconomic indicator of labor market slack and business cycle health.",
        "frequency": "Monthly",
        "units": "Percent (%)",
        "category": "Labor & Growth",
    },
}


@dataclass
class SeriesAnalysis:
    """Detailed analytical interpretation for a specific FRED series."""
    series_id: str
    name: str
    latest_date: str
    latest_value: Optional[float]
    prev_value: Optional[float]
    change_1p: Optional[float]
    trend_direction: str  # "Increasing", "Decreasing", "Stable"
    frequency: str
    units: str
    observations_count: int
    observable_fact: str
    economic_interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Series": self.series_id,
            "Name": self.name,
            "Latest Date": self.latest_date,
            "Value": f"{self.latest_value:.2f} {self.units}" if self.latest_value is not None else "N/A",
            "Change (1-Period)": f"{self.change_1p:+.2f}" if self.change_1p is not None else "N/A",
            "Trend": self.trend_direction,
            "Frequency": self.frequency,
            "Observable Fact": self.observable_fact,
            "Interpretation": self.economic_interpretation,
        }


@dataclass
class YieldCurveAnalysis:
    """Yield curve slope and inversion analytics (10Y minus 3M)."""
    dgs10_value: Optional[float]
    dtb3_value: Optional[float]
    spread_value: Optional[float]  # DGS10 - DTB3 in percentage points
    spread_bps: Optional[float]    # in basis points
    status: str                    # "Inverted", "Flat", "Normal / Upward Sloping"
    observable_fact: str
    economic_interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "10-Year Yield (DGS10)": f"{self.dgs10_value:.2f}%" if self.dgs10_value is not None else "N/A",
            "3-Month Yield (DTB3)": f"{self.dtb3_value:.2f}%" if self.dtb3_value is not None else "N/A",
            "Spread (10Y - 3M)": f"{self.spread_value:+.2f}% ({self.spread_bps:+.0f} bps)" if self.spread_value is not None else "N/A",
            "Curve Regime": self.status,
            "Observable Fact": self.observable_fact,
            "Economic Interpretation": self.economic_interpretation,
        }


def analyze_single_series(series_id: str, df_series: pd.DataFrame) -> SeriesAnalysis:
    """
    Analyze trajectory, rate-of-change, and macro significance of a single FRED series.
    """
    sid = series_id.strip().upper()
    meta = FRED_SERIES_CATALOG.get(sid, {
        "name": f"FRED Series {sid}",
        "frequency": "Variable",
        "units": "Value",
    })

    if df_series is None or df_series.empty:
        return SeriesAnalysis(
            series_id=sid,
            name=meta.get("name", sid),
            latest_date="N/A",
            latest_value=None,
            prev_value=None,
            change_1p=None,
            trend_direction="Unavailable",
            frequency=meta.get("frequency", "N/A"),
            units=meta.get("units", ""),
            observations_count=0,
            observable_fact=f"No observations available for series {sid}.",
            economic_interpretation="Data unavailable for macro interpretation.",
        )

    # Locate numeric column
    val_cols = [c for c in df_series.columns if "value" in c.lower() or "yield" in c.lower()]
    val_col = val_cols[0] if val_cols else df_series.columns[-1]

    clean_df = df_series.dropna(subset=[val_col]).copy()
    if clean_df.empty:
        return SeriesAnalysis(
            series_id=sid,
            name=meta.get("name", sid),
            latest_date="N/A",
            latest_value=None,
            prev_value=None,
            change_1p=None,
            trend_direction="Unavailable",
            frequency=meta.get("frequency", "N/A"),
            units=meta.get("units", ""),
            observations_count=0,
            observable_fact=f"All observations for {sid} are non-numeric or holiday markers.",
            economic_interpretation="Cannot determine economic trajectory.",
        )

    clean_df["Date"] = pd.to_datetime(clean_df["Date"])
    clean_df = clean_df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    n_obs = len(clean_df)
    latest_row = clean_df.iloc[-1]
    prev_row = clean_df.iloc[-2] if n_obs >= 2 else latest_row

    latest_val = float(latest_row[val_col])
    prev_val = float(prev_row[val_col]) if n_obs >= 2 else latest_val
    latest_date_str = latest_row["Date"].strftime("%Y-%m-%d")
    chg = latest_val - prev_val if n_obs >= 2 else 0.0

    # Trend direction
    if chg > 0.02:
        trend = "Increasing"
    elif chg < -0.02:
        trend = "Decreasing"
    else:
        trend = "Stable / Flat"

    # Domain-specific factual and interpretive statements
    if sid == "DTB3":
        obs_fact = f"3-Month T-Bill yield stands at {latest_val:.2f}% as of {latest_date_str} ({chg:+.2f}% 1-day change)."
        if latest_val > 4.5:
            interp = "Elevated short-term risk-free yield provides substantial cash returns on T-bills, setting a high hurdle rate for equity risk premiums."
        elif latest_val < 2.5:
            interp = "Low short-term yields suppress cash returns, driving institutional capital into higher-yielding equities and real assets."
        else:
            interp = "Moderate short-term yield environment consistent with neutral monetary policy."

    elif sid == "DGS10":
        obs_fact = f"10-Year Benchmark Treasury rate closed at {latest_val:.2f}% on {latest_date_str} ({chg:+.2f}% change)."
        if chg > 0.10:
            interp = "Rapid rise in benchmark borrowing costs increases equity discount rates, creating valuation compression headwinds for duration assets like XLK."
        elif chg < -0.10:
            interp = "Easing benchmark yields support equity multiples and fixed income bond prices (AGG capital appreciation)."
        else:
            interp = "Stable long-term sovereign yield indicates anchored inflation and growth expectations."

    elif sid == "FEDFUNDS":
        obs_fact = f"Federal Funds Effective Rate reported at {latest_val:.2f}% for {latest_date_str}."
        if trend == "Increasing":
            interp = "Active monetary policy tightening restricts interbank liquidity to cool aggregate demand."
        elif trend == "Decreasing":
            interp = "Monetary policy easing cycle underway, lowering borrowing costs across commercial credit."
        else:
            interp = "Central bank policy rate pause; markets anticipate potential future pivot based on upcoming inflation data."

    elif sid == "CPIAUCSL":
        pct_chg_ann = ((latest_val / prev_val) ** 12 - 1) * 100.0 if prev_val > 0 else 0.0
        obs_fact = f"Consumer Price Index stands at {latest_val:.2f} for {latest_date_str} (Annualized monthly run-rate: {pct_chg_ann:+.2f}%)."
        if pct_chg_ann > 3.5:
            interp = "Elevated inflation exceeds the endowment's 2.5% CPI hurdle, requiring real-asset hedging through Gold (GLD) and Energy (XLE)."
        elif pct_chg_ann < 2.0:
            interp = "Inflation remains below historical averages, supporting fixed-income purchasing power and real distribution sustainability."
        else:
            interp = "Inflation trajectory aligns closely with central bank targets and the endowment's 2.5% baseline assumption."

    elif sid == "UNRATE":
        obs_fact = f"Civilian Unemployment Rate is {latest_val:.1f}% as of {latest_date_str} ({chg:+.1f}% month-over-month)."
        if latest_val < 4.0:
            interp = "Very tight labor market with low unemployment; supportive of consumer spending but creates wage pressure."
        elif latest_val > 5.0:
            interp = "Elevated labor market slack signaling cyclical economic slowdown or recessionary risks."
        else:
            interp = "Healthy, balanced labor market near the natural rate of unemployment (NAIRU)."
    else:
        obs_fact = f"{meta.get('name', sid)} stands at {latest_val:.2f} as of {latest_date_str}."
        interp = f"Macroeconomic indicator tracking {trend.lower()} trajectory."

    return SeriesAnalysis(
        series_id=sid,
        name=meta.get("name", sid),
        latest_date=latest_date_str,
        latest_value=latest_val,
        prev_value=prev_val,
        change_1p=chg,
        trend_direction=trend,
        frequency=meta.get("frequency", "N/A"),
        units=meta.get("units", ""),
        observations_count=n_obs,
        observable_fact=obs_fact,
        economic_interpretation=interp,
    )


def analyze_yield_curve_slope(
    dtb3_series: Any = None,
    dgs10_series: Any = None,
    dtb3_value: Optional[float] = None,
    dgs10_value: Optional[float] = None,
) -> YieldCurveAnalysis:
    """
    Evaluate the 10-Year minus 3-Month Treasury yield spread (T10Y3M).
    Accepts either DataFrames or pre-extracted float values.
    """
    dtb3_val = float(dtb3_value) if dtb3_value is not None else None
    dgs10_val = float(dgs10_value) if dgs10_value is not None else None

    if dtb3_val is None:
        if isinstance(dtb3_series, (int, float)):
            dtb3_val = float(dtb3_series)
        elif isinstance(dtb3_series, pd.DataFrame) and not dtb3_series.empty:
            cols = [c for c in dtb3_series.columns if "value" in c.lower() or "yield" in c.lower()]
            if cols:
                c_df = dtb3_series.dropna(subset=[cols[0]]).copy()
                if not c_df.empty:
                    if "Date" in c_df.columns:
                        c_df["Date"] = pd.to_datetime(c_df["Date"])
                        c_df = c_df.sort_values(by="Date", ascending=True).reset_index(drop=True)
                    dtb3_val = float(c_df.iloc[-1][cols[0]])

    if dgs10_val is None:
        if isinstance(dgs10_series, (int, float)):
            dgs10_val = float(dgs10_series)
        elif isinstance(dgs10_series, pd.DataFrame) and not dgs10_series.empty:
            cols = [c for c in dgs10_series.columns if "value" in c.lower() or "yield" in c.lower()]
            if cols:
                c_df = dgs10_series.dropna(subset=[cols[0]]).copy()
                if not c_df.empty:
                    if "Date" in c_df.columns:
                        c_df["Date"] = pd.to_datetime(c_df["Date"])
                        c_df = c_df.sort_values(by="Date", ascending=True).reset_index(drop=True)
                    dgs10_val = float(c_df.iloc[-1][cols[0]])

    if dtb3_val is None or dgs10_val is None:
        return YieldCurveAnalysis(
            dgs10_value=dgs10_val,
            dtb3_value=dtb3_val,
            spread_value=None,
            spread_bps=None,
            status="Unavailable",
            observable_fact="Insufficient yield data to compute 10Y-3M spread.",
            economic_interpretation="Ensure both DTB3 and DGS10 series are populated.",
        )

    spread = round(dgs10_val - dtb3_val, 2)
    bps = round(spread * 100.0, 0)

    if spread < -0.10:
        status = "Inverted Yield Curve"
        obs = f"10-Year yield ({dgs10_val:.2f}%) trades below 3-Month yield ({dtb3_val:.2f}%), creating an inversion of {bps:.0f} bps."
        interp = (
            "Yield curve inversion is historically one of the most reliable leading indicators of macroeconomic recession (12-18 month lag). "
            "In this regime, unhedged high-beta equities face earnings compression, warranting protective allocations in AGG bonds and GLD safe-haven assets."
        )
    elif spread < 0.15:
        status = "Flat Yield Curve"
        obs = f"10-Year yield ({dgs10_val:.2f}%) and 3-Month yield ({dtb3_val:.2f}%) are virtually identical (Spread: {bps:+.0f} bps)."
        interp = "A flat yield curve reflects transitional macroeconomic uncertainty, with market participants pricing in imminent monetary policy shifts."
    else:
        status = "Normal / Upward Sloping"
        obs = f"10-Year yield ({dgs10_val:.2f}%) exceeds 3-Month yield ({dtb3_val:.2f}%) by {bps:+.0f} bps."
        interp = "Normal upward-sloping yield curve indicates healthy term premium and expectations of economic growth."

    return YieldCurveAnalysis(
        dgs10_value=dgs10_val,
        dtb3_value=dtb3_val,
        spread_value=spread,
        spread_bps=bps,
        status=status,
        observable_fact=obs,
        economic_interpretation=interp,
    )


def analyze_all_economic_data(fred_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Run comprehensive macro intelligence across all FRED series.
    """
    series_analyses: Dict[str, SeriesAnalysis] = {}
    for sid in ["DTB3", "DGS10", "FEDFUNDS", "CPIAUCSL", "UNRATE"]:
        df = fred_data_dict.get(sid)
        series_analyses[sid] = analyze_single_series(sid, df)

    yc_analysis = analyze_yield_curve_slope(
        dtb3_series=fred_data_dict.get("DTB3"),
        dgs10_series=fred_data_dict.get("DGS10"),
    )

    return {
        "series_analyses": series_analyses,
        "yield_curve": yc_analysis,
        "active_series_count": sum(1 for s in series_analyses.values() if s.latest_value is not None),
    }
