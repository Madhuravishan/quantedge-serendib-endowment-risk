"""
Institutional Quantitative Engine Configuration
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'
"""

from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Workspace Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = BASE_DIR / "report"
REPORTS_DIR = REPORT_DIR  # Alias for backward compatibility
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Client Mandate Specification
# ---------------------------------------------------------------------------
CLIENT_NAME = "Serendib University Endowment Fund"
MANDATE_SIZE_USD = 10_000_000.0  # $10 Million AUM
ANNUAL_SPENDING_RATE = 0.045      # 4.5% target annual real payout for scholarships & ops
INFLATION_BENCHMARK = 0.025       # 2.5% long-term expected CPI
REQUIRED_RETURN_NOMINAL = ANNUAL_SPENDING_RATE + INFLATION_BENCHMARK  # 7.0% nominal hurdle
MAX_TOLERABLE_DRAWDOWN = 0.15     # 15.0% maximum peak-to-trough drawdown threshold
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Prescribed 7-ETF Universe & Meta Data
# ---------------------------------------------------------------------------
TICKERS: List[str] = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]
BENCHMARK_EQUITY = "SPY"
BENCHMARK_BOND = "AGG"
RISK_FREE_FRED_ID = "DTB3"  # 3-Month Treasury Bill: Secondary Market Rate (FRED)
RISK_FREE_YF_SYMBOL = "^IRX" # 13-Week Treasury Bill (Yahoo Finance fallback)

ASSET_METADATA: Dict[str, Dict[str, str]] = {
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": "US Large-Cap Core Equity",
        "role": "Core domestic capital compounding and real purchasing power growth.",
        "color": "#1f77b4",
    },
    "EFA": {
        "name": "iShares MSCI EAFE ETF",
        "asset_class": "Developed Markets Equities",
        "role": "International developed market geographical and currency diversifier.",
        "color": "#aec7e8",
    },
    "EEM": {
        "name": "iShares MSCI Emerging Markets ETF",
        "asset_class": "Emerging Market Equities",
        "role": "High-beta emerging economic growth kicker (capped exposure).",
        "color": "#ff7f0e",
    },
    "AGG": {
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "asset_class": "US Investment Grade Fixed Income",
        "role": "Defensive anchor, capital preservation, and liquid operational yield.",
        "color": "#2ca02c",
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "asset_class": "Precious Metals / Real Asset",
        "role": "Safe haven, tail-risk shock absorber, and monetary debasement hedge.",
        "color": "#d62728",
    },
    "XLE": {
        "name": "Energy Select Sector SPDR Fund",
        "asset_class": "Natural Resources / Energy Equities",
        "role": "Commodity inflation and geopolitical stagflation hedge.",
        "color": "#9467bd",
    },
    "XLK": {
        "name": "Technology Select Sector SPDR Fund",
        "asset_class": "US Technology Sector",
        "role": "Secular innovation growth factor (tactical/stress testing).",
        "color": "#8c564b",
    },
}

# ---------------------------------------------------------------------------
# Prescribed Benchmark & Model Portfolios
# ---------------------------------------------------------------------------
# Serendib Conservative Allocation: 
# Human-justified endowment allocation balancing 4.5% payout with 15% drawdown cap.
SERENDIB_CONSERVATIVE_WEIGHTS: Dict[str, float] = {
    "AGG": 0.35,  # 35% Fixed Income anchor (low-volatility stability)
    "SPY": 0.25,  # 25% Core US large-cap equity
    "GLD": 0.15,  # 15% Real asset & geopolitical hedge
    "EFA": 0.10,  # 10% Developed international diversification
    "XLE": 0.10,  # 10% Commodity inflation / stagflation insurance
    "EEM": 0.05,  # 5% Controlled emerging market exposure
    "XLK": 0.00,  # 0% Direct tech (already ~30% embedded inside SPY)
}

TARGET_WEIGHTS = SERENDIB_CONSERVATIVE_WEIGHTS
TICKER_NAMES = {t: ASSET_METADATA[t]["name"] for t in ASSET_METADATA}

CLASSIC_60_40_WEIGHTS: Dict[str, float] = {
    "SPY": 0.60,
    "AGG": 0.40,
    "EFA": 0.00,
    "EEM": 0.00,
    "GLD": 0.00,
    "XLE": 0.00,
    "XLK": 0.00,
}

EQUAL_WEIGHTS: Dict[str, float] = {
    t: 1.0 / len(TICKERS) for t in TICKERS
}

# ---------------------------------------------------------------------------
# Historical Analysis & Out-of-Sample Split Windows
# ---------------------------------------------------------------------------
FULL_SAMPLE_START = "2006-01-01"
FULL_SAMPLE_END = "2026-10-20"

IN_SAMPLE_TRAIN_END = "2023-12-31"   # In-sample calibration window (2006 to 2023)
OUT_OF_SAMPLE_START = "2024-01-01"  # Out-of-sample backtesting window (2024 to present)

# Historical Stress Period Windows
STRESS_PERIODS: Dict[str, Dict[str, str]] = {
    "2008_GFC": {
        "name": "2008 Global Financial Crisis (Lehman Crash)",
        "start": "2008-09-01",
        "end": "2008-11-30",
        "narrative": "Subprime mortgage collapse, Lehman Brothers bankruptcy, and global interbank credit freeze.",
    },
    "2020_COVID": {
        "name": "March 2020 COVID-19 Liquidity Shock",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "narrative": "Global pandemic lockdown, rapid dash for cash, extreme equity sell-off, and cross-asset volatility spike.",
    },
    "2022_RATES": {
        "name": "2022 Inflation & Aggressive Rate Hike Shock",
        "start": "2022-01-03",
        "end": "2022-10-14",
        "narrative": "Surging global inflation, Russia-Ukraine war, aggressive Fed tightening, simultaneous stock-bond correlation breakdown.",
    },
}

# Hypothetical Scenario: 1970s-Style Geopolitical Stagflation Shock
HYPOTHETICAL_STAGFLATION_SHOCKS: Dict[str, float] = {
    "SPY": -0.20,  # US Equities de-rate on compressed multiples & earnings squeeze
    "EFA": -0.22,  # Developed international hit by energy import dependence
    "EEM": -0.25,  # Emerging markets suffer USD surge and high debt service
    "AGG": -0.12,  # Core bonds drop as yields spike across the curve
    "GLD": +0.15,  # Gold surges as sovereign store-of-value & inflation hedge
    "XLE": +0.25,  # Energy equities surge on crude supply constriction
    "XLK": -0.28,  # High-duration tech assets heavily devalued by discount rates
}

# ---------------------------------------------------------------------------
# Basel Regulatory Backtesting Standards (99% 1-Day VaR, 250 Trading Days)
# ---------------------------------------------------------------------------
BASEL_WINDOW_DAYS = 250
BASEL_CONFIDENCE_LEVEL = 0.99

# Traffic Light Zones (Basel Committee on Banking Supervision, 1996)
BASEL_ZONES = {
    "GREEN": {
        "max_exceptions": 4,
        "multiplier": 3.00,
        "status": "Acceptable",
        "description": "Model is deemed accurate and mathematically sound. No capital penalty.",
        "color": "#10b981",
    },
    "YELLOW": {
        "min_exceptions": 5,
        "max_exceptions": 9,
        "multiplier_range": (3.40, 3.85),
        "status": "Supervisory Monitoring",
        "description": "Supervisory inquiry triggered. Capital multiplier scaled up due to potential under-forecasting.",
        "color": "#f59e0b",
    },
    "RED": {
        "min_exceptions": 10,
        "multiplier": 4.00,
        "status": "Model Rejected",
        "description": "Model definitively rejected. Automatic supervisory intervention and immediate recalibration.",
        "color": "#ef4444",
    },
}

# ---------------------------------------------------------------------------
# Quantitative Engine Defaults
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MONTE_CARLO_SIMULATIONS = 10_000
DEFAULT_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
EVT_DEFAULT_PERCENTILE_THRESHOLD = 0.95
