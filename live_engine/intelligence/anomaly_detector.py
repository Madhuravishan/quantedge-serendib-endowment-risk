"""
Statistical Anomaly Detection Module
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 3)

Identifies empirical statistical market anomalies using rolling z-scores and boundary thresholds:
1. Unusually large daily price returns (|Z| >= 2.0)
2. Abnormal trading volume surges (Volume >= 2.0x 20-day mean or Z_vol >= 2.0)
3. Volatility spikes (Current 20-day rolling vol >= 1.5x historical baseline)
4. Extreme deviations from intermediate 50-day moving average (|Z_dev| >= 2.0)
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class AnomalyAlert:
    """Standardized representation of a detected statistical market anomaly."""
    asset: str
    anomaly_type: str  # "Price Return Shock", "Abnormal Volume Surge", "Volatility Spike", "Extreme Moving Average Dislocation"
    severity: str      # "High", "Medium", "Low"
    date: str
    current_value: float
    baseline_value: float
    z_score: Optional[float]
    threshold_description: str
    narrative: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Asset": self.asset,
            "Anomaly Type": self.anomaly_type,
            "Severity": self.severity,
            "Date": self.date,
            "Observed Value": f"{self.current_value:.2f}",
            "Baseline": f"{self.baseline_value:.2f}",
            "Z-Score": f"{self.z_score:+.2f}σ" if self.z_score is not None else "N/A",
            "Threshold": self.threshold_description,
            "Description": self.narrative,
        }


def detect_asset_anomalies(
    symbol: str,
    df_history: pd.DataFrame,
    z_threshold_price: float = 2.0,
    volume_multiplier_threshold: float = 2.0,
    vol_spike_threshold: float = 1.5,
) -> List[AnomalyAlert]:
    """
    Detect statistical anomalies in an asset's historical price & volume time series.
    
    Parameters
    ----------
    symbol : str
        Asset ticker (e.g. 'SPY').
    df_history : pd.DataFrame
        DataFrame with 'Date', 'Close', and optionally 'Volume'.
    z_threshold_price : float
        Z-score threshold for daily return shocks (default: 2.0).
    volume_multiplier_threshold : float
        Multiple of 20-day moving average volume to trigger volume anomaly (default: 2.0).
    vol_spike_threshold : float
        Multiple of 60-day baseline volatility to trigger volatility spike (default: 1.5).
        
    Returns
    -------
    List[AnomalyAlert]
        List of identified anomalies for the asset.
    """
    alerts: List[AnomalyAlert] = []
    if df_history is None or df_history.empty or "Close" not in df_history.columns or len(df_history) < 25:
        return alerts

    df = df_history.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Return"] = df["Close"].pct_change()

    latest_date_str = df["Date"].iloc[-1].strftime("%Y-%m-%d") if "Date" in df.columns else "Latest"
    latest_close = float(df["Close"].iloc[-1])
    latest_ret = float(df["Return"].iloc[-1]) if pd.notna(df["Return"].iloc[-1]) else 0.0

    # 1. Price Return Anomaly (Rolling 30-day Z-score)
    ret_window = df["Return"].iloc[-31:-1]
    if len(ret_window) >= 20:
        mu_ret = float(ret_window.mean())
        sigma_ret = float(ret_window.std(ddof=1))
        if sigma_ret > 1e-6:
            z_ret = (latest_ret - mu_ret) / sigma_ret
            if abs(z_ret) >= z_threshold_price:
                sev = "High" if abs(z_ret) >= 3.0 else "Medium"
                direction = "drop" if z_ret < 0 else "surge"
                alerts.append(
                    AnomalyAlert(
                        asset=symbol,
                        anomaly_type="Price Return Shock",
                        severity=sev,
                        date=latest_date_str,
                        current_value=latest_ret * 100.0,
                        baseline_value=mu_ret * 100.0,
                        z_score=round(z_ret, 2),
                        threshold_description=f"|Z| >= {z_threshold_price:.1f}σ (Observed: {z_ret:+.2f}σ)",
                        narrative=f"Unusually large daily price {direction} of {latest_ret*100:+.2f}% ({abs(z_ret):.1f} standard deviations from 30-day mean).",
                    )
                )

    # 2. Abnormal Trading Volume (20-day volume multiple)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        vol_clean = df["Volume"].dropna()
        if len(vol_clean) >= 21:
            latest_vol = float(vol_clean.iloc[-1])
            vol_hist_20 = vol_clean.iloc[-21:-1]
            mean_vol_20 = float(vol_hist_20.mean())
            std_vol_20 = float(vol_hist_20.std(ddof=1))

            if mean_vol_20 > 0:
                vol_ratio = latest_vol / mean_vol_20
                z_vol = (latest_vol - mean_vol_20) / std_vol_20 if std_vol_20 > 0 else 0.0
                if vol_ratio >= volume_multiplier_threshold:
                    sev = "High" if vol_ratio >= 3.0 else "Medium"
                    alerts.append(
                        AnomalyAlert(
                            asset=symbol,
                            anomaly_type="Abnormal Volume Surge",
                            severity=sev,
                            date=latest_date_str,
                            current_value=latest_vol,
                            baseline_value=mean_vol_20,
                            z_score=round(z_vol, 2),
                            threshold_description=f"Volume >= {volume_multiplier_threshold:.1f}x 20-day mean ({vol_ratio:.1f}x observed)",
                            narrative=f"Trading volume spiked to {latest_vol:,.0f} shares ({vol_ratio:.1f}x 20-day baseline of {mean_vol_20:,.0f}), indicating heavy institutional turnover.",
                        )
                    )

    # 3. Volatility Spike (20-day rolling vol vs 60-day baseline)
    if len(df) >= 60:
        ret_60 = df["Return"].iloc[-60:]
        ret_20 = df["Return"].iloc[-20:]

        vol_20_ann = float(ret_20.std(ddof=1) * math.sqrt(252) * 100.0)
        vol_60_ann = float(ret_60.std(ddof=1) * math.sqrt(252) * 100.0)

        if vol_60_ann > 0:
            vol_ratio = vol_20_ann / vol_60_ann
            if vol_ratio >= vol_spike_threshold:
                sev = "High" if vol_ratio >= 2.0 else "Medium"
                alerts.append(
                    AnomalyAlert(
                        asset=symbol,
                        anomaly_type="Volatility Spike",
                        severity=sev,
                        date=latest_date_str,
                        current_value=vol_20_ann,
                        baseline_value=vol_60_ann,
                        z_score=round((vol_ratio - 1.0) * 2.0, 2),
                        threshold_description=f"20-day Vol >= {vol_spike_threshold:.1f}x 60-day baseline ({vol_ratio:.2f}x observed)",
                        narrative=f"Realized annualized volatility surged to {vol_20_ann:.1f}% vs 60-day baseline of {vol_60_ann:.1f}% ({vol_ratio:.2f}x expansion).",
                    )
                )

    # 4. Extreme Deviation from 50-Day Moving Average
    if len(df) >= 50:
        sma_50 = float(df["Close"].iloc[-50:].mean())
        std_50 = float(df["Close"].iloc[-50:].std(ddof=1))
        if std_50 > 0:
            z_ma = (latest_close - sma_50) / std_50
            if abs(z_ma) >= 2.2:
                sev = "High" if abs(z_ma) >= 3.0 else "Medium"
                dir_label = "above" if z_ma > 0 else "below"
                alerts.append(
                    AnomalyAlert(
                        asset=symbol,
                        anomaly_type="Extreme Moving Average Dislocation",
                        severity=sev,
                        date=latest_date_str,
                        current_value=latest_close,
                        baseline_value=sma_50,
                        z_score=round(z_ma, 2),
                        threshold_description=f"|Z_dev| >= 2.2σ ({z_ma:+.2f}σ observed)",
                        narrative=f"Price (${latest_close:.2f}) is stretched {abs(z_ma):.1f}σ {dir_label} the 50-day SMA (${sma_50:.2f}), presenting mean-reversion pressure.",
                    )
                )

    return alerts


def scan_universe_anomalies(history_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Scan all 7 ETFs for statistical market anomalies.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with all alerts, breakdown by severity, and counts.
    """
    all_alerts: List[AnomalyAlert] = []
    asset_alert_counts: Dict[str, int] = {}

    for sym in ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]:
        df = history_dict.get(sym)
        alerts = detect_asset_anomalies(sym, df)
        all_alerts.extend(alerts)
        asset_alert_counts[sym] = len(alerts)

    # Sort alerts by severity (High first, then Medium, Low)
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    all_alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

    high_c = sum(1 for a in all_alerts if a.severity == "High")
    med_c = sum(1 for a in all_alerts if a.severity == "Medium")
    low_c = sum(1 for a in all_alerts if a.severity == "Low")

    return {
        "total_anomalies": len(all_alerts),
        "high_severity_count": high_c,
        "medium_severity_count": med_c,
        "low_severity_count": low_c,
        "alerts": all_alerts,
        "asset_counts": asset_alert_counts,
        "has_active_anomalies": len(all_alerts) > 0,
    }
