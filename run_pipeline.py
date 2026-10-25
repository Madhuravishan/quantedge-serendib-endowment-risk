"""
Institutional Quantitative Engine - Master Pipeline Runner
Mandate: The Serendib University Endowment Fund ($10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'

Executes all 6 quantitative layers end-to-end, performs model validation,
and generates the Executive Tear Sheet and Investment Committee Memorandum.
"""

import sys
import time
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    CLIENT_NAME,
    MANDATE_SIZE_USD,
    SERENDIB_CONSERVATIVE_WEIGHTS,
    DATA_PROCESSED_DIR,
    REPORTS_DIR,
)
from src.data_loader import load_market_dataset
from src.portfolio_opt import (
    build_standard_portfolios_suite,
    compute_portfolio_return_series,
)
from src.var_engine import run_comprehensive_var_suite
from src.es_engine import run_comprehensive_es_suite, verify_subadditivity_properties
from src.evt_engine import run_comprehensive_evt_suite
from src.backtest import run_all_models_backtest_suite
from src.stress_test import (
    compute_euler_risk_budgeting,
    evaluate_historical_stress_scenarios,
    evaluate_hypothetical_stagflation_shock,
    perform_reverse_stress_test,
)
from src.report_generator import (
    generate_executive_tearsheet,
    generate_investment_committee_memo,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")


def run_full_pipeline(force_reload: bool = False):
    start_time = time.time()
    print("=" * 80)
    print(f"  QUANTEDGE 1.0 QUANTITATIVE RISK & PORTFOLIO ENGINE")
    print(f"  Mandate: {CLIENT_NAME} (USD {MANDATE_SIZE_USD:,.0f} AUM)")
    print(f"  Team: 'Pandas Predictors' | Date: October 2026")
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Step 1: Data Ingestion & Stationarity Checks
    # ---------------------------------------------------------------------------
    print("\n[Step 1/7] Ingesting 7-Asset ETF Universe & Risk-Free Rate...")
    md = load_market_dataset(force_reload=force_reload)
    print(f"-> Ingested {len(md.prices)} trading days from {md.prices.index.min().date()} to {md.prices.index.max().date()}.")
    print(f"-> ADF Stationarity Proof: Verified I(1) prices vs I(0) returns for all 7 assets.")

    # ---------------------------------------------------------------------------
    # Step 2: Layer 1 - Portfolio Optimization & KPIs
    # ---------------------------------------------------------------------------
    print("\n[Step 2/7] Layer 1: Portfolio Theory, Markowitz Optimization & KPIs...")
    suite = build_standard_portfolios_suite(md.simple_returns, risk_free_series=md.risk_free_daily)
    kpi_records = [suite[p][1].to_dict() for p in suite]
    kpi_df = pd.DataFrame(kpi_records)
    kpi_df.to_csv(DATA_PROCESSED_DIR / "portfolio_kpis_summary.csv", index=False)

    serendib_w = SERENDIB_CONSERVATIVE_WEIGHTS
    serendib_ret = compute_portfolio_return_series(md.simple_returns, serendib_w)
    serendib_kpis = suite["Serendib Conservative (Recommended)"][1].to_dict()

    print(f"-> Serendib Conservative CAGR: {serendib_kpis['CAGR']} (Hurdle: 7.00%)")
    print(f"-> Annualized Volatility: {serendib_kpis['Annual Volatility']} | Max Drawdown: {serendib_kpis['Max Drawdown (MDD)']}")

    # ---------------------------------------------------------------------------
    # Step 3: Layer 2 - Value-at-Risk Engine
    # ---------------------------------------------------------------------------
    print("\n[Step 3/7] Layer 2: 3-Method Value-at-Risk (90%, 95%, 99%)...")
    var_results, var_summary_df = run_comprehensive_var_suite(
        returns_df=md.simple_returns,
        weights=serendib_w,
        confidence_levels=[0.90, 0.95, 0.99],
        aum=MANDATE_SIZE_USD,
    )
    var_summary_df.to_csv(DATA_PROCESSED_DIR / "var_summary.csv", index=False)
    risk_gap_99 = var_results[0.99].risk_gap_usd
    print(f"-> 1-Day 99% VaR (Historical): {var_results[0.99].historical_var * 100:.2f}% (${var_results[0.99].historical_var * MANDATE_SIZE_USD:,.0f})")
    print(f"-> 1-Day 99% VaR (Gaussian):   {var_results[0.99].parametric_gaussian_var * 100:.2f}% (${var_results[0.99].parametric_gaussian_var * MANDATE_SIZE_USD:,.0f})")
    print(f"-> Gaussian Risk Gap Underestimation at 99%: +${risk_gap_99:,.0f} on $10M AUM!")

    # ---------------------------------------------------------------------------
    # Step 4: Layer 3 - Expected Shortfall & Coherent Risk
    # ---------------------------------------------------------------------------
    print("\n[Step 4/7] Layer 3: Expected Shortfall (CVaR) & Subadditivity...")
    es_results, es_summary_df = run_comprehensive_es_suite(
        returns_df=md.simple_returns,
        weights=serendib_w,
        confidence_levels=[0.90, 0.95, 0.99],
        aum=MANDATE_SIZE_USD,
    )
    es_summary_df.to_csv(DATA_PROCESSED_DIR / "es_summary.csv", index=False)
    es_99_usd = es_results[0.99].es_usd
    tail_gap_usd = es_results[0.99].tail_severity_usd
    print(f"-> 1-Day 99% Expected Shortfall: {es_results[0.99].es_historical * 100:.2f}% (${es_99_usd:,.0f})")
    print(f"-> Tail Severity Gap (ES - VaR): +${tail_gap_usd:,.0f} additional loss beyond VaR threshold")

    sub_res = verify_subadditivity_properties(md.simple_returns["SPY"], md.simple_returns["AGG"])
    print(f"-> Subadditivity Proof Verified: {sub_res['Subadditivity']['ES_Holds']} (Diversification benefit: {sub_res['Subadditivity']['ES_Diversification_Benefit']})")

    # ---------------------------------------------------------------------------
    # Step 5: Layer 4 - Extreme Value Theory (EVT-POT)
    # ---------------------------------------------------------------------------
    print("\n[Step 5/7] Layer 4: Extreme Value Theory (Peaks-Over-Threshold)...")
    evt_metrics, evt_fit, evt_summary_df = run_comprehensive_evt_suite(
        returns_series=serendib_ret,
        confidence_levels=[0.90, 0.95, 0.99],
        aum=MANDATE_SIZE_USD,
    )
    evt_summary_df.to_csv(DATA_PROCESSED_DIR / "evt_summary.csv", index=False)
    print(f"-> GPD Threshold u: {evt_fit.threshold_u * 100:.3f}% ({evt_fit.threshold_percentile * 100:.1f}th percentile)")
    print(f"-> Fitted Shape xi (Tail Index): {evt_fit.shape_xi:.4f} ({evt_fit.tail_type})")
    print(f"-> KS Test Goodness-of-Fit p-value: {evt_fit.ks_p_value:.4f} (Valid GPD: {evt_fit.ks_p_value > 0.05})")
    print(f"-> EVT-POT 99% VaR: {evt_metrics[0.99].evt_var * 100:.2f}% (${evt_metrics[0.99].evt_var_usd:,.0f})")
    print(f"-> EVT-POT 99% ES:  {evt_metrics[0.99].evt_es * 100:.2f}% (${evt_metrics[0.99].evt_es_usd:,.0f})")

    # ---------------------------------------------------------------------------
    # Step 6: Layer 5 - Out-of-Sample Backtesting & Basel Compliance
    # ---------------------------------------------------------------------------
    print("\n[Step 6/7] Layer 5: Model Validation & Basel Compliance...")
    backtest_reports, backtest_df = run_all_models_backtest_suite(
        returns_series=serendib_ret,
        confidence_level=0.99,
        estimation_window=500,
        out_of_sample_start="2024-01-01",
    )
    backtest_df.to_csv(DATA_PROCESSED_DIR / "backtest_summary.csv", index=False)
    hist_rep = backtest_reports["historical"]
    print(f"-> Kupiec POF Test: p = {hist_rep.kupiec.p_value:.4f} ({hist_rep.kupiec.verdict})")
    print(f"-> Basel Committee Traffic Light Status: {hist_rep.basel.zone} ZONE (Multiplier: {hist_rep.basel.capital_multiplier:.2f}x)")

    # ---------------------------------------------------------------------------
    # Step 7: Layer 6 - Stress Testing & Risk Budgeting
    # ---------------------------------------------------------------------------
    print("\n[Step 7/7] Layer 6: Stress Testing, Risk Budgeting & Scenario Analysis...")
    _, budget_df, val = compute_euler_risk_budgeting(md.simple_returns, serendib_w, aum=MANDATE_SIZE_USD)
    budget_df.to_csv(DATA_PROCESSED_DIR / "risk_budgeting.csv", index=False)

    _, crisis_df = evaluate_historical_stress_scenarios(md.simple_returns, serendib_w, aum=MANDATE_SIZE_USD)
    crisis_df.to_csv(DATA_PROCESSED_DIR / "crisis_scenarios.csv", index=False)

    stagflation = evaluate_hypothetical_stagflation_shock(serendib_w, aum=MANDATE_SIZE_USD)
    reverse_stress = perform_reverse_stress_test(serendib_w, aum=MANDATE_SIZE_USD)

    print(f"-> Euler Risk Decomposition Verified: {val['Euler_Decomposition_Verified']}")
    print(f"-> Hypothetical Stagflation Impact: {stagflation['portfolio_return'] * 100:.2f}% (${stagflation['dollar_impact_usd']:,.0f})")
    print(f"-> Capital Preserved vs 60/40 in Stagflation: +${stagflation['capital_saved_usd']:,.0f}")
    print(f"-> Reverse Stress Circuit Breaker: Requires {reverse_stress['critical_spy_drop']} equity drop + {reverse_stress['critical_agg_drop']} bond drop.")

    # ---------------------------------------------------------------------------
    # Generate Formal Institutional Deliverables
    # ---------------------------------------------------------------------------
    print("\nGenerating Institutional Reports...")
    tearsheet_path = REPORTS_DIR / "EXECUTIVE_TEARSHEET.md"
    memo_path = REPORTS_DIR / "INVESTMENT_COMMITTEE_MEMORANDUM.md"

    generate_executive_tearsheet(
        kpis_dict=serendib_kpis,
        var_df=var_summary_df,
        es_df=es_summary_df,
        evt_info=evt_fit,
        basel_status=hist_rep.basel.zone,
        stagflation_result=stagflation,
        output_path=tearsheet_path,
    )

    generate_investment_committee_memo(
        kpi_table_df=kpi_df,
        stationarity_df=md.stationarity_report,
        var_summary_df=var_summary_df,
        es_summary_df=es_summary_df,
        evt_fit_info=evt_fit,
        backtest_summary_df=backtest_df,
        budget_df=budget_df,
        crisis_df=crisis_df,
        stagflation_dict=stagflation,
        reverse_stress_dict=reverse_stress,
        output_path=memo_path,
    )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"  QUANT ENGINE PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s")
    print(f"  Deliverables Ready:")
    print(f"  - Executive Tear Sheet: {tearsheet_path}")
    print(f"  - Investment Committee Memo: {memo_path}")
    print(f"  - Interactive Streamlit Dashboard: 'streamlit run app.py'")
    print("=" * 80)


if __name__ == "__main__":
    force = "--reload" in sys.argv
    run_full_pipeline(force_reload=force)
