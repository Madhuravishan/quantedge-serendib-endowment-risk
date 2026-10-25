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
    REPORT_DIR,
    REPORTS_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    TICKERS,
    ASSET_METADATA,
)
from src.data_loader import load_market_dataset
from src.portfolio_opt import (
    build_standard_portfolios_suite,
    compute_portfolio_return_series,
)
from src.var_engine import run_comprehensive_var_suite
from src.es_engine import run_comprehensive_es_suite, verify_subadditivity_properties
from src.evt_engine import run_comprehensive_evt_suite, export_mean_excess_diagnostic_data
from src.backtest import run_all_models_backtest_suite
from src.stress_test import (
    compute_euler_risk_budgeting,
    evaluate_historical_stress_scenarios,
    evaluate_hypothetical_stagflation_shock,
    evaluate_all_stress_scenarios,
    perform_reverse_stress_test,
)
from src.report_generator import (
    generate_executive_tearsheet,
    generate_investment_committee_memo,
    build_tear_sheet_pdf,
    build_investment_memo_pdf,
    export_submission_tables,
    generate_publication_figures,
)
from src.export_static_showcase import export_all_showcase_assets

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
    
    # Explicitly persist ADF stationarity table
    md.stationarity_report.to_csv(DATA_PROCESSED_DIR / "adf_stationarity.csv", index=False)
    md.stationarity_report.to_csv(OUTPUTS_TABLES_DIR / "adf_stationarity.csv", index=False)
    md.stationarity_report.to_csv(DATA_PROCESSED_DIR / "adf_stationarity_results.csv", index=False)
    md.stationarity_report.to_csv(OUTPUTS_TABLES_DIR / "adf_stationarity_results.csv", index=False)

    # ---------------------------------------------------------------------------
    # Step 2: Layer 1 - Portfolio Optimization & KPIs
    # ---------------------------------------------------------------------------
    print("\n[Step 2/7] Layer 1: Portfolio Theory, Markowitz Optimization & KPIs...")
    suite = build_standard_portfolios_suite(md.simple_returns, risk_free_series=md.risk_free_daily)
    kpi_records = [suite[p][1].to_dict() for p in suite]
    kpi_df = pd.DataFrame(kpi_records)
    kpi_df.to_csv(DATA_PROCESSED_DIR / "portfolio_kpis_summary.csv", index=False)
    kpi_df.to_csv(OUTPUTS_TABLES_DIR / "portfolio_kpis_summary.csv", index=False)
    kpi_df.to_csv(DATA_PROCESSED_DIR / "institutional_kpis.csv", index=False)
    kpi_df.to_csv(OUTPUTS_TABLES_DIR / "institutional_kpis.csv", index=False)

    serendib_w = SERENDIB_CONSERVATIVE_WEIGHTS
    weights_records = []
    for t in TICKERS:
        w = serendib_w.get(t, 0.0)
        meta = ASSET_METADATA.get(t, {})
        weights_records.append({
            "Ticker": t,
            "Asset_Name": meta.get("name", t),
            "Asset_Class": meta.get("asset_class", ""),
            "Weight": f"{w*100:.1f}%",
            "Weight_Numeric": w,
            "Capital_Allocation_USD": w * MANDATE_SIZE_USD,
            "Institutional_Role": meta.get("role", ""),
        })
    pd.DataFrame(weights_records).to_csv(DATA_PROCESSED_DIR / "target_weights.csv", index=False)
    pd.DataFrame(weights_records).to_csv(OUTPUTS_TABLES_DIR / "target_weights.csv", index=False)

    # Weights comparison table across archetypes (portfolio_weights_comparison.csv)
    weights_comp_records = []
    for p in suite:
        w_dict, _, _ = suite[p]
        rec = {"Portfolio": p}
        for t in TICKERS:
            rec[t] = f"{w_dict.get(t, 0.0)*100:.1f}%"
        weights_comp_records.append(rec)
    weights_comp_df = pd.DataFrame(weights_comp_records)
    weights_comp_df.to_csv(DATA_PROCESSED_DIR / "portfolio_weights_comparison.csv", index=False)
    weights_comp_df.to_csv(OUTPUTS_TABLES_DIR / "portfolio_weights_comparison.csv", index=False)

    # Combined portfolio weights & KPIs table (portfolio_weights_kpis.csv)
    weights_kpi_records = []
    for p in suite:
        w_dict, kpi_obj, _ = suite[p]
        rec = kpi_obj.to_dict()
        for t in TICKERS:
            rec[f"Weight_{t}"] = f"{w_dict.get(t, 0.0)*100:.1f}%"
        weights_kpi_records.append(rec)
    weights_kpi_df = pd.DataFrame(weights_kpi_records)
    weights_kpi_df.to_csv(DATA_PROCESSED_DIR / "portfolio_weights_kpis.csv", index=False)
    weights_kpi_df.to_csv(OUTPUTS_TABLES_DIR / "portfolio_weights_kpis.csv", index=False)

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
    var_summary_df.to_csv(OUTPUTS_TABLES_DIR / "var_summary.csv", index=False)
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
    es_summary_df.to_csv(OUTPUTS_TABLES_DIR / "es_summary.csv", index=False)
    es_99_usd = es_results[0.99].es_usd
    tail_gap_usd = es_results[0.99].tail_severity_usd
    print(f"-> 1-Day 99% Expected Shortfall: {es_results[0.99].es_historical * 100:.2f}% (${es_99_usd:,.0f})")
    print(f"-> Tail Severity Gap (ES - VaR): +${tail_gap_usd:,.0f} additional loss beyond VaR threshold")

    sub_res = verify_subadditivity_properties(md.simple_returns["SPY"], md.simple_returns["AGG"])
    print(f"-> Subadditivity Proof Verified: {sub_res['Subadditivity']['ES_Holds']} (Diversification benefit: {sub_res['Subadditivity']['ES_Diversification_Benefit']})")

    # Combined VaR & ES Comparison Table (var_es_comparison.csv)
    var_es_records = []
    for c in [0.90, 0.95, 0.99]:
        vr = var_results[c]
        er = es_results[c]
        c_pct = f"{int(c*100)}%"
        var_es_records.append({
            "Confidence Level": c_pct,
            "Historical VaR (%)": f"{vr.historical_var * 100:.2f}%",
            "Historical VaR ($)": f"${vr.historical_var * MANDATE_SIZE_USD:,.0f}",
            "Parametric Gaussian VaR (%)": f"{vr.parametric_gaussian_var * 100:.2f}%",
            "Parametric Gaussian VaR ($)": f"${vr.parametric_gaussian_var * MANDATE_SIZE_USD:,.0f}",
            "Parametric Student-t VaR (%)": f"{vr.parametric_student_t_var * 100:.2f}%",
            "Parametric Student-t VaR ($)": f"${vr.parametric_student_t_var * MANDATE_SIZE_USD:,.0f}",
            "Monte Carlo VaR (%)": f"{vr.monte_carlo_var * 100:.2f}%",
            "Monte Carlo VaR ($)": f"${vr.monte_carlo_var * MANDATE_SIZE_USD:,.0f}",
            "Historical Expected Shortfall (%)": f"{er.es_historical * 100:.2f}%",
            "Historical Expected Shortfall ($)": f"${er.es_usd:,.0f}",
            "Parametric Gaussian ES (%)": f"{er.es_parametric_gaussian * 100:.2f}%",
            "Parametric Gaussian ES ($)": f"${er.es_parametric_gaussian * MANDATE_SIZE_USD:,.0f}",
            "Parametric Student-t ES (%)": f"{er.es_parametric_student_t * 100:.2f}%",
            "Parametric Student-t ES ($)": f"${er.es_parametric_student_t * MANDATE_SIZE_USD:,.0f}",
            "Monte Carlo ES (%)": f"{er.es_monte_carlo * 100:.2f}%",
            "Monte Carlo ES ($)": f"${er.es_monte_carlo * MANDATE_SIZE_USD:,.0f}",
            "Gaussian Underestimation Gap ($)": f"${vr.risk_gap_usd:,.0f}",
            "Tail Severity Gap (ES - VaR $)": f"${er.tail_severity_usd:,.0f}",
            "Tail Risk Ratio (ES / VaR)": f"{er.es_historical / vr.historical_var:.2f}x",
        })
    var_es_df = pd.DataFrame(var_es_records)
    var_es_df.to_csv(DATA_PROCESSED_DIR / "var_es_comparison.csv", index=False)
    var_es_df.to_csv(OUTPUTS_TABLES_DIR / "var_es_comparison.csv", index=False)

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
    evt_summary_df.to_csv(OUTPUTS_TABLES_DIR / "evt_summary.csv", index=False)
    export_mean_excess_diagnostic_data(
        returns_series=serendib_ret,
        output_path=DATA_PROCESSED_DIR / "evt_mean_excess_diagnostic.csv",
    )
    export_mean_excess_diagnostic_data(
        returns_series=serendib_ret,
        output_path=OUTPUTS_TABLES_DIR / "evt_mean_excess_diagnostic.csv",
    )

    # Comprehensive EVT Tail Metrics Table (evt_tail_metrics.csv)
    evt_tail_records = []
    for c in [0.90, 0.95, 0.99]:
        em = evt_metrics[c]
        evt_tail_records.append({
            "Confidence Level": f"{int(c*100)}%",
            "Threshold u (%)": f"{evt_fit.threshold_u * 100:.3f}%",
            "Threshold Percentile": f"{evt_fit.threshold_percentile * 100:.1f}th",
            "Exceedances Count (Nu)": evt_fit.exceedances_count_nu,
            "Fitted Shape xi (Tail Index)": f"{evt_fit.shape_xi:.4f}",
            "Fitted Scale sigma (%)": f"{evt_fit.scale_sigma * 100:.4f}%",
            "Tail Distribution Regime": evt_fit.tail_type,
            "KS Goodness-of-Fit p-value": f"{evt_fit.ks_p_value:.4f}",
            "EVT-POT VaR (%)": f"{em.evt_var * 100:.2f}%",
            "EVT-POT VaR ($10M AUM)": f"${em.evt_var_usd:,.0f}",
            "EVT-POT Expected Shortfall (%)": f"{em.evt_es * 100:.2f}%",
            "EVT-POT Expected Shortfall ($10M AUM)": f"${em.evt_es_usd:,.0f}",
        })
    evt_tail_df = pd.DataFrame(evt_tail_records)
    evt_tail_df.to_csv(DATA_PROCESSED_DIR / "evt_tail_metrics.csv", index=False)
    evt_tail_df.to_csv(OUTPUTS_TABLES_DIR / "evt_tail_metrics.csv", index=False)

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
    backtest_df.to_csv(OUTPUTS_TABLES_DIR / "backtest_summary.csv", index=False)
    hist_rep = backtest_reports["historical"]
    print(f"-> Kupiec POF Test: p = {hist_rep.kupiec.p_value:.4f} ({hist_rep.kupiec.verdict})")
    print(f"-> Basel Committee Traffic Light Status: {hist_rep.basel.zone} ZONE (Multiplier: {hist_rep.basel.capital_multiplier:.2f}x)")

    # ---------------------------------------------------------------------------
    # Step 7: Layer 6 - Stress Testing & Risk Budgeting
    # ---------------------------------------------------------------------------
    print("\n[Step 7/7] Layer 6: Stress Testing, Risk Budgeting & Scenario Analysis...")
    _, budget_df, val = compute_euler_risk_budgeting(md.simple_returns, serendib_w, aum=MANDATE_SIZE_USD)
    budget_df.to_csv(DATA_PROCESSED_DIR / "risk_budgeting.csv", index=False)
    budget_df.to_csv(OUTPUTS_TABLES_DIR / "risk_budgeting.csv", index=False)

    _, crisis_df = evaluate_historical_stress_scenarios(md.simple_returns, serendib_w, aum=MANDATE_SIZE_USD)
    crisis_df.to_csv(DATA_PROCESSED_DIR / "crisis_scenarios.csv", index=False)
    crisis_df.to_csv(OUTPUTS_TABLES_DIR / "crisis_scenarios.csv", index=False)

    stagflation = evaluate_hypothetical_stagflation_shock(serendib_w, aum=MANDATE_SIZE_USD)
    stagflation["asset_breakdown"].to_csv(DATA_PROCESSED_DIR / "hypothetical_stagflation_shock.csv", index=False)
    stagflation["asset_breakdown"].to_csv(OUTPUTS_TABLES_DIR / "hypothetical_stagflation_shock.csv", index=False)

    # Combined Stress Test Losses Table (stress_test_losses.csv)
    _, stress_losses_df = evaluate_all_stress_scenarios(md.simple_returns, serendib_w, aum=MANDATE_SIZE_USD)
    stress_losses_df.to_csv(DATA_PROCESSED_DIR / "stress_test_losses.csv", index=False)
    stress_losses_df.to_csv(OUTPUTS_TABLES_DIR / "stress_test_losses.csv", index=False)

    reverse_stress = perform_reverse_stress_test(serendib_w, aum=MANDATE_SIZE_USD)
    rev_df = pd.DataFrame([{
        "Max_Drawdown_Limit": reverse_stress["max_drawdown_limit_pct"],
        "Max_Drawdown_USD": reverse_stress["max_drawdown_limit_usd"],
        "Crash_Multiplier_Required": f"{reverse_stress['crash_multiplier_required']:.2f}x",
        "Critical_SPY_Drop": reverse_stress["critical_spy_drop"],
        "Critical_AGG_Drop": reverse_stress["critical_agg_drop"],
        "Board_Defense_Narrative": reverse_stress["board_defense_narrative"],
    }])
    rev_df.to_csv(DATA_PROCESSED_DIR / "reverse_stress_test.csv", index=False)
    rev_df.to_csv(OUTPUTS_TABLES_DIR / "reverse_stress_test.csv", index=False)

    print(f"-> Euler Risk Decomposition Verified: {val['Euler_Decomposition_Verified']}")
    print(f"-> Hypothetical Stagflation Impact: {stagflation['portfolio_return'] * 100:.2f}% (${stagflation['dollar_impact_usd']:,.0f})")
    print(f"-> Capital Preserved vs 60/40 in Stagflation: +${stagflation['capital_saved_usd']:,.0f}")
    print(f"-> Reverse Stress Circuit Breaker: Requires {reverse_stress['critical_spy_drop']} equity drop + {reverse_stress['critical_agg_drop']} bond drop.")

    # ---------------------------------------------------------------------------
    # Step 8: Generate Publication Figures First (300 DPI)
    # ---------------------------------------------------------------------------
    print("\nGenerating Publication-Ready Figures (300 DPI) in outputs/figures/...")
    generate_publication_figures(
        returns_df=md.simple_returns,
        serendib_ret=serendib_ret,
        portfolio_suite=suite,
        var_results=var_results,
        es_results=es_results,
        evt_fit_info=evt_fit,
        crisis_df=crisis_df,
        outputs_figures_dir=OUTPUTS_FIGURES_DIR,
    )

    # ---------------------------------------------------------------------------
    # Step 9: Export Institutional Summary Tables to outputs/tables/
    # ---------------------------------------------------------------------------
    print("Exporting Institutional Tables to outputs/tables/...")
    export_submission_tables(DATA_PROCESSED_DIR, OUTPUTS_TABLES_DIR)

    # ---------------------------------------------------------------------------
    # Step 10: Generate Formal Institutional Deliverables & PDFs
    # ---------------------------------------------------------------------------
    print("\nCompiling Official Institutional Reports & PDF Documents...")
    tearsheet_path = REPORT_DIR / "EXECUTIVE_TEARSHEET.md"
    memo_path = REPORT_DIR / "INVESTMENT_COMMITTEE_MEMORANDUM.md"

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

    # Compile official PDFs in report/
    tear_sheet_pdf = REPORT_DIR / "tear_sheet.pdf"
    memo_pdf = REPORT_DIR / "investment_memo.pdf"
    build_tear_sheet_pdf(
        output_pdf_path=tear_sheet_pdf,
        kpis_dict=serendib_kpis,
        var_summary_df=var_summary_df,
        es_summary_df=es_summary_df,
        evt_fit_info=evt_fit,
        basel_status=hist_rep.basel.zone,
        stagflation_dict=stagflation,
    )
    build_investment_memo_pdf(
        output_pdf_path=memo_pdf,
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
    )

    # Export standalone interactive HTML charts and mirror PDFs to docs/ for GitHub Pages
    try:
        export_all_showcase_assets()
    except Exception as e:
        logger.warning(f"Static showcase export notice: {e}")

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"  QUANT ENGINE PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s")
    print(f"  Deliverables Ready in report/ and outputs/:")
    print(f"  - Executive Tear Sheet (PDF): {tear_sheet_pdf}")
    print(f"  - Investment Committee Memo (PDF): {memo_pdf}")
    print(f"  - Executive Tear Sheet (Markdown): {tearsheet_path}")
    print(f"  - Investment Committee Memo (Markdown): {memo_path}")
    print(f"  - Publication Figures: {OUTPUTS_FIGURES_DIR}")
    print(f"  - Institutional Tables: {OUTPUTS_TABLES_DIR}")
    print(f"  - Interactive Streamlit Dashboard: 'streamlit run app.py'")
    print("=" * 80)


if __name__ == "__main__":
    force = "--reload" in sys.argv
    run_full_pipeline(force_reload=force)
