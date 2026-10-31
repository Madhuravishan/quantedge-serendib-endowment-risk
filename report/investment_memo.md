# FORMAL INVESTMENT COMMITTEE MEMORANDUM

**TO:** Investment Committee & Board of Trustees, The Serendib University Endowment Fund  
**FROM:** Team "Pandas Predictors" (Quantitative Asset Management Division)  
**DATE:** October 2026  
**SUBJECT:** Institutional Quantitative Risk, Portfolio Optimization, and Tail-Risk Validation for USD 10,000,000 Mandate  
**REGULATORY CLASSIFICATION:** Basel III & FRTB Compliant / CFA Institute Global Investment Standards  

---

## TABLE OF CONTENTS
- Executive Summary & Investment Recommendation
- Section 1: Data Ingestion & Stationarity Analysis
- Section 2: Modern Portfolio Theory & Efficient Frontier
- Section 3: Downside Tail-Risk Modeling (3-Method VaR)
- Section 4: Coherent Risk Measures (Expected Shortfall)
- Section 5: Extreme Value Theory (EVT / POT)
- Section 6: Model Validation & Basel III Regulatory Compliance
- Section 7: Macroeconomic Stress Testing & Euler Risk Budgeting
- Section 8: The Endowment Drawdown Guardrail Protocol
- Section 9: Model Assumptions & Operational Limitations (Plain English)
- Section 10: Final Investment Committee Recommendations & Sign-off

---

## EXECUTIVE SUMMARY & INVESTMENT RECOMMENDATION

### The Mandate Decision
Team "Pandas Predictors" was commissioned to design, implement, empirically backtest, and mathematically defend an institutional portfolio management engine for the **Serendib University Endowment Fund** ($10,000,000 AUM).

Following exhaustive multi-layer quantitative evaluation across a 20-year empirical window (2006 to 2026), our unequivocal recommendation to the Investment Committee is:

> **RECOMMENDATION: FULL MANDATE DEPLOYMENT INTO THE SERENDIB CONSERVATIVE ALLOCATION.**  
> Target Weights: **35% AGG, 25% SPY, 15% GLD, 10% EFA, 10% XLE, 5% EEM, 0% XLK.**

### Key Empirical Justifications
1. **Target Return Surplus:** Delivers a **7.91% CAGR**, surpassing the Endowment's statutory **7.00% nominal hurdle** (4.5% real operational spending + 2.5% expected inflation) by **+91 basis points**, compounding $10,000,000 into **$48,556,769** over 20 years.
2. **Superior Downside Protection:** Truncates 20-year Maximum Drawdown to **-31.05%**, outperforming the Classic 60/40 benchmark (-34.70%) and Equal Weight (-42.35%).
3. **Stagflation Immunization:** Under our forward-looking 1970s-style geopolitical stagflation crisis (Equities -20%, Bonds -12%), the Serendib allocation restricts portfolio drawdown to **-7.90% (-$790,000)**, saving **$890,000** in capital compared to a 60/40 portfolio (-$1,680,000) due to structural Gold (+15%) and Energy (+25%) hedges.
4. **Basel Regulatory Certification:** Achieves **GREEN ZONE** status under Basel Committee backtesting standards with 10 exceptions over 702 out-of-sample trading days (Kupiec p = 0.2879), incurring zero capital multiplier penalties (k = 3.00x).

---

## SECTION 1: DATA INGESTION & STATIONARITY ANALYSIS
Stationarity is mathematically required before estimating covariance matrices and tail-risk parameters. Augmented Dickey-Fuller (ADF) unit-root tests were conducted on all 7 ETF price and return series, alongside the FRED 3-Month Treasury Bill rate (DTB3). All log price series are confirmed non-stationary I(1) (p > 0.20), whereas all log return series are strictly stationary I(0) (p < 0.0001, far below 0.01).

---

## SECTION 2: MODERN PORTFOLIO THEORY & EFFICIENT FRONTIER
Markowitz quadratic programming balances variance and return under strict institutional constraints. While the unconstrained Max Sharpe portfolio produces 14.55% CAGR, its -34.94% max drawdown violates the endowment risk budget. The Serendib Conservative Allocation provides the optimal balance.

---

## SECTION 3: DOWNSIDE TAIL-RISK MODELING (3-METHOD VaR)
Historical Simulation, Parametric (Gaussian and Student-t), and 10,000-path Monte Carlo simulations reveal that standard normal Gaussian models underestimate 99% tail losses by +$32,915 on $10M AUM due to severe negative skewness and leptokurtosis.

---

## SECTION 4: COHERENT RISK MEASURES (EXPECTED SHORTFALL)
Unlike VaR, Expected Shortfall (CVaR) is mathematically subadditive (ES(X+Y) <= ES(X) + ES(Y)), guaranteeing that diversification never penalizes the portfolio. At 99% confidence, 1-day Expected Shortfall reaches $303,954 (3.04%), exposing a 1.56x post-breach loss severity.

---

## SECTION 5: EXTREME VALUE THEORY (EVT / POT)
Via the Pickands-Balkema-de Haan theorem, excess losses above the 95th percentile threshold (u = 1.011%) follow a heavy-tailed Generalized Pareto Distribution (xi = 0.2938, p = 0.8694). EVT calculates 99% 1-day VaR at 2.00% ($199,593) and Expected Shortfall at 3.08% ($308,209).

---

## SECTION 6: MODEL VALIDATION & BASEL III REGULATORY COMPLIANCE
Over 702 out-of-sample days (2024–2026), the Kupiec POF Likelihood Ratio test (LR = 1.13, p = 0.2879) and Christoffersen independence test confirm model validity. Basel Committee backtesting places the fund in the GREEN ZONE with a baseline multiplier of k = 3.00x.

---

## SECTION 7: MACROECONOMIC STRESS TESTING & EULER RISK BUDGETING
Replaying the 2008 GFC, 2020 COVID crash, 2022 rate shock, and 1970s stagflation confirms robust capital preservation. Euler decomposition shows SPY represents 25% of capital but 37.95% of VaR risk, while AGG anchors portfolio stability.

---

## SECTION 8: THE ENDOWMENT DRAWDOWN GUARDRAIL PROTOCOL
Reconciles the 15% annual operational drawdown circuit breaker against the -31.05% unmanaged 2008 GFC trough. Governed by a 3-tier action policy (Normal <10%, Watch 10%-15%, Circuit Breaker >15% de-risking into US T-Bills).

---

## SECTION 9: MODEL ASSUMPTIONS & OPERATIONAL LIMITATIONS (PLAIN ENGLISH)
Transparently audits 6 critical real-world frictions: non-normality, macro regime shifts, ETF tracking error, liquidity frictions, rebalancing turnover costs, and linear instrument payoffs.

---

## SECTION 10: FINAL INVESTMENT COMMITTEE RECOMMENDATIONS & SIGN-OFF
Actionable 4-phase rollout schedule, algorithmic TWAP execution, quarterly rebalancing corridors, AI usage disclosure (Section 8.7 compliance), and formal fiduciary attestation.
