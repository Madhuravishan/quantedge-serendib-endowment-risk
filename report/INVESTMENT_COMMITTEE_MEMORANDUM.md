# FORMAL INVESTMENT COMMITTEE MEMORANDUM

**TO:** Investment Committee & Board of Trustees, The Serendib University Endowment Fund  
**FROM:** Team "Pandas Predictors" (Quantitative Asset Management Division)  
**DATE:** October 22, 2026  
**SUBJECT:** Institutional Quantitative Risk, Portfolio Optimization, and Tail-Risk Validation for USD 10,000,000 Mandate  
**REGULATORY CLASSIFICATION:** Basel III & FRTB Compliant / CFA Institute Global Investment Standards  

---

## TABLE OF CONTENTS
1. Executive Summary & Investment Recommendation
2. Client Mandate, Institutional Philosophy & Governance
3. Investable Universe, Data Integrity & Stationarity Proofs
4. Layer 1: Portfolio Theory, Markowitz Optimization & Institutional KPIs
5. Layer 2: Value-at-Risk (VaR) Engine & The "Risk Gap" Phenomenon
6. Layer 3: Expected Shortfall (CVaR) & Coherent Risk Measure Subadditivity
7. Layer 4: Extreme Value Theory (EVT) via Peaks-Over-Threshold (POT)
8. Layer 5: Out-of-Sample Model Validation & Basel Regulatory Compliance
9. Layer 6: Stress Testing, Risk Budgeting & Scenario Analysis
10. Mandatory Audit: Model Assumptions, Operational Limitations & Epistemic Uncertainty
11. Implementation Plan & Ongoing Risk Governance

---

## 1. EXECUTIVE SUMMARY & INVESTMENT RECOMMENDATION

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

## 2. CLIENT MANDATE, INSTITUTIONAL PHILOSOPHY & GOVERNANCE

### Client Profile
- **Institution:** The Serendib University Endowment Fund.
- **Corpus:** USD 10,000,000 ($10 Million AUM).
- **Time Horizon:** Perpetuity (intergenerational endowment).
- **Core Purpose:** Sustainable annual distributions to fund faculty chairs, STEM laboratory infrastructure, and need-based undergraduate scholarships.

### Institutional Guiding Axiom
The Investment Committee's core philosophy is established on two foundational axioms:
> *"Avoiding large losses is more important than maximizing gains."*  
> *"A good investment decision is not the one with the highest expected return, but the one whose downside has been measured, empirically tested, and mathematically defended."*

In accordance with institutional endowment governance, an endowment portfolio must withstand systemic deflationary crashes, liquidity panics, and stagflationary shocks without depleting principal.

---

## 3. INVESTABLE UNIVERSE, DATA INTEGRITY & STATIONARITY PROOFS

### 7-ETF Universe Specification
In accordance with competition rules, the engine restricts operations strictly to the prescribed 7-ETF liquid universe plus macro risk-free series:
1. **SPY (S&P 500 ETF Trust):** Core US Large-Cap Equity engine.
2. **EFA (iShares MSCI EAFE ETF):** Developed International Equity diversifier.
3. **EEM (iShares MSCI Emerging Markets ETF):** High-beta growth factor (strictly capped at 5%).
4. **AGG (iShares Core US Aggregate Bond ETF):** Deflation anchor and fixed income liquidity foundation.
5. **GLD (SPDR Gold Shares):** Monetary debasement and geopolitical tail-risk hedge.
6. **XLE (Energy Select Sector SPDR):** Commodity inflation and supply constriction hedge.
7. **XLK (Technology Select Sector SPDR):** High-multiple growth factor (allocated 0% to prevent double-counting SPY's ~30% tech exposure).
8. **Risk-Free Rate Proxy:** St. Louis Federal Reserve (FRED DTB3: 3-Month US Treasury Bill Secondary Market Rate) with failover to Yahoo Finance ^IRX.

### Data Cleaning Discipline
- **Adjusted Close Prices:** All price series utilize dividend- and split-adjusted daily closes to prevent artificial return distortions.
- **Calendar Alignment:** Cross-market trading calendar discrepancies are forward-filled, ensuring complete cross-sectional alignment across all 5,231 trading days without data leakage.
- **Return Conventions:** 
  - Logarithmic returns ($r_t = \ln(P_t / P_{t-1})$) are utilized for stationarity testing, distributional fitting, and EVT tail modeling due to time-additivity.
  - Simple returns ($R_t = (P_t - P_{t-1}) / P_{t-1}$) are used for cross-sectional portfolio weighting ($R_{p,t} = \sum w_i R_{i,t}$) and accounting reconciliation.

### Formal Augmented Dickey-Fuller (ADF) Unit-Root Proofs
To establish mathematical validity before modeling, Augmented Dickey-Fuller unit-root tests were conducted on both log price series and return series:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \varepsilon_t$$

**Hypotheses:**  
$H_0: \gamma = 0$ (Unit root exists -> Non-Stationary $I(1)$)  
$H_1: \gamma < 0$ (Stationary $I(0)$)

```
Ticker            Series  ADF Statistic    p-value  Critical 1%  Critical 5%  Stationary (5%) Order of Integration
   SPY Log Price ln(P_t)         0.6396 9.8855e-01      -3.4316      -2.8621            False       I(1) Unit Root
   SPY    Log Return r_t       -16.2786 3.4304e-29      -3.4316      -2.8621             True      I(0) Stationary
   EFA Log Price ln(P_t)        -0.5089 8.9027e-01      -3.4316      -2.8621            False       I(1) Unit Root
   EFA    Log Return r_t       -15.7019 1.3929e-28      -3.4316      -2.8621             True      I(0) Stationary
   EEM Log Price ln(P_t)        -1.6025 4.8232e-01      -3.4316      -2.8621            False       I(1) Unit Root
   EEM    Log Return r_t       -15.8520 9.4242e-29      -3.4316      -2.8621             True      I(0) Stationary
   AGG Log Price ln(P_t)        -2.1647 2.1929e-01      -3.4316      -2.8621            False       I(1) Unit Root
   AGG    Log Return r_t       -15.3816 3.4019e-28      -3.4316      -2.8621             True      I(0) Stationary
   GLD Log Price ln(P_t)        -0.8001 8.1909e-01      -3.4316      -2.8621            False       I(1) Unit Root
   GLD    Log Return r_t       -73.6040 0.0000e+00      -3.4316      -2.8621             True      I(0) Stationary
   XLE Log Price ln(P_t)        -0.8871 7.9220e-01      -3.4316      -2.8621            False       I(1) Unit Root
   XLE    Log Return r_t       -76.4927 0.0000e+00      -3.4316      -2.8621             True      I(0) Stationary
   XLK Log Price ln(P_t)         1.0385 9.9466e-01      -3.4316      -2.8621            False       I(1) Unit Root
   XLK    Log Return r_t       -17.9873 2.7619e-30      -3.4316      -2.8621             True      I(0) Stationary
```

**Mathematical Conclusion:**  
Every asset's log price series exhibits an ADF p-value > 0.20, failing to reject $H_0$ and proving that raw prices are integrated of order 1 ($I(1)$). Conversely, all return series reject $H_0$ with p-values < 1e-28, proving that returns are strictly stationary ($I(0)$). This validates the statistical foundation of our subsequent covariance and tail-risk estimations.

---

## 4. LAYER 1: PORTFOLIO THEORY, MARKOWITZ OPTIMIZATION & INSTITUTIONAL KPIS

### Markowitz Mean-Variance Formulation
The optimization problem is formulated as a quadratic program with no short-selling constraints ($w_i \ge 0, \sum w_i = 1$):

$$\min_w \quad w^T \Sigma w \quad \text{s.t.} \quad w^T \mathbf{1} = 1, \quad w_i \ge 0, \quad w^T \mu = r_{\text{target}}$$

### Model Portfolios Performance Comparison

```
                          Portfolio   CAGR Annual Volatility Sharpe Ratio (Rf) Downside Volatility Sortino Ratio CAPM Beta (vs SPY) Max Drawdown (MDD) End Wealth ($10M AUM)
Serendib Conservative (Recommended)  7.91%            11.31%             0.578               8.00%         0.818              0.528            -31.05%        $48,556,769.19
               Markowitz Max Sharpe 14.55%            15.66%             0.837              10.91%         1.201              0.637            -34.94%       $167,792,963.67
                   Minimum Variance  3.70%             5.08%             0.402               3.74%         0.547              0.070            -16.74%        $21,240,321.01
            Classic 60/40 Benchmark  8.25%            11.71%             0.589               8.25%         0.836              0.600            -34.70%        $51,838,705.31
                 Equal Weight (1/N)  9.60%            16.11%             0.543              11.38%         0.769              0.776            -42.35%        $67,031,829.55
```

### Why the Serendib Conservative Allocation Outclasses Rivals
1. **Vs. Max Sharpe (14.55% CAGR, 15.66% Vol, -34.94% MDD):** Max Sharpe over-weights tech/growth assets, creating severe vulnerability to high-inflation regimes and rate shocks. Its higher drawdown violates the endowment's risk budget.
2. **Vs. Minimum Variance (3.70% CAGR, 5.08% Vol, -16.74% MDD):** While Min Vol achieves low volatility, its 3.70% CAGR fails to meet the required 7.00% payout hurdle, leading to principal erosion in real terms.
3. **Vs. Classic 60/40 (8.25% CAGR, 11.71% Vol, -34.70% MDD):** The 60/40 lacks direct inflation hedges. During the 2022 rate shock, stock-bond correlation flipped positive, causing 60/40 to suffer a -20.26% loss. Serendib Conservative held drawdown to -13.38% through its 15% Gold and 10% Energy allocations.

---

## 5. LAYER 2: VALUE-AT-RISK (VaR) ENGINE & THE "RISK GAP" PHENOMENON

### 3-Method Comparative Framework
We implement three distinct Value-at-Risk methodologies on the exact same portfolio across 90%, 95%, and 99% confidence levels:
1. **Method A: Historical Simulation:** Non-parametric empirical quantile of realized returns.
2. **Method B: Parametric (Variance-Covariance):**
   - Gaussian normal: $\text{VaR}_\alpha = -\mu_p + z_\alpha \sigma_p$
   - Student-t heavy tail: $\text{VaR}_\alpha = -\mu_p + t_{\alpha, \nu}^{-1} \sqrt{(\nu-2)/\nu} \sigma_p$ (fitted $\nu = 8.46$)
   - Cornish-Fisher expansion: Quantile adjusted for skewness $S$ and excess kurtosis $K$.
3. **Method C: Monte Carlo Simulation:** 10,000 multivariate simulated paths using Cholesky decomposition of the covariance matrix ($\Sigma = L L^T$).

```
Confidence Level Historical Sim VaR Parametric Gaussian VaR Parametric Student-t VaR Cornish-Fisher VaR Monte Carlo VaR (10k) Risk Gap (Hist - Gauss) Fitted Student-t DoF
             90%    0.68% ($68,124)         0.88% ($88,069)          0.83% ($83,298)    0.65% ($65,233)       0.89% ($89,025)       -0.20% ($-19,944)                 8.46
             95%   1.01% ($101,127)        1.14% ($113,964)         1.12% ($111,731)    0.90% ($89,794)      1.15% ($114,812)       -0.13% ($-12,837)                 8.46
             99%   1.95% ($195,453)        1.63% ($162,538)         1.75% ($174,822)   4.57% ($457,023)      1.66% ($166,416)         0.33% ($32,915)                 8.46
```

### The "Risk Gap" Analysis
The **Risk Gap** represents the discrepancy between true empirical losses and Gaussian parametric estimates:

$$\text{Risk Gap}_\alpha = \text{VaR}_\alpha^{\text{Historical}} - \text{VaR}_\alpha^{\text{Gaussian}}$$

- At **90% Confidence:** Risk Gap is -0.20% (-$19,944). The Gaussian model overstates risk in the central shoulders of the distribution.
- At **99% Confidence:** Risk Gap flips to **+0.33% (+$32,915 underestimation)**.
- **Institutional Takeaway:** A standard normal distribution dangerously blinds the Investment Committee to tail risk. The portfolio exhibits positive excess kurtosis and negative skewness, meaning 99% 1-day losses in reality are over $32,000 worse than Gaussian models project!

---

## 6. LAYER 3: EXPECTED SHORTFALL (CVaR) & COHERENT RISK MEASURE SUBADDITIVITY

### Expected Shortfall Formulation
Value-at-Risk merely answers *"What is the minimum loss on bad days?"* but remains silent on *"How bad do losses get once the threshold is crossed?"*  
Expected Shortfall (CVaR) measures the conditional tail expectation:

$$\text{ES}_\alpha = -\mathbb{E}[R_p \mid R_p \le -\text{VaR}_\alpha]$$

```
Confidence Level 1-Day VaR (Empirical) 1-Day Expected Shortfall (ES) Parametric Gaussian ES Parametric Student-t ES Monte Carlo ES (10k) Tail Severity Gap (ES - VaR) Tail Risk Ratio (ES / VaR)
             90%       0.68% ($68,124)              1.26% ($125,724)       1.22% ($121,813)        1.24% ($123,749)     1.23% ($122,731)              0.58% ($57,599)                      1.85x
             95%      1.01% ($101,127)              1.68% ($167,842)       1.44% ($143,747)        1.51% ($151,461)     1.45% ($145,353)              0.67% ($66,715)                      1.66x
             99%      1.95% ($195,453)              3.04% ($303,954)       1.87% ($186,691)        2.16% ($216,385)     1.93% ($192,619)             1.09% ($108,500)                      1.56x
```

### Key Board Findings on Expected Shortfall
1. **The Tail Severity Gap:** At 99% confidence, while VaR is $195,453 (1.95%), Expected Shortfall is **$303,954 (3.04%)**. Once the 99% threshold is breached, the average loss is **+$108,500 worse than VaR**!
2. **Tail Risk Ratio (1.56x):** The empirical Tail Risk Ratio ($\text{ES} / \text{VaR} = 1.56$) significantly exceeds the Gaussian ratio (1.14).
3. **Mathematical Proof of Subadditivity:**  
   Artzner et al. (1999) proved that a risk measure $\rho$ is *coherent* if and only if it satisfies Monotonicity, Subadditivity, Positive Homogeneity, and Translation Invariance.
   - **VaR Fails Subadditivity:** For non-normal, fat-tailed assets, $\text{VaR}(X + Y) > \text{VaR}(X) + \text{VaR}(Y)$ can occur, falsely indicating that diversification creates risk.
   - **ES Strictly Satisfies Subadditivity:** $\text{ES}(X + Y) \le \text{ES}(X) + \text{ES}(Y)$ is guaranteed for all distributions. In our empirical test between SPY and AGG, combined ES (1.89%) was **53 basis points lower** than the weighted sum of individual risks (2.42%), mathematically proving the diversification benefit.

---

## 7. LAYER 4: EXTREME VALUE THEORY (EVT) VIA PEAKS-OVER-THRESHOLD (POT)

### Mathematical Justification of Threshold Selection
Rather than selecting an arbitrary cutoff, threshold $u$ was mathematically justified using:
1. **Mean Excess Function (MEF) Plot:** Identified the threshold where empirical mean excess $e_n(u) = \mathbb{E}[L - u \mid L > u]$ transitions into a stable linear slope.
2. **Hill Plot:** Verified stability of the tail index across upper order statistics.
3. **Kolmogorov-Smirnov Test:** Confirmed goodness-of-fit ($p = 0.8694 > 0.05$), failing to reject the Generalized Pareto Distribution null hypothesis.

### Generalized Pareto Distribution (GPD) Calibration Results
- **Threshold $u$ (95th percentile):** 1.011%
- **Exceedances ($N_u / N$):** 262 / 5230 (5.01%)
- **Tail Index $\xi$ (Shape):** **0.2938** (Heavy-Tailed (Fréchet, xi > 0))
- **Scale Parameter $\sigma$:** 0.0048

### EVT Closed-Form Risk Estimates

$$\text{VaR}_\alpha^{\text{EVT}} = u + \frac{\sigma}{\xi} \left[ \left( \frac{N}{N_u} (1 - \alpha) \right)^{-\xi} - 1 \right] = \mathbf{2.00\% \quad (\$199,593)}$$

$$\text{ES}_\alpha^{\text{EVT}} = \frac{\text{VaR}_\alpha^{\text{EVT}} + \sigma - \xi u}{1 - \xi} = \mathbf{3.08\% \quad (\$308,209)}$$

EVT provides the true asymptotic tail limit, proving that Gaussian models underestimate tail severity by **over 39%** ($186k vs $308k).

---

## 8. LAYER 5: OUT-OF-SAMPLE MODEL VALIDATION & BASEL REGULATORY COMPLIANCE

### Backtesting Methodology
To ensure zero look-ahead bias, models were calibrated on historical data and tested out-of-sample over **702 trading days (2024 to 2026)**.

```
          Model Name Confidence    Obs / Exceptions Kupiec POF p-val Independence p-val Conditional Coverage             Basel Traffic Light Capital Multiplier
Historical VaR (99%)        99% 702 / 10 (Exp: 7.0)    0.2879 (Pass)      0.0058 (Fail)        0.0127 (Fail)              GREEN (Acceptable)              3.00x
  Gaussian VaR (99%)        99% 702 / 11 (Exp: 7.0)    0.1633 (Pass)      0.0091 (Fail)        0.0126 (Fail) YELLOW (Supervisory Monitoring)              3.40x
 Student_t VaR (99%)        99% 702 / 11 (Exp: 7.0)    0.1633 (Pass)      0.0091 (Fail)        0.0126 (Fail) YELLOW (Supervisory Monitoring)              3.40x
       Evt VaR (99%)        99% 702 / 10 (Exp: 7.0)    0.2879 (Pass)      0.0058 (Fail)        0.0127 (Fail)              GREEN (Acceptable)              3.00x
```

### Regulatory Audit Findings
1. **Basel Committee Traffic Light Framework:** 
   - **Historical VaR and EVT VaR** achieved **GREEN ZONE** certification (10 exceptions over 702 days, with only 3 exceptions in the trailing 250-day window). Regulatory multiplier remains at baseline **$k = 3.00x$** with zero capital penalty.
   - **Gaussian VaR** drifted into the **YELLOW ZONE** on trailing stress intervals, triggering supervisory review and an escalated capital multiplier ($k = 3.40x$).
2. **Kupiec POF Test:** With 10 realized exceptions versus 7.0 expected, the Kupiec Likelihood Ratio test yielded $LR = 1.13$ ($p = 0.2879$), comfortably passing unconditional coverage ($p \ge 0.05$).
3. **Christoffersen Independence Test:** Independence tests yielded $p = 0.0058$, indicating slight temporal clustering of exceptions during macroeconomic shock dates. This empirical finding reinforces the necessity of dynamic volatility scaling or EVT overlays.

---

## 9. LAYER 6: STRESS TESTING, RISK BUDGETING & SCENARIO ANALYSIS

### Euler Risk Decomposition & Position Sizing
Using Euler's homogeneous function theorem, portfolio risk was decomposed into asset-level contributions ($\sum \text{PCR}_i = 100\%$):

```
Asset Weight Marginal VaR (MVaR) Component VaR (CVaR) % Risk Contrib (VaR) Marginal ES (MES) Component ES (CES) % Risk Contrib (ES) Dollar Risk Contrib ($)
  SPY  25.0%               2.47%                0.62%               37.95%             4.83%              1.21%              39.71%                 $61,684
  EFA  10.0%               2.78%                0.28%               17.13%             5.23%              0.52%              17.20%                 $27,849
  EEM   5.0%               3.46%                0.17%               10.65%             6.48%              0.32%              10.66%                 $17,314
  AGG  35.0%               0.15%                0.05%                3.24%             0.27%              0.09%               3.10%                  $5,262
  GLD  15.0%               1.05%                0.16%                9.71%             1.26%              0.19%               6.22%                 $15,790
  XLE  10.0%               3.46%                0.35%               21.31%             7.03%              0.70%              23.11%                 $34,639
  XLK   0.0%               2.56%                0.00%                0.00%             4.75%              0.00%               0.00%                      $0
```

**Key Allocation Insight:**  
Although US Equities (SPY) represent only **25.0% of portfolio capital**, they account for **37.95% of total portfolio VaR** and **39.71% of Expected Shortfall**. Conversely, Fixed Income (AGG) represents **35.0% of capital** but contributes only **3.24% of portfolio risk**. This confirms that risk budgeting—not capital weighting—governs endowment solvency.

### Historical Crisis Replay

```
                            Crisis Scenario                   Window Portfolio Return Endowment Impact ($10M) Max Crisis Drawdown Capital Preserved vs 60/40
2008 Global Financial Crisis (Lehman Crash) 2008-09-01 to 2008-11-30          -16.02%             $-1,602,276             -24.79%                   $214,306
        March 2020 COVID-19 Liquidity Shock 2020-02-19 to 2020-03-23          -21.21%             $-2,120,974             -22.19%                    $29,087
2022 Inflation & Aggressive Rate Hike Shock 2022-01-03 to 2022-10-14          -13.38%             $-1,338,312             -15.00%                   $687,325
```

### Hypothetical Scenario: 1970s-Style Geopolitical Stagflation Shock
We designed an adverse stagflation scenario featuring an oil supply shock, stubborn inflation, and monetary tightening:
- Equities (SPY -20%, EFA -22%, EEM -25%) drop on margin compression.
- Bonds (AGG -12%) suffer as yields spike.
- Hedges (GLD +15%, XLE +25%) surge.

**Results:**
- **Serendib Conservative Return:** **-7.90% (-$790,000)**
- **Classic 60/40 Benchmark:** **-16.80% (-$1,680,000)**
- **Capital Preserved:** **+$890,000 USD**  
The strategic 15% Gold and 10% Energy allocations provided an $890,000 cushion, completely insulating the endowment from a double-digit catastrophe.

### Reverse Stress Testing
Setting the Board's maximum tolerable drawdown threshold at **15.0% ($1,500,000)**:
- The Serendib portfolio requires an extraordinary systemic equity collapse of **-23.6% SPY drop** combined with a concurrent **-9.4% bond drop** before the $1.5M circuit breaker is breached.
- This demonstrates robust institutional structural resilience.

---

## 10. MANDATORY AUDIT: MODEL ASSUMPTIONS, OPERATIONAL LIMITATIONS & EPISTEMIC UNCERTAINTY

In compliance with competition guidelines and CFA Institute professional standards, the quantitative engine's limitations are explicitly disclosed:

1. **Stationarity & Parameter Stability Assumption:**  
   The Markowitz optimizer and parametric VaR assume that covariance structures $\Sigma$ and mean returns $\mu$ are stationary over the evaluation horizon. In reality, cross-asset correlations spike toward 1.0 during severe liquidity panics (as seen in March 2020), temporarily reducing diversification benefits.
2. **Distributional Truncation (Gaussian Model Blindspot):**  
   Standard Gaussian VaR severely underestimates tail probability due to zero excess kurtosis. While our Student-t and EVT-POT engines successfully mitigate this, EVT assumes asymptotic threshold stability that may be challenged by unprecedented black swan events.
3. **Execution & Liquidity Friction:**  
   Our backtests and rebalancing simulations assume continuous daily liquidity at closing prices with zero bid-ask spread and zero market impact. During an interbank liquidity freeze, liquidating large ETF positions could incur execution slippage.
4. **Non-Linear Payoff & Convexity Absence:**  
   The current investable universe consists of linear delta-one ETF instruments. It does not incorporate asymmetric derivative overlays (such as out-of-the-money put swaptions or tail-risk collars) that could provide convexity during sudden crash events.
5. **Macroeconomic Regime Shift Risk:**  
   A prolonged 1970s stagflationary regime or sovereign debt crisis could break the historical negative correlation between equities and government bonds for multiple years, elevating the baseline portfolio volatility.

---

## 11. IMPLEMENTATION PLAN & ONGOING RISK GOVERNANCE

### Phase 1: Execution & Mandate Funding (Month 1)
- Deploy $10,000,000 across the 7-asset universe strictly at target weights using TWAP (Time-Weighted Average Price) algorithmic execution over 5 trading days to minimize market impact.

### Phase 2: Ongoing Rebalancing Corridors
- Adopt a **Tolerance Band Corridor of +/- 3.0%** around strategic weights. Rebalancing is triggered quarterly or dynamically whenever any asset breaches its corridor.

### Phase 3: Supervisory Risk Committee Reporting
- Weekly dashboard reporting of Historical, Student-t, and EVT Expected Shortfall.
- Monthly tracking of Basel Traffic Light exceptions. If the trailing 250-day window registers >= 5 exceptions (entering Yellow Zone), automatic risk committee review and position recalibration are mandated.

---

**RESPECTFULLY SUBMITTED BY:**  
**Team "Pandas Predictors"**  
Quantitative Asset Management & Risk Engineering Division  
QuantEdge 1.0 National STEM Competition  
SAIFA (University of Colombo) & CFA Society Sri Lanka
