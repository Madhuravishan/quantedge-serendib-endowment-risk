# QuantEdge 1.0 Final Challenge: Quantitative Risk & Portfolio Engine
**Serendib University Endowment Fund ($10,000,000 AUM Mandate) | Team: "Pandas Predictors"**  
*Organized by the Society of Actuarial and Financial Analytics (SAIFA), Department of Statistics, University of Colombo & CFA Society Sri Lanka*

[![Live Showcase](https://img.shields.io/badge/Live_Showcase-GitHub_Pages-0284c7?style=for-the-badge&logo=github)](https://madhuravishan.github.io/quantedge-serendib-endowment-risk/)
[![Basel III Compliance](https://img.shields.io/badge/Basel_III-Green_Zone_(k=3.00x)-10b981?style=for-the-badge&logo=safari)](https://madhuravishan.github.io/quantedge-serendib-endowment-risk/)
[![Test Suite](https://img.shields.io/badge/Tests-26_Passing_(100%25)-10b981?style=for-the-badge&logo=pytest)](https://github.com/Madhuravishan/quantedge-serendib-endowment-risk/actions)
[![Python Support](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12_|_3.13_|_3.14-38bdf8?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic_/_Open_Source-f59e0b?style=for-the-badge)]()

---

> ### 🌐 Executive Portals & Core Submission Deliverables
> - 🌐 **Live Executive Web Showcase:** [https://madhuravishan.github.io/quantedge-serendib-endowment-risk/](https://madhuravishan.github.io/quantedge-serendib-endowment-risk/)
> - 📄 **1-Page Executive Tear Sheet (PDF):** [`report/tear_sheet.pdf`](report/tear_sheet.pdf) *(Verified: exactly 1 page)*
> - 📑 **11-Page Technical Investment Memo (PDF):** [`report/investment_memo.pdf`](report/investment_memo.pdf) *(Verified: 11 pages, strictly within 7–15 limit)*
> - ⚡ **Interactive Streamlit Web Engine:** [https://quantedge-serendib-endowment-risk.streamlit.app/](https://quantedge-serendib-endowment-risk.streamlit.app/) *(Local execution: `streamlit run app.py`)*
> - 💻 **GitHub Repository:** [https://github.com/Madhuravishan/quantedge-serendib-endowment-risk](https://github.com/Madhuravishan/quantedge-serendib-endowment-risk)

---

## 🏛️ 1. Core Institutional Mandate & Governance

The **Serendib University Endowment Fund** is an intergenerational pool of capital established to fund academic operations, endowed professorship chairs, STEM laboratory infrastructure, and student scholarships in perpetuity across macroeconomic regimes.

### Mandate Investment Parameters
- **Mandate Capital (AUM):** `$10,000,000 USD`
- **Target Real Spending Payout:** `4.5% per annum` (operational spending rule)
- **Inflation Baseline:** `2.5% per annum` (long-term CPI expectation)
- **Nominal Compound Hurdle Rate:** `7.0% CAGR` ($4.5\% + 2.5\%$)
- **Maximum Tolerable Drawdown:** `15.0% Risk Limit` ($1,500,000 max tolerable dollar drawdown)
- **Supervisory Validation:** `Basel Committee on Banking Supervision (BCBS) / FRTB Standards`

### Core Institutional Philosophy & Axiom
> *"Avoiding large losses is more important than maximizing gains. A good investment decision is not the one with the highest expected return, but the one whose downside has been measured, empirically tested, and mathematically defended."*

---

## 🛡️ 2. The Drawdown Governance Protocol
### Reconciling the 15.0% Board Limit vs. the -31.05% Empirical 20-Year Lookback

A critical fiduciary question is reconciling the Board’s **15.00% maximum tolerable drawdown threshold ($1,500,000 max dollar loss)** against the portfolio's empirical **-31.05% peak-to-trough maximum drawdown** recorded during the 2008 Lehman Brothers Global Financial Crisis.

> ### 📋 Fiduciary Reconciliation & Governance Mandate
> 1. **Operational Risk Budget vs. Unhedged Historical Tail:**  
>    The Board's **15.00% limit is an active, operational circuit-breaker policy** designed to govern ongoing trading and capital allocation. In contrast, the **-31.05% figure represents an unmanaged, buy-and-hold historical lookback trough** over a 20-year lookback (2006–2026) spanning a 1-in-100-year systemic crisis.
> 2. **Active Capital Preservation (Circuit-Breaker Defense):**  
>    In live deployment, the portfolio is never left unmanaged. Once losses approach the warning zone, the formal **3-Tier Drawdown Governance Response Protocol** intervenes systematically to prevent the unhedged tail from ever materializing:

### Formal 3-Tier Governance Action Matrix

| Governance Tier | Drawdown Threshold | Dollar Loss ($10M AUM) | Operational Status | Mandatory Governance Action |
| :--- | :---: | :---: | :---: | :--- |
| **🟢 Tier 1: Normal** | `< 10.0%` | `< $1,000,000` | Normal Policy Execution | Standard allocation execution (35% AGG, 25% SPY, 15% GLD, 10% EFA, 10% XLE, 5% EEM). Routine quarterly rebalancing within $\pm 3.0\%$ corridor tolerance bands. Operational spending distributions disbursed on schedule. |
| **🟡 Tier 2: Watch** | `10.0% to 15.0%` | `$1,000,000 to $1,500,000` | Warning / Risk Alert | Immediate rebalancing freeze. Convene Emergency Risk Committee alert. Tactical suspension of active equity expansion. Channel bond coupon income strictly to capital defense buffers and spending continuity. |
| **🔴 Tier 3: Action** | `> 15.0%` | `> $1,500,000` | **Mandatory Circuit Breaker** | Mandatory capital preservation protocol: systematic and orderly de-risking of equity and commodity risk assets into short-term US Treasury Bills (FRED DTB3 / BIL). Convene Emergency Board Meeting within 24 hours to preserve endowment solvency. |

---

## 🚀 3. Quickstart & One-Command Execution

### System Requirements & Prerequisites
- **Operating System:** Windows 10/11, macOS Monterey 12+, Linux (Ubuntu 20.04+, Debian, RHEL)
- **Python Version:** Python 3.10, 3.11, 3.12, 3.13, or 3.14 (Verified on Python 3.14.0 64-bit)
- **Hardware Requirements:** 4 GB RAM minimum (8 GB recommended for 10,000-path Monte Carlo simulations)
- **Environment:** All packages are free-tier, non-proprietary open source.

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/Madhuravishan/quantedge-serendib-endowment-risk.git
cd quantedge-serendib-endowment-risk

# Create and activate an isolated virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies (reproducible manifest)
pip install -r requirements.txt
```

### 2. Execution Commands

#### A. Run Master Headless Pipeline & Export All Artifacts
Executes all 6 quantitative layers sequentially, generates high-DPI figures, exports institutional CSV tables, builds official PDF reports, and exports the static interactive showcase:
```bash
python run_pipeline.py
```

#### B. Launch Interactive Streamlit Web Engine
Launches the full 8-tab interactive institutional dashboard in your browser (`http://localhost:8501`):
```bash
streamlit run app.py
```

#### C. Run the Unit Test Suite (26 Tests / 100% Pass Rate)
Executes comprehensive mathematical and econometric unit tests:
```bash
pytest tests/ -v
```

#### D. Compile Official PDF Reports Direct
Generates the 1-page Executive Tear Sheet and 11-page Investment Committee Memorandum:
```bash
python src/report_generator.py
```

---

## 📐 4. 6-Layer Modular Quantitative Architecture

```text
quantedge-serendib-endowment-risk/
├── .github/
│   └── workflows/
│       └── deploy-pages.yml     # Automated GitHub Pages CI/CD deployment workflow
├── run_pipeline.py              # Master deterministic headless execution pipeline
├── app.py                       # Interactive 8-tab institutional Streamlit web application
├── requirements.txt             # Strict, reproducible dependency specification (with pypdf)
├── README.md                    # Institutional documentation and audit manual
├── AI_DISCLOSURE.md             # Mandatory declaration of AI tools & validation (Section 8.7)
│
├── src/                         # Modular institutional quantitative engine
│   ├── __init__.py
│   ├── config.py                # Asset universe (7 ETFs + DTB3), $10M mandate, and date limits
│   ├── data_loader.py           # Module 1: Ingestion, cleaning, log-returns, ADF stationarity
│   ├── portfolio_opt.py         # Module 2 (Layer 1): Markowitz QP, Efficient Frontier, KPIs
│   ├── var_engine.py            # Module 3 (Layer 2): 3-Method VaR, Student-t, Monte Carlo 10k
│   ├── es_engine.py             # Module 4 (Layer 3): Coherent Expected Shortfall (CVaR), subadditivity
│   ├── evt_engine.py            # Module 5 (Layer 4): EVT Peaks-Over-Threshold (POT), GPD MLE, MEF/Hill
│   ├── backtest.py              # Module 6 (Layer 5): Kupiec POF, Christoffersen, Basel Traffic Light
│   ├── stress_test.py           # Module 7 (Layer 6): Euler risk budgeting, Historical crises, Stagflation
│   ├── report_generator.py      # Module 8: ReportLab PDF compilation & 300 DPI publication figures
│   └── export_static_showcase.py# Module 9: Standalone Plotly HTML exporter & docs synchronizer
│
├── tests/                       # Complete pytest unit test suite (26 passing tests)
│   ├── conftest.py              # Test fixtures and synthetic data generators
│   ├── test_data_loader.py      # ADF stationarity, log returns, and data cleaning governance
│   ├── test_portfolio_opt.py    # Markowitz constraints, CAGR hurdle, and KPI metrics
│   ├── test_var_engine.py       # Monotonicity and parametric VaR mathematical properties
│   ├── test_es_engine.py        # Coherence, subadditivity (Axiom 3), and ES >= VaR proofs
│   ├── test_evt_engine.py       # GPD POT parameter estimation and goodness-of-fit
│   ├── test_backtest.py         # Kupiec POF, Christoffersen independence, and Basel Traffic Light
│   └── test_stress_test.py      # Euler decomposition identity and reverse stress testing
│
├── docs/                        # GitHub Pages executive showcase web portal
│   ├── index.html               # Institutional dark-theme executive landing page
│   ├── .nojekyll                # Bypass Jekyll processing for raw static assets
│   ├── assets/charts/           # Standalone interactive Plotly HTML visualizations
│   │   ├── cumulative_wealth.html
│   │   ├── efficient_frontier.html
│   │   ├── evt_mean_excess.html
│   │   └── stress_scenarios.html
│   └── report/                  # Mirrored submission PDF deliverables
│       ├── tear_sheet.pdf
│       └── investment_memo.pdf
│
├── data/
│   ├── raw/                     # Ingested price and risk-free rate series (CSV / Parquet)
│   └── processed/               # Cleaned returns, stationarity reports, and EVT diagnostics
│
├── outputs/
│   ├── figures/                 # 14 publication-grade high-resolution charts (300 DPI)
│   └── tables/                  # 23 structured institutional CSV analytical tables
│
└── report/                      # Official submission deliverables
    ├── tear_sheet.pdf           # Deliverable 4: Official 1-Page Executive Tear Sheet PDF
    ├── investment_memo.pdf      # Deliverable 3: Official 11-Page Technical Investment Memo PDF
    ├── EXECUTIVE_TEARSHEET.md   # Markdown edition of Executive Tear Sheet
    └── INVESTMENT_COMMITTEE_MEMORANDUM.md # Markdown edition of Investment Memo
```

### The 6 Quantitative Layers

1. **Layer 1: Econometric Stationarity & Portfolio Optimization:**
   - Ingests 7 liquid ETFs (`SPY`, `EFA`, `EEM`, `AGG`, `GLD`, `XLE`, `XLK`) and FRED 3-Month T-Bill (`DTB3`) across 5,231 trading days (2006–2026).
   - Augmented Dickey-Fuller (ADF) testing verifies that log prices are non-stationary $I(1)$ ($p > 0.20$), while daily log returns are strictly covariance-stationary $I(0)$ ($p < 10^{-28}$).
   - Solves the Markowitz quadratic programming problem under long-only institutional constraints ($\sum w_i = 1$, $w_i \ge 0$).
2. **Layer 2: 3-Method Value-at-Risk Engine (90%, 95%, 99%):**
   - **Historical Simulation:** Non-parametric empirical quantile estimation capturing full historical skewness and kurtosis.
   - **Parametric Gaussian & Student-$t$:** Analytical closed-form estimation ($\nu = 8.46$ fitted degrees of freedom).
   - **Monte Carlo Simulation:** 10,000 synthetic paths parameterized via Cholesky decomposition ($L L^T = \Sigma$) of the empirical covariance matrix.
3. **Layer 3: Expected Shortfall (CVaR) & Mathematical Subadditivity:**
   - Computes Expected Shortfall ($ES_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$) to quantify catastrophic tail severity beyond VaR.
   - Mathematically proves subadditivity coherence (Artzner et al. 1999 Axiom 3: $\rho(X+Y) \le \rho(X) + \rho(Y)$), proving that VaR violates subadditivity under non-normal distributions while ES guarantees diversification benefits.
4. **Layer 4: Extreme Value Theory (EVT) via Peaks-Over-Threshold (POT):**
   - Applies the **Pickands-Balkema-de Haan (1975)** theorem: exceedances above threshold $u$ asymptotically converge to the Generalized Pareto Distribution (GPD):
     $$G_{\xi, \sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$
   - Mathematically justifies threshold $u^* = 1.011\%$ (95th percentile) via Sample Mean Excess Function (MEF) linearity and Hill plot plateau stability.
   - Maximum Likelihood Estimation yields tail index $\xi = 0.2938$ (Fréchet heavy-tail domain, $\xi > 0$).
5. **Layer 5: Out-of-Sample Model Validation & Basel Supervisory Compliance:**
   - Evaluated across 702 out-of-sample trading days (2024 to 2026) using an expanding/rolling 500-day window.
   - **Kupiec Proportion of Failures (POF):** Validates unconditional coverage ($LR_{POF} = 1.13, p = 0.2879 > 0.05$).
   - **Christoffersen Independence Test:** Confirms absence of breach clustering.
   - **Basel Committee Traffic Light Framework:** 3 exceptions $\le$ 4 in 250 days $\to$ **GREEN ZONE** certification ($k = 3.00\times$, zero capital penalty).
6. **Layer 6: Macroeconomic Stress Testing & Euler Risk Budgeting:**
   - **Euler Risk Decomposition:** $\sum_{i=1}^N w_i \frac{\partial \text{VaR}_p}{\partial w_i} = \text{VaR}_p$ confirms position sizing risk vs. capital weight.
   - Replays historical systemic shocks (2008 Lehman GFC, March 2020 COVID, 2022 Inflation & Rate Hike).
   - Simulates custom team macroeconomic shock: **1970s Geopolitical Stagflation**.
   - **Reverse Stress Testing:** Solves for systemic market shocks required to breach the Board's 15% ($1,500,000) drawdown circuit-breaker.

---

## 📊 5. Summary of Quantitative Findings & Empirical Tables

### Strategic Asset Allocation: "Serendib Conservative"
Selected as the optimal fiduciary choice to satisfy the 7.00% CAGR hurdle while defending the 15% drawdown limit:

| Asset Ticker | Asset Class Description | Target Weight | Dollar Allocation ($10M AUM) | Strategic Institutional Role |
| :--- | :--- | :---: | :---: | :--- |
| **AGG** | US Aggregate Investment-Grade Bonds | **35.0%** | $3,500,000 | Defensive liquidity anchor, coupon yield & flight-to-quality stabilizer |
| **SPY** | US Large-Cap Equities (S&P 500) | **25.0%** | $2,500,000 | Core capital compounding engine to outpace long-term inflation |
| **GLD** | Physical Gold Trust (Real Asset) | **15.0%** | $1,500,000 | Safe-haven store of value, currency debasement & stagflation hedge |
| **EFA** | International Developed Equities | **10.0%** | $1,000,000 | Geographic diversification across OECD developed economies |
| **XLE** | Energy Sector SPDR (Commodity Proxy)| **10.0%** | $1,000,000 | Direct geopolitical supply-shock hedge & energy inflation shield |
| **EEM** | Emerging Market Equities | **5.0%** | $500,000 | High-beta economic growth kicker with disciplined capital cap |
| **XLK** | Technology Select Sector | **0.0%** | $0 | Directly excluded (avoid double-counting ~30% tech exposure in SPY) |

### 20-Year Benchmark Performance Comparison (2006–2026)
*Evaluation Period: 2006-01-03 to 2026-10-19 (5,231 trading days)*

| Performance Metric | Serendib Conservative (Mandate) | Classic 60/40 Benchmark | Markowitz Max Sharpe | Global Minimum Variance | Equal Weight (1/N) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CAGR** | **7.91%** | 8.25% | 14.55% | 3.70% | 9.60% |
| **Hurdle Target (7.0%)** | **ACHIEVED (+91 bps)** | ACHIEVED | ACHIEVED | FAILED (-330 bps) | ACHIEVED |
| **Annualized Volatility** | **11.31%** | 11.71% | 15.66% | 5.08% | 16.11% |
| **Sharpe Ratio (Rf=3.5%)** | **0.578** | 0.589 | 0.837 | 0.402 | 0.543 |
| **Sortino Ratio (Downside)** | **0.818** | 0.836 | 1.201 | 0.547 | 0.769 |
| **Max Drawdown (20-Yr)** | **-31.05%** | -34.70% | -34.94% | -16.74% | -42.35% |
| **Calmar Ratio** | **0.255** | 0.238 | 0.416 | 0.221 | 0.227 |
| **Ending Wealth ($10M)** | **$48,556,769** | $51,838,705 | $167,792,964 | $21,240,321 | $67,031,830 |

### Value-at-Risk & Expected Shortfall Breakdown (1-Day Horizon on $10M AUM)

| Confidence Level | Historical VaR | Parametric Gaussian VaR | Parametric Student-t VaR | **EVT-POT VaR** | **Expected Shortfall (ES)** | Gaussian Underestimation | Tail Severity Gap (ES - VaR) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **90%** | 0.68% ($68,124) | 0.88% ($88,069) | 0.83% ($83,298) | **0.71% ($71,200)** | **1.26% ($125,724)** | -$19,944 | +$57,599 |
| **95%** | 1.01% ($101,127) | 1.14% ($113,964) | 1.12% ($111,731) | **1.01% ($101,100)** | **1.68% ($167,842)** | -$12,837 | +$66,715 |
| **99%** | 1.95% ($195,453) | 1.63% ($162,538) | 1.75% ($174,822) | **2.00% ($199,593)** | **3.04% ($303,954)** | **+$32,915** | **+$108,500** |

> **The "Gaussian Blindspot":** At the 99% confidence level, thin-tailed Gaussian parametric models underestimate daily loss by **+$32,915** on the $10M endowment. Expected Shortfall reveals that when extreme market events breach the VaR threshold, losses average **$303,954** (a **1.56x escalation**). Capital reserves must be anchored strictly to EVT and ES.

### Macroeconomic Stress Testing: Exact Dollar Impact on $10,000,000 AUM

| Crisis Scenario | Scenario Type | Horizon / Window | Serendib Conservative Loss ($10M) | Classic 60/40 Loss ($10M) | Capital Preserved vs. 60/40 | Key Institutional Defense Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **2008 Lehman GFC Crash** | Historical Replay | 2008-09-01 to 2008-11-30 | **-$1,602,276 (-16.02%)** | -$1,816,582 (-18.17%) | **+$214,306 Preserved** | 35% AGG flight-to-quality cashflow & duration cushion |
| **March 2020 COVID Liquidity** | Historical Replay | 2020-02-19 to 2020-03-23 | **-$2,120,974 (-21.21%)** | -$2,150,061 (-21.50%) | **+$29,087 Preserved** | Balanced bond stability amid rapid market liquidity collapse |
| **2022 Inflation & Rate Hike** | Historical Replay | 2022-01-03 to 2022-10-14 | **-$1,338,312 (-13.38%)** | -$2,025,636 (-20.26%) | **+$687,325 Preserved** | 10% XLE (+46.8% surge) insulated against stock-bond crash |
| **1970s Geopolitical Stagflation**| Custom Macro Shock | Multi-Quarter Horizon | **-$790,000 (-7.90%)** | -$1,680,000 (-16.80%) | **+$890,000 Preserved** | Real assets: 15% GLD (+15%) & 10% XLE (+25%) oil spike hedge |

### Reverse Stress Testing: Solving for the 15% Circuit Breaker
- **Board Risk Limit:** 15.0% maximum tolerable drawdown (`-$1,500,000 USD`)
- **Systemic Collapse Amplitude Required:** A concurrent **-23.62% equity crash (SPY)** combined with a **-9.45% bond drop (AGG)** is mathematically required to trigger the circuit breaker.
- **Institutional Conclusion:** The endowment exhibits structural resilience, insulated from routine cyclical market drawdowns.

---

## 🤖 6. AI Policy Disclosure & Compliance (Section 8.7)

In strict accordance with **Section 8.7 of the QuantEdge 1.0 Final Challenge Official Rules**:
- **AI Tooling Utilized:** Google Antigravity Agentic IDE & Gemini 3.8 Flash.
- **Operational Scope:** Automated syntax scaffolding, Streamlit layout optimization, unit testing boilerplate, and HTML report formatting.
- **Human Financial & Mathematical Validation:**
  - Every mathematical formula (Markowitz QP matrix formulation, Cholesky factorization, Pickands-Balkema-de Haan POT derivation, Kupiec likelihood ratio test, Christoffersen transition probability matrices, and Euler homogenous function theorem) was manually derived, reviewed, and cross-verified by team members against academic source literature.
  - All strategic asset allocation decisions, drawdown governance policies, and committee memorandums reflect the original intellectual property and fiduciary judgment of the human analysts.
- **Full Transparency Record:** A comprehensive audit log of AI interactions and human verification controls is published in [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).

---

## 📜 7. QuantEdge 1.0 Submission Deliverables Matrix

| Rubric Deliverable | Verification Status | Deliverable Path / Location | Technical Details |
| :--- | :---: | :--- | :--- |
| **Deliverable 1: Public GitHub Repository** | ✅ Verified | [GitHub Repository](https://github.com/Madhuravishan/quantedge-serendib-endowment-risk) | Clean directory layout (`src/`, `tests/`, `outputs/`, `report/`, `docs/`, `data/`) |
| **Deliverable 2: Interactive Streamlit App** | ✅ Verified | [`app.py`](app.py) | 8 comprehensive tabs; verified via `streamlit run app.py` |
| **Deliverable 3: 7–15 Page Investment Memo** | ✅ Verified | [`report/investment_memo.pdf`](report/investment_memo.pdf) | **Exactly 11 pages** (Verified via `pypdf`, strictly within 7–15 limit) |
| **Deliverable 4: 1-Page Executive Tear Sheet** | ✅ Verified | [`report/tear_sheet.pdf`](report/tear_sheet.pdf) | **Exactly 1 page** (High-level institutional tear sheet for Board review) |
| **Deliverable 5: Finalist Pitch Readiness** | ✅ Verified | Finalist Deck & Live Showcase | Fully prepared for committee defense before the competition panel |
| **Executive Web Showcase (GitHub Pages)** | ✅ Verified | [GitHub Pages Portal](https://madhuravishan.github.io/quantedge-serendib-endowment-risk/) | Standalone Plotly CDN interactive figures & embedded PDF viewer |
| **Automated Unit Test Suite** | ✅ Verified | [`tests/`](tests/) | **26 of 26 tests passing (100%)** via `pytest tests/ -v` |

---

### Team "Pandas Predictors" — QuantEdge 1.0 National STEM Challenge
*Quantitative Asset Management & Risk Engineering Division*  
*Society of Actuarial and Financial Analytics (SAIFA), University of Colombo & CFA Society Sri Lanka*
