# QuantEdge 1.0 Final Challenge: Quantitative Risk & Portfolio Engine
**Organizers:** SAIFA (University of Colombo) & CFA Society Sri Lanka  
**Team:** "Pandas Predictors"  
**Client Mandate:** The Serendib University Endowment Fund (USD 10,000,000 AUM)  
**Regulatory Compliance:** Basel Committee on Banking Supervision (BCBS) / FRTB Standards  

---

## 🏛️ Executive Summary & Client Mandate
The **Serendib University Endowment Fund** is an intergenerational pool of capital ($10M AUM) established to fund university academic operations, faculty research chairs, STEM laboratory infrastructure, and student scholarships in perpetuity.

### Core Institutional Philosophy & Axiom
> *"Avoiding large losses is more important than maximizing gains. A good investment decision is not the one with the highest expected return, but the one whose downside has been measured, empirically tested, and mathematically defended."*

### Key Mandate Targets
- **AUM:** $10,000,000 USD
- **Target Real Spending Payout:** 4.5% annual real distribution
- **Inflation Baseline:** 2.5% expected long-term CPI
- **Nominal Hurdle Rate:** 7.0% nominal CAGR
- **Maximum Drawdown Circuit-Breaker:** 15.0% maximum peak-to-trough tolerance

---

## 🚀 Quickstart & One-Command Execution

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies (Python 3.10–3.14):
```bash
git clone https://github.com/Madhuravishan/quantedge-serendib-endowment-risk.git
cd saifa_quant_round_02

# Install dependencies (All free-tier, non-proprietary open source)
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Quantitative Pipeline
To ingest data, run stationarity tests, calculate all 6 quantitative layers, validate Basel compliance, and generate executive reports in one command:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

### 4. Run the Full Unit Test Suite
```bash
pytest tests/ -v
```

---

## 📐 Quantitative Architecture & 6-Layer Modular Pipeline

```
saifa_quant_round_02/
├── run_pipeline.py          # Master deterministic execution runner
├── app.py                   # Multi-panel interactive Streamlit application
├── requirements.txt         # Open-source free-tier dependency manifest
├── README.md                # Comprehensive institutional documentation
├── src/
│   ├── config.py            # Mandate parameters, tickers, date ranges, Basel limits
│   ├── data_loader.py       # Module 1: Ingestion, cleaning, log-returns, ADF stationarity
│   ├── portfolio_opt.py     # Module 2 (Layer 1): Markowitz QP, Efficient Frontier, Institutional KPIs
│   ├── var_engine.py        # Module 3 (Layer 2): 3-Method VaR, Cornish-Fisher, Monte Carlo 10k, Risk Gap
│   ├── es_engine.py         # Module 4 (Layer 3): Expected Shortfall (CVaR), subadditivity proof
│   ├── evt_engine.py        # Module 5 (Layer 4): EVT Peaks-Over-Threshold (POT), GPD MLE, MEF/Hill plots
│   ├── backtest.py          # Module 6 (Layer 5): Kupiec POF, Christoffersen Independence, Basel Traffic Light
│   ├── stress_test.py       # Module 7 (Layer 6): Euler risk budgeting, Historical crises, Stagflation, Reverse stress
│   └── report_generator.py  # Automated generator for Tear Sheet and Investment Memo
├── tests/
│   ├── test_data_loader.py  # Unit tests for data cleaning & ADF testing
│   ├── test_portfolio_opt.py# Unit tests for Markowitz QP & KPIs
│   ├── test_var_engine.py   # Unit tests for 3 VaR methods & monotonicity
│   ├── test_es_engine.py    # Unit tests for ES & subadditivity
│   ├── test_evt_engine.py   # Unit tests for GPD fitting & EVT metrics
│   ├── test_backtest.py     # Unit tests for Kupiec, Christoffersen & Basel zones
│   └── test_stress_test.py  # Unit tests for Euler budgeting & stress tests
├── data/
│   ├── raw/                 # Ingested price and risk-free series
│   └── processed/           # Processed log/simple returns, ADF reports, VaR/ES matrices
└── reports/
    ├── EXECUTIVE_TEARSHEET.md              # 1-page executive tear sheet (<5 min read)
    └── INVESTMENT_COMMITTEE_MEMORANDUM.md  # 12-page comprehensive institutional memorandum
```

---

## 📊 Summary of Quantitative Results

### Recommended Asset Allocation: "Serendib Conservative"
- **AGG (35%):** US Investment Grade Fixed Income (Defensive liquidity & yield anchor)
- **SPY (25%):** US Large-Cap Equities (Core capital compounding engine)
- **GLD (15%):** Physical Gold (Monetary debasement & geopolitical safe haven)
- **EFA (10%):** International Developed Equities (Geographical diversification)
- **XLE (10%):** Energy Equities (Commodity inflation & stagflation insurance)
- **EEM (5%):** Emerging Market Equities (High-beta growth kicker, strictly capped)
- **XLK (0%):** Technology Sector (Excluded directly to avoid double-counting ~30% tech exposure inside SPY)

### Performance & Risk Benchmark Comparison (2006–2026)
| Metric | Serendib Conservative (Recommended) | Classic 60/40 Benchmark | Markowitz Max Sharpe | Minimum Variance | Equal Weight (1/N) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CAGR** | **7.91%** | 8.25% | 14.55% | 3.70% | 9.60% |
| **Annual Volatility** | **11.31%** | 11.71% | 15.66% | 5.08% | 16.11% |
| **Sharpe Ratio (Rf=3.5%)** | **0.578** | 0.589 | 0.837 | 0.402 | 0.543 |
| **Sortino Ratio** | **0.818** | 0.836 | 1.201 | 0.547 | 0.769 |
| **Max Drawdown (20-Yr)** | **-31.05%** | -34.70% | -34.94% | -16.74% | -42.35% |
| **Calmar Ratio** | **0.255** | 0.238 | 0.416 | 0.221 | 0.227 |
| **Ending Wealth ($10M Start)**| **$48,556,769** | $51,838,705 | $167,792,964 | $21,240,321 | $67,031,830 |

---

## 🛡️ Risk Management & Model Validation Highlights

### 1. 3-Method VaR & Expected Shortfall (1-Day Horizon on $10M AUM)
- **Historical Simulation VaR (99%):** 1.95% ($195,453)
- **Parametric Gaussian VaR (99%):** 1.63% ($162,538)
- **Parametric Student-t VaR (99%):** 1.75% ($174,822, fitted $\nu = 8.46$)
- **Extreme Value Theory (EVT-POT VaR 99%):** 2.00% ($199,593)
- **Expected Shortfall (CVaR 99%):** **3.04% ($303,954)**
- **The "Risk Gap":** Gaussian parametric models underestimate 99% tail losses by **+$32,915** on a single day. Expected Shortfall reveals that when the 99% barrier is breached, losses escalate by **1.56x** ($304,000 expected loss).

### 2. Basel Committee Traffic Light Backtesting Status
- **Window:** 702 Out-of-Sample Trading Days (2024 to 2026)
- **Traffic Light Result:** **GREEN ZONE** (Trailing 250-day exceptions: 3 $\le$ 4).
- **Capital Multiplier:** Baseline $k = 3.00x$ (Zero regulatory capital penalty).
- **Kupiec POF Test:** Passed ($LR = 1.13, p = 0.2879 \ge 0.05$).

### 3. Stress Testing & Stagflation Resilience
- **1970s Geopolitical Stagflation Shock:** (Equities -20%, Bonds -12%, Gold +15%, Energy +25%)
  - Serendib Conservative Drawdown: **-7.90% (-$790,000)**
  - Classic 60/40 Drawdown: **-16.80% (-$1,680,000)**
  - **Capital Preserved by Serendib Allocation: +$890,000 USD**
- **Reverse Stress Circuit Breaker:** Solves for the systemic market crash amplitude that breaches the Board's 15% ($1.5M) drawdown limit: Requires a **-23.6% SPY equity crash** concurrent with a **-9.4% bond drop**.

---

## 🤖 AI Policy Compliance & Disclosure (Section 8.7)
In accordance with Section 8.7 of the QuantEdge Competition Brief:
- **AI Tool Used:** Google Antigravity / Gemini 3.8 Flash.
- **Purpose:** Vectorized pipeline scaffolding, Streamlit CSS layout, and automated unit testing (`pytest`).
- **Human Analytical Validation:** Every quantitative model (Cholesky factorization, GPD Peaks-Over-Threshold MLE, Kupiec LR test, Christoffersen Markov transition probabilities, Euler risk decomposition) was independently derived, verified, and cross-referenced with closed-form mathematical equations by the human team.
- **Ownership:** The human quantitative research team takes 100% fiduciary responsibility for all assumptions, portfolio recommendations, and committee memorandums.

---

## 📜 Deliverables Checklist
- [x] **Reproducible GitHub Repository:** Clean, modular structure with `run_pipeline.py`.
- [x] **Interactive Streamlit Web App:** Runnable locally via `streamlit run app.py`.
- [x] **Executive Tear Sheet:** [reports/EXECUTIVE_TEARSHEET.md](file:///c:/Users/Madhura/OneDrive/Desktop/saifa_quant_round_02/reports/EXECUTIVE_TEARSHEET.md) (<5 min read).
- [x] **Formal Investment Committee Memorandum:** [reports/INVESTMENT_COMMITTEE_MEMORANDUM.md](file:///c:/Users/Madhura/OneDrive/Desktop/saifa_quant_round_02/reports/INVESTMENT_COMMITTEE_MEMORANDUM.md) (12-page comprehensive report).
- [x] **Full Pytest Suite:** 22 passing unit tests covering all mathematical modules.
