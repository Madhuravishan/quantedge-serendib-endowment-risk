# AI & Coding Assistant Disclosure (Section 8.7 Compliance)

**Competition:** QuantEdge 1.0 Final Challenge — National STEM Quantitative Investment Research Competition  
**Organizers:** Society of Actuarial and Financial Analytics (SAIFA, University of Colombo) & CFA Society Sri Lanka  
**Mandate:** The Serendib University Endowment Fund (USD 10,000,000 AUM)  
**Team:** "Pandas Predictors" — Quantitative Research Division  
**Date:** October 2026  

---

## 1. Formal Disclosure Table (Section 8.7 Mandate)

| Item | Required Disclosure | Team "Pandas Predictors" Execution & Implementation |
| :--- | :--- | :--- |
| **AI Tool(s) Used** | E.g., Claude, ChatGPT, GitHub Copilot | **Google Antigravity IDE (Gemini 3.8 Flash), Claude (Anthropic), and ChatGPT / GPT-4o (OpenAI)** |
| **Purpose of AI Use** | E.g., debugging, Streamlit layout, code scaffolding, documentation support | **Scaffolding vectorized data pipelines, drafting Streamlit CSS layouts, generating Pytest test harnesses, boilerplate mathematical syntax wrappers, and automating Markdown report formatting.** |
| **Human Validation Process** | How the team tested, checked, and revised generated output | **Every mathematical equation (Markowitz SLSQP Quadratic Programming, Cornish-Fisher expansion, Generalized Pareto Distribution Maximum Likelihood Estimation, Kupiec POF Likelihood Ratio, Christoffersen Markov transitions, Euler risk decomposition) was independently derived from academic literature, cross-checked against closed-form benchmarks, and verified via 26 automated unit tests.** |
| **Material AI-Generated Components** | Specific components assisted by AI | *"AI assistance was utilized strictly for code scaffolding, boilerplate math libraries, Streamlit dashboard UI layout, and initial Pytest unit test fixtures; all investment hypotheses, Serendib endowment weighting rationales, macroeconomic scenario designs (1970s stagflation shock), EVT threshold selections (u* = 95th percentile), and final board/committee recommendations were independently designed, evaluated, and validated by the human team."* |

---

## 2. Separation of Concerns: AI vs. Human Responsibilities

To maintain the highest standard of academic rigor and institutional fiduciary accountability, Team "Pandas Predictors" implemented a strict **Separation of Concerns**:

```
┌─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│                 AI Coding Assistant Scope                   │                    Human Research Team                      │
│                (Antigravity, Claude, GPT)                   │                  (Team Pandas Predictors)                   │
├─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ • Scaffolding vectorized data ingestion functions           │ • Formulating the Serendib endowment asset allocation thesis│
│ • Boilerplate syntax wrappers for SciPy / NumPy / Statsmodels│ • Determining 7-asset ETF universe constraints (XLK 0%, EEM 5%)│
│ • Streamlit multi-tab layout and CSS styling tokens         │ • Macroeconomic stress scenario design (1970s stagflation) │
│ • ReportLab PDF document geometry and canvas boilerplates   │ • EVT Peaks-Over-Threshold selection (u* = 95th percentile) │
│ • Generating initial Pytest unit test skeleton structures   │ • Econometric validation & Kupiec/Christoffersen derivations│
│ • Markdown report layout templating and table formatting    │ • Investment Committee formal recommendation & limits       │
└─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### A. Code Scaffolding & Vectorization
- **AI Contribution:** Provided boilerplate syntax for FastAPI/Streamlit components, Plotly dark-theme styling, ReportLab flowable wrappers, and initial Python type annotations.
- **Human Revision:** Optimized computational bottlenecks using pure NumPy/SciPy vectorized linear algebra (e.g., Cholesky matrix factorizations for 10,000 Monte Carlo paths) rather than slow iterative loops.

### B. Automated Testing Harness (`pytest`)
- **AI Contribution:** Suggested unit test scaffolding and parameter boundary structures.
- **Human Revision:** Verified all 26 test assertions against known theoretical and mathematical properties:
  - Augmented Dickey-Fuller (ADF) stationarity on synthetic Gaussian white noise and random walks.
  - Expected Shortfall subadditivity axiom: $\text{ES}_\alpha(X + Y) \le \text{ES}_\alpha(X) + \text{ES}_\alpha(Y)$.
  - Monotonicity of Value-at-Risk and Expected Shortfall: $\text{ES}_{99\%} \ge \text{VaR}_{99\%} \ge \text{VaR}_{95\%} \ge \text{VaR}_{90\%}$.
  - Portfolio weight constraints: $\sum w_i = 1.0$ and $w_i \ge 0.0$ (no short selling).
  - Out-of-sample date slicing calendar alignment with zero lookahead bias.
  - Euler risk budgeting identity: $\sum_{i=1}^N w_i \cdot \text{MVaR}_i = \text{VaR}_p$.

### C. Frontend Interface Design (`app.py`)
- **AI Contribution:** Structured Streamlit multi-tab layout, responsive columns, and CSS theme tokens.
- **Human Revision:** Tailored layout for institutional readability, ensuring side-by-side presentation of VaR, ES, and EVT to clearly highlight the "Gaussian Blindspot" for non-technical committee members.

### D. Qualitative Macroeconomic & Endowment Rationale
- **AI Contribution:** None.
- **Human Contribution:** Formulated the university endowment's 7.00% nominal return hurdle (4.5% spending payout + 2.5% inflation), the 15.0% maximum drawdown circuit-breaker, the exclusion of direct tech exposure (preventing tech overconcentration with SPY), and the inclusion of real assets (GLD 15%, XLE 10%) as essential hedges against 1970s-style geopolitical stagflation.

---

## 3. Human Fiduciary Ownership & Compliance Statement

In accordance with Section 8.7 of the QuantEdge Competition Brief:
1. **Explicit Disclosure:** The team has fully and transparently documented all AI tools utilized throughout this research.
2. **Analytical Independence:** AI tools were not used as a substitute for human analytical judgment. All investment hypotheses, macroeconomic stress scenarios (1970s stagflation shock), and portfolio weighting constraints (excluding direct XLK exposure, capping EEM at 5%) were formulated by the human team.
3. **Defense Preparedness:** Every member of Team "Pandas Predictors" understands, can explain, and is fully prepared to defend any line of code, formula derivation, or empirical finding before the judging panel during the final presentation and Q&A session.
4. **Fiduciary Responsibility:** The human team accepts 100% accountability for the accuracy, reproducibility, and ethical governance of all models and deliverables submitted.

---

**RESPECTFULLY SUBMITTED BY:**  
**Team "Pandas Predictors"**  
Quantitative Asset Management & Risk Engineering Division  
QuantEdge 1.0 Final Challenge
