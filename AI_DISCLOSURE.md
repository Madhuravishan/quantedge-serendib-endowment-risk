# 🤖 AI & Coding Assistant Disclosure (Section 8.7 Compliance)

**Competition:** QuantEdge 1.0 Final Challenge — National STEM Quantitative Investment Research Competition  
**Organizers:** Society of Actuarial and Financial Analytics (SAIFA, University of Colombo) & CFA Society Sri Lanka  
**Mandate:** The Serendib University Endowment Fund (USD 10,000,000 AUM)  
**Team:** "Pandas Predictors" — Quantitative Research Division  
**Date:** October 2026  

---

## 1. Formal Disclosure Table (Section 8.7 Mandate)

| Item | Required Disclosure | Team "Pandas Predictors" Execution & Implementation |
| :--- | :--- | :--- |
| **AI Tool(s) Used** | E.g., Claude, ChatGPT, GitHub Copilot | **Google Antigravity / Gemini 3.8 Flash** |
| **Purpose of AI Use** | E.g., debugging, Streamlit layout, code scaffolding, documentation support | **Scaffolding vectorized pipelines, drafting Streamlit CSS layouts, generating Pytest test harnesses, and automating markdown report formatting.** |
| **Human Validation Process** | How the team tested, checked, and revised generated output | **Every mathematical equation (Markowitz SLSQP QP, Cornish-Fisher expansion, GPD MLE, Kupiec LR, Christoffersen Markov transitions, Euler risk decomposition) was independently derived from academic literature, cross-checked against closed-form benchmarks, and verified via 22 automated unit tests.** |
| **Material AI-Generated Components** | Specific components assisted by AI | *"Initial Streamlit dashboard UI structure and Pytest test fixtures were AI-assisted; all quantitative risk engines, parameter fits, EVT threshold justifications, stress testing scenarios, and fiduciary recommendations were independently designed, evaluated, and validated by the team."* |

---

## 2. Detailed Technical Breakdown of AI Assistance

### A. Code Scaffolding & Vectorization
- **AI Contribution:** Provided boilerplate for FastAPI/Streamlit components, Plotly dark-theme styling, and initial function type annotations.
- **Human Revision:** Rewrote computational bottlenecks to use pure NumPy/SciPy vectorized linear algebra (e.g., Cholesky matrix factorizations for 10,000 Monte Carlo paths) rather than slow iterative loops.

### B. Automated Testing Harness (`pytest`)
- **AI Contribution:** Suggested unit test scaffolding and parameter ranges.
- **Human Revision:** Verified test assertions against known theoretical properties:
  - Augmented Dickey-Fuller (ADF) stationarity on synthetic Gaussian white noise.
  - Expected Shortfall subadditivity axiom: $\text{ES}_\alpha(X + Y) \le \text{ES}_\alpha(X) + \text{ES}_\alpha(Y)$.
  - Monotonicity of Value-at-Risk across confidence levels: $\text{VaR}_{99\%} > \text{VaR}_{95\%} > \text{VaR}_{90\%}$.
  - Euler risk budgeting identity: $\sum_{i=1}^N w_i \cdot \text{MVaR}_i = \text{VaR}_p$.

### C. Frontend Interface Design (`app.py`)
- **AI Contribution:** Structured Streamlit multi-tab layout and CSS theme tokens.
- **Human Revision:** Tailored layout for institutional readability, ensuring side-by-side presentation of VaR, ES, and EVT to clearly highlight the "Gaussian Blindspot" for non-technical committee members.

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
