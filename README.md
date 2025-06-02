# Smooth Transition Regression (STR) for Macroeconomic Drivers of S&P 500

A Python workflow that models U.S. macro-financial data with a logistic Smooth Transition Regression to explain monthly S&P 500 returns.
> **Heads-up:** `Code.py` is a **concatenation of several smaller scripts/notebooks that were originally run separately**, so you’ll notice duplicated data-loading, repeated tests and alternative model blocks. They don’t change the results but could be refactored for clarity.

## Repo layout
| Path        | Purpose |
|-------------|---------|
| `Code.py`   | End-to-end script: load data → ADF tests → OLS benchmark → STR estimation & checks.|
| `STR.pdf`   | 20-page write-up with theory, data description and discussion of results.
| `figures/`  | Auto-generated plots |
| `data/`     | Raw CSV/XLSX inputs from FRED & S&P Global |

## Highlights
- Integrated stationarity checks, visual diagnostics and model evaluation.  
- Logistic STR with time-varying coefficients plus residual, normality and heteroskedasticity tests.
## License
MIT

---

© 2024-25 Bruno Sciascia
