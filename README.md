# Smooth Transition Regression (STR) for Macroeconomic Drivers of S&P 500

A Python workflow that models U.S. macro-financial data with a logistic Smooth Transition Regression to explain monthly S&P 500 returns.

## Repo layout
| Path        | Purpose |
|-------------|---------|
| `Code.py`   | End-to-end script: load data → ADF tests → OLS benchmark → STR estimation & checks. :contentReference[oaicite:0]{index=0} |
| `STR.pdf`   | 20-page write-up with theory, data description and discussion of results. :contentReference[oaicite:1]{index=1} |
| `figures/`  | Auto-generated plots |
| `data/`     | Raw CSV/XLSX inputs from FRED & S&P Global |

## Highlights
- Integrated stationarity checks, visual diagnostics and model evaluation. :contentReference[oaicite:2]{index=2}  
- Logistic STR with time-varying coefficients plus residual, normality and heteroskedasticity tests. :contentReference[oaicite:3]{index=3}  

## License
MIT

---

© 2024-25 Bruno Sciascia
