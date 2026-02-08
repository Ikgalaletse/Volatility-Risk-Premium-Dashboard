# S&P 500 Volatility Risk Premium (VRP) Dashboard (Streamlit)

A Streamlit dashboard that computes and visualizes the **Volatility Risk Premium (VRP)** for the S&P 500 using:

- **Implied volatility:** VIX (annualized %)
- **Realized volatility:** Yang–Zhang estimator from OHLC prices (annualized %)
- **VRP:** `Implied Vol − Realized Vol` (in **percentage points**)

The app is robust to Yahoo Finance data issues, automatically switching between **^GSPC (index)** and **SPY (ETF)** when OHLC data is missing, and includes trading signals, cross-asset VRP comparison, a simple backtest, and macro correlations.

---

## What Is the Volatility Risk Premium (VRP)?

The **Volatility Risk Premium (VRP)** measures the difference between:

- **Implied volatility** – the market’s *expected* future volatility, inferred from option prices (e.g. the VIX), and  
- **Realized volatility** – the *actual* volatility that materializes in the underlying asset over time.

Formally:

VRP = Implied Volatility − Realized Volatility

### Intuition
- Options typically **price in a premium** because sellers demand compensation for bearing volatility risk.
- As a result, implied volatility tends to be **higher than realized volatility on average**, leading to a **positive VRP**.

### Interpretation
- **High VRP**  
  → Implied volatility is expensive relative to realized volatility  
  → Often interpreted as *overpriced volatility*  
  → Historically associated with favorable conditions for **selling volatility**

- **Low or negative VRP**  
  → Implied volatility is cheap relative to realized volatility  
  → Often occurs before or during market stress  
  → Can favor **buying volatility** for protection or convexity

In this dashboard:
- Implied volatility is proxied by the **VIX** (≈ 30-day forward-looking volatility).
- Realized volatility is estimated using the **Yang–Zhang estimator**, which efficiently combines overnight, intraday, and range-based information from OHLC prices.
- VRP is expressed in **percentage points**, not percent.

---

## Features

### Core VRP Analytics
- OHLC price download with **automatic fallback** (^GSPC ↔ SPY)
- VIX alignment and cleaning
- Yang–Zhang realized volatility (rolling window, annualized)
- VRP computation and visualization

### Enhancements Included
1. **VRP Percentile & Regime Signals**
   - Percentile ranking of VRP (0–100)
   - Regime classification:
     - > 75th percentile → High VRP (Sell Volatility)
     - < 25th percentile → Low VRP (Buy Volatility)
     - Otherwise → Neutral
   - Conditional forward monthly return analysis

2. **Cross-Asset VRP Comparison**
   - S&P 500: (^GSPC / SPY + ^VIX)
   - Nasdaq: (QQQ + ^VXN)
   - Bonds: (TLT + ^MOVE)
   - Relative VRP heatmap for asset rotation

3. **Simple VRP-Based Strategy Backtest**
   - Rule-based exposure when VRP exceeds a threshold
   - Strategy vs buy-and-hold performance
   - Sharpe ratio and maximum drawdown

4. **Macro Correlations**
   - VRP vs yield curve slope (10Y − 2Y)
   - Scatter plots with OLS trendlines
   - Regime-based macro interpretation

---

## Project Structure


---

## Installation

### Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run VRPDasboard.py
```

**Disclaimer:** This dashboard is for educational and research purposes only and does not constitute financial or investment advice.
