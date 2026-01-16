# Bayesian Quant Strategy for NASDAQ Futures

### Project Overview
This project implements an **Event-Driven Backtesting Engine** for NASDAQ 100 Futures (NQ/MNQ). Unlike simple vectorized backtests, this system simulates trade execution bar-by-bar to eliminate look-ahead bias and accurately model real-world trading conditions.

The strategy employs a **Hybrid Logic (Sniper & Surfer)** combining **Bayesian Probability** and **Z-Score Normalization** to capture both mean reversion opportunities and trend-following signals.

---

### Key Performance (2025.03 - 2026.01)
* **Net Profit:** +65.63% (Benchmark Alpha)
* **Profit Factor:** **2.22** (Excellent Risk/Reward Ratio)
* **Win Rate:** 56.52%
* **Max Drawdown (MDD):** **-11.47%** (Robust against volatility)

<img width="1120" height="358" alt="커브" src="https://github.com/user-attachments/assets/31c879b6-42fc-4882-a393-375f35e3dc7a" />

> *Note: Validated on 60-minute bars with strict transaction logic.*


---

### Strategy Logic (Hypothesis)
The core philosophy is to filter out "noise" and trade only when a **Statistical Edge** is confirmed.

#### 1. Sniper Strategy (Mean Reversion)
* **Concept:** Catching "falling knives" safely by identifying statistical extremes.
* **Entry Logic:**
    * **Z-Score < -2.0:** Price is in the bottom 2.2% (Oversold).
    * **Bayesian Prob ≥ 40%:** Statistical probability of a rebound increases.
    * **Price Action:** Long lower shadow candle (>30%) confirms buying pressure.
* **Exit Logic:** Wide trailing stop (3.0 ATR) to withstand volatility and capture full reversals.

#### 2. Surfer Strategy (Trend Following)
* **Concept:** Buying the dip within a long-term uptrend.
* **Entry Logic:**
    * **Trend Filter:** Price > MA200.
    * **Bayesian Prob ≥ 55%:** Strong momentum confirmed.
    * **Z-Score < 0:** Pullback to the mean (Buying opportunity).
* **Exit Logic:** Tight trailing stop (1.0 ATR) for quick profit-taking ("Hit & Run").

---

### Technical Implementation
* **Language:** Python 3.10+
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Architecture:** **Functional Programming (Procedural)**
    * **Design Choice:** Implemented using pure functions (`fetch_data`, `calc_indicators`, `run_backtest`) to prioritize **readability** and **reproducibility** of the trading logic.
    * **Modularity:** Distinct separation between Data ETL, Alpha Generation, and Execution Engine.
* **Simulation Engine:**
    * **Iterative Loop:** Iterates through dataframe rows to strictly separate "Observation Time" (i) and "Execution Time" (i+1).
    * **Risk Management:** Implemented dynamic Position Sizing and variable Trailing Stops based on ATR volatility.

---

### 📂 File Structure
```text
├── main.py                     # Core logic (Data fetching, Strategy, Backtest Engine)
├── Backtest_Result.csv         # Detailed trade logs
└── README.md                   # Project documentation
