# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

## Project Overview
This project builds a dynamic asset allocation engine that shifts portfolio weights between equities, bonds, and gold based on the current market "regime" (Bull, Bear, or Crisis). Instead of relying on a static 60/40 allocation, the system uses a Hidden Markov Model (HMM) to detect underlying market conditions and Convex Optimization to maximize risk-adjusted returns accordingly.

## Methodology

### 1. Feature Engineering
The HMM relies on two core features:
*   **Realized Volatility:** 21-day rolling standard deviation of Nifty 50 log returns.
*   **Market Fear:** The India VIX index.
Both features are standardized using an *expanding window* Z-score with a 63-day minimum period. This ensures the model only normalizes using data available at time *t*, strictly preventing lookahead bias.

### 2. Regime Classification (HMM)
I used a Gaussian HMM to infer 3 hidden states. The states are dynamically mapped at each step by sorting their learned variances:
*   **State 2 (Bull):** Lowest volatility.
*   **State 0 (Bear):** Moderate volatility.
*   **State 1 (Crisis):** Highest volatility.

### 3. Walk-Forward Validation & Optimization
To simulate real-world trading, the engine uses a 2-year rolling window. The HMM is refit every 5 days, but predictions and weight optimizations occur daily. 
Using `cvxpy`, the portfolio weights are optimized based on the detected regime:
*   **Crisis:** Strictly minimize portfolio variance.
*   **Bull:** Maximize returns with low risk aversion.
*   **Bear:** Maximize returns with high risk aversion (defensive growth).
*   **Constraints:** Long-only, fully invested, with a 5% minimum and 80% maximum cap per asset to ensure diversification.

## Results
The backtest includes a realistic 10 basis point transaction cost per unit of turnover to penalize excessive trading. 

**Performance Summary (2017-2023):**
*   **Dynamic Engine:** 441.72% Total Return | 0.54 Calmar Ratio | -56.28% Max Drawdown
*   **Static 60/40 Benchmark:** 109.85% Total Return | 0.47 Calmar Ratio | -26.02% Max Drawdown

While the dynamic strategy experienced higher volatility, its ability to capture massive upside runs during Bull regimes resulted in severe outperformance and a superior Calmar ratio compared to the static benchmark.

## How to Run
1. Install dependencies: `pip install numpy pandas yfinance hmmlearn cvxpy matplotlib tqdm`
2. Run the Jupyter Notebook cell by cell to download data, fit the model, and view the equity curves.