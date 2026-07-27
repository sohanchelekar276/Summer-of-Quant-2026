# Regime-Shift Asset Allocation

An HMM-based market regime classifier driving a regime-conditional convex portfolio optimiser,
validated walk-forward with transaction costs, on Indian equity / gold / G-Sec.

```
data → integrity audit → features → walk-forward HMM → CVXPY weights → costed backtest → tearsheet
```

## Run it

```bash
pip install yfinance hmmlearn cvxpy scikit-learn pandas numpy matplotlib scipy tqdm
jupyter notebook SoQ_RegimeShift_Final.ipynb   # Runtime → Run all
```

Runs top-to-bottom from a fresh kernel in roughly 3–5 minutes. All parameters live in the `CFG`
dict in Cell 1; nothing is hardcoded downstream. Outputs `performance_summary.csv`,
`transition_matrix.csv`, `regimes_and_weights.csv`, `daily_returns.csv`.

## Universe

| Role | Ticker | Notes |
|---|---|---|
| Equity | `^NSEI` | Nifty 50 |
| Gold | `GOLDBEES.NS` | Nippon India Gold BeES |
| Bonds | `LICNETFGSC.NS` | LIC MF G-Sec ETF |
| Feature only | `^INDIAVIX` | India VIX, not investable |

Downloaded with `auto_adjust=True` so splits and distributions are back-adjusted.

## Key decisions

### Why a data integrity audit before anything else

Thinly-traded Indian ETF series contain occasional bad prints. One mis-scaled close produces a
mirrored return pair — roughly `−99%` on day `t` and `+8700%` on `t+1`. Cumulative growth jumps
~30x in a single bar, annualised volatility reads several hundred percent, and every Sharpe,
drawdown and Calmar downstream is fiction.

Cell 2 flags every single-day log move above 25% on any asset. Mirrored pairs are treated as data
errors: the offending close is nulled and interpolated. Anything non-mirrored is printed for manual
inspection rather than silently patched. The audit table is part of the output — if it says
"clean", that is a claim the notebook makes and shows its working for.

**No regime model or backtest is meaningful until this table is clean.**

### Why 3 regimes

Bull / Bear / Crisis is the smallest set that separates the three behaviours that demand different
allocations: trending-up (take risk), grinding-down (reduce risk), and violent high-volatility
(preserve capital). Two states collapse Bear and Crisis, which are economically distinct — a slow
drawdown and a liquidity event call for different books. Four or more states split on noise and
produce regimes with expected durations of days rather than weeks, which transaction costs then eat.

Model selection check: expected durations from the fitted transition matrix should read in weeks
or months. If any diagonal entry drops below 0.80, Cell 7 warns explicitly.

### Why these features

Nine causal features in three families:

- **Momentum** (`mom_5`, `mom_21`, `mom_63`) — sign and persistence of the trend, which is what
  separates Bull from Bear at similar volatility levels.
- **Volatility** (`vol_21`, `vol_63`, `vol_ratio`, `downside_21`) — level and, importantly,
  *acceleration*. `vol_ratio = vol_21/vol_63` is the early-warning term: it spikes before the
  long-window measure does. `downside_21` separates downside dispersion from a rally that is
  simply volatile.
- **Fear** (`vix`, `vix_chg_5`) — the only forward-looking input in the set. India VIX is an
  option-implied expectation, so it carries information realised volatility cannot.

Every feature at date `t` uses only data at or before `t`.

### Why the HMM is configured the way it is

**Sticky transition prior.** A default `GaussianHMM` fit on smooth, autocorrelated features can
converge to a degenerate solution — a transition matrix like `[[0,1,0],[0.99,0,0.01],[0,0.005,0.995]]`
says the market alternates state every single day. That is a 2-cycle, not a regime model. A
Dirichlet prior weighted on the diagonal (`transmat_prior = 50·I + 1`) encodes the correct prior
belief that regimes persist. This is a modelling assumption, stated rather than smuggled in.

**States are ranked by the MEAN of the volatility feature.** `model.covars_[s]` is the dispersion
of a feature *within* state `s` — sorting on it answers "which state has the most erratic
volatility", which is unrelated to which state is the crisis. `model.means_[s]` is the correct
statistic: highest mean `vol_21` ⇒ Crisis, lowest ⇒ Bull. Because `StandardScaler` is monotone, the
ordering is the same in scaled or raw space. Cell 7 prints state means in original units so the
labelling can be verified by eye.

**Regimes are decoded over a sequence, not a single day.** `model.predict(X[i:i+1])` on one row
discards the transition matrix entirely — it collapses to a per-day emission-likelihood argmax,
which is why naive implementations produce flickering regime charts. The correct causal call is
`model.predict(X[:i+1])[-1]`: Viterbi over everything up to and including today, take today's
label. This uses no future data but does use the persistence structure the model learned.

### Why the regime objectives are what they are

| Regime | Objective | Rationale |
|---|---|---|
| Bull | max `μ'w − 2·w'Σw` | Return-seeking; low risk penalty |
| Bear | max `μ'w − 8·w'Σw` | Same form, 4× the penalty |
| Crisis | min `w'Σw` | Capital preservation; μ ignored entirely |

Constraints: long-only, fully invested, 70% per-asset cap.

μ is dropped in Crisis deliberately. Expected returns estimated during a crisis are the least
reliable numbers in the whole pipeline, and a min-variance objective needs no return forecast at all.

**Mean shrinkage (`MU_SHRINK = 0.75`) matters more than the objectives.** A sample mean over a
2-year window is close to pure noise. Fed raw into a max-return objective it produces corner
solutions that flip at every rebalance. Shrinking μ toward its cross-sectional mean cut annualised
turnover from ~1200% to ~310% with essentially no loss of signal. Σ uses an EWMA estimate
(63-day half-life) shrunk 15% toward its diagonal, which keeps it well-conditioned for CVXPY.

## Walk-forward contract

| Quantity | Computed from |
|---|---|
| `StandardScaler` mean/std | `X[:i]` — training window only |
| HMM parameters | `X[:i]`, re-fit every 21 days on an expanding window |
| Regime label for day `i` | Viterbi over `X[:i+1]`, last element |
| μ, Σ for the optimiser | returns strictly before day `i` |
| Weights applied | held over day `i+1` (one-day implementation lag) |

No right-hand side ever references an index greater than the left. Standardisation happens *inside*
the loop — z-scoring the full series up front, even with an expanding window, bakes future
information into the scale of past observations.

Cell 8 makes the cost of getting this wrong visible: it plots the walk-forward regime chart against
a single HMM fit on the full sample and asked to label history. The full-sample version marks the
COVID crash cleanly because it has already seen it. Label agreement between the two is the size of
the hindsight advantage.

## Transaction costs

7.5bps per unit of turnover, charged on the rebalance day.

- Weights **drift** with returns between rebalances. Turnover is measured against the drifted book,
  not against the previous target — otherwise you charge yourself for trades you never made.
- Rebalance triggers: regime change, or every 21 days, whichever comes first.
- A 5% no-trade band suppresses cosmetic rebalances that cost money and change nothing.

Results are reported gross and net so the cost drag is legible as a single number.

## Benchmarks

Static 60/40 (60% Nifty, 40% G-Sec) and equal-weight across all three assets, both constant-weight
over the identical out-of-sample window. Compared on Sharpe, Sortino, max drawdown, Calmar and
turnover. Sharpe and Sortino are excess of a 6% Indian risk-free rate.

## Known limitations

- Three assets is a small universe; the optimiser's degrees of freedom are correspondingly limited
  and results are sensitive to the bond ETF's liquidity.
- Single walk-forward path. Randomised seeds and start dates would give a distribution of outcomes
  rather than a point estimate.
- No cash/T-bill asset, so the Crisis book is still fully invested. Adding LIQUIDBEES would let
  min-variance actually de-risk rather than just rotate.
- Regime labels are model output, not ground truth. There is no way to compute a classification
  accuracy without hand-labelling, which the brief explicitly rules out.
