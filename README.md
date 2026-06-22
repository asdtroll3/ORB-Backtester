# Opening Range Breakout (ORB) Backtester

A self-contained Python backtester for an **Opening Range Breakout** strategy on
intraday futures data. It runs the full study end-to-end and writes an
interactive HTML report plus a per-trade CSV.

Built and tested against E-mini Nasdaq-100 (**NQ**) 5-minute data, but it works on
**any instrument** — you just change the contract specs (see
[Adapting to other instruments](#adapting-to-other-instruments)).

---

## The strategy

| | |
|---|---|
| **Opening range** | First 30 min of the New York session (09:30–10:00 ET). Record the range high (**ORH**) and low (**ORL**). |
| **Entry** | **Long only.** When a 5-min candle *closes* above ORH. Entry = that candle's close. |
| **Stop loss** | ORL. |
| **Profit target** | Entry + R, where **R = Entry − ORL** (1R / 1:1 by default). |
| **Time exit** | If neither stop nor target is hit, exit at 14:00 NY time (configurable). |

One trade per day (the first breakout). The opening-range duration, target
multiple, and exit time are all configurable, and the sensitivity analysis sweeps
them automatically.

## What it produces

- **Full statistics** — total trades, win rate, average win/loss, profit factor,
  expectancy, average R, max drawdown, return on drawdown, Sharpe, Sortino, CAGR,
  consecutive wins/losses, and a frictionless ("no costs") profit figure.
- **Annual breakdown** — per-year trades, win rate, net P&L, net R, max drawdown,
  and return-on-drawdown.
- **Charts** (interactive, in one HTML file): equity curve + drawdown, monthly-returns
  heatmap, trade-distribution analysis, Monte-Carlo simulation of the trade sequence,
  and sensitivity heatmaps (opening-range duration × target multiple).
- **Worst-10 drawdowns** table.
- **Statistical significance** test of whether the edge survives costs & slippage.

---

## Requirements

- Python 3.9+ (developed on 3.14)
- `pip install pandas numpy plotly`

## Quick start

1. Put your price CSV next to `Backtest.py` (see [Data format](#data-format)).
   The repo ships **without** data — supply your own file.
2. Open `Backtest.py` and set `CSV_FILE`, the timezone, and the contract specs.
3. Run:

   ```bash
   python Backtest.py
   ```

4. The console prints the full report; `Backtest_Report.html` opens in your
   browser and `Backtest_trades.csv` holds every trade.

> `DrawDay.py` is a small helper to plot a single trading day from the same CSV.

---

## Data format

The loader expects the format of the original NQ export (it auto-detects nothing —
match this or adjust `load_sessions`):

| Field | Example | Notes |
|---|---|---|
| Delimiter | `;` | semicolon-separated |
| `Date` | `1/30/2026 3:55 PM` | `M/D/YYYY h:MM AM/PM` |
| `Open/High/Low/Close` | `25.640,75` | **European** numbers: `.` thousands, `,` decimal |
| `Volume` | `739` | |

- **Bar timestamp = the bar's _start_ time.**
- Rows may be in any order (the loader sorts them).

⚠️ The price data and the generated `*.html` / `*.csv` files are **git-ignored**
on purpose, so the repository stays small and free of licensed data.

---

## Configuration

Everything lives in the clearly-flagged CONFIG block at the top of `Backtest.py`.
The three `▶ CHANGE ME ◀` blocks are the ones you touch most.

### 1. Data file
```python
CSV_FILE = "NQ_5Min.csv"
```

### 2. Timezone
The strategy is defined in **New York** time. Tell it what zone the *file* is in,
as a UTC offset; the code shifts every bar by `NY_UTC_OFFSET − FILE_UTC_OFFSET`.
```python
FILE_UTC_OFFSET = -5   # source data timezone (e.g. Chicago / Central)
NY_UTC_OFFSET   = -4   # New York / Eastern
# file already in NY time -> set both equal (shift = 0)
```

### 3. Year range
```python
START_YEAR = 2020   # None = from file start
END_YEAR   = 2023   # None = to file end
```

### Strategy parameters
```python
OR_START_TIME     = "09:30"   # session open / start of the opening range
OR_MINUTES        = 30        # opening-range duration
TARGET_R_MULTIPLE = 1.0       # target = entry + (this × R)
TRADE_EXIT_TIME   = "15:00"   # hard time-exit (NY time)
```

### Contract specs & costs
```python
POINT_VALUE_USD   = 20.0      # $ per 1.0 index point
TICK_SIZE         = 0.25      # minimum price increment
SLIPPAGE_TICKS    = 1.0       # slippage per market fill, in ticks
COMMISSION_RT_USD = 4.20      # commission per round-turn, per contract
STARTING_CAPITAL  = 100_000.0
```

### Position sizing
```python
RISK_PER_TRADE_USD    = 5000.0  # $ risked per trade (loss if stop is hit)
USE_VOL_TARGET_SIZING = True    # True  -> size each trade so a stop-out loses
                                #          exactly RISK_PER_TRADE_USD: wide/volatile
                                #          days get FEWER contracts, quiet days MORE.
                                # False -> trade a fixed CONTRACTS every time.
MAX_CONTRACTS         = 20      # hard cap on contracts per trade
CONTRACTS             = 1       # used only when USE_VOL_TARGET_SIZING = False
```

---

## Adapting to other instruments

To backtest something other than E-mini NQ, change **`POINT_VALUE_USD`** and
**`TICK_SIZE`** to that contract's spec (and point `CSV_FILE` at its data). These
two numbers are all that convert price movement into dollars, so the rest of the
report adjusts automatically.

| Instrument | Symbol | `POINT_VALUE_USD` | `TICK_SIZE` | (tick value) |
|---|---|---|---|---|
| E-mini Nasdaq-100 | NQ | `20.0` | `0.25` | $5.00 |
| Micro E-mini Nasdaq-100 | MNQ | `2.0` | `0.25` | $0.50 |
| E-mini S&P 500 | ES | `50.0` | `0.25` | $12.50 |
| Micro E-mini S&P 500 | MES | `5.0` | `0.25` | $1.25 |
| E-mini Dow | YM | `5.0` | `1.0` | $5.00 |
| E-mini Russell 2000 | RTY | `50.0` | `0.10` | $5.00 |
| Crude Oil | CL | `1000.0` | `0.01` | $10.00 |
| Micro Crude Oil | MCL | `100.0` | `0.01` | $1.00 |
| Gold | GC | `100.0` | `0.10` | $10.00 |

> Tick value = `POINT_VALUE_USD × TICK_SIZE`. Always confirm the current spec on the
> exchange — contract specs change.

**Also check, per instrument:**
- **Timezone** of your data file (set `FILE_UTC_OFFSET`).
- **Session times** — `OR_START_TIME` / `TRADE_EXIT_TIME` default to the US equity
  session. For products you want to trade around a different open, change them.
- **Slippage/commission** — a wider, less-liquid market deserves more than 1 tick.

---

## Outputs

| File | Contents |
|---|---|
| `Backtest_Report.html` | Full interactive report (stats + all charts). Auto-opens. |
| `Backtest_trades.csv` | Every trade: entry/exit, prices, R, contracts, P&L, exit reason. |

Set `OPEN_REPORT = False` to stop the browser auto-opening.

---

## Notes

- Intrabar ambiguity (a bar that spans both stop and target) is resolved
  **stop-first** by default (`STOP_BEFORE_TARGET = True`) — conservative.
- Volatility-target sizing uses **fractional contracts** so the dollar risk is held
  exactly constant; round to whole contracts before trading live.
- Monte-Carlo and significance tests describe the *historical sample*; they are not
  a guarantee of future performance.

## Disclaimer

For research and educational purposes only. This is **not** financial advice, and
nothing here is a recommendation to trade. Backtested results are hypothetical.
