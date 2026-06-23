"""
Opening Range Breakout (ORB) backtest

═══════════════════════════════════════════════════════════════════════════════
STRATEGY
─────────────────────────────────────────────────────────────────────────────────
  • Opening Range : first 30 min of the New York session (09:30–10:00 ET).
                    ORH = highest high, ORL = lowest low of those bars.
  • Entry         : LONG only. When a 5-min candle CLOSES above ORH.
                    Entry price = closing price of the breakout candle.
  • Stop loss     : ORL.
  • Profit target : Entry + R, where R = Entry − ORL   (1R, i.e. 1:1).
  • Time exit     : if neither stop nor target hit, exit at 15:00 NY time.

═══════════════════════════════════════════════════════════════════════════════
DATA FORMAT
─────────────────────────────────────────────────────────────────────────────────
  Delimiter   : ';'
  Date field  : M/D/YYYY h:MM AM/PM   e.g.  1/30/2026 3:55 PM
  Numbers     : European  – '.' thousands sep, ',' decimal sep   (25.640,75 → 25640.75)
  Bar stamp   : the timestamp is the bar's *START* time.
                (Verified: last bar before the 16:00–17:00 CT CME maintenance halt
                 is stamped 15:55, the first bar after it is stamped 17:00.)
  Timezone    : the file is in CHICAGO / Central time.  New York = Chicago + 1h.
                Both cities share US DST rules, so the offset is a constant +1h
                regardless of the time of year.

  → To use a DIFFERENT file or a DIFFERENT source timezone, change ONLY the two
    blocks flagged  ▶ CHANGE ME ◀  below.

Requirements:  pip install pandas numpy plotly
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import sys
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        ▶ CHANGE ME ◀   (1) DATA FILE                       ║
# ╠══════════════════════════════════════════════════════════════════════════╣
CSV_FILE = "NQ_5Min.csv"          # <-- which CSV to read
# ╚══════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        ▶ CHANGE ME ◀   (2) TIMEZONE                        ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# The file's timestamps are in this zone; the strategy is defined in NY time.   #
# Expressed as a UTC offset in hours. The code shifts every bar by              #
#   (NY_UTC_OFFSET − FILE_UTC_OFFSET) hours  to obtain New York wall-clock time.#
FILE_UTC_OFFSET = -5              # Chicago / Central (CDT = −5).  CST = −6.
NY_UTC_OFFSET   = -4              # New York / Eastern (EDT = −4). EST = −5.
#   Examples:                                                                   #
#     file already in NY time          -> set both equal (shift = 0)            #
#     file in UTC, want NY             -> FILE = 0,  NY = -4                     #
# ╚══════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        ▶ CHANGE ME ◀   (3) YEAR RANGE                      ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# Restrict the backtest to a span of years (inclusive, by NY-session year).     #
# Use None for an open end.   Examples:                                         #
#     2020 .. 2023  ->  START_YEAR = 2020,  END_YEAR = 2023                      #
#     from 2018 on  ->  START_YEAR = 2018,  END_YEAR = None                      #
#     whole file    ->  START_YEAR = None,  END_YEAR = None                      #
START_YEAR = 2021                 # first year to include  (None = file start)
END_YEAR   = 2025                 # last  year to include  (None = file end)
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────── STRATEGY PARAMETERS ────────────────────────────
OR_START_TIME    = "09:30"        # NY session open  (start of opening range)
OR_MINUTES       = 30             # opening-range duration in minutes
TARGET_R_MULTIPLE = 1.0           # profit target = entry + (this × R)
TRADE_EXIT_TIME  = "15:00"        # hard time-exit (NY time)
STOP_BEFORE_TARGET = True         # if a single bar spans BOTH stop & target,
                                  # assume the STOP filled first (conservative).

# ─────────────────────────── CONTRACT / COSTS ───────────────────────────────
POINT_VALUE_USD   = 20.0          # Example: E-mini NQ = $20 per index point, Mini Crude Oil Futures = 1000$ per index point
TICK_SIZE         = 0.25          # NQ tick = 0.25 pt ( = $5 ), CL tick = 0.01 pt ( = 10$ )
SLIPPAGE_TICKS    = 1.0           # slippage per *market* fill (entry / stop / time)
COMMISSION_RT_USD = 4.20          # commission per round-turn, per contract
STARTING_CAPITAL  = 100_000.0     # account size used for %-returns / equity curve

# ─────────────────────────── POSITION SIZING ────────────────────────────────
# Risk per trade can be a FIXED dollar amount, or a PERCENT OF CURRENT EQUITY.
USE_PCT_EQUITY_RISK   = False     # True  -> risk RISK_PCT_OF_EQUITY of the *current*
                                  #          equity on every trade (COMPOUNDING): it
                                  #          starts at 1% of STARTING_CAPITAL and grows
                                  #          in $ as the account grows (and shrinks in
                                  #          drawdowns).  Overrides RISK_PER_TRADE_USD.
                                  # False -> use the fixed RISK_PER_TRADE_USD below.
RISK_PCT_OF_EQUITY    = 0.01      # fraction of equity risked per trade (0.01 = 1%)

RISK_PER_TRADE_USD    = 5000.0    # $ risked per trade (the loss taken if the stop
                                  # at ORL is hit).  Held constant on every trade.
USE_VOL_TARGET_SIZING = True      # True  -> Volatility-Target sizing: size each
                                  #          trade so a stop-out loses exactly the
                                  #          risk budget (fixed $ or % of equity).
                                  #          Wide-range / high-volatility days
                                  #          (big entry−ORL) get FEWER contracts;
                                  #          quiet/tight days get MORE.  Trade count
                                  #          is unchanged.
                                  # False -> trade a fixed CONTRACTS on every trade
                                  #          (dollar risk then varies day to day).
MAX_CONTRACTS         = 20        # hard cap on contracts per trade (both modes)
CONTRACTS             = 1         # fixed size used when USE_VOL_TARGET_SIZING = False

# ─────────────────────────── ANALYSIS SETTINGS ──────────────────────────────
SENS_OR_DURATIONS = [15, 30, 45, 60]          # sensitivity: OR length (minutes)
SENS_TARGET_RS    = [0.5, 1.0, 1.5, 2.0, 3.0] # sensitivity: target multiples
MC_RUNS           = 5000          # Monte-Carlo bootstrap resamples
MC_PLOT_PATHS     = 200           # how many MC equity paths to draw
RANDOM_SEED       = 42
TRADING_DAYS_YR   = 252

# ─────────────────────────── OUTPUT ─────────────────────────────────────────
HTML_REPORT  = "Backtest_Report.html"
TRADES_CSV   = "Backtest_trades.csv"
OPEN_REPORT  = True               # auto-open the HTML report in the browser


# ════════════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def _eu_floats(s: pd.Series) -> pd.Series:
    """Vectorised European number parse: '25.640,75' -> 25640.75"""
    return (s.str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False)
             .astype("float64"))


def parse_time_to_min(hhmm: str) -> int:
    """'09:30' -> 570 (minutes since midnight)"""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def load_sessions(csv_file: str, tz_shift_hours: int, win_end_min: int,
                  start_year=None, end_year=None):
    """
    Read the CSV, convert to New York time, and pre-slice each trading day into
    numpy arrays covering 09:30 .. TRADE_EXIT_TIME.  Returns (sessions, meta).

    sessions: list of dicts {date, min, O, H, L, C} sorted chronologically,
              one per calendar trading day, bars sorted ascending by time.

    start_year / end_year restrict the run to that span of years (inclusive,
    by NY-session year). None means open-ended.
    """
    print(f"  Loading  {csv_file} ...")
    df = pd.read_csv(csv_file, sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    df["DT"] = pd.to_datetime(df["Date"].str.strip(), format="%m/%d/%Y %I:%M %p")
    for col in ("Open", "High", "Low", "Close"):
        df[col] = _eu_floats(df[col])

    # drop unparseable rows
    df = df.dropna(subset=["DT", "Open", "High", "Low", "Close"])

    # ── convert file-local time -> New York wall-clock time ──────────────────
    df["NY"] = df["DT"] + pd.Timedelta(hours=tz_shift_hours)

    # ── restrict to the configured year range (inclusive) ────────────────────
    if start_year is not None:
        df = df[df["NY"].dt.year >= start_year]
    if end_year is not None:
        df = df[df["NY"].dt.year <= end_year]
    if df.empty:
        sys.exit(f"  No data in the requested year range "
                 f"{start_year or 'start'}–{end_year or 'end'}.")
    if start_year is not None or end_year is not None:
        print(f"  Year filter      : {start_year or 'start'}–{end_year or 'end'}")

    # sort ascending & remove any duplicate timestamps (e.g. contract-roll overlap)
    df = (df.sort_values("NY")
            .drop_duplicates(subset="NY", keep="first")
            .reset_index(drop=True))

    df["SessDate"] = df["NY"].dt.normalize()                  # midnight Timestamp
    df["MinOfDay"] = df["NY"].dt.hour * 60 + df["NY"].dt.minute

    or_start_min = parse_time_to_min(OR_START_TIME)
    rth = df[(df["MinOfDay"] >= or_start_min) & (df["MinOfDay"] < win_end_min)]

    sessions = []
    for sess_date, g in rth.groupby("SessDate", sort=True):
        sessions.append({
            "date": sess_date.date(),
            "min":  g["MinOfDay"].to_numpy(),
            "O":    g["Open"].to_numpy(),
            "H":    g["High"].to_numpy(),
            "L":    g["Low"].to_numpy(),
            "C":    g["Close"].to_numpy(),
        })

    meta = SimpleNamespace(
        n_rows=len(df),
        first=df["NY"].iloc[0],
        last=df["NY"].iloc[-1],
        symbol_first=df["Symbol"].iloc[0] if "Symbol" in df else "",
        symbol_last=df["Symbol"].iloc[-1] if "Symbol" in df else "",
    )
    print(f"  Loaded   {meta.n_rows:,} bars  |  {meta.first}  ..  {meta.last}")
    print(f"  Sessions {len(sessions):,} trading days in the 09:30–{TRADE_EXIT_TIME} window")
    return sessions, meta


# ════════════════════════════════════════════════════════════════════════════
#  2.  CORE BACKTEST  (one trade per day, first breakout)
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(sessions, or_minutes, target_r, cfg):
    """
    Returns (trades, day_pnls):
      trades   : list of trade dicts (only days that produced a trade)
      day_pnls : {date: net_usd_pnl}  for EVERY valid session day (0 if no trade)
                 -> the correct denominator for daily-return / Sharpe maths.
    """
    or_start = cfg.or_start_min
    or_end   = or_start + or_minutes
    win_end  = cfg.win_end_min
    expected = or_minutes // 5

    pt   = cfg.point_value
    slp  = cfg.slippage_pts
    comm = cfg.commission_rt

    trades, day_pnls = [], {}
    equity = cfg.starting_capital          # running equity (for % -of-equity sizing)

    for s in sessions:
        mins, O, H, L, C = s["min"], s["O"], s["H"], s["L"], s["C"]

        # --- opening range -----------------------------------------------------
        or_mask = (mins >= or_start) & (mins < or_end)
        if or_mask.sum() != expected or mins[or_mask][0] != or_start:
            continue                       # incomplete / irregular open -> skip day
        ORH = H[or_mask].max()
        ORL = L[or_mask].min()

        day_pnls[s["date"]] = 0.0          # valid trading day (flat unless we trade)

        # --- trade window bars -------------------------------------------------
        idx = np.nonzero((mins >= or_end) & (mins < win_end))[0]
        if idx.size == 0:
            continue

        # --- first breakout (close > ORH) -------------------------------------
        entry_i = -1
        for j in idx:
            if C[j] > ORH:
                entry_i = j
                break
        if entry_i == -1:
            continue                       # no breakout today

        theo_entry = C[entry_i]
        R_pts = theo_entry - ORL
        if R_pts <= 0:
            continue                       # degenerate (flat) range
        stop   = ORL
        target = theo_entry + target_r * R_pts

        # --- position size -----------------------------------------------------
        # The risk budget is either a fixed $ amount or a % of CURRENT equity
        # (compounding). Vol-target sizing turns that budget into contracts so a
        # stop-out (R_pts × $/pt × qty) loses exactly the budget -> fewer
        # contracts on wide/volatile ranges, more on quiet/tight ranges.
        equity_before = equity
        if cfg.use_pct_equity_risk:
            risk_budget = cfg.risk_pct_of_equity * max(equity_before, 0.0)
            qty = min(risk_budget / (R_pts * pt), cfg.max_contracts)
        elif cfg.use_vol_sizing:
            qty = min(cfg.risk_per_trade / (R_pts * pt), cfg.max_contracts)
        else:
            qty = min(cfg.contracts, cfg.max_contracts)
        if qty <= 0:
            continue                       # no risk budget (ruin) -> skip, stay flat

        # --- manage trade on subsequent bars ----------------------------------
        reason = gross_exit = sell_fill = None
        exit_i = entry_i
        for j in idx:
            if j <= entry_i:
                continue
            o, h, l = O[j], H[j], L[j]
            if o <= stop:                                  # gap down through stop
                reason, gross_exit, sell_fill, exit_i = "stop", stop, o - slp, j
                break
            if o >= target:                                # gap up through target
                reason, gross_exit, sell_fill, exit_i = "target", target, o, j
                break
            hit_stop, hit_tgt = (l <= stop), (h >= target)
            if hit_stop and hit_tgt:                       # ambiguous bar
                if cfg.stop_before_target:
                    reason, gross_exit, sell_fill, exit_i = "stop", stop, stop - slp, j
                else:
                    reason, gross_exit, sell_fill, exit_i = "target", target, target, j
                break
            if hit_stop:
                reason, gross_exit, sell_fill, exit_i = "stop", stop, stop - slp, j
                break
            if hit_tgt:
                reason, gross_exit, sell_fill, exit_i = "target", target, target, j
                break

        if reason is None:                                 # ---- time exit ----
            last = idx[-1]
            reason, gross_exit, sell_fill, exit_i = "time", C[last], C[last] - slp, last

        # --- realised P&L ------------------------------------------------------
        buy_fill = theo_entry + slp
        pnl_pts  = sell_fill - buy_fill
        pnl_usd  = pnl_pts * pt * qty - comm * qty
        # frictionless P&L: entry/exit at the theoretical levels, no slip, no commission
        pnl_usd_nocost = (gross_exit - theo_entry) * pt * qty
        gross_R  = (gross_exit - theo_entry) / R_pts          # +target_r / -1 / frac
        net_R    = pnl_usd / (R_pts * pt * qty)

        day_pnls[s["date"]] = pnl_usd
        ret_frac = pnl_usd / equity_before if equity_before > 0 else 0.0
        equity += pnl_usd                                  # compound the running equity
        trades.append({
            "date":        s["date"],
            "entry_min":   int(mins[entry_i]) + 5,            # bar-close = entry time
            "exit_min":    int(mins[exit_i]) + 5,
            "ORH": ORH, "ORL": ORL, "R_pts": R_pts,
            "entry": theo_entry, "stop": stop, "target": target,
            "exit_price": gross_exit, "reason": reason,
            "gross_R": gross_R, "net_R": net_R, "contracts": qty,
            "equity_before": equity_before, "ret_frac": ret_frac,
            "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "pnl_usd_nocost": pnl_usd_nocost,
        })

    return trades, day_pnls


# ════════════════════════════════════════════════════════════════════════════
#  3.  METRICS
# ════════════════════════════════════════════════════════════════════════════

def _max_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def drawdown_episodes(equity: np.ndarray, dates):
    """Identify every peak-to-trough-to-recovery drawdown episode."""
    eps = []
    peak = equity[0]; peak_i = 0; in_dd = False; trough = peak; trough_i = 0
    for i, e in enumerate(equity):
        if e >= peak:
            if in_dd:
                eps.append({
                    "peak_date": dates[peak_i], "trough_date": dates[trough_i],
                    "recover_date": dates[i], "depth_usd": peak - trough,
                    "depth_pct": (peak - trough) / peak * 100,
                    "len_days": (dates[i] - dates[peak_i]).days,
                    "recovered": True,
                })
                in_dd = False
            peak, peak_i = e, i
        else:
            if not in_dd:
                in_dd, trough, trough_i = True, e, i
            elif e < trough:
                trough, trough_i = e, i
    if in_dd:
        eps.append({
            "peak_date": dates[peak_i], "trough_date": dates[trough_i],
            "recover_date": None, "depth_usd": peak - trough,
            "depth_pct": (peak - trough) / peak * 100,
            "len_days": (dates[-1] - dates[peak_i]).days, "recovered": False,
        })
    eps.sort(key=lambda d: d["depth_usd"], reverse=True)
    return eps


def compute_metrics(trades, day_pnls, cfg):
    """Full statistics dict from a finished backtest."""
    m = {"n_trades": len(trades)}
    if not trades:
        return m

    tdf = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    pnl = tdf["pnl_usd"].to_numpy()
    net_R = tdf["net_R"].to_numpy()
    gross_R = tdf["gross_R"].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    m["wins"], m["losses"] = len(wins), len(losses)
    m["win_rate"] = len(wins) / len(pnl) * 100
    m["avg_win_usd"]  = wins.mean()  if len(wins)  else 0.0
    m["avg_loss_usd"] = losses.mean() if len(losses) else 0.0
    m["avg_win_R"]  = net_R[pnl > 0].mean()  if len(wins)  else 0.0
    m["avg_loss_R"] = net_R[pnl <= 0].mean() if len(losses) else 0.0
    gp, gl = wins.sum(), -losses.sum()
    m["gross_profit"], m["gross_loss"] = gp, gl
    m["profit_factor"] = (gp / gl) if gl > 0 else math.inf
    m["net_profit"] = pnl.sum()
    m["profit_no_costs"] = tdf["pnl_usd_nocost"].sum()      # zero slippage & commission
    m["cost_drag_usd"]   = m["profit_no_costs"] - m["net_profit"]
    m["expectancy_usd"] = pnl.mean()
    m["expectancy_R"]   = net_R.mean()
    m["avg_R_net"]      = net_R.mean()
    m["avg_R_gross"]    = gross_R.mean()
    m["std_R"]          = net_R.std(ddof=1) if len(net_R) > 1 else 0.0
    m["max_win_usd"], m["max_loss_usd"] = pnl.max(), pnl.min()
    m["max_consec_wins"]   = _max_streak(pnl > 0)
    m["max_consec_losses"] = _max_streak(pnl <= 0)

    rc = tdf["reason"].value_counts().to_dict()
    m["exit_target"] = rc.get("target", 0)
    m["exit_stop"]   = rc.get("stop", 0)
    m["exit_time"]   = rc.get("time", 0)

    # position-sizing diagnostics
    m["avg_contracts"]      = tdf["contracts"].mean()
    m["max_contracts_used"] = tdf["contracts"].max()
    m["pct_capped"] = float((tdf["contracts"] >= cfg.max_contracts - 1e-9).mean() * 100)

    # ── daily equity curve over EVERY valid session day ──────────────────────
    ddates = sorted(day_pnls.keys())
    dpnl = np.array([day_pnls[d] for d in ddates])
    eq = cfg.starting_capital + np.cumsum(dpnl)
    ts = pd.to_datetime(pd.Series(ddates))
    m["equity_dates"] = ts
    m["equity"] = eq
    m["daily_pnl"] = dpnl
    m["final_equity"] = eq[-1]
    m["total_return_pct"] = (eq[-1] / cfg.starting_capital - 1) * 100
    m["n_session_days"] = len(ddates)

    # drawdown
    run_max = np.maximum.accumulate(eq)
    dd = eq - run_max
    m["max_dd_usd"] = -dd.min()
    m["max_dd_pct"] = (dd / run_max).min() * 100
    m["dd_series"] = dd / run_max * 100
    m["drawdowns"] = drawdown_episodes(eq, list(ts))
    # return on max drawdown = net profit / max drawdown ($)
    m["return_on_dd"] = (m["net_profit"] / m["max_dd_usd"]) if m["max_dd_usd"] > 0 else math.inf

    # daily returns -> Sharpe / Sortino.
    #  • %-of-equity sizing is COMPOUNDING -> returns on the prior running equity.
    #  • fixed-$ / fixed-contract sizing is ADDITIVE -> returns on INITIAL capital
    #    (also robust if aggressive sizing drives equity below zero).
    compounding = cfg.use_pct_equity_risk
    if compounding:
        prev_eq = np.concatenate([[cfg.starting_capital], eq[:-1]])
        dret = dpnl / prev_eq
    else:
        dret = dpnl / cfg.starting_capital
    mu, sd = dret.mean(), dret.std(ddof=1)
    downside = dret[dret < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    m["sharpe"]  = (mu / sd * math.sqrt(TRADING_DAYS_YR)) if sd > 0 else 0.0
    m["sortino"] = (mu / dsd * math.sqrt(TRADING_DAYS_YR)) if dsd > 0 else 0.0

    # CAGR / years
    yrs = max((ts.iloc[-1] - ts.iloc[0]).days / 365.25, 1e-9)
    m["years"] = yrs
    base = eq[-1] / cfg.starting_capital
    m["cagr_pct"] = ((base ** (1 / yrs) - 1) * 100) if base > 0 else -100.0

    # ── monthly & yearly tables ──────────────────────────────────────────────
    # compounding mode -> geometric link of daily returns; additive -> simple sum.
    rser = pd.Series(dret, index=ts)
    if compounding:
        agg = lambda r: (np.prod(1 + r) - 1) * 100
        monthly = rser.resample("ME").apply(agg)
        yearly  = rser.resample("YE").apply(agg)
    else:
        monthly = rser.resample("ME").sum() * 100
        yearly  = rser.resample("YE").sum() * 100
    m["monthly_returns"] = monthly
    m["yearly_returns"]  = yearly

    # per-year max drawdown (intra-year, seeded by the prior year-end equity so an
    # early-January decline from the previous peak still counts)
    eq_years = ts.dt.year.to_numpy()
    mdd_by_year = {}
    for y in np.unique(eq_years):
        idx = np.nonzero(eq_years == y)[0]
        seed = eq[idx[0] - 1] if idx[0] > 0 else cfg.starting_capital
        seg = np.concatenate([[seed], eq[idx[0]: idx[-1] + 1]])
        mdd_by_year[int(y)] = float(-(seg - np.maximum.accumulate(seg)).min())

    # per-year trade table
    tdf["year"] = pd.to_datetime(tdf["date"]).dt.year
    yr_rows = []
    for y, g in tdf.groupby("year"):
        w = (g["pnl_usd"] > 0).sum()
        net_y = g["pnl_usd"].sum()
        mdd_y = mdd_by_year.get(int(y), 0.0)
        yr_rows.append({
            "year": int(y), "trades": len(g), "win_rate": w / len(g) * 100,
            "net_usd": net_y, "net_R": g["net_R"].sum(),
            "max_dd_usd": mdd_y,
            "romdd": (net_y / mdd_y) if mdd_y > 0 else math.inf,
        })
    m["year_table"] = pd.DataFrame(yr_rows)

    # per-weekday (Mon–Fri) table
    tdf["dow"] = pd.to_datetime(tdf["date"]).dt.weekday        # Mon=0 .. Sun=6
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_rows = []
    for d in range(5):
        g = tdf[tdf["dow"] == d]
        if len(g):
            w = (g["pnl_usd"] > 0).sum()
            dow_rows.append({
                "dow": d, "day": dow_names[d], "trades": len(g),
                "win_rate": w / len(g) * 100, "net_usd": g["pnl_usd"].sum(),
                "net_R": g["net_R"].sum(), "expectancy_R": g["net_R"].mean(),
            })
        else:
            dow_rows.append({"dow": d, "day": dow_names[d], "trades": 0,
                             "win_rate": 0.0, "net_usd": 0.0, "net_R": 0.0,
                             "expectancy_R": 0.0})
    m["dow_table"] = pd.DataFrame(dow_rows)

    m["trades_df"] = tdf
    return m


# ════════════════════════════════════════════════════════════════════════════
#  4.  STATISTICAL SIGNIFICANCE  (numpy only – normal approx + bootstrap)
# ════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def significance(trades, rng):
    """One-sample test that mean net-R > 0, plus a bootstrap CI."""
    r = np.array([t["net_R"] for t in trades], dtype=float)
    n = len(r)
    if n < 2:
        return {}
    mean, sd = r.mean(), r.std(ddof=1)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else 0.0
    p_one = 1 - _norm_cdf(t)                 # H1: mean > 0  (large-n normal approx)
    ci95 = (mean - 1.96 * se, mean + 1.96 * se)

    boot = rng.choice(r, size=(10000, n), replace=True).mean(axis=1)
    boot_ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    p_boot = (boot <= 0).mean()              # bootstrap P(mean <= 0)

    return {"n": n, "mean_R": mean, "se": se, "t_stat": t,
            "p_one_sided": p_one, "ci95": ci95,
            "boot_ci": boot_ci, "p_boot": p_boot}


# ════════════════════════════════════════════════════════════════════════════
#  5.  MONTE-CARLO  (bootstrap the trade sequence)
# ════════════════════════════════════════════════════════════════════════════

def monte_carlo(trades, cfg, rng):
    # Resample the trade sequence (with replacement) MC_RUNS times.
    #  • compounding (%-equity) sizing: resample per-trade RETURN FRACTIONS and
    #    grow equity multiplicatively, so the path reflects compounding.
    #  • additive (fixed-$) sizing: resample $ P&L and accumulate.
    n = len(trades)
    if cfg.use_pct_equity_risk:
        rf = np.array([t["ret_frac"] for t in trades], dtype=float)
        sample = rng.choice(rf, size=(cfg.mc_runs, n), replace=True)
        eq = cfg.starting_capital * np.cumprod(1 + sample, axis=1)
        paths = cfg.starting_capital * np.cumprod(
            1 + sample[:min(cfg.mc_plot_paths, cfg.mc_runs)], axis=1)
    else:
        pnl = np.array([t["pnl_usd"] for t in trades], dtype=float)
        sample = rng.choice(pnl, size=(cfg.mc_runs, n), replace=True)
        eq = cfg.starting_capital + np.cumsum(sample, axis=1)
        paths = cfg.starting_capital + np.cumsum(
            sample[:min(cfg.mc_plot_paths, cfg.mc_runs)], axis=1)
    finals = eq[:, -1]
    run_max = np.maximum.accumulate(eq, axis=1)
    max_dd = ((eq - run_max) / run_max).min(axis=1) * 100      # worst %DD per run
    return {
        "final_equity": finals,
        "final_return_pct": (finals / cfg.starting_capital - 1) * 100,
        "max_dd_pct": max_dd,
        "p_profit": (finals > cfg.starting_capital).mean() * 100,
        "median_final": np.median(finals),
        "p05_final": np.percentile(finals, 5),
        "p95_final": np.percentile(finals, 95),
        "median_dd": np.median(max_dd),
        "p05_dd": np.percentile(max_dd, 5),       # 5th pct = deep DD tail
        "paths": paths,
    }


# ════════════════════════════════════════════════════════════════════════════
#  6.  SENSITIVITY  (OR duration × target multiple)
# ════════════════════════════════════════════════════════════════════════════

def sensitivity(sessions, cfg):
    rows = []
    for dur in SENS_OR_DURATIONS:
        for tr in SENS_TARGET_RS:
            trd, dp = run_backtest(sessions, dur, tr, cfg)
            if trd:
                mm = compute_metrics(trd, dp, cfg)
                rows.append({
                    "or_min": dur, "target_r": tr, "trades": mm["n_trades"],
                    "win_rate": mm["win_rate"], "expectancy_R": mm["expectancy_R"],
                    "profit_factor": mm["profit_factor"],
                    "total_return_pct": mm["total_return_pct"],
                    "max_dd_pct": mm["max_dd_pct"], "sharpe": mm["sharpe"],
                })
            else:
                rows.append({"or_min": dur, "target_r": tr, "trades": 0,
                             "win_rate": 0, "expectancy_R": 0, "profit_factor": 0,
                             "total_return_pct": 0, "max_dd_pct": 0, "sharpe": 0})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  7.  TEXT REPORT
# ════════════════════════════════════════════════════════════════════════════

def fmt_money(x):  return f"${x:,.0f}"
def fmt_pct(x):    return f"{x:,.2f}%"


def build_text_report(meta, cfg, m, sig, mc, sens) -> str:
    L = []
    P = L.append
    P("=" * 78)
    P("  OPENING RANGE BREAKOUT — BACKTEST REPORT  (NQ, 5-min, LONG only)")
    P("=" * 78)
    P(f"  Data file        : {CSV_FILE}")
    P(f"  Symbols          : {meta.symbol_last} .. {meta.symbol_first} (continuous front month)")
    P(f"  Period           : {meta.first.date()}  ->  {meta.last.date()}   ({m['years']:.2f} yrs)")
    P(f"  Source timezone  : UTC{FILE_UTC_OFFSET:+d}  ->  NY UTC{NY_UTC_OFFSET:+d}"
      f"   (shift {cfg.tz_shift:+d}h)")
    P(f"  Opening range    : {OR_START_TIME} + {OR_MINUTES}min  |  Target {TARGET_R_MULTIPLE}R"
      f"  |  Time-exit {TRADE_EXIT_TIME} NY")
    P(f"  Costs            : {SLIPPAGE_TICKS:g} tick slip/side"
      f"  +  {fmt_money(COMMISSION_RT_USD)}/round-turn   (${POINT_VALUE_USD:g}/pt)")
    if USE_PCT_EQUITY_RISK:
        P(f"  Position sizing  : vol-target  {RISK_PCT_OF_EQUITY * 100:g}% of equity/trade"
          f"  (compounding)  |  cap {MAX_CONTRACTS} contracts")
    elif USE_VOL_TARGET_SIZING:
        P(f"  Position sizing  : vol-target  {fmt_money(RISK_PER_TRADE_USD)} risk/trade"
          f"  |  cap {MAX_CONTRACTS} contracts")
    else:
        P(f"  Position sizing  : fixed {CONTRACTS} contract(s)"
          f"  |  cap {MAX_CONTRACTS} contracts")
    P("")
    P("-" * 78)
    P("  CORE PERFORMANCE")
    P("-" * 78)
    P(f"  Session days (opportunities) : {m['n_session_days']:,}")
    P(f"  Total trades                 : {m['n_trades']:,}")
    P(f"  Win rate                     : {fmt_pct(m['win_rate'])}"
      f"   ({m['wins']}W / {m['losses']}L)")
    P(f"  Exit breakdown               : target {m['exit_target']}"
      f"  |  stop {m['exit_stop']}  |  time {m['exit_time']}")
    P(f"  Contracts / trade            : avg {m['avg_contracts']:.2f}"
      f"  |  max {m['max_contracts_used']:.1f}  |  cap hit {m['pct_capped']:.0f}% of trades")
    P(f"  Average win                  : {fmt_money(m['avg_win_usd'])}"
      f"   ({m['avg_win_R']:+.2f}R)")
    P(f"  Average loss                 : {fmt_money(m['avg_loss_usd'])}"
      f"   ({m['avg_loss_R']:+.2f}R)")
    P(f"  Profit factor                : {m['profit_factor']:.3f}")
    P(f"  Expectancy / trade           : {fmt_money(m['expectancy_usd'])}"
      f"   ({m['expectancy_R']:+.4f}R)")
    P(f"  Avg R / trade (net | gross)  : {m['avg_R_net']:+.4f}R | {m['avg_R_gross']:+.4f}R")
    P(f"  Std-dev of R                 : {m['std_R']:.3f}")
    P(f"  Best / worst trade           : {fmt_money(m['max_win_usd'])}"
      f"  /  {fmt_money(m['max_loss_usd'])}")
    P(f"  Max consecutive W / L        : {m['max_consec_wins']} / {m['max_consec_losses']}")
    rodd = f"{m['return_on_dd']:.2f}" if math.isfinite(m['return_on_dd']) else "∞"
    P(f"  Return on max drawdown       : {rodd}   (net profit / max DD)")
    P("")
    P("-" * 78)
    P("  EQUITY / RISK")
    P("-" * 78)
    P(f"  Starting capital             : {fmt_money(cfg.starting_capital)}")
    P(f"  Final equity                 : {fmt_money(m['final_equity'])}")
    P(f"  Net profit                   : {fmt_money(m['net_profit'])}"
      f"   ({fmt_pct(m['total_return_pct'])})")
    P(f"  Profit if no costs           : {fmt_money(m['profit_no_costs'])}"
      f"   (zero slippage & commission)")
    P(f"  Cost drag (slip + comm)      : {fmt_money(-m['cost_drag_usd'])}"
      f"   ({-m['cost_drag_usd'] / cfg.starting_capital * 100:+.1f}% of capital)")
    P(f"  CAGR                         : {fmt_pct(m['cagr_pct'])}")
    P(f"  Max drawdown                 : {fmt_money(m['max_dd_usd'])}"
      f"   ({fmt_pct(m['max_dd_pct'])})")
    P(f"  Sharpe (daily, ann.)         : {m['sharpe']:.2f}")
    P(f"  Sortino (daily, ann.)        : {m['sortino']:.2f}")
    P("")
    P("-" * 78)
    P("  ANNUAL RETURN BREAKDOWN")
    P("-" * 78)
    P(f"  {'Year':<6}{'Trades':>7}{'Win%':>8}{'Net $':>13}{'Net R':>8}"
      f"{'MaxDD $':>12}{'RoDD':>8}{'Return%':>9}")
    for _, r in m["year_table"].iterrows():
        yr_ret = m["yearly_returns"][m["yearly_returns"].index.year == r["year"]]
        rp = yr_ret.iloc[0] if len(yr_ret) else 0.0
        romdd = f"{r['romdd']:.2f}" if np.isfinite(r['romdd']) else "∞"
        P(f"  {int(r['year']):<6}{int(r['trades']):>7}{r['win_rate']:>7.1f}%"
          f"{r['net_usd']:>13,.0f}{r['net_R']:>8.1f}{r['max_dd_usd']:>12,.0f}"
          f"{romdd:>8}{rp:>8.1f}%")
    P("")
    P("-" * 78)
    P("  DAY-OF-WEEK BREAKDOWN  (Mon–Fri)")
    P("-" * 78)
    P(f"  {'Day':<11}{'Trades':>8}{'Win%':>8}{'Net $':>14}{'Net R':>9}{'Exp.R/trade':>13}")
    for _, r in m["dow_table"].iterrows():
        P(f"  {r['day']:<11}{int(r['trades']):>8}{r['win_rate']:>7.1f}%"
          f"{r['net_usd']:>14,.0f}{r['net_R']:>9.1f}{r['expectancy_R']:>+13.4f}")
    P("")
    P("-" * 78)
    P("  WORST 10 DRAWDOWNS")
    P("-" * 78)
    P(f"  {'#':<3}{'Depth$':>12}{'Depth%':>9}{'Peak':>13}{'Trough':>13}"
      f"{'Recovered':>13}{'Days':>7}")
    for i, d in enumerate(m["drawdowns"][:10], 1):
        rec = d["recover_date"].date().isoformat() if d["recovered"] else "—(open)"
        P(f"  {i:<3}{d['depth_usd']:>12,.0f}{d['depth_pct']:>8.1f}%"
          f"{str(d['peak_date'].date()):>13}{str(d['trough_date'].date()):>13}"
          f"{rec:>13}{d['len_days']:>7}")
    P("")
    P("-" * 78)
    P("  MONTE-CARLO  (bootstrap resample of trade sequence, "
      f"{cfg.mc_runs:,} runs)")
    P("-" * 78)
    P(f"  P(profitable)                : {mc['p_profit']:.1f}%")
    P(f"  Median final equity          : {fmt_money(mc['median_final'])}"
      f"   ({(mc['median_final']/cfg.starting_capital-1)*100:+.1f}%)")
    P(f"   5th–95th pct final equity   : {fmt_money(mc['p05_final'])}"
      f"  ..  {fmt_money(mc['p95_final'])}")
    P(f"  Median max-drawdown          : {fmt_pct(mc['median_dd'])}")
    P(f"  5th-pct (deep) max-drawdown  : {fmt_pct(mc['p05_dd'])}")
    P("")
    P("-" * 78)
    P("  SENSITIVITY — Expectancy (net R/trade)   [rows=OR min, cols=target R]")
    P("-" * 78)
    piv = sens.pivot(index="or_min", columns="target_r", values="expectancy_R")
    hdr = "  OR\\R " + "".join(f"{c:>10g}" for c in piv.columns)
    P(hdr)
    for idx_, row in piv.iterrows():
        P(f"  {idx_:>4} " + "".join(f"{v:>+10.4f}" for v in row.values))
    P("")
    P("  SENSITIVITY — Profit factor")
    pivf = sens.pivot(index="or_min", columns="target_r", values="profit_factor")
    P(hdr)
    for idx_, row in pivf.iterrows():
        P(f"  {idx_:>4} " + "".join(f"{v:>10.3f}" for v in row.values))
    P("")

    # ── final assessment ────────────────────────────────────────────────────
    P("=" * 78)
    P("  FINAL ASSESSMENT — Is there a statistically significant edge?")
    P("=" * 78)
    if sig:
        P(f"  Net expectancy per trade : {sig['mean_R']:+.4f} R")
        P(f"  Std error                : {sig['se']:.4f} R   (n = {sig['n']:,} trades)")
        P(f"  t-statistic (H1: >0)     : {sig['t_stat']:.2f}")
        P(f"  p-value (one-sided)      : {sig['p_one_sided']:.4f}")
        P(f"  95% CI on mean R         : [{sig['ci95'][0]:+.4f}, {sig['ci95'][1]:+.4f}]")
        P(f"  Bootstrap 95% CI         : [{sig['boot_ci'][0]:+.4f}, {sig['boot_ci'][1]:+.4f}]"
          f"   (P[mean<=0] = {sig['p_boot']*100:.1f}%)")
        P("")
        gross_e = m["avg_R_gross"]
        net_e = m["expectancy_R"]
        P(f"  Gross expectancy (no costs): {gross_e:+.4f} R")
        P(f"  Cost drag                  : {net_e - gross_e:+.4f} R per trade")
        P("")
        significant = (sig["p_one_sided"] < 0.05 and sig["ci95"][0] > 0
                       and m["profit_factor"] > 1 and net_e > 0)
        if significant:
            P("  VERDICT:  The strategy shows a STATISTICALLY SIGNIFICANT positive")
            P("            edge at the 5% level AFTER costs & slippage.")
        elif net_e > 0 and sig["p_one_sided"] < 0.10:
            P("  VERDICT:  The edge is POSITIVE but only MARGINALLY significant")
            P("            (weak — sensitive to cost assumptions). Treat with caution.")
        elif gross_e > 0 >= net_e:
            P("  VERDICT:  A positive GROSS edge is ERODED TO ZERO/NEGATIVE by")
            P("            transaction costs & slippage. NOT tradeable as-is.")
        else:
            P("  VERDICT:  NO statistically significant edge. The results are")
            P("            consistent with random (1:1 R with ~50% hit rate).")
        P("")
        P(f"  Supporting Monte-Carlo: P(profit over the full sequence) = {mc['p_profit']:.1f}%.")
    P("=" * 78)
    return "\n".join(L)


# ════════════════════════════════════════════════════════════════════════════
#  8.  HTML REPORT  (plotly, single self-contained file)
# ════════════════════════════════════════════════════════════════════════════

BG, GRID, TEXT, ACC = "#131722", "#2a2e39", "#d1d4dc", "#26a69a"


def _layout(fig, title, h=420):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color=TEXT)),
        height=h, plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT, size=11), margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def build_html(text_report, m, mc, sens):
    figs = []

    # 1) equity curve + drawdown
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                      vertical_spacing=0.04)
    f.add_trace(go.Scatter(x=m["equity_dates"], y=m["equity"], name="Equity",
                           line=dict(color=ACC, width=1.5)), row=1, col=1)
    f.add_trace(go.Scatter(x=m["equity_dates"], y=m["dd_series"], name="Drawdown %",
                           fill="tozeroy", line=dict(color="#ef5350", width=1)),
                row=2, col=1)
    _layout(f, "Equity Curve  &  Drawdown (%)", h=560)
    f.update_yaxes(title_text="Equity $", row=1, col=1)
    f.update_yaxes(title_text="DD %", row=2, col=1)
    figs.append(f)

    # 2) monthly returns heatmap
    mr = m["monthly_returns"]
    pdf = pd.DataFrame({"y": mr.index.year, "m": mr.index.month, "v": mr.values})
    piv = pdf.pivot(index="y", columns="m", values="v").reindex(columns=range(1, 13))
    mon = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    f = go.Figure(go.Heatmap(
        z=piv.values, x=mon, y=piv.index.astype(str),
        colorscale=[[0, "#ef5350"], [0.5, "#1a1d28"], [1, ACC]], zmid=0,
        text=np.round(piv.values, 1), texttemplate="%{text}",
        textfont=dict(size=9), colorbar=dict(title="%")))
    _layout(f, "Monthly Returns Heatmap (%)", h=420)
    figs.append(f)

    # 3) trade R distribution
    tdf = m["trades_df"]
    f = go.Figure()
    f.add_trace(go.Histogram(x=tdf["net_R"], nbinsx=60, marker_color=ACC))
    f.add_vline(x=tdf["net_R"].mean(), line_color="#f0c040", line_dash="dash",
                annotation_text=f"mean {tdf['net_R'].mean():+.3f}R")
    _layout(f, "Trade Distribution — net R multiples", h=360)
    f.update_xaxes(title_text="net R per trade")
    f.update_yaxes(title_text="count")
    figs.append(f)

    # 3b) exit-reason + win-rate-by-year
    f = make_subplots(rows=1, cols=2, subplot_titles=("Exit reason", "Win-rate by year"))
    rc = tdf["reason"].value_counts()
    f.add_trace(go.Bar(x=rc.index, y=rc.values,
                       marker_color=["#26a69a", "#ef5350", "#f0c040"][:len(rc)],
                       showlegend=False), row=1, col=1)
    yt = m["year_table"]
    f.add_trace(go.Bar(x=yt["year"].astype(str), y=yt["win_rate"], marker_color=ACC,
                       showlegend=False), row=1, col=2)
    f.add_hline(y=50, line_dash="dot", line_color="#888", row=1, col=2)
    _layout(f, "Trade Analysis", h=340)
    figs.append(f)

    # 3c) day-of-week performance
    dt = m["dow_table"]
    short = [d[:3] for d in dt["day"]]
    f = make_subplots(rows=1, cols=2, subplot_titles=("Net P&L by weekday ($)",
                                                      "Win-rate by weekday (%)"))
    f.add_trace(go.Bar(x=short, y=dt["net_usd"],
                       marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in dt["net_usd"]],
                       showlegend=False), row=1, col=1)
    f.add_trace(go.Bar(x=short, y=dt["win_rate"], marker_color=ACC,
                       showlegend=False), row=1, col=2)
    f.add_hline(y=50, line_dash="dot", line_color="#888", row=1, col=2)
    _layout(f, "Day-of-Week Performance (Mon–Fri)", h=340)
    figs.append(f)

    # 4) Monte-Carlo paths + final-return histogram
    f = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                      subplot_titles=("Monte-Carlo equity paths",
                                      "Final return distribution (%)"))
    xs = np.arange(mc["paths"].shape[1])
    for p in mc["paths"]:
        f.add_trace(go.Scatter(x=xs, y=p, line=dict(width=0.4, color="rgba(38,166,154,0.18)"),
                               showlegend=False, hoverinfo="skip"), row=1, col=1)
    f.add_hline(y=STARTING_CAPITAL, line_dash="dot", line_color="#888", row=1, col=1)
    f.add_trace(go.Histogram(x=mc["final_return_pct"], nbinsx=60, marker_color=ACC,
                             showlegend=False), row=1, col=2)
    f.add_vline(x=0, line_dash="dash", line_color="#ef5350", row=1, col=2)
    _layout(f, f"Monte-Carlo  ({len(mc['final_equity']):,} runs,  "
               f"P(profit)={mc['p_profit']:.0f}%)", h=420)
    f.update_xaxes(title_text="trade #", row=1, col=1)
    figs.append(f)

    # 5) sensitivity heatmaps
    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("Expectancy (net R/trade)", "Profit factor"))
    pe = sens.pivot(index="or_min", columns="target_r", values="expectancy_R")
    pf = sens.pivot(index="or_min", columns="target_r", values="profit_factor")
    f.add_trace(go.Heatmap(z=pe.values, x=[f"{c:g}R" for c in pe.columns],
                           y=[f"{i}m" for i in pe.index],
                           colorscale=[[0,"#ef5350"],[0.5,"#1a1d28"],[1,ACC]], zmid=0,
                           text=np.round(pe.values, 4), texttemplate="%{text}",
                           textfont=dict(size=9), colorbar=dict(x=0.45)), row=1, col=1)
    f.add_trace(go.Heatmap(z=pf.values, x=[f"{c:g}R" for c in pf.columns],
                           y=[f"{i}m" for i in pf.index],
                           colorscale=[[0,"#ef5350"],[0.5,"#1a1d28"],[1,ACC]], zmid=1,
                           text=np.round(pf.values, 2), texttemplate="%{text}",
                           textfont=dict(size=9), colorbar=dict(x=1.0)), row=1, col=2)
    _layout(f, "Sensitivity — OR duration × target multiple", h=380)
    figs.append(f)

    # assemble single HTML
    parts = []
    for i, fig in enumerate(figs):
        parts.append(fig.to_html(full_html=False,
                                 include_plotlyjs=("cdn" if i == 0 else False)))
    charts = "".join(f'<div class="chart">{p}</div>' for p in parts)

    # colour the "Return on max drawdown" line red in the HTML (console stays plain)
    report_html = text_report
    for ln in text_report.splitlines():
        if ln.lstrip().startswith("Return on max drawdown"):
            report_html = report_html.replace(
                ln, f'<span style="color:#ef5350">{ln}</span>', 1)
            break

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ORB Backtest Report</title>
<style>
 body{{background:{BG};color:{TEXT};font-family:-apple-system,Segoe UI,Roboto,sans-serif;
      margin:0;padding:24px;}}
 h1{{color:{ACC};font-weight:600;}}
 pre{{background:#0d1017;border:1px solid {GRID};border-radius:8px;padding:18px;
      font-family:'Consolas','Courier New',monospace;font-size:12.5px;line-height:1.4;
      color:#cfd3dc;overflow-x:auto;white-space:pre;}}
 .chart{{background:{BG};border:1px solid {GRID};border-radius:8px;margin:18px 0;}}
</style></head><body>
<h1>Opening Range Breakout — NQ 5-min — Backtest Report</h1>
<pre>{report_html}</pre>
{charts}
</body></html>"""
    with open(HTML_REPORT, "w", encoding="utf-8") as fh:
        fh.write(html)
    return HTML_REPORT


# ════════════════════════════════════════════════════════════════════════════
#  9.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    try:                                   # render Unicode cleanly on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rng = np.random.default_rng(RANDOM_SEED)
    tz_shift = NY_UTC_OFFSET - FILE_UTC_OFFSET
    win_end_min = parse_time_to_min(TRADE_EXIT_TIME)

    cfg = SimpleNamespace(
        or_start_min=parse_time_to_min(OR_START_TIME),
        win_end_min=win_end_min, tz_shift=tz_shift,
        point_value=POINT_VALUE_USD, slippage_pts=SLIPPAGE_TICKS * TICK_SIZE,
        commission_rt=COMMISSION_RT_USD, contracts=CONTRACTS,
        starting_capital=STARTING_CAPITAL, stop_before_target=STOP_BEFORE_TARGET,
        use_vol_sizing=USE_VOL_TARGET_SIZING, risk_per_trade=RISK_PER_TRADE_USD,
        max_contracts=MAX_CONTRACTS,
        use_pct_equity_risk=USE_PCT_EQUITY_RISK, risk_pct_of_equity=RISK_PCT_OF_EQUITY,
        mc_runs=MC_RUNS, mc_plot_paths=MC_PLOT_PATHS,
    )

    sessions, meta = load_sessions(CSV_FILE, tz_shift, win_end_min,
                                   START_YEAR, END_YEAR)

    print("  Running primary backtest ...")
    trades, day_pnls = run_backtest(sessions, OR_MINUTES, TARGET_R_MULTIPLE, cfg)
    if not trades:
        sys.exit("  No trades generated — check timezone / session settings.")

    m   = compute_metrics(trades, day_pnls, cfg)
    sig = significance(trades, rng)
    mc  = monte_carlo(trades, cfg, rng)
    print(f"  Running sensitivity grid "
          f"({len(SENS_OR_DURATIONS)}x{len(SENS_TARGET_RS)} combos) ...")
    sens = sensitivity(sessions, cfg)

    text = build_text_report(meta, cfg, m, sig, mc, sens)
    print("\n" + text + "\n")

    m["trades_df"].to_csv(TRADES_CSV, index=False)
    path = build_html(text, m, mc, sens)
    print(f"  Trades written : {TRADES_CSV}")
    print(f"  HTML report    : {path}")
    if OPEN_REPORT:
        try:
            webbrowser.open(Path(path).resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
