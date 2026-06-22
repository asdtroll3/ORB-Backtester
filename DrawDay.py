"""
───────────────────
Delimiter  : semicolon (;)
Date field : M/D/YYYY H:MM AM  (e.g. 1/21/2026 4:40 AM)
Number fmt : European  –  dot as thousands sep, comma as decimal sep
             e.g.  25.181,75  →  25181.75

Requirements
────────────
    pip install pandas plotly
"""

import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ╔══════════════════════════════════════════════════════════════════╗
# ║                  ▶  CHANGE THESE TWO LINES  ◀                   ║
CSV_FILE    = "NQ_5Min.csv"    # path to your CSV file
TARGET_DATE = "1/21/2026"   # day to plot  (M/D/YYYY)
# ╚══════════════════════════════════════════════════════════════════╝


# ── helpers ──────────────────────────────────────────────────────────────────

def eu_float(s: str) -> float:
    """'25.181,75'  →  25181.75"""
    return float(str(s).strip().replace(".", "").replace(",", "."))

def eu_int(s) -> int:
    """'1.234' or '1234'  →  1234"""
    return int(str(s).strip().replace(".", "").replace(",", ""))


# ── 1.  Load & parse ──────────────────────────────────────────────────────────

df = pd.read_csv(CSV_FILE, sep=";", dtype=str, encoding="utf-8-sig")

# Normalise column names: strip whitespace + any invisible/BOM characters
df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

# Debug: uncomment the line below if you still get a KeyError to see exact column names
print("Columns found:", df.columns.tolist())

# Parse datetime – pandas handles single-digit month/day without explicit format
df["Datetime"] = pd.to_datetime(
    df["Date"].str.strip(),
    format="%m/%d/%Y %I:%M %p"
)

for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].apply(eu_float)

df["Volume"] = df["Volume"].apply(eu_int)


# ── 2.  Filter to the requested day ──────────────────────────────────────────

target_date = pd.to_datetime(TARGET_DATE, dayfirst=False).date()
day = (
    df[df["Datetime"].dt.date == target_date]
    .sort_values("Datetime")
    .reset_index(drop=True)
)

if day.empty:
    sys.exit(
        f"\n  No bars found for  {TARGET_DATE}\n"
        f"  Check the date or the CSV path: {CSV_FILE!r}\n"
    )

symbol   = day["Symbol"].iloc[0] if "Symbol" in day.columns else ""
date_str = target_date.strftime("%B %d, %Y")
print(f"  Loaded {len(day)} bars  |  {symbol}  |  {date_str}")


# ── 3.  Compute a few stats for the title ────────────────────────────────────

day_open  = day["Open"].iloc[0]
day_close = day["Close"].iloc[-1]
day_high  = day["High"].max()
day_low   = day["Low"].min()
day_range = day_high - day_low
chg       = day_close - day_open
chg_pct   = chg / day_open * 100
chg_sign  = "▲" if chg >= 0 else "▼"
chg_color = "#26a69a" if chg >= 0 else "#ef5350"

title_text = (
    f"{symbol}  ·  {date_str}  (5-min)   "
    f"O {day_open:,.2f}  H {day_high:,.2f}  L {day_low:,.2f}  C {day_close:,.2f}  "
    f"Range {day_range:,.2f}"
)


# ── 4.  Build the chart ───────────────────────────────────────────────────────

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.02,
)

# --- candlesticks ---
fig.add_trace(
    go.Candlestick(
        x=day["Datetime"],
        open=day["Open"],
        high=day["High"],
        low=day["Low"],
        close=day["Close"],
        name=symbol,
        increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
        whiskerwidth=0.5,
    ),
    row=1, col=1,
)

# --- VWAP (simple running average weighted by volume) ---
day["TypicalPrice"] = (day["High"] + day["Low"] + day["Close"]) / 3
day["CumTPV"]       = (day["TypicalPrice"] * day["Volume"]).cumsum()
day["CumVol"]       = day["Volume"].cumsum()
day["VWAP"]         = day["CumTPV"] / day["CumVol"]

fig.add_trace(
    go.Scatter(
        x=day["Datetime"],
        y=day["VWAP"],
        name="VWAP",
        line=dict(color="#f0c040", width=1.2, dash="dot"),
        hovertemplate="VWAP: %{y:,.2f}<extra></extra>",
    ),
    row=1, col=1,
)

# --- volume bars (colour-matched to candle direction) ---
bar_colors = [
    "#26a69a" if c >= o else "#ef5350"
    for c, o in zip(day["Close"], day["Open"])
]

fig.add_trace(
    go.Bar(
        x=day["Datetime"],
        y=day["Volume"],
        name="Volume",
        marker_color=bar_colors,
        opacity=0.7,
        showlegend=False,
    ),
    row=2, col=1,
)

# ── 5.  Styling ───────────────────────────────────────────────────────────────

BG      = "#131722"
GRID    = "#1e222d"
TEXT    = "#d1d4dc"
BORDER  = "#2a2e39"

fig.update_layout(
    title=dict(
        text=title_text,
        x=0.5,
        font=dict(size=13, color=TEXT),
    ),
    height=700,
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    font=dict(color=TEXT, size=11, family="'Courier New', monospace"),
    legend=dict(
        bgcolor=BORDER,
        bordercolor=BORDER,
        borderwidth=1,
        x=0.01,
        y=0.99,
    ),
    xaxis_rangeslider_visible=False,
    margin=dict(l=70, r=40, t=60, b=40),
    hovermode="x unified",
)

# Price pane
fig.update_yaxes(
    gridcolor=GRID,
    gridwidth=0.5,
    zeroline=False,
    tickformat=",.2f",
    showline=True,
    linecolor=BORDER,
    row=1, col=1,
)
fig.update_xaxes(
    gridcolor=GRID,
    showgrid=False,
    showline=True,
    linecolor=BORDER,
    row=1, col=1,
)

# Volume pane
fig.update_yaxes(
    title_text="Volume",
    title_font=dict(size=10),
    gridcolor=GRID,
    gridwidth=0.5,
    zeroline=False,
    showline=True,
    linecolor=BORDER,
    row=2, col=1,
)
fig.update_xaxes(
    gridcolor=GRID,
    gridwidth=0.5,
    showgrid=True,
    showline=True,
    linecolor=BORDER,
    tickformat="%H:%M",
    row=2, col=1,
)

# ── 6.  Show ──────────────────────────────────────────────────────────────────

fig.show()