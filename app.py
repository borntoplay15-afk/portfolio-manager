"""
Portfolio Manager — live Streamlit web app.
Upload (or demo) a portfolio, answer a quick risk questionnaire, and get a
personalised Markowitz efficient frontier plus explicit invest / divest guidance.

Backend + frontend in one file. Designed to run from Colab via a Cloudflare tunnel,
or deploy to Streamlit Community Cloud later.

Quant stack:
  - Ledoit-Wolf shrinkage covariance (robust vs raw sample cov)
  - Black-Litterman posterior returns driven by broker price targets (stocks only)
  - PyPortfolioOpt efficient frontier over HOLDINGS + candidate STOCKS
  - Risk questionnaire -> the recommended point on the frontier MOVES per individual
    (Conservative = min-variance · Moderate = max-Sharpe · Aggressive = max-utility)
  - One unified conviction list (value quality + broker consensus + optimiser target)
"""
import warnings
warnings.filterwarnings("ignore")
import io

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio

from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel

# Global Plotly theme to match the dark UI (applies to every chart, no per-fig edits)
pio.templates.default = "plotly_dark"
_t = pio.templates["plotly_dark"].layout
_t.paper_bgcolor = "rgba(0,0,0,0)"
_t.plot_bgcolor = "rgba(0,0,0,0)"
_t.font.family = "Inter, sans-serif"
_t.font.color = "#e8edf7"
_t.colorway = ["#6366f1", "#22d3ee", "#22c55e", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399", "#fb7185"]

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Manager", page_icon="📈", layout="wide")

# ── Theme / design system ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

:root{
  --bg:#0b0f1a; --panel:#141a2a; --panel2:#1b2336; --line:#263149;
  --ink:#e8edf7; --muted:#93a0bd; --brand:#6366f1; --brand2:#22d3ee;
  --up:#22c55e; --down:#f43f5e; --gold:#f59e0b;
}
html, body, [class*="css"], .stApp{ font-family:'Inter',sans-serif; }
.stApp{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,.18), transparent 60%),
    radial-gradient(1000px 500px at 100% 0%, rgba(34,211,238,.12), transparent 55%),
    var(--bg);
  color:var(--ink);
}
.block-container{ padding-top:1.4rem; max-width:1400px; }

/* Headings */
h1,h2,h3{ font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-.02em; }
h2,h3{ color:var(--ink); }

/* Hero banner */
.hero{
  background:linear-gradient(135deg, rgba(99,102,241,.22), rgba(34,211,238,.10));
  border:1px solid var(--line); border-radius:18px; padding:22px 26px; margin-bottom:18px;
  box-shadow:0 10px 30px rgba(0,0,0,.35);
}
.hero h1{ font-size:1.7rem; margin:0; }
.hero .sub{ color:var(--muted); font-size:.9rem; margin-top:2px; }
.hero-row{ display:flex; gap:34px; flex-wrap:wrap; align-items:flex-end; margin-top:14px; }
.hero-stat .lbl{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
.hero-stat .val{ font-family:'Space Grotesk'; font-size:1.7rem; font-weight:700; line-height:1.1; }
.up{ color:var(--up); } .down{ color:var(--down); }
.pill{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
       background:rgba(34,197,94,.14); color:var(--up); border:1px solid rgba(34,197,94,.3); }

/* Metric cards */
[data-testid="stMetric"]{
  background:linear-gradient(180deg,var(--panel),var(--panel2));
  border:1px solid var(--line); border-radius:14px; padding:14px 16px;
  box-shadow:0 6px 18px rgba(0,0,0,.28);
}
[data-testid="stMetricLabel"]{ color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"]{ font-family:'Space Grotesk'; font-weight:700; }

/* Tabs as pills */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:none; }
.stTabs [data-baseweb="tab"]{
  background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:9px 16px; color:var(--muted); font-weight:600;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,var(--brand),#4f46e5)!important;
  color:#fff!important; border-color:transparent!important;
}

/* Sidebar */
[data-testid="stSidebar"]{ background:#0d1322; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .stButton>button{ width:100%; }

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,var(--brand),#4f46e5); color:#fff; border:none;
  border-radius:10px; padding:.5rem 1rem; font-weight:600; transition:.15s;
}
.stButton>button:hover{ filter:brightness(1.1); transform:translateY(-1px); }

/* Dataframes */
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }

/* Section captions */
.stCaption, [data-testid="stCaptionContainer"]{ color:var(--muted); }
hr{ border-color:var(--line); }

/* Insight chips */
.chips{ display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 14px; }
.chip{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
       padding:10px 14px; min-width:150px; }
.chip .k{ color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; }
.chip .v{ font-family:'Space Grotesk'; font-weight:700; font-size:1.05rem; margin-top:2px; }

/* Custom holdings table */
.ptable{ width:100%; border-collapse:separate; border-spacing:0; font-size:.9rem;
         border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.ptable thead th{ background:#10182a; color:var(--muted); font-weight:600; text-align:right;
                  padding:11px 14px; font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; }
.ptable thead th:first-child{ text-align:left; }
.ptable tbody td{ padding:11px 14px; text-align:right; border-top:1px solid var(--line);
                  font-variant-numeric:tabular-nums; }
.ptable tbody td:first-child{ text-align:left; }
.ptable tbody tr:hover{ background:rgba(99,102,241,.07); }
.asset{ display:flex; flex-direction:column; }
.asset .tkr{ font-weight:700; color:var(--ink); }
.asset .nm{ color:var(--muted); font-size:.74rem; }
.badge{ font-size:.62rem; padding:2px 7px; border-radius:6px; font-weight:700; margin-left:8px; }
.badge.etf{ background:rgba(99,102,241,.16); color:#a5b4fc; }
.badge.stock{ background:rgba(245,158,11,.16); color:#fcd34d; }
.dot{ display:inline-block; width:8px; height:8px; border-radius:50%; }
.dot.live{ background:var(--up); box-shadow:0 0 6px var(--up); }
.dot.stale{ background:var(--gold); }
.num.up{ color:var(--up); font-weight:600; } .num.down{ color:var(--down); font-weight:600; }

/* Conviction cards */
.cards{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }
.ccard{ background:linear-gradient(180deg,var(--panel),var(--panel2)); border:1px solid var(--line);
        border-radius:16px; padding:16px; box-shadow:0 6px 18px rgba(0,0,0,.25); }
.ccard .top{ display:flex; justify-content:space-between; align-items:flex-start; }
.ccard .tk{ font-family:'Space Grotesk'; font-weight:700; font-size:1.1rem; }
.ccard .co{ color:var(--muted); font-size:.74rem; }
.act{ font-size:.7rem; font-weight:700; padding:4px 10px; border-radius:999px; }
.act.buy{ background:rgba(34,197,94,.16); color:var(--up); border:1px solid rgba(34,197,94,.35); }
.act.watch{ background:rgba(245,158,11,.16); color:var(--gold); border:1px solid rgba(245,158,11,.35); }
.act.avoid{ background:rgba(244,63,94,.16); color:var(--down); border:1px solid rgba(244,63,94,.35); }
.score{ font-family:'Space Grotesk'; font-weight:800; font-size:2rem; margin:8px 0 2px; }
.bar{ height:7px; background:#0e1626; border-radius:999px; overflow:hidden; margin:8px 0 12px; }
.bar > i{ display:block; height:100%; border-radius:999px;
          background:linear-gradient(90deg,#f43f5e,#f59e0b,#22c55e); }
.ccard .row{ display:flex; justify-content:space-between; font-size:.8rem; padding:3px 0;
             border-top:1px dashed var(--line); }
.ccard .row span:first-child{ color:var(--muted); }

/* Invest / divest action lists */
.acts{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:900px){ .acts{ grid-template-columns:1fr; } }
.actcol{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
.actcol h4{ margin:0 0 8px; font-size:.95rem; }
.actrow{ display:flex; justify-content:space-between; align-items:center; padding:8px 0;
         border-top:1px solid var(--line); font-size:.9rem; }
.actrow:first-of-type{ border-top:none; }
.actrow .amt{ font-family:'Space Grotesk'; font-weight:700; }
.invest .amt{ color:var(--up); } .divest .amt{ color:var(--down); }
</style>
""", unsafe_allow_html=True)

RISK_FREE = 0.045
FX_USD_GBP = 1 / 1.34472  # fallback GBPUSD if live FX fetch fails

CCY_SYMBOL = {"GBP": "£", "USD": "$", "EUR": "€"}
# Fallback FX (units of TO per 1 FROM) if the live pair fetch fails
_FX_FALLBACK = {
    "USDGBP": 0.744, "GBPUSD": 1.345, "USDEUR": 0.92, "EURUSD": 1.09,
    "GBPEUR": 1.16, "EURGBP": 0.86,
}

# ── DEMO PORTFOLIO (generic example — safe to show publicly) ───────────────────
# A neutral, diversified sample so anyone can try the app instantly.
# avg_gbp  = average cost per unit in the BASE currency (cost basis)
# last_gbp = last known price per unit in the BASE currency (fallback if live fails)
# y_ccy    = the currency YAHOO quotes this ticker in (LSE = GBp/pence; US = USD).
DEMO_HOLDINGS = {
    "VWRP.L": {"name": "Vanguard FTSE All-World",   "qty": 12, "avg_gbp": 95.0,  "last_gbp": 108.0, "y_ccy": "GBp", "kind": "ETF"},
    "AAPL":   {"name": "Apple Inc.",                "qty": 6,  "avg_gbp": 130.0, "last_gbp": 150.0, "y_ccy": "USD", "kind": "Stock"},
    "MSFT":   {"name": "Microsoft Corp.",           "qty": 4,  "avg_gbp": 260.0, "last_gbp": 320.0, "y_ccy": "USD", "kind": "Stock"},
    "AZN.L":  {"name": "AstraZeneca plc",           "qty": 2,  "avg_gbp": 105.0, "last_gbp": 120.0, "y_ccy": "GBp", "kind": "Stock"},
    "SHEL.L": {"name": "Shell plc",                 "qty": 15, "avg_gbp": 25.0,  "last_gbp": 28.0,  "y_ccy": "GBp", "kind": "Stock"},
    "IGLT.L": {"name": "iShares Core UK Gilts",     "qty": 20, "avg_gbp": 11.0,  "last_gbp": 11.4,  "y_ccy": "GBp", "kind": "ETF"},
    "SGLN.L": {"name": "iShares Physical Gold",     "qty": 5,  "avg_gbp": 45.0,  "last_gbp": 52.0,  "y_ccy": "GBp", "kind": "ETF"},
}

# Optional PRIVATE override (gitignored, NEVER pushed/deployed). If a local
# _private_holdings.py exists exposing a HOLDINGS dict, use it — so the owner's
# real portfolio loads locally and in Colab, while the public deploy (which has
# no such file) safely shows the generic demo above.
try:
    from _private_holdings import HOLDINGS as _PRIV
    if isinstance(_PRIV, dict) and _PRIV:
        DEMO_HOLDINGS = _PRIV
except Exception:
    pass

# Candidate stocks the optimiser may recommend buying into (generic examples;
# uploaded holdings always enter the universe regardless of this list).
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "BRK-B", "JPM", "JNJ", "PG", "KO",
    "V", "HD", "MCD", "WMT", "CVX", "XOM", "BAC", "UNH",
    "AZN.L", "SHEL.L", "HSBA.L", "BP.L", "GSK.L", "LLOY.L", "BARC.L", "RIO.L", "ULVR.L",
]

# ── Session state ─────────────────────────────────────────────────────────────
if "holdings" not in st.session_state:
    st.session_state.holdings = dict(DEMO_HOLDINGS)
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(WATCHLIST)
if "base_ccy" not in st.session_state:
    st.session_state.base_ccy = "GBP"
if "pf_name" not in st.session_state:
    st.session_state.pf_name = "Demo portfolio"
if "risk" not in st.session_state:
    st.session_state.risk = "Moderate"
if "risk_aversion" not in st.session_state:
    st.session_state.risk_aversion = 1.0
if "risk_assessed" not in st.session_state:
    st.session_state.risk_assessed = False
if "risk_score" not in st.session_state:
    st.session_state.risk_score = None


# ─────────────────────────── DATA LAYER ──────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fx_rate(frm: str, to: str) -> float:
    """Units of `to` per 1 unit of `frm` (e.g. fx_rate('USD','GBP') ≈ 0.79)."""
    if frm == to:
        return 1.0
    pair = f"{frm}{to}=X"
    try:
        v = float(yf.Ticker(pair).fast_info["last_price"])
        if v and v > 0:
            return v
    except Exception:
        pass
    try:
        d = yf.download(pair, period="5d", progress=False)
        cl = d["Close"] if isinstance(d.columns, pd.MultiIndex) or "Close" in getattr(d, "columns", []) else d
        s = cl.dropna() if hasattr(cl, "dropna") else None
        if s is not None and len(s):
            v = float(s.iloc[-1] if getattr(s, "ndim", 1) == 1 else s.iloc[-1, 0])
            if v > 0:
                return v
    except Exception:
        pass
    return _FX_FALLBACK.get(f"{frm}{to}", 1.0)


def to_base(price, quote_ccy, base="GBP"):
    """Convert a Yahoo quote into the user's BASE currency.
    Handles London pence (GBp/GBX -> GBP) then any FX cross."""
    if price is None:
        return None
    q = quote_ccy
    if q in ("GBX", "GBp"):     # London pence -> pounds
        price = float(price) / 100.0
        q = "GBP"
    if q == base:
        return float(price)
    return float(price) * fx_rate(q, base)


# Back-compat alias (demo base is GBP)
def to_gbp(price, currency):
    return to_base(price, currency, "GBP")


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def latest_closes(tickers: tuple) -> dict:
    """One batched download for ALL tickers — reliable, avoids per-ticker
    rate-limiting. Returns {ticker: last close in Yahoo's quote currency}."""
    try:
        raw = yf.download(list(tickers), period="5d", progress=False)
        cl = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(cl, pd.Series):
            cl = cl.to_frame(tickers[0])
        out = {}
        for t in cl.columns:
            s = cl[t].dropna()
            if not s.empty:
                out[t] = float(s.iloc[-1])
        return out
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_history(tickers: tuple, period: str = "3y") -> pd.DataFrame:
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
    px_ = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(px_, pd.Series):
        px_ = px_.to_frame(tickers[0])
    return px_.dropna(how="all")


def rec_label(key):
    return {
        "strong_buy": "🟢 Strong Buy", "buy": "🟩 Buy", "hold": "🟡 Hold",
        "sell": "🟥 Sell", "strong_sell": "🔴 Strong Sell",
    }.get(str(key).lower().replace(" ", "_"), "—")


# ─────────────────── BUILD HOLDINGS FROM UPLOAD / MANUAL ──────────────────────
def build_holdings(rows, base: str) -> tuple:
    """rows: iterable of (ticker, qty, avg_cost). Returns (holdings_dict, errors)."""
    holds, errors = {}, []
    clean = []
    for tk, qty, avg in rows:
        tk = str(tk).strip().upper()
        if not tk:
            continue
        try:
            qty = float(qty); avg = float(avg)
        except (TypeError, ValueError):
            errors.append(f"{tk}: bad quantity/cost")
            continue
        if qty <= 0:
            errors.append(f"{tk}: quantity must be > 0")
            continue
        clean.append((tk, qty, avg))

    if not clean:
        return {}, errors

    closes = latest_closes(tuple(t for t, _, _ in clean))
    for tk, qty, avg in clean:
        info = get_info(tk)
        ccy = info.get("currency") or ("GBp" if tk.endswith(".L") else "USD")
        qt = (info.get("quoteType") or "").upper()
        kind = "ETF" if qt in ("ETF", "MUTUALFUND") else "Stock"
        name = info.get("shortName") or info.get("longName") or tk
        raw = closes.get(tk)
        last = to_base(raw, ccy, base) if raw else avg
        holds[tk] = {"name": str(name)[:34], "qty": qty, "avg_gbp": avg,
                     "last_gbp": last, "y_ccy": ccy, "kind": kind}
    return holds, errors


def _aggregate_transactions(df, base: str) -> tuple:
    """Aggregate a broker TRANSACTION export (e.g. Trading 212 / most brokers) into
    holdings. Sums buys/sells per ticker into a net quantity + weighted-average buy
    cost. Uses the 'Total' column (already in account currency) for cost, so GBX/USD
    price quirks don't matter. LSE tickers (GBX/GBP-priced) get a '.L' suffix for Yahoo."""
    cols = {c: c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return n
        return None
    c_act = col("action")
    c_tk = col("ticker", "symbol")
    c_sh = col("no. of shares", "no of shares", "shares", "quantity", "qty")
    c_total = col("total", "total amount")
    c_pcur = col("currency (price / share)", "currency (price/share)", "price currency")
    c_pps = col("price / share", "price/share", "price")
    if not (c_tk and c_sh):
        raise ValueError("Transaction export needs 'Ticker' and 'No. of shares' columns.")

    agg, errors = {}, []
    for _, r in df.iterrows():
        action = str(r.get(c_act, "")).lower()
        if c_act and not ("buy" in action or "sell" in action):
            continue  # skip interest, deposits, dividends, fx, etc.
        raw = str(r.get(c_tk) or "").strip().upper()
        if not raw or raw == "NAN":
            continue
        try:
            sh = float(r.get(c_sh))
        except (TypeError, ValueError):
            continue
        if sh <= 0:
            continue
        # Map to a Yahoo ticker: LSE listings (priced in GBX/GBP) need a ".L" suffix
        pcur = str(r.get(c_pcur, "") or "").strip().upper()
        yk = raw if "." in raw else (raw + ".L" if pcur in ("GBX", "GBP") else raw)
        # Cost of this line in the account/base currency
        total = None
        try:
            total = float(r.get(c_total))
        except (TypeError, ValueError):
            try:  # fall back to shares × price/share
                total = sh * float(r.get(c_pps))
            except (TypeError, ValueError):
                total = None
        a = agg.setdefault(yk, {"net": 0.0, "buy_sh": 0.0, "buy_cost": 0.0})
        if "sell" in action:
            a["net"] -= sh
        else:
            a["net"] += sh
            a["buy_sh"] += sh
            if total is not None:
                a["buy_cost"] += total

    rows = []
    for yk, a in agg.items():
        if a["net"] <= 1e-9:
            continue  # fully sold out
        avg = (a["buy_cost"] / a["buy_sh"]) if a["buy_sh"] else 0.0
        rows.append((yk, a["net"], avg))
    if not rows:
        raise ValueError("No open buy positions found in this export.")
    holds, errs = build_holdings(rows, base)
    return holds, errors + errs


def parse_csv(file, base: str) -> tuple:
    """Accept either a simple holdings list (ticker/quantity/avg_cost) OR a broker
    transaction export (Trading 212 etc. — Action/Ticker/No. of shares/Total)."""
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = set(df.columns)

    # Broker transaction export? (has an Action column + shares/ticker)
    if "action" in cols and ("ticker" in cols or "symbol" in cols) and \
       any(c in cols for c in ("no. of shares", "no of shares", "shares")):
        return _aggregate_transactions(df, base)

    # Otherwise a simple holdings list
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None
    c_tk = pick("ticker", "symbol", "stock")
    c_qty = pick("quantity", "qty", "shares", "units")
    c_avg = pick("avg_cost", "avg cost", "cost", "price", "avg", "average cost", "avg_price")
    if not (c_tk and c_qty):
        raise ValueError("CSV needs either 'ticker' + 'quantity' columns, or a broker "
                         "transaction export (Action / Ticker / No. of shares / Total).")
    rows = []
    for _, r in df.iterrows():
        avg = r[c_avg] if c_avg else 0
        rows.append((r[c_tk], r[c_qty], avg))
    return build_holdings(rows, base)


SAMPLE_CSV = "ticker,quantity,avg_cost\nAAPL,10,180\nMSFT,5,330\nVWRP.L,3,95\nJPM,8,150\n"


# ─────────────────────────── VALUE METRICS ───────────────────────────────────
def graham_number(info):
    eps = info.get("trailingEps")
    bvps = info.get("bookValue")
    if eps and bvps and eps > 0 and bvps > 0:
        return (22.5 * eps * bvps) ** 0.5
    return None


def quality_score(info):
    """Cheap Piotroski-style 0-100 quality proxy from .info fields."""
    pts, total = 0, 0
    checks = [
        (info.get("returnOnEquity"), lambda v: v > 0.12),     # strong ROE
        (info.get("returnOnAssets"), lambda v: v > 0.05),
        (info.get("grossMargins"),   lambda v: v > 0.30),     # pricing power
        (info.get("profitMargins"),  lambda v: v > 0.08),
        (info.get("operatingMargins"), lambda v: v > 0.10),
        (info.get("currentRatio"),   lambda v: v > 1.2),      # liquidity
        (info.get("debtToEquity"),   lambda v: v < 100),      # yfinance reports as %
        (info.get("revenueGrowth"),  lambda v: v > 0.0),      # growing
        (info.get("freeCashflow"),   lambda v: v > 0),        # FCF positive
    ]
    for val, test in checks:
        if val is not None:
            total += 1
            if test(val):
                pts += 1
    return round(100 * pts / total) if total else None


def broker_score(info):
    """0-100 from recommendationMean (1=Strong Buy .. 5=Strong Sell)."""
    rm = info.get("recommendationMean")
    if rm is None:
        return None
    return round(max(0, min(100, (5 - rm) / 4 * 100)))


def analyst_upside(info):
    cur = info.get("currentPrice") or info.get("regularMarketPrice")
    tgt = info.get("targetMeanPrice")
    if cur and tgt:
        return (tgt / cur - 1) * 100
    return None


# ───────────────────────── PORTFOLIO VALUATION ───────────────────────────────
def portfolio_table(holdings, base="GBP"):
    closes = latest_closes(tuple(holdings.keys()))   # one batched call for all tickers
    rows = []
    for tk, h in holdings.items():
        raw = closes.get(tk)
        if raw:
            live = to_base(raw, h.get("y_ccy", "GBp"), base)
            source = "🟢 live"
        else:
            live = h.get("last_gbp") or h["avg_gbp"]   # fallback (NOT cost — gain still shows)
            source = "🟡 last-known"
        cost = h["qty"] * h["avg_gbp"]
        val = h["qty"] * live
        rows.append({
            "Ticker": tk, "Name": h["name"], "Kind": h["kind"],
            "Qty": round(h["qty"], 4), "Avg": round(h["avg_gbp"], 4),
            "Live": round(live, 4), "Cost": round(cost, 2), "Value": round(val, 2),
            "P&L": round(val - cost, 2),
            "Return %": round((val - cost) / cost * 100, 2) if cost else 0,
            "Source": source,
        })
    return pd.DataFrame(rows)


def holdings_html(df, sym="£"):
    """Render the holdings DataFrame as a styled HTML table."""
    rows = ""
    for _, r in df.sort_values("Value", ascending=False).iterrows():
        pcls = "up" if r["P&L"] >= 0 else "down"
        badge = "etf" if r["Kind"] == "ETF" else "stock"
        dot = "live" if "live" in r["Source"] else "stale"
        sign = "+" if r["P&L"] >= 0 else "−"
        rows += f"""<tr>
          <td><div class="asset"><span class="tkr">{r['Ticker']}<span class="badge {badge}">{r['Kind']}</span></span>
              <span class="nm">{r['Name']}</span></div></td>
          <td>{r['Qty']:.3f}</td>
          <td>{sym}{r['Avg']:,.2f}</td>
          <td>{sym}{r['Live']:,.2f}</td>
          <td>{sym}{r['Value']:,.2f}</td>
          <td class="num {pcls}">{sign}{sym}{abs(r['P&L']):,.2f}</td>
          <td class="num {pcls}">{r['Return %']:+.1f}%</td>
          <td><span class="dot {dot}"></span></td>
        </tr>"""
    return f"""<table class="ptable"><thead><tr>
        <th>Asset</th><th>Qty</th><th>Avg</th><th>Price</th><th>Value</th>
        <th>P&amp;L</th><th>Return</th><th>Live</th></tr></thead><tbody>{rows}</tbody></table>"""


# ───────────────────────── OPTIMISATION ENGINE ───────────────────────────────
def optimise(prices, kinds, risk, risk_aversion=1.0, use_bl=True):
    """Returns dict with frontier curve, max-sharpe / min-vol / RISK-TARGET weights.
    The 'w_target' is the point each individual should hold, and it MOVES with risk:
      Conservative -> min-variance · Moderate -> max-Sharpe · Aggressive -> max-utility.
    Returns None if the cleaned price matrix is too small to optimise."""
    # ── Bulletproof data cleaning (US + LSE calendars differ → align & fill) ──
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(axis=1, how="all")                 # drop dead tickers
    prices = prices.ffill().dropna()                          # align trading days
    prices = prices.loc[:, prices.nunique() > 1]              # drop constant cols
    if prices.shape[1] < 2 or prices.shape[0] < 30:
        return None
    tickers = list(prices.columns)

    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(prices)
    mu_hist = expected_returns.mean_historical_return(prices)

    # Black-Litterman: blend broker targets into expected returns for the STOCKS
    mu = mu_hist.copy()
    bl_used = False
    if use_bl:
        try:
            stock_tks = [t for t in tickers if kinds.get(t) == "Stock"]
            if len(stock_tks) >= 2:
                S_st = S.loc[stock_tks, stock_tks]
                mcaps, views = {}, {}
                for t in stock_tks:
                    info = get_info(t)
                    mc = info.get("marketCap")
                    up = analyst_upside(info)
                    if mc and up is not None:
                        mcaps[t] = mc
                        views[t] = up / 100.0  # annual view = analyst upside
                if len(mcaps) == len(stock_tks) and views:
                    prior = black_litterman.market_implied_prior_returns(pd.Series(mcaps), 2.5, S_st)
                    bl = BlackLittermanModel(S_st, pi=prior, absolute_views=views)
                    post = bl.bl_returns()
                    for t in stock_tks:
                        mu[t] = post[t]
                    bl_used = True
        except Exception:
            bl_used = False

    # Weight bounds by risk appetite (concentration cap)
    bound = {"Conservative": 0.15, "Moderate": 0.25, "Aggressive": 0.40}[risk]

    def fresh_ef():
        return EfficientFrontier(mu, S, weight_bounds=(0, bound))

    try:
        ef = fresh_ef(); ef.max_sharpe(risk_free_rate=RISK_FREE)
        w_sharpe = ef.clean_weights()
        perf_sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE)
    except Exception:
        w_sharpe = {t: 1 / len(tickers) for t in tickers}
        perf_sharpe = (float(mu.mean()), float(np.sqrt(np.diag(S).mean())), 0)

    try:
        ef2 = fresh_ef(); ef2.min_volatility()
        w_minvol = ef2.clean_weights()
        perf_minvol = ef2.portfolio_performance(risk_free_rate=RISK_FREE)
    except Exception:
        w_minvol = {t: 1 / len(tickers) for t in tickers}
        perf_minvol = (float(mu.mean()), float(np.sqrt(np.diag(S).mean())), 0)

    # ── RISK-DRIVEN TARGET: the point THIS individual should hold ──────────────
    # This is the change that makes the questionnaire actually move the answer.
    if risk == "Conservative":
        w_target, perf_target, target_label = w_minvol, perf_minvol, "Min-variance"
    elif risk == "Aggressive":
        try:
            ef4 = fresh_ef(); ef4.max_quadratic_utility(risk_aversion=max(0.3, risk_aversion))
            w_target = ef4.clean_weights()
            perf_target = ef4.portfolio_performance(risk_free_rate=RISK_FREE)
            target_label = "Max utility (return-seeking)"
        except Exception:
            w_target, perf_target, target_label = w_sharpe, perf_sharpe, "Max Sharpe"
    else:  # Moderate
        w_target, perf_target, target_label = w_sharpe, perf_sharpe, "Max Sharpe"

    # Smooth analytic frontier
    fr_ret, fr_vol = [], []
    lo, hi = float(mu.min()), float(mu.max())
    for target in np.linspace(lo, hi, 40):
        try:
            ef3 = fresh_ef(); ef3.efficient_return(target_return=target)
            r, v, _ = ef3.portfolio_performance(risk_free_rate=RISK_FREE)
            fr_ret.append(r); fr_vol.append(v)
        except Exception:
            pass

    return {
        "tickers": tickers, "mu": mu, "S": S,
        "w_sharpe": w_sharpe, "perf_sharpe": perf_sharpe,
        "w_minvol": w_minvol, "perf_minvol": perf_minvol,
        "w_target": w_target, "perf_target": perf_target, "target_label": target_label,
        "fr_ret": fr_ret, "fr_vol": fr_vol, "bl_used": bl_used,
    }


# ── Risk questionnaire scoring ────────────────────────────────────────────────
RISK_QUESTIONS = [
    ("When do you expect to need most of this money?",
     [("Within 3 years", 1), ("3 to 10 years", 2), ("More than 10 years", 3)]),
    ("Your portfolio drops 20% in a month. You…",
     [("Sell to stop further losses", 1), ("Hold and wait it out", 2), ("Buy more at lower prices", 3)]),
    ("What is your main goal for this money?",
     [("Preserve what I have", 1), ("Balanced, steady growth", 2), ("Maximise long-term growth", 3)]),
    ("How stable is your income / job?",
     [("Unstable / variable", 1), ("Stable", 2), ("Very secure, with cash savings", 3)]),
    ("How would you rate your investing experience?",
     [("Beginner", 1), ("Some experience", 2), ("Experienced", 3)]),
]


def score_to_profile(score):
    if score <= 8:
        return "Conservative", 4.0    # high risk aversion (only used by max-utility path)
    if score >= 13:
        return "Aggressive", 0.6      # low risk aversion -> return-seeking
    return "Moderate", 1.5


# ════════════════════════════════ UI ═════════════════════════════════════════
base = st.session_state.base_ccy
SYM = CCY_SYMBOL.get(base, "£")

with st.sidebar:
    st.title("📈 Portfolio Manager")
    st.caption(st.session_state.pf_name)
    st.divider()

    # ── 1. Load your portfolio ────────────────────────────────────────────────
    st.subheader("1 · Your portfolio")
    src = st.radio("Source", ["Demo portfolio", "Upload CSV", "Enter manually"],
                   captions=["One-click sample", "Holdings list OR broker export", "Type your holdings"])

    if src != "Demo portfolio":
        base = st.selectbox("Base currency", ["GBP", "USD", "EUR"],
                            index=["GBP", "USD", "EUR"].index(st.session_state.base_ccy))

    if src == "Demo portfolio":
        if st.button("Load demo portfolio", use_container_width=True):
            st.session_state.holdings = dict(DEMO_HOLDINGS)
            st.session_state.base_ccy = "GBP"
            st.session_state.pf_name = "Demo portfolio"
            st.rerun()

    elif src == "Upload CSV":
        st.download_button("⬇ Sample CSV", SAMPLE_CSV, "portfolio_template.csv",
                           "text/csv", use_container_width=True)
        st.caption("Works with a simple holdings list **or** a broker transaction "
                   "export (e.g. Trading 212). For a transaction export, use your "
                   "**full history** — a short date range only captures recent trades.")
        up = st.file_uploader("Upload portfolio CSV", type=["csv"])
        if up is not None and st.button("Load this portfolio", use_container_width=True):
            try:
                holds, errs = parse_csv(up, base)
                if holds:
                    st.session_state.holdings = holds
                    st.session_state.base_ccy = base
                    st.session_state.pf_name = "Your uploaded portfolio"
                    if errs:
                        st.warning("Skipped: " + "; ".join(errs))
                    st.rerun()
                else:
                    st.error("No valid rows found. " + ("; ".join(errs) if errs else ""))
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    else:  # Enter manually
        seed = pd.DataFrame({"ticker": ["AAPL", "MSFT", ""],
                             "quantity": [10.0, 5.0, None],
                             "avg_cost": [180.0, 330.0, None]})
        edited = st.data_editor(seed, num_rows="dynamic", use_container_width=True,
                                key="manual_editor")
        if st.button("Load this portfolio", use_container_width=True):
            rows = [(r["ticker"], r["quantity"], r["avg_cost"]) for _, r in edited.iterrows()
                    if str(r.get("ticker") or "").strip()]
            holds, errs = build_holdings(rows, base)
            if holds:
                st.session_state.holdings = holds
                st.session_state.base_ccy = base
                st.session_state.pf_name = "Your portfolio"
                if errs:
                    st.warning("Skipped: " + "; ".join(errs))
                st.rerun()
            else:
                st.error("No valid rows. " + ("; ".join(errs) if errs else ""))

    st.divider()

    # ── 2. Risk questionnaire ─────────────────────────────────────────────────
    st.subheader("2 · Your risk profile")
    with st.expander("📋 Answer 5 quick questions", expanded=not st.session_state.risk_assessed):
        answers = []
        for i, (q, opts) in enumerate(RISK_QUESTIONS):
            labels = [o[0] for o in opts]
            choice = st.radio(q, labels, key=f"rq_{i}", index=1)
            answers.append(dict(opts)[choice])
        if st.button("Assess my risk profile", use_container_width=True):
            sc = sum(answers)
            prof, ra = score_to_profile(sc)
            st.session_state.risk = prof
            st.session_state.risk_aversion = ra
            st.session_state.risk_score = sc
            st.session_state.risk_assessed = True
            st.rerun()

    if st.session_state.risk_assessed:
        st.success(f"**{st.session_state.risk}** · score {st.session_state.risk_score}/15")

    risk = st.select_slider("Risk appetite (override)",
                            ["Conservative", "Moderate", "Aggressive"],
                            value=st.session_state.risk)
    if risk != st.session_state.risk:
        st.session_state.risk = risk
        _, st.session_state.risk_aversion = score_to_profile(
            {"Conservative": 6, "Moderate": 11, "Aggressive": 14}[risk])
    st.caption({
        "Conservative": "Min-variance target · max 15% per asset",
        "Moderate": "Max-Sharpe target · max 25% per asset",
        "Aggressive": "Return-seeking target · max 40% per asset",
    }[risk])

    new_cash = st.number_input(f"New cash to invest ({SYM})", min_value=0, value=2000, step=250)
    st.divider()

    # ── 3. Candidate stocks ───────────────────────────────────────────────────
    st.subheader("3 · Candidate stocks")
    st.caption("Optional — names the optimiser may recommend buying into. Your own "
               "holdings are always included.")
    candidates = st.multiselect(
        "Stocks the optimiser may recommend buying",
        options=st.session_state.watchlist,
        default=["AAPL", "MSFT", "BRK-B", "JNJ", "AZN.L", "SHEL.L", "HSBA.L", "GSK.L"],
    )
    extra = st.text_input("Add a ticker", placeholder="NVDA or ULVR.L").upper().strip()
    if st.button("Add to candidates", use_container_width=True) and extra:
        if extra not in st.session_state.watchlist:
            st.session_state.watchlist.append(extra)
            st.rerun()

risk_aversion = st.session_state.risk_aversion

# ── Hero banner (computed once, reused across tabs) ───────────────────────────
pf_df = portfolio_table(st.session_state.holdings, base)
tc, tv = pf_df["Cost"].sum(), pf_df["Value"].sum()
pnl = tv - tc
ret_pct = (pnl / tc * 100) if tc else 0
live_n = int((pf_df["Source"] == "🟢 live").sum())
dirn = "up" if pnl >= 0 else "down"
arrow = "▲" if pnl >= 0 else "▼"

st.markdown(f"""
<div class="hero">
  <h1>📈 Portfolio Manager</h1>
  <div class="sub">{st.session_state.pf_name} · {risk} profile · {live_n}/{len(pf_df)} prices live</div>
  <div class="hero-row">
    <div class="hero-stat"><div class="lbl">Total Value</div><div class="val">{SYM}{tv:,.0f}</div></div>
    <div class="hero-stat"><div class="lbl">Invested</div><div class="val">{SYM}{tc:,.0f}</div></div>
    <div class="hero-stat"><div class="lbl">Total P&amp;L</div>
        <div class="val {dirn}">{arrow} {SYM}{abs(pnl):,.0f}</div></div>
    <div class="hero-stat"><div class="lbl">Return</div>
        <div class="val {dirn}">{ret_pct:+.1f}%</div></div>
    <div class="hero-stat"><div class="lbl">Positions</div><div class="val">{len(pf_df)}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Portfolio", "🎯 Conviction Buy List", "📈 Efficient Frontier", "🔎 Value Screener"
])

# ── TAB 1: PORTFOLIO ──────────────────────────────────────────────────────────
with tab1:
    df = pf_df
    top = df.loc[df["Value"].idxmax()]
    best = df.loc[df["Return %"].idxmax()]
    worst = df.loc[df["Return %"].idxmin()]

    # Insight chips
    st.markdown(f"""
    <div class="chips">
      <div class="chip"><div class="k">Largest position</div>
          <div class="v">{top['Ticker']} · {top['Value']/tv*100:.0f}%</div></div>
      <div class="chip"><div class="k">Best performer</div>
          <div class="v up">{best['Ticker']} {best['Return %']:+.1f}%</div></div>
      <div class="chip"><div class="k">Weakest</div>
          <div class="v {'up' if worst['Return %']>=0 else 'down'}">{worst['Ticker']} {worst['Return %']:+.1f}%</div></div>
      <div class="chip"><div class="k">Diversification</div>
          <div class="v">{len(df)} holdings</div></div>
    </div>
    """, unsafe_allow_html=True)

    if top["Value"] / tv > 0.4:
        st.warning(f"⚠️ **Concentration risk:** {top['Ticker']} is {top['Value']/tv*100:.0f}% of your "
                   f"portfolio — a large single-name exposure. The optimiser below will suggest trimming it. "
                   f"(Tip: if this is your employer's stock, your salary and this holding share the same risk.)")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("###### Holdings")
        st.markdown(holdings_html(df, SYM), unsafe_allow_html=True)
    with right:
        st.markdown("###### Allocation")
        fig = px.pie(df, values="Value", names="Ticker", hole=0.62)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="#0b0f1a", width=2)))
        fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10, l=10, r=10),
                          annotations=[dict(text=f"<b>{SYM}{tv:,.0f}</b><br><span style='color:#93a0bd'>total</span>",
                                            x=0.5, y=0.5, font_size=18, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"🟢 live price · 🟡 last-known (fallback). Values shown in {base}.")

    st.markdown("###### Return by position")
    ds = df.sort_values("Return %")
    fig2 = go.Figure(go.Bar(x=ds["Return %"], y=ds["Ticker"], orientation="h",
                            marker_color=["#22c55e" if x >= 0 else "#f43f5e" for x in ds["Return %"]],
                            text=[f"{x:+.1f}%" for x in ds["Return %"]], textposition="outside",
                            marker_line_width=0))
    fig2.update_layout(height=330, margin=dict(t=10, b=10), xaxis_title="Return %", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

# Build combined universe once for tabs 2 & 3
universe = list(st.session_state.holdings.keys()) + candidates
kinds = {t: st.session_state.holdings.get(t, {}).get("kind", "Stock") for t in universe}

# ── TAB 2: CONVICTION BUY LIST ────────────────────────────────────────────────
with tab2:
    st.subheader("🎯 One ranked buy list")
    st.caption("Blends value-quality + broker consensus + analyst upside + optimiser target into a single conviction score (0-100).")

    if not candidates:
        st.info("Pick candidate stocks in the sidebar to build your buy list.")
    else:
        with st.spinner("Scoring candidates & running optimiser…"):
            prices = get_history(tuple(universe), "3y")
            avail = [t for t in universe if t in prices.columns]
            opt = optimise(prices[avail], kinds, risk, risk_aversion) if len(avail) >= 2 else None
            target_w = dict(opt["w_target"]) if opt else {}

            rows = []
            for t in candidates:
                info = get_info(t)
                q = quality_score(info)
                b = broker_score(info)
                up = analyst_upside(info)
                ow = target_w.get(t, 0) * 100
                gnum = graham_number(info)
                cur = info.get("currentPrice") or info.get("regularMarketPrice")
                mos = ((gnum - cur) / cur * 100) if (gnum and cur) else None  # margin of safety

                # composite conviction (weighted; skips missing components)
                parts, wts = [], []
                if q is not None:  parts.append(q);                       wts.append(0.30)
                if b is not None:  parts.append(b);                       wts.append(0.25)
                if up is not None: parts.append(max(0, min(100, 50 + up)));wts.append(0.25)
                if ow:             parts.append(min(100, ow * 4));         wts.append(0.20)
                conv = round(sum(p * w for p, w in zip(parts, wts)) / sum(wts)) if parts else None

                action = "—"
                if conv is not None:
                    action = "🟢 BUY" if conv >= 65 else "🟡 WATCH" if conv >= 50 else "🔴 AVOID"

                rows.append({
                    "Conviction": conv if conv is not None else 0,
                    "Action": action, "Ticker": t,
                    "Company": (info.get("shortName") or t)[:24],
                    "Quality": q if q is not None else "—",
                    "Broker": rec_label(info.get("recommendationKey", "")),
                    "Upside %": round(up, 1) if up is not None else "—",
                    "Margin of Safety %": round(mos, 1) if mos is not None else "—",
                    "Optimiser %": round(ow, 1),
                })

            cdf = pd.DataFrame(rows).sort_values("Conviction", ascending=False)

        # Render as score cards
        cards = ""
        for _, r in cdf.iterrows():
            conv = r["Conviction"]
            acls = "buy" if conv >= 65 else "watch" if conv >= 50 else "avoid"
            alabel = "BUY" if conv >= 65 else "WATCH" if conv >= 50 else "AVOID"
            up_s = f"{r['Upside %']:+.1f}%" if isinstance(r["Upside %"], (int, float)) else "—"
            mos_s = f"{r['Margin of Safety %']:+.1f}%" if isinstance(r["Margin of Safety %"], (int, float)) else "—"
            cards += f"""<div class="ccard">
              <div class="top"><div><div class="tk">{r['Ticker']}</div><div class="co">{r['Company']}</div></div>
                <div class="act {acls}">{alabel}</div></div>
              <div class="score">{conv}<span style="font-size:1rem;color:#93a0bd">/100</span></div>
              <div class="bar"><i style="width:{max(2,min(100,conv))}%"></i></div>
              <div class="row"><span>Quality</span><span>{r['Quality']}</span></div>
              <div class="row"><span>Broker</span><span>{r['Broker']}</span></div>
              <div class="row"><span>Analyst upside</span><span>{up_s}</span></div>
              <div class="row"><span>Margin of safety</span><span>{mos_s}</span></div>
              <div class="row"><span>Optimiser target</span><span>{r['Optimiser %']}%</span></div>
            </div>"""
        st.markdown(f'<div class="cards">{cards}</div>', unsafe_allow_html=True)

        with st.expander("📋 View as table"):
            st.dataframe(cdf, use_container_width=True, hide_index=True)

        if opt and opt.get("bl_used"):
            st.caption("✅ Expected returns blended with broker price targets via Black-Litterman.")
        else:
            st.caption("ℹ️ Optimiser/Black-Litterman inputs incomplete — conviction uses value + broker signals.")

        st.markdown("**How to read it:** Conviction ≥65 = BUY signal across value + analysts + optimiser. "
                    "Margin of Safety = how far below the Graham fair value the stock trades (higher = cheaper).")

# ── TAB 3: EFFICIENT FRONTIER ─────────────────────────────────────────────────
with tab3:
    st.subheader("📈 Markowitz Efficient Frontier")
    st.caption(f"Universe: your {len(st.session_state.holdings)} holdings + {len(candidates)} candidate stocks · "
               f"Ledoit-Wolf covariance · risk-free {RISK_FREE*100:.1f}% · **{risk}** target")

    px_hist = get_history(tuple(universe), "3y")
    valid = [t for t in universe if t in px_hist.columns]
    opt = optimise(px_hist[valid], kinds, risk, risk_aversion) if len(valid) >= 3 else None
    if opt is None:
        st.info("Need at least 3 instruments with usable price history. Add candidate stocks in the sidebar.")
    else:
        mu, S = opt["mu"], opt["S"]
        w_target = opt["w_target"]

        # current weights (by value across holdings in the optimisable universe)
        pf = pf_df
        tv = pf["Value"].sum()
        cur_w = {r["Ticker"]: r["Value"] / tv for _, r in pf.iterrows() if r["Ticker"] in valid}
        wvec = np.array([cur_w.get(t, 0) for t in opt["tickers"]])
        if wvec.sum() > 0:
            wvec = wvec / wvec.sum()
            cur_ret = float(wvec @ mu.values)
            cur_vol = float(np.sqrt(wvec @ S.values @ wvec))
            cur_sh = (cur_ret - RISK_FREE) / cur_vol if cur_vol else 0
        else:
            cur_ret = cur_vol = cur_sh = None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[v*100 for v in opt["fr_vol"]], y=[r*100 for r in opt["fr_ret"]],
                                 mode="lines", line=dict(color="#6366f1", width=3), name="Efficient frontier"))
        sr, sv, ss = opt["perf_sharpe"]
        mr, mv, _ = opt["perf_minvol"]
        tr, tvol, tsh = opt["perf_target"]
        fig.add_trace(go.Scatter(x=[sv*100], y=[sr*100], mode="markers",
                                 marker=dict(size=11, color="#16a34a", symbol="diamond"), name="Max Sharpe"))
        fig.add_trace(go.Scatter(x=[mv*100], y=[mr*100], mode="markers",
                                 marker=dict(size=11, color="#06b6d4", symbol="pentagon"), name="Min Variance"))
        # The recommended point for THIS user's risk profile
        fig.add_trace(go.Scatter(x=[tvol*100], y=[tr*100], mode="markers+text",
                                 marker=dict(size=20, color="#a78bfa", symbol="star",
                                             line=dict(color="#fff", width=1.5)),
                                 text=[f"Your target ({opt['target_label']})"], textposition="top center",
                                 name="Your target"))
        if cur_ret is not None:
            fig.add_trace(go.Scatter(x=[cur_vol*100], y=[cur_ret*100], mode="markers+text",
                                     marker=dict(size=16, color="#f59e0b", symbol="circle"),
                                     text=[f"You now ({cur_sh:.2f})"], textposition="bottom center", name="Current"))
        fig.update_layout(title=f"Efficient frontier — recommended point moves with your {risk} profile",
                          xaxis_title="Annual volatility %", yaxis_title="Expected annual return %", height=520)
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        if cur_ret is not None:
            m1.metric("Your portfolio now", f"{cur_ret*100:.1f}% return", f"Vol {cur_vol*100:.1f}% · Sharpe {cur_sh:.2f}")
        m2.metric(f"Your target ({risk})", f"{tr*100:.1f}% return", f"Vol {tvol*100:.1f}% · Sharpe {tsh:.2f}")
        m3.metric("Min Variance (floor)", f"{mr*100:.1f}% return", f"Vol {mv*100:.1f}%")

        st.divider()
        st.subheader("💸 Where to invest & where to divest")
        st.caption(f"Rebalancing your current {SYM}{tv:,.0f} from where you are now → your {risk} target. "
                   "Amounts are indicative, not advice.")

        # Build the rebalance: current value vs target value per ticker
        reb = []
        for t in opt["tickers"]:
            cur_val = cur_w.get(t, 0) * tv
            tgt_val = w_target.get(t, 0) * tv
            delta = tgt_val - cur_val
            reb.append({"Ticker": t, "Kind": kinds.get(t, "Stock"),
                        "Current %": round(cur_w.get(t, 0) * 100, 1),
                        "Target %": round(w_target.get(t, 0) * 100, 1),
                        "Δ %": round((w_target.get(t, 0) - cur_w.get(t, 0)) * 100, 1),
                        "Move": round(delta, 2)})
        rebdf = pd.DataFrame(reb)
        thresh = max(0.005 * tv, 1.0)   # ignore moves under 0.5% of book
        invest = rebdf[rebdf["Move"] > thresh].sort_values("Move", ascending=False)
        divest = rebdf[rebdf["Move"] < -thresh].sort_values("Move")

        inv_rows = "".join(
            f'<div class="actrow"><span><b>{r["Ticker"]}</b> '
            f'<span style="color:#93a0bd">{r["Current %"]}%→{r["Target %"]}%</span></span>'
            f'<span class="amt">+{SYM}{r["Move"]:,.0f}</span></div>'
            for _, r in invest.iterrows()) or '<div class="actrow"><span>Nothing to add — you are at/above target.</span></div>'
        div_rows = "".join(
            f'<div class="actrow"><span><b>{r["Ticker"]}</b> '
            f'<span style="color:#93a0bd">{r["Current %"]}%→{r["Target %"]}%</span></span>'
            f'<span class="amt">−{SYM}{abs(r["Move"]):,.0f}</span></div>'
            for _, r in divest.iterrows()) or '<div class="actrow"><span>Nothing to trim — no overweights.</span></div>'

        st.markdown(f"""
        <div class="acts">
          <div class="actcol invest"><h4>🟢 Invest (buy / add)</h4>{inv_rows}</div>
          <div class="actcol divest"><h4>🔴 Divest (trim / sell)</h4>{div_rows}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Full target allocation table"):
            show = rebdf.copy()
            show["Action"] = ["BUY ↑" if d > 2 else "TRIM ↓" if d < -2 else "hold" for d in show["Δ %"]]
            show = show.rename(columns={"Move": f"Move {SYM}"}).sort_values("Target %", ascending=False)
            st.dataframe(
                show.style.map(lambda v: f"color: {'#16a34a' if 'BUY' in str(v) else '#dc2626' if 'TRIM' in str(v) else '#888'}",
                               subset=["Action"]),
                use_container_width=True, hide_index=True,
            )

        # Discrete allocation for new cash
        if new_cash > 0:
            try:
                latest = get_latest_prices(px_hist[opt["tickers"]])
                base_prices = {}
                for t in opt["tickers"]:
                    info = get_info(t)
                    base_prices[t] = to_base(latest.get(t), info.get("currency", base), base) or latest.get(t)
                da = DiscreteAllocation(w_target, pd.Series(base_prices), total_portfolio_value=new_cash)
                alloc, leftover = da.greedy_portfolio()
                if alloc:
                    buy_df = pd.DataFrame([
                        {"Ticker": t, "Shares to buy": s, f"≈ {SYM}": round(s * base_prices.get(t, 0), 2)}
                        for t, s in alloc.items()
                    ]).sort_values(f"≈ {SYM}", ascending=False)
                    st.markdown(f"**Deploying {SYM}{new_cash:,} of new cash** (≈{SYM}{leftover:.2f} left over) "
                                f"into your {risk} target:")
                    st.dataframe(buy_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Discrete allocation unavailable: {e}")

# ── TAB 4: VALUE SCREENER ─────────────────────────────────────────────────────
with tab4:
    st.subheader("🔎 Value Screener")
    c1, c2, c3 = st.columns(3)
    max_pe = c1.slider("Max P/E", 5, 50, 25)
    max_pb = c1.slider("Max P/B", 0.5, 10.0, 3.0, 0.5)
    min_roe = c2.slider("Min ROE %", 0, 40, 10)
    max_de = c2.slider("Max Debt/Equity", 0.0, 300.0, 150.0, 10.0)
    min_margin = c3.slider("Min gross margin %", 0, 60, 15)
    min_upside = c3.slider("Min analyst upside %", -20, 50, 0)

    pick = st.multiselect("Screen these tickers", st.session_state.watchlist,
                          default=st.session_state.watchlist[:14])
    if st.button("Run screener", use_container_width=True):
        rows = []
        prog = st.progress(0)
        for i, t in enumerate(pick):
            info = get_info(t)
            cur = info.get("currentPrice") or info.get("regularMarketPrice")
            pe, pb = info.get("trailingPE"), info.get("priceToBook")
            roe = (info.get("returnOnEquity") or 0) * 100
            de = info.get("debtToEquity")
            margin = (info.get("grossMargins") or 0) * 100
            up = analyst_upside(info)

            fails = []
            # Missing data FAILS (does not silently pass)
            if pe is None: fails.append("no P/E")
            elif pe > max_pe: fails.append(f"P/E {pe:.0f}")
            if pb is None: fails.append("no P/B")
            elif pb > max_pb: fails.append(f"P/B {pb:.1f}")
            if not roe: fails.append("no ROE")
            elif roe < min_roe: fails.append(f"ROE {roe:.0f}%")
            if de is None: fails.append("no D/E")
            elif de > max_de: fails.append(f"D/E {de:.0f}")
            if not margin: fails.append("no margin")
            elif margin < min_margin: fails.append(f"margin {margin:.0f}%")
            if up is None: fails.append("no target")
            elif up < min_upside: fails.append(f"upside {up:.0f}%")

            rows.append({
                "Pass": "✅" if not fails else "❌", "Ticker": t,
                "Company": (info.get("shortName") or t)[:22],
                "Sector": info.get("sector", "—"),
                "P/E": round(pe, 1) if pe else "—", "P/B": round(pb, 2) if pb else "—",
                "ROE %": round(roe, 1) if roe else "—", "D/E": round(de, 0) if de else "—",
                "Margin %": round(margin, 1) if margin else "—",
                "Upside %": round(up, 1) if up is not None else "—",
                "Quality": quality_score(info) or "—",
                "Consensus": rec_label(info.get("recommendationKey", "")),
                "Why failed": "; ".join(fails) if fails else "—",
            })
            prog.progress((i + 1) / max(len(pick), 1))
        prog.empty()
        sdf = pd.DataFrame(rows)
        passed = sdf[sdf["Pass"] == "✅"]
        st.success(f"{len(passed)} of {len(sdf)} passed all value filters")
        st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("""
**Value criteria** · P/E < 25 (Buffett deep-value < 15) · P/B < 3 (asset margin of safety) ·
ROE > 10% (capital efficiency) · D/E < 150% (low leverage) · Gross margin > 15% (moat) ·
*Missing data fails the filter — it never silently passes.*
""")
