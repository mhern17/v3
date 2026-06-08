import math
import string
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Strong Buy Stock Scanner",
    page_icon="🟢",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
:root {
    --green:#00c853;
    --red:#ff3b30;
    --yellow:#ffd60a;
    --bg:#0b0f14;
    --panel:#111820;
    --soft:#1b2530;
}

.stApp {
    background: #0b0f14;
    color: #e8edf2;
}

.block-container {
    padding-top: 4rem;
}

[data-testid="stSidebar"] {
    background: #080b0f;
}

.big-title {
    font-size: 40px;
    font-weight: 900;
    line-height: 1.05;
}

.small {
    color:#9aa7b2;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@st.cache_data(ttl=300)
def load_stock(ticker, period="1y", interval="1d"):

    t = yf.Ticker(ticker)

    hist = t.history(
        period=period,
        interval=interval,
        auto_adjust=True
    )

    info = {}

    try:
        info = t.info or {}
    except Exception:
        info = {}

    return hist, info


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0).rolling(period).mean()

    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss

    return 100 - (100 / (1 + rs))


def add_indicators(df):

    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]

    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["RSI"] = rsi(df["Close"])

    df["VOL20"] = df["Volume"].rolling(20).mean()

    df["RET20"] = df["Close"].pct_change(20)

    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    return df


def signal_engine(df, info):

    clean = df.dropna()

    if clean.empty or len(clean) < 2:
        return None

    latest = clean.iloc[-1]
    prev = clean.iloc[-2]

    score = 0
    reasons = []

    # Trend
    if latest.Close > latest.SMA20 > latest.SMA50:
        score += 18
        reasons.append("Bullish short-term trend")

    elif latest.Close < latest.SMA20 < latest.SMA50:
        score -= 18
        reasons.append("Bearish short-term trend")

    if latest.Close > latest.SMA200:
        score += 12
        reasons.append("Above 200 SMA")

    else:
        score -= 12
        reasons.append("Below 200 SMA")

    # Momentum
    if latest.MACD > latest.MACD_SIGNAL and prev.MACD <= prev.MACD_SIGNAL:
        score += 16
        reasons.append("Fresh MACD bullish crossover")

    elif latest.MACD < latest.MACD_SIGNAL and prev.MACD >= prev.MACD_SIGNAL:
        score -= 16
        reasons.append("Fresh MACD bearish crossover")

    elif latest.MACD > latest.MACD_SIGNAL:
        score += 8
        reasons.append("MACD positive")

    else:
        score -= 8
        reasons.append("MACD negative")

    # RSI
    if 50 <= latest.RSI <= 68:
        score += 12
        reasons.append("Healthy RSI")

    elif latest.RSI > 75:
        score -= 10
        reasons.append("Overbought RSI")

    elif latest.RSI < 30:
        score += 6
        reasons.append("Oversold bounce")

    elif latest.RSI < 45:
        score -= 6
        reasons.append("Weak RSI")

    # Volume
    if latest.Volume > latest.VOL20 * 1.25 and latest.Close > prev.Close:
        score += 10
        reasons.append("Volume confirmation")

    elif latest.Volume > latest.VOL20 * 1.25 and latest.Close < prev.Close:
        score -= 10
        reasons.append("Heavy sell volume")

    # Fundamentals
    pe = safe_float(info.get("trailingPE"))
    fpe = safe_float(info.get("forwardPE"))
    rev_growth = safe_float(info.get("revenueGrowth"))
    profit_margin = safe_float(info.get("profitMargins"))
    debt_equity = safe_float(info.get("debtToEquity"))

    if not math.isnan(rev_growth):
        if rev_growth > 0.10:
            score += 8
        elif rev_growth < 0:
            score -= 8

    if not math.isnan(profit_margin):
        if profit_margin > 0.12:
            score += 6
        elif profit_margin < 0:
            score -= 8

    if not math.isnan(debt_equity):
        if debt_equity < 80:
            score += 4
        elif debt_equity > 180:
            score -= 6

    if not math.isnan(pe) and not math.isnan(fpe):
        if fpe < pe:
            score += 5
        elif fpe > pe * 1.25:
            score -= 4

    score = max(-100, min(100, score))

    confidence = min(
        99,
        max(1, int(abs(score) * 0.9 + 10))
    )

    return {
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "latest": latest,
        "prev": prev
    }


# ---------- MASSIVE TICKER UNIVERSE ----------
def get_large_ticker_universe():

    base = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD",
        "NFLX","AVGO","PLTR","SMCI","JPM","BAC","WFC","GS","MS",
        "C","SCHW","BLK","AXP","V","MA","PYPL","SOFI","HOOD",
        "LLY","UNH","JNJ","MRK","ABBV","PFE","XOM","CVX",
        "CAT","DE","BA","WMT","COST","HD","CRM","ORCL",
        "ADBE","COIN","MARA","RIOT","RIVN","LCID","ABAT"
    ]

    generated = []
    letters = string.ascii_uppercase

    for a in letters:
        for b in letters:
            generated.append(a + b)

    for a in letters:
        for b in letters:
            for c in letters:
                generated.append(a + b + c)

    universe = sorted(list(set(base + generated)))

    # Spread the scan across the alphabet instead of only A tickers
    np.random.seed(42)
    np.random.shuffle(universe)

    return universe[:5000]


# ---------- SIDEBAR ----------
st.sidebar.title("🟢 Strong Buy Scanner")

scanner_symbols = st.sidebar.text_area(
    "Ticker universe",
    value="\n".join(get_large_ticker_universe()),
    height=420
)

min_score = st.sidebar.slider(
    "Minimum Signal Score",
    0,
    100,
    60
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0,
    100,
    70
)

st.sidebar.caption(
    "Educational model only. Not financial advice."
)

# ---------- MAIN ----------
st.markdown(
    "<div class='big-title'>🟢 Strong Buy Stock Scanner</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<span class='small'>Finds stocks with strong bullish technical + fundamental alignment.</span>",
    unsafe_allow_html=True
)

st.divider()

scanner_tickers = [
    x.strip().upper()
    for x in scanner_symbols.splitlines()
    if x.strip()
]

c1, c2, c3 = st.columns(3)

c1.metric(
    "Tickers Loaded",
    len(scanner_tickers)
)

c2.metric(
    "Minimum Score",
    min_score
)

c3.metric(
    "Minimum Confidence",
    f"{min_confidence}%"
)

run_scan = st.button(
    "Run Strong Buy Scanner",
    type="primary",
    use_container_width=True
)

if run_scan:

    rows = []

    progress = st.progress(0)

    total = len(scanner_tickers)

    for idx, symbol in enumerate(scanner_tickers):

        try:

            hist, info = load_stock(symbol)

            if hist.empty or len(hist) < 220:
                continue

            df = add_indicators(hist)

            result = signal_engine(df, info)

            if result is None:
                continue

            latest = result["latest"]
            prev = result["prev"]

            score = result["score"]
            confidence = result["confidence"]

            if score >= min_score and confidence >= min_confidence:

                change_pct = (
                    (latest.Close - prev.Close)
                    / prev.Close
                ) * 100

                rows.append({

                    "Ticker": symbol,

                    "Company":
                        info.get("shortName")
                        or info.get("longName")
                        or symbol,

                    "Price":
                        round(latest.Close, 2),

                    "Daily %":
                        round(change_pct, 2),

                    "Signal Score":
                        round(score, 0),

                    "Confidence":
                        f"{confidence}%",

                    "RSI":
                        round(latest.RSI, 1),

                    "Volume vs Avg":
                        f"{latest.Volume / latest.VOL20:.2f}x",

                    "Top Reason":
                        result["reasons"][0]
                        if len(result["reasons"]) > 0
                        else ""
                })

        except Exception:
            pass

        progress.progress((idx + 1) / total)

    st.divider()

    if len(rows) == 0:

        st.warning(
            "No strong-buy stocks found."
        )

    else:

        results_df = pd.DataFrame(rows)

        results_df = results_df.sort_values(
            "Signal Score",
            ascending=False
        )

        st.success(
            f"Found {len(results_df)} strong-buy stock(s)."
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name="strong_buy_scanner_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:

    st.info(
        "Click Run Strong Buy Scanner to begin scanning."
    )
