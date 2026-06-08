import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Strong Buy Stock Scanner", page_icon="🟢", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b0f14; color: #e8edf2; }
.block-container { padding-top: 4rem; }
[data-testid="stSidebar"] { background: #080b0f; }
.big-title { font-size: 38px; font-weight: 900; line-height: 1.05; }
.small { color:#9aa7b2; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


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
    hist = t.history(period=period, interval=interval, auto_adjust=True)
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

    if latest.Close > latest.SMA20 > latest.SMA50:
        score += 18
        reasons.append("Bullish short-term trend: price > 20SMA > 50SMA")
    elif latest.Close < latest.SMA20 < latest.SMA50:
        score -= 18
        reasons.append("Bearish short-term trend: price < 20SMA < 50SMA")

    if latest.Close > latest.SMA200:
        score += 12
        reasons.append("Price above 200SMA: long-term trend supportive")
    else:
        score -= 12
        reasons.append("Price below 200SMA: long-term trend weak")

    if latest.MACD > latest.MACD_SIGNAL and prev.MACD <= prev.MACD_SIGNAL:
        score += 16
        reasons.append("Fresh MACD bullish crossover")
    elif latest.MACD < latest.MACD_SIGNAL and prev.MACD >= prev.MACD_SIGNAL:
        score -= 16
        reasons.append("Fresh MACD bearish crossover")
    elif latest.MACD > latest.MACD_SIGNAL:
        score += 8
        reasons.append("MACD momentum positive")
    else:
        score -= 8
        reasons.append("MACD momentum negative")

    if 50 <= latest.RSI <= 68:
        score += 12
        reasons.append("RSI in healthy bullish range")
    elif latest.RSI > 75:
        score -= 10
        reasons.append("RSI overbought")
    elif latest.RSI < 30:
        score += 6
        reasons.append("RSI oversold bounce potential")
    elif latest.RSI < 45:
        score -= 6
        reasons.append("RSI below bullish range")

    if latest.Volume > latest.VOL20 * 1.25 and latest.Close > prev.Close:
        score += 10
        reasons.append("Up move confirmed by above-average volume")
    elif latest.Volume > latest.VOL20 * 1.25 and latest.Close < prev.Close:
        score -= 10
        reasons.append("Sell pressure confirmed by above-average volume")

    pe = safe_float(info.get("trailingPE"))
    fpe = safe_float(info.get("forwardPE"))
    rev_growth = safe_float(info.get("revenueGrowth"))
    profit_margin = safe_float(info.get("profitMargins"))
    debt_equity = safe_float(info.get("debtToEquity"))

    if not math.isnan(rev_growth):
        if rev_growth > 0.10:
            score += 8
            reasons.append("Revenue growth over 10%")
        elif rev_growth < 0:
            score -= 8
            reasons.append("Negative revenue growth")

    if not math.isnan(profit_margin):
        if profit_margin > 0.12:
            score += 6
            reasons.append("Solid profit margin")
        elif profit_margin < 0:
            score -= 8
            reasons.append("Negative profit margin")

    if not math.isnan(debt_equity):
        if debt_equity < 80:
            score += 4
            reasons.append("Debt/equity appears manageable")
        elif debt_equity > 180:
            score -= 6
            reasons.append("Debt/equity elevated")

    if not math.isnan(pe) and not math.isnan(fpe):
        if fpe < pe:
            score += 5
            reasons.append("Forward P/E below trailing P/E")
        elif fpe > pe * 1.25:
            score -= 4
            reasons.append("Forward P/E materially higher than trailing P/E")

    score = max(-100, min(100, score))
    confidence = min(99, max(1, int(abs(score) * 0.9 + 10)))

    if score >= 60:
        signal_label = "🔥 Strong Buy"
    elif score >= 35:
        signal_label = "🟢 Buy"
    elif score >= 10:
        signal_label = "🟡 Weak Bullish"
    elif score > -10:
        signal_label = "⚪ Neutral"
    elif score > -35:
        signal_label = "🟠 Weak Bearish"
    elif score > -60:
        signal_label = "🔴 Sell"
    else:
        signal_label = "🚨 Strong Sell"

    if confidence >= 85:
        confidence_label = "🎯 Extremely High"
    elif confidence >= 70:
        confidence_label = "🟢 High"
    elif confidence >= 55:
        confidence_label = "🟡 Moderate"
    elif confidence >= 40:
        confidence_label = "🟠 Low"
    else:
        confidence_label = "🔴 Weak"

    atr = latest.ATR if not np.isnan(latest.ATR) else latest.Close * 0.03

    return {
        "score": score,
        "confidence": confidence,
        "signal_label": signal_label,
        "confidence_label": confidence_label,
        "reasons": reasons,
        "latest": latest,
        "prev": prev,
        "entry": latest.Close,
        "stop": latest.Close - atr * 1.5,
        "target1": latest.Close + atr * 2,
        "target2": latest.Close + atr * 3.5,
    }


@st.cache_data(ttl=900)
def run_strong_buy_scanner(tickers):
    rows = []

    for symbol in tickers:
        try:
            hist, info = load_stock(symbol, "1y", "1d")
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

            if score > 60 and confidence > 70:
                daily_change = ((latest.Close - prev.Close) / prev.Close) * 100
                volume_vs_avg = latest.Volume / latest.VOL20 if latest.VOL20 else np.nan

                rows.append({
                    "Ticker": symbol,
                    "Company": info.get("shortName") or info.get("longName") or symbol,
                    "Price": round(latest.Close, 2),
                    "Daily %": round(daily_change, 2),
                    "Signal": result["signal_label"],
                    "Signal Score": round(score, 0),
                    "Confidence": confidence,
                    "Confidence Rating": result["confidence_label"],
                    "RSI": round(latest.RSI, 1),
                    "Volume vs Avg": f"{volume_vs_avg:.2f}x",
                    "Entry": round(result["entry"], 2),
                    "Stop Loss": round(result["stop"], 2),
                    "Target 1": round(result["target1"], 2),
                    "Target 2": round(result["target2"], 2),
                    "Reason 1": result["reasons"][0] if len(result["reasons"]) > 0 else "",
                    "Reason 2": result["reasons"][1] if len(result["reasons"]) > 1 else "",
                    "Reason 3": result["reasons"][2] if len(result["reasons"]) > 2 else "",
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


st.sidebar.title("🟢 Strong Buy Scanner")

@st.cache_data
def get_large_ticker_universe():

    return [
        # Mega Caps
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX","AVGO","PLTR","SMCI",

        # Financial
        "JPM","BAC","WFC","GS","MS","C","SCHW","BLK","AXP","V","MA","PYPL","SOFI","HOOD",

        # Healthcare
        "LLY","UNH","JNJ","MRK","ABBV","PFE","TMO","ISRG","VRTX","REGN","NVO",

        # Energy
        "XOM","CVX","SLB","COP","OXY","MPC","PSX","VLO",

        # Industrials
        "CAT","DE","GE","RTX","BA","LMT","ETN","PH","HON",

        # Retail
        "WMT","COST","HD","LOW","TGT","TJX","ROST",

        # Software
        "CRM","ORCL","ADBE","SNOW","MDB","DDOG","NET","CRWD","ZS","PANW","SHOP",

        # EV / Speculative
        "RIVN","LCID","NIO","XPEV","LI","QS","CHPT","BLNK",

        # Crypto / High Beta
        "COIN","MARA","RIOT","CLSK","IREN","CIFR",

        # Lithium / Mining
        "ABAT","ALB","LAC","SQM","PLL","MP","FCX",

        # ETFs
        "SPY","QQQ","IWM","DIA","SMH","XLF","XLE","ARKK"
    ]
    import string

# Add thousands of additional tickers automatically
for first in string.ascii_uppercase:
    for second in string.ascii_uppercase:
        for third in string.ascii_uppercase:
            ticker = first + second + third
            get_large_ticker_universe().append(ticker)

scanner_symbols = st.sidebar.text_area("Ticker universe", value="\n".join(get_large_ticker_universe()),, height=420)
st.sidebar.caption("Educational model only. Not financial advice.")

st.markdown("<div class='big-title'>🟢 Strong Buy Stock Scanner</div>", unsafe_allow_html=True)
st.markdown(
    "<span class='small'>Scans for stocks with Signal Score over 60 and Confidence over 70 using trend, momentum, volume, and fundamental checks.</span>",
    unsafe_allow_html=True
)

st.divider()

scanner_tickers = [x.strip().upper() for x in scanner_symbols.replace(",", "\n").splitlines() if x.strip()]

c1, c2, c3 = st.columns(3)
c1.metric("Tickers Loaded", len(scanner_tickers))
c2.metric("Signal Filter", ">60 Strong Buy")
c3.metric("Confidence Filter", ">70 High")

run_scan = st.button("Run Strong Buy Scanner", type="primary", use_container_width=True)

if run_scan:
    with st.spinner("Scanning tickers..."):
        results = run_strong_buy_scanner(scanner_tickers)

    st.divider()

    if results.empty:
        st.warning("No stocks found with Signal Score over 60 and Confidence over 70.")
    else:
        results = results.sort_values(["Signal Score", "Confidence"], ascending=[False, False])
        display_results = results.copy()
        display_results["Confidence"] = display_results["Confidence"].astype(str) + "%"

        st.success(f"Found {len(results)} strong-buy, high-confidence stock(s).")

        st.dataframe(display_results, use_container_width=True, hide_index=True)

        csv = display_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Scanner Results CSV",
            data=csv,
            file_name="strong_buy_scanner_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("Paste or edit your ticker list in the sidebar, then click Run Strong Buy Scanner.")
