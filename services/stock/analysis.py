"""
services/stock/analysis.py — Capital Pulse stock intelligence (memory-safe).

Why no yfinance/pandas/torch/Prophet?
  * yfinance is rate-limited/blocked on cloud server IPs (HTTP 429 from Yahoo),
    so it fails for every ticker on Render.
  * pandas (pulled in by yfinance) adds ~150 MB, which pushed the whole service
    over Render's 512 MB cap and was killing document uploads mid-request.

So this module uses:
  * Stooq  (free CSV endpoint, no API key, works from datacenter IPs) for
           historical daily closes  -> prediction + price summary.
  * Finnhub (lightweight REST, optional key) for company profile, live quote
            and news -> analytical chatbot.
  * numpy + scikit-learn (already core deps) for the lightweight forecaster.

The original LSTM + Prophet Streamlit version still lives in the GDG repo for
local use; this gives the same experience without exceeding the memory budget.
"""

import csv
import io
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np

# ── Company name -> ticker shortcuts ──────────────────────────────────────────
TICKER_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "disney": "DIS", "ibm": "IBM", "oracle": "ORCL", "adobe": "ADBE",
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS", "wipro": "WIPRO.NS", "tata motors": "TATAMOTORS.NS",
    "sbi": "SBIN.NS", "adani": "ADANIENT.NS", "itc": "ITC.NS",
}

# Words that look like tickers but aren't (avoids matching "FOR", "SHORT", etc.)
_STOPWORDS = {
    "THE", "FOR", "AND", "WHY", "DID", "HAS", "WAS", "ARE", "WHAT", "HOW",
    "WHEN", "WHO", "DOES", "DID", "IS", "IN", "ON", "OF", "TO", "A", "AN",
    "STOCK", "STOCKS", "PRICE", "SHARE", "SHARES", "SHORT", "LONG", "BUY",
    "SELL", "UP", "DOWN", "DROP", "RISE", "NEWS", "TODAY", "NOW", "ABOUT",
    "WITH", "THIS", "THAT", "TELL", "ME", "GET", "RECENT", "RECENTLY", "MOVE",
    "MOVED", "MARKET", "HAPPENING", "GOING",
}


def _http_get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36",
        "Accept": "application/json, text/csv, */*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ── Yahoo v8 chart API (primary; no key, global + Indian coverage) ─────────────
# NOTE: this is the lightweight /v8/finance/chart endpoint, NOT the
# quoteSummary/crumb endpoints that yfinance uses and that get HTTP-429'd on
# datacenter IPs. This one is far more permissive from servers.
def _yahoo_symbols(ticker: str):
    t = ticker.strip().upper()
    cands = [t]
    # Convenience: allow "RELIANCE" to also try the NSE listing.
    if "." not in t:
        cands.append(f"{t}.NS")
    out, seen = [], set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fetch_yahoo(ticker: str):
    for sym in _yahoo_symbols(ticker):
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{urllib.parse.quote(sym)}?range=1y&interval=1d")
        try:
            data = json.loads(_http_get(url))
        except Exception as e:
            print(f"[stock] yahoo fetch error for {sym}: {e}")
            continue
        try:
            res = data["chart"]["result"][0]
            ts = res["timestamp"]
            closes = res["indicators"]["quote"][0]["close"]
        except (KeyError, IndexError, TypeError):
            continue
        series = []
        for t, c in zip(ts, closes):
            if c is None:
                continue
            d = datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
            series.append((d, float(c)))
        if len(series) >= 30:
            return series
    return None


# ── Stooq historical prices (fallback CSV source) ─────────────────────────────
def _stooq_candidates(ticker: str):
    """Stooq uses lowercase + exchange suffix: US -> aapl.us, India -> reliance.in."""
    t = ticker.strip().lower()
    cands = []
    if "." in t:
        base, suf = t.rsplit(".", 1)
        if suf in ("ns", "bo"):          # Indian NSE/BSE
            cands += [f"{base}.in", base]
        else:
            cands += [t, base]
    else:
        cands += [f"{t}.us", f"{t}.in", t]
    out, seen = [], set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fetch_stooq(ticker: str):
    for sym in _stooq_candidates(ticker):
        url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(sym)}&i=d"
        try:
            text = _http_get(url)
        except Exception as e:
            print(f"[stock] stooq fetch error for {sym}: {e}")
            continue
        lines = text.strip().splitlines()
        if not lines or not lines[0].lower().startswith("date,"):
            continue  # bot challenge / invalid symbol -> html
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except Exception:
            continue
        series = []
        for r in rows:
            c, d = r.get("Close"), r.get("Date")
            if not c or c in ("N/D", "null") or not d:
                continue
            try:
                series.append((d, float(c)))
            except ValueError:
                continue
        if len(series) >= 30:
            return series
    return None


def fetch_history(ticker: str):
    """Return list of (date_str, close_float) oldest->newest, or None.
    Tries Yahoo chart API first (reliable from servers), then Stooq."""
    return _fetch_yahoo(ticker) or _fetch_stooq(ticker)


# ── Finnhub (news / profile / quote) ──────────────────────────────────────────
def _finnhub_key():
    return os.getenv("FINNHUB_API_KEY") or os.getenv("finnhub_api_key")


def _finnhub_get(path: str, params: dict):
    key = _finnhub_key()
    if not key:
        return None
    params = {**params, "token": key}
    url = f"https://finnhub.io/api/v1/{path}?{urllib.parse.urlencode(params)}"
    try:
        return json.loads(_http_get(url))
    except Exception as e:
        print(f"[stock] finnhub {path} error: {e}")
        return None


def get_company_info(ticker: str) -> str:
    data = _finnhub_get("stock/profile2", {"symbol": ticker})
    if not data or not data.get("name"):
        return ""
    mcap = data.get("marketCapitalization")
    mcap_str = f"${mcap:,.0f}M" if isinstance(mcap, (int, float)) else "N/A"
    return (
        "COMPANY INFO:\n"
        f"Name: {data.get('name', ticker)}\n"
        f"Industry: {data.get('finnhubIndustry', 'N/A')}\n"
        f"Exchange: {data.get('exchange', 'N/A')}\n"
        f"Market Cap: {mcap_str}\n"
    )


def get_history_summary(series, days: int = 30) -> str:
    """Build a price-trend summary from an already-fetched series."""
    if not series:
        return ""
    window = series[-days:]
    start, end = window[0][1], window[-1][1]
    change = ((end - start) / start) * 100 if start else 0
    highs = max(v for _, v in window)
    lows = min(v for _, v in window)
    return (
        f"PRICE TREND (last {len(window)} trading days):\n"
        f"Start: {start:.2f}  ->  Latest: {end:.2f}\n"
        f"Change: {change:+.2f}%\n"
        f"High: {highs:.2f}  Low: {lows:.2f}\n"
    )


def get_news(ticker: str, days: int = 21) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    articles = _finnhub_get("company-news", {"symbol": ticker, "from": start, "to": today})
    if not articles or not isinstance(articles, list):
        return ""
    out = "RECENT NEWS:\n"
    for a in articles[:10]:
        d = datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d")
        out += f"\n[{d}] {a.get('headline', 'No title')}\n"
        if a.get("summary"):
            out += f"{a['summary'][:280]}\n"
    return out


def build_context(ticker: str) -> str:
    # Price history comes from the reliable Yahoo chart API (no key needed).
    series = fetch_history(ticker)
    parts = [
        get_company_info(ticker),       # Finnhub (optional)
        get_history_summary(series),    # always available if ticker is valid
        get_news(ticker),               # Finnhub (optional)
    ]
    return "\n".join(p for p in parts if p)


# ── Ticker resolution ─────────────────────────────────────────────────────────
def resolve_ticker(query: str, llm_client=None) -> str:
    q = (query or "").lower()

    # 1. Known company names (longest first so "tata motors" beats "tata")
    for name in sorted(TICKER_MAP, key=len, reverse=True):
        if name in q:
            return TICKER_MAP[name]

    # 2. Explicit uppercase ticker typed in the ORIGINAL text (e.g. "AAPL"),
    #    skipping common English words.
    for raw in (query or "").replace("?", " ").replace(",", " ").split():
        token = raw.strip(".:;!").upper()
        if (raw.strip(".:;!").isupper()
                and 1 <= len(token) <= 5
                and token.isalpha()
                and token not in _STOPWORDS):
            return token

    # 3. LLM fallback (optional)
    if llm_client is not None:
        try:
            from services.document_chat.config import retrieval_config
            resp = llm_client.models.generate_content(
                model=retrieval_config.gemini_model,
                contents=(
                    "Extract ONLY the stock ticker symbol from this query "
                    "(e.g. Apple -> AAPL, Reliance -> RELIANCE.NS). Reply with "
                    f"just the symbol, or UNKNOWN.\nQuery: {query}\nTicker:"
                ),
            )
            cand = (resp.text or "").strip().upper().split()[0] if resp.text else ""
            cand = cand.strip(".:;!")
            if cand and cand != "UNKNOWN" and cand not in _STOPWORDS:
                return cand
        except Exception as e:
            print(f"[stock] ticker LLM fallback failed: {e}")
    return ""


# ── Analytical chatbot ────────────────────────────────────────────────────────
STOCK_PROMPT = """You are Capital Pulse, an expert financial analyst.

CONTEXT (live market data & news):
{context}

QUESTION: {question}

Write a clear, evidence-based answer that:
1. References specific price movements (with % changes / levels) when relevant.
2. Cites concrete news headlines or facts from the context.
3. Explains the likely link between news/fundamentals and price action.
4. Is concise, well-structured, and easy to read.

If the context lacks the needed data, say so briefly and answer with general
financial reasoning. Do NOT give personalised investment advice; add a short
neutral disclaimer at the end.

Answer:"""


def stock_chat(question: str, llm_client) -> dict:
    ticker = resolve_ticker(question, llm_client)
    if not ticker:
        return {
            "ticker": None,
            "answer": (
                "I couldn't identify a stock from that. Mention a company or "
                "ticker, e.g. \"Why did Apple drop?\" or \"What's up with TSLA?\""
            ),
        }

    context = build_context(ticker)
    if len(context) < 40:
        note = "" if _finnhub_key() else (
            " (Tip: set FINNHUB_API_KEY on the server to enable live news & "
            "company data — price analysis works without it.)"
        )
        return {
            "ticker": ticker,
            "answer": (
                f"I couldn't pull enough live data for {ticker} right now."
                f"{note} Please double-check the ticker (US e.g. AAPL; Indian "
                "e.g. RELIANCE.NS) or try again shortly."
            ),
        }

    prompt = STOCK_PROMPT.format(context=context, question=question)
    try:
        # Reuse the version-robust Gemini caller from the document chat module
        # so we don't 500 on google-genai version differences.
        from services.document_chat.processor import _generate
        resp = _generate(llm_client, prompt)
        answer = (getattr(resp, "text", None) or "").strip() or (
            "I couldn't generate an analysis for that. Please rephrase."
        )
    except Exception as e:
        print(f"[stock] gemini error: {e}")
        answer = f"Analysis engine error: {e}"

    return {"ticker": ticker, "answer": answer}


# ── Lightweight price prediction (numpy + scikit-learn) ───────────────────────
def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    nonzero = actual != 0
    mape = (float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100)
            if nonzero.any() else 0.0)
    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape": round(mape, 2),
        "accuracy": round(max(0.0, 100.0 - mape), 2),
    }


def _trend_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Linear trend (least squares) on a recent window blended with recent
    momentum, anchored to the last observed price. Fast and memory-safe."""
    from sklearn.linear_model import LinearRegression

    n = len(series)
    window = series[-min(n, 120):]
    x = np.arange(len(window)).reshape(-1, 1)
    y = window.reshape(-1, 1)

    reg = LinearRegression().fit(x, y)
    future_x = np.arange(len(window), len(window) + horizon).reshape(-1, 1)
    trend = reg.predict(future_x).flatten()

    last = float(series[-1])
    recent = series[-min(n, 10):]
    drift = float(np.mean(np.diff(recent))) if len(recent) > 1 else 0.0
    drift_path = last + drift * np.arange(1, horizon + 1)

    forecast = 0.6 * trend + 0.4 * drift_path
    forecast = forecast + (last - forecast[0]) * np.linspace(1, 0, horizon)
    return forecast


def predict_prices(ticker: str, horizon: int = 7) -> dict:
    series = fetch_history(ticker)
    if not series:
        return {
            "error": (
                f"No price data found for '{ticker}'. Check the symbol "
                "(US e.g. AAPL; Indian e.g. RELIANCE.NS)."
            )
        }

    dates = [d for d, _ in series]
    values = np.array([v for _, v in series], dtype=float)
    if len(values) < 30:
        return {"error": f"Not enough price history for '{ticker}'."}

    # Backtest on the last `horizon` days for honest metrics.
    train, test = values[:-horizon], values[-horizon:]
    backtest = _trend_forecast(train, horizon)
    metrics = _metrics(test, backtest)

    # Real future forecast.
    future = _trend_forecast(values, horizon)
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, horizon + 1)]

    return {
        "ticker": ticker.upper(),
        "model": "Trend + momentum regression (memory-safe)",
        "history": {
            "dates": dates[-180:],
            "prices": [round(float(v), 2) for v in values[-180:]],
        },
        "forecast": {
            "dates": future_dates,
            "prices": [round(float(p), 2) for p in future],
        },
        "metrics": metrics,
        "current_price": round(float(values[-1]), 2),
        "next_day": round(float(future[0]), 2),
    }
