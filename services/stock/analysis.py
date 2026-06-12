"""
services/stock/analysis.py — Capital Pulse stock intelligence (memory-safe port).

Adapted from the GDG "Capital Pulse" project (Vijay Chamyal) to run within
Render's 512 MB free-tier budget:

  * Analytical chatbot  -> yfinance + Finnhub + the EXISTING google-genai client
                           (no LangChain / FAISS / HuggingFace embeddings).
  * Price prediction    -> lightweight trend forecast with numpy + scikit-learn
                           (no PyTorch / Prophet, which alone exceed the RAM cap).

The full LSTM + Prophet Streamlit version still lives in the GDG repo and can be
run locally; this module gives the same *experience* (forecast chart + metrics +
explanatory chatbot) inside CortexOS without blowing the memory limit.
"""

import os
from datetime import datetime, timedelta

import numpy as np

# ── Ticker shortcuts (same idea as the original) ──────────────────────────────
TICKER_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "disney": "DIS", "ibm": "IBM", "oracle": "ORCL", "adobe": "ADBE",
}


# ── Lazy imports so the doc pipeline never pays for these unless used ──────────
def _yf():
    import yfinance as yf
    return yf


def _finnhub_client():
    api_key = os.getenv("FINNHUB_API_KEY") or os.getenv("finnhub_api_key")
    if not api_key:
        return None
    try:
        import finnhub
        return finnhub.Client(api_key=api_key)
    except Exception as e:
        print(f"[stock] finnhub init failed: {e}")
        return None


# ── Ticker resolution ─────────────────────────────────────────────────────────
def resolve_ticker(query: str, llm_client=None) -> str:
    q = (query or "").lower()
    for name, tk in TICKER_MAP.items():
        if name in q:
            return tk

    # If it already looks like a ticker (1-5 uppercase letters), accept it.
    for token in (query or "").replace("?", " ").split():
        t = token.strip().upper()
        if 1 <= len(t) <= 5 and t.isalpha():
            return t

    # LLM fallback (optional)
    if llm_client is not None:
        try:
            from services.document_chat.config import retrieval_config
            resp = llm_client.models.generate_content(
                model=retrieval_config.gemini_model,
                contents=(
                    "Extract ONLY the stock ticker symbol from this query "
                    "(e.g. Apple -> AAPL). Reply with just the symbol, or "
                    f"UNKNOWN if none.\nQuery: {query}\nTicker:"
                ),
            )
            cand = (resp.text or "").strip().upper().split()[0] if resp.text else ""
            if cand and cand != "UNKNOWN" and cand.isalpha():
                return cand
        except Exception as e:
            print(f"[stock] ticker LLM fallback failed: {e}")
    return ""


# ── Data fetching helpers ─────────────────────────────────────────────────────
def _history(ticker: str, period: str = "1y"):
    yf = _yf()
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print(f"[stock] history error for {ticker}: {e}")
        return None


def get_company_info(ticker: str) -> str:
    try:
        info = _yf().Ticker(ticker).info or {}
        return (
            "COMPANY INFO:\n"
            f"Name: {info.get('longName', ticker)}\n"
            f"Sector: {info.get('sector', 'N/A')}\n"
            f"Industry: {info.get('industry', 'N/A')}\n"
            f"Market Cap: ${info.get('marketCap', 0):,.0f}\n"
        )
    except Exception:
        return ""


def get_price_summary(ticker: str, days: int = 30) -> str:
    df = _history(ticker, period="3mo")
    if df is None:
        return ""
    closes = df["Close"].dropna()
    if closes.empty:
        return ""
    window = closes.tail(days)
    change = ((window.iloc[-1] - window.iloc[0]) / window.iloc[0]) * 100
    return (
        f"PRICE DATA (last {len(window)} trading days):\n"
        f"Current: ${window.iloc[-1]:.2f}\n"
        f"Start: ${window.iloc[0]:.2f}\n"
        f"Change: {change:+.2f}%\n"
        f"High: ${window.max():.2f}\n"
        f"Low: ${window.min():.2f}\n"
    )


def get_finnhub_news(ticker: str, days: int = 21) -> str:
    client = _finnhub_client()
    if client is None:
        return ""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        articles = client.company_news(ticker, _from=start, to=today) or []
        if not articles:
            return ""
        out = "RECENT NEWS:\n"
        for a in articles[:10]:
            d = datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d")
            out += f"\n[{d}] {a.get('headline', 'No title')}\n"
            if a.get("summary"):
                out += f"{a['summary'][:300]}\n"
        return out
    except Exception as e:
        print(f"[stock] finnhub news error: {e}")
        return ""


def get_yfinance_news(ticker: str) -> str:
    try:
        news = _yf().Ticker(ticker).news or []
        if not news:
            return ""
        out = "\nADDITIONAL HEADLINES:\n"
        for item in news[:8]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title") or "No title"
            out += f"\n- {title}\n"
        return out
    except Exception as e:
        print(f"[stock] yfinance news error: {e}")
        return ""


def build_context(ticker: str) -> str:
    parts = [
        get_company_info(ticker),
        get_price_summary(ticker),
        get_finnhub_news(ticker),
        get_yfinance_news(ticker),
    ]
    return "\n".join(p for p in parts if p)


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
    if len(context) < 80:
        return {
            "ticker": ticker,
            "answer": (
                f"I couldn't pull enough live data for {ticker} right now "
                "(it may be an invalid ticker or a temporary data/API limit). "
                "Please try again shortly."
            ),
        }

    from services.document_chat.config import retrieval_config
    from google.genai import types as genai_types

    prompt = STOCK_PROMPT.format(context=context, question=question)
    try:
        resp = llm_client.models.generate_content(
            model=retrieval_config.gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
        answer = (resp.text or "").strip() or (
            "I couldn't generate an analysis for that. Please rephrase."
        )
    except Exception as e:
        print(f"[stock] gemini error: {e}")
        answer = f"Analysis engine error: {e}"

    return {"ticker": ticker, "answer": answer}


# ── Lightweight price prediction (no torch / Prophet) ─────────────────────────
def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    # avoid division by zero
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) \
        if nonzero.any() else 0.0
    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape": round(mape, 2),
        "accuracy": round(max(0.0, 100.0 - mape), 2),
    }


def _trend_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """
    Lightweight forecaster: linear trend (least squares) on a recent window
    blended with the last observed value, plus the average recent daily change.
    Memory-safe and fast — a sensible stand-in for the heavy LSTM/Prophet models.
    """
    from sklearn.linear_model import LinearRegression

    n = len(series)
    window = series[-min(n, 120):]
    x = np.arange(len(window)).reshape(-1, 1)
    y = window.reshape(-1, 1)

    reg = LinearRegression().fit(x, y)
    future_x = np.arange(len(window), len(window) + horizon).reshape(-1, 1)
    trend = reg.predict(future_x).flatten()

    last = float(series[-1])
    # recent average daily change (momentum)
    recent = series[-min(n, 10):]
    drift = float(np.mean(np.diff(recent))) if len(recent) > 1 else 0.0

    drift_path = last + drift * np.arange(1, horizon + 1)

    # Blend trend and momentum, anchored to the last actual price.
    forecast = 0.6 * trend + 0.4 * drift_path
    # ensure continuity: shift so day-1 connects smoothly to last value
    forecast = forecast + (last - forecast[0]) * np.linspace(1, 0, horizon)
    return forecast


def predict_prices(ticker: str, horizon: int = 7) -> dict:
    df = _history(ticker, period="1y")
    if df is None:
        return {"error": f"No data found for '{ticker}'. Check the ticker symbol."}

    closes = df["Close"].dropna()
    if len(closes) < 30:
        return {"error": f"Not enough price history for '{ticker}'."}

    dates = [d.strftime("%Y-%m-%d") for d in closes.index]
    values = closes.values.astype(float)

    # Backtest on the last `horizon` days for honest metrics.
    train, test = values[:-horizon], values[-horizon:]
    backtest = _trend_forecast(train, horizon)
    metrics = _metrics(test, backtest)

    # Real future forecast.
    future = _trend_forecast(values, horizon)
    last_date = closes.index[-1]
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, horizon + 1)]

    # Trim history sent to the client to keep payload light.
    hist_dates = dates[-180:]
    hist_values = [round(v, 2) for v in values[-180:]]

    return {
        "ticker": ticker.upper(),
        "model": "Lightweight trend + momentum (memory-safe)",
        "history": {"dates": hist_dates, "prices": hist_values},
        "forecast": {
            "dates": future_dates,
            "prices": [round(float(p), 2) for p in future],
        },
        "metrics": metrics,
        "current_price": round(float(values[-1]), 2),
        "next_day": round(float(future[0]), 2),
    }
