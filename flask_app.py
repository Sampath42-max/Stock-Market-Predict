from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
import pytz
import json
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests for Live Server compatibility

# Market Open & Close Time (Eastern Time)
MARKET_OPEN = dt.strptime("09:30", "%H:%M").time()
MARKET_CLOSE = dt.strptime("16:00", "%H:%M").time()

# Comprehensive List of Stock Tickers (US Tech, Global Giants, Indian Markets, ETFs & Crypto)
TICKERS = [
    # US Tech & AI Leaders
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC", 
    "CRM", "ORCL", "ADBE", "QCOM", "AVGO", "IBM", "PLTR", "ARM", "SMCI", "SNOW", "UBER", "ABNB", "SHOP", "COIN",
    # Global Giants & Finance
    "V", "MA", "JPM", "BAC", "GS", "MS", "BLK", "WMT", "COST", "DIS", "PYPL", "KO", "PEP", "NKE", "SBUX", "MCD",
    # Healthcare & Energy
    "JNJ", "PFE", "LLY", "UNH", "ABBV", "CVX", "XOM", "CAT", "BA", "GE",
    # Indian Stock Market Leaders (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS", "MOTILALOFS.NS",
    # Market ETFs & Crypto
    "SPY", "QQQ", "BTC-USD", "ETH-USD"
]

# Simple in-memory cache for market summary and stock data
CACHE = {
    'market_summary': None,
    'market_summary_time': 0,
    'stock_data': {}
}
CACHE_TTL = 300  # 5 minutes TTL

def is_market_open():
    """Checks if US stock market is currently open."""
    try:
        est = pytz.timezone("US/Eastern")
        now_est = dt.now(est)
        if now_est.weekday() >= 5:  # Saturday or Sunday
            return False
        now_time = now_est.time()
        return MARKET_OPEN <= now_time <= MARKET_CLOSE
    except Exception:
        return False

def calculate_rsi(series, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def fetch_single_ticker_info(ticker):
    """Fetches summary data for a single ticker safely."""
    try:
        t = yf.Ticker(ticker.strip())
        df = t.history(period='5d', interval='1d')
        if df.empty or len(df) < 1:
            return None

        current_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_close
        change = current_close - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0.0

        volume = int(df["Volume"].iloc[-1]) if "Volume" in df.columns and not pd.isna(df["Volume"].iloc[-1]) else 0
        open_price = float(df["Open"].iloc[-1]) if "Open" in df.columns else current_close
        high_price = float(df["High"].iloc[-1]) if "High" in df.columns else current_close
        low_price = float(df["Low"].iloc[-1]) if "Low" in df.columns else current_close

        sparkline = [round(val, 2) for val in df["Close"].tolist()]

        return {
            "Ticker": ticker.strip(),
            "Open": round(open_price, 2),
            "High": round(high_price, 2),
            "Low": round(low_price, 2),
            "Close": round(current_close, 2),
            "Change": round(change, 2),
            "ChangePct": round(change_pct, 2),
            "Value": round(current_close * volume, 2) if volume else 0,
            "Volume": volume,
            "Sparkline": sparkline
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def fetch_all_market_summary():
    """Fetches overview metrics for all tickers in parallel with caching."""
    now = time.time()
    if CACHE['market_summary'] and (now - CACHE['market_summary_time'] < CACHE_TTL):
        return CACHE['market_summary']

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_single_ticker_info, ticker) for ticker in TICKERS]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)

    CACHE['market_summary'] = results
    CACHE['market_summary_time'] = now
    return results

def predict_stock_ml(ticker, future_days=5):
    """
    Advanced & Fast Ensemble Machine Learning Model for Stock Forecasting.
    Uses Ridge Regression + Trend Momentum + Volatility Bounds.
    """
    stock = yf.Ticker(ticker.strip())
    df = stock.history(period='1y', interval='1d')

    if df.empty or len(df) < 30:
        return None, None, None

    df = df[['Close', 'High', 'Low', 'Volume']].copy()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    # Feature Engineering
    df['Time_Index'] = np.arange(len(df))
    X = df[['Time_Index', 'SMA_10', 'SMA_30', 'RSI', 'Daily_Return']].values
    y = df['Close'].values

    # Train / Test split to calculate accuracy metrics
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    # Refit on full dataset for future projection
    model.fit(X, y)

    last_index = df['Time_Index'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    last_sma10 = df['SMA_10'].iloc[-1]
    last_sma30 = df['SMA_30'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    recent_volatility = float(df['Daily_Return'].std() * last_close)

    future_dates = []
    future_predictions = []
    upper_bounds = []
    lower_bounds = []

    current_date = df.index[-1]
    curr_sma10 = last_sma10
    curr_sma30 = last_sma30
    curr_rsi = last_rsi

    for i in range(1, future_days + 1):
        next_date = current_date + timedelta(days=i)
        if next_date.weekday() >= 5:  # Skip weekends for realistic date mapping
            next_date += timedelta(days=2 if next_date.weekday() == 5 else 1)
        current_date = next_date

        next_time_index = last_index + i
        feat = np.array([[next_time_index, curr_sma10, curr_sma30, curr_rsi, 0.001]])
        pred_val = float(model.predict(feat)[0])

        # Apply exponential momentum smoothing
        smoothed_pred = (pred_val * 0.4) + (last_close * (1 + (curr_rsi - 50) * 0.0005) * 0.6)
        margin = recent_volatility * np.sqrt(i) * 0.65

        future_dates.append(current_date.strftime("%Y-%m-%d"))
        future_predictions.append(round(smoothed_pred, 2))
        upper_bounds.append(round(smoothed_pred + margin, 2))
        lower_bounds.append(round(max(0.1, smoothed_pred - margin), 2))

        last_close = smoothed_pred
        curr_sma10 = (curr_sma10 * 9 + smoothed_pred) / 10
        curr_sma30 = (curr_sma30 * 29 + smoothed_pred) / 30

    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "last_price": round(float(df['Close'].iloc[-1]), 2),
        "target_price": future_predictions[-1],
        "expected_change_pct": round(((future_predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100, 2),
        "signal": "BULLISH" if future_predictions[-1] > df['Close'].iloc[-1] else "BEARISH"
    }

    historical = {
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "close": [round(c, 2) for c in df['Close'].tolist()]
    }

    prediction = {
        "dates": future_dates,
        "predicted": future_predictions,
        "upper_bound": upper_bounds,
        "lower_bound": lower_bounds
    }

    return historical, prediction, metrics

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def market():
    stock_market = fetch_all_market_summary()
    market_open = is_market_open()
    return render_template('market.html', stock_market=stock_market, is_market_open=market_open)

@app.route('/share-info')
def plot():
    return render_template('plot.html')

@app.route('/api/market-summary')
def api_market_summary():
    data = fetch_all_market_summary()
    return jsonify({
        "status": "success",
        "market_open": is_market_open(),
        "stocks": data
    })

@app.route('/api/get_stock_data')
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL').upper().strip()
    period = request.args.get('period', '6mo')
    interval = request.args.get('interval', '1d')

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)

        if data.empty:
            return jsonify({"error": f"No data found for ticker '{ticker}'!"}), 404

        dates = data.index.strftime("%Y-%m-%d" if interval in ['1d', '1wk'] else "%Y-%m-%d %H:%M").tolist()
        close_prices = pd.Series(data["Close"])

        # Technical Calculations
        sma_20 = close_prices.rolling(window=20).mean().fillna(close_prices).tolist()
        sma_50 = close_prices.rolling(window=50).mean().fillna(close_prices).tolist()
        ema_20 = close_prices.ewm(span=20, adjust=False).mean().tolist()
        rsi_14 = calculate_rsi(close_prices, window=14).tolist()

        result = {
            "ticker": ticker,
            "dates": dates,
            "open": [round(x, 2) for x in data["Open"].tolist()],
            "high": [round(x, 2) for x in data["High"].tolist()],
            "low": [round(x, 2) for x in data["Low"].tolist()],
            "close": [round(x, 2) for x in data["Close"].tolist()],
            "volume": [int(x) for x in data["Volume"].tolist()],
            "sma_20": [round(x, 2) for x in sma_20],
            "sma_50": [round(x, 2) for x in sma_50],
            "ema_20": [round(x, 2) for x in ema_20],
            "rsi_14": [round(x, 2) for x in rsi_14]
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_stock')
def predict_stock_route():
    ticker = request.args.get('ticker', 'AAPL').upper().strip()
    try:
        future_days = int(request.args.get('future_days', 5))
    except ValueError:
        future_days = 5

    historical, prediction, metrics = predict_stock_ml(ticker, future_days=future_days)

    if historical is None:
        return jsonify({"error": f"Unable to generate prediction for ticker '{ticker}'!"}), 404

    return jsonify({
        "ticker": ticker,
        "historical": historical,
        "prediction": prediction,
        "metrics": metrics
    })

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').upper().strip()
    if not query:
        return jsonify([])

    matches = [t for t in TICKERS if query in t]
    if not matches and len(query) >= 2:
        # Check if query is a valid yfinance ticker
        try:
            t = yf.Ticker(query)
            hist = t.history(period='1d')
            if not hist.empty:
                matches.append(query)
        except Exception:
            pass

    return jsonify(matches)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
