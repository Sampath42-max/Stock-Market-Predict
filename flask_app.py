from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
import pytz
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Market Open & Close Time (Eastern Time)
MARKET_OPEN = dt.strptime("09:30", "%H:%M").time()
MARKET_CLOSE = dt.strptime("16:00", "%H:%M").time()

# List of Stock Tickers
Tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "V", "JNJ",
    "WMT", "JPM", "UNH", "PG", "MA", "DIS", "HD", "PYPL", "INTC", "ASML",
    "ADBE", "CMCSA", "NFLX", "KO", "PEP", "T", "MRK", "PFE", "ABBV", "COST",
    "AVGO", "TXN", "TMO", "NKE", "ORCL", "ACN", "AMD", "MDT", "CRM", "UPS",
    "IBM", "LLY", "QCOM", "SBUX", "CVX", "GS", "CAT", "BA", "MCD", "AXP",
    "DHR", "RTX", "BAC", "BLK", "GE"," MOTILALOFS.NS"
]

# Check if market is open
def is_market_open():
    est = pytz.timezone("US/Eastern")
    now = dt.now(est).time()
    return MARKET_OPEN <= now <= MARKET_CLOSE

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def market():
    stock_market = []
    
    for ticker in Tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')

        if not data.empty:
            close_price = round(data["Close"].iloc[-1], 2) if "Close" in data.columns else None
            volume = int(data["Volume"].iloc[-1]) if "Volume" in data.columns and not pd.isna(data["Volume"].iloc[-1]) else None
            
            stock_market.append({
                "Ticker": ticker,
                "Open": round(data["Open"].iloc[-1], 2) if "Open" in data.columns else None,
                "High": round(data["High"].iloc[-1], 2) if "High" in data.columns else None,
                "Low": round(data["Low"].iloc[-1], 2) if "Low" in data.columns else None,
                "Close": close_price,
                "Value": round(close_price * volume, 2) if close_price and volume else "N/A",
                "Volume": volume if volume else "N/A",
            })
    
    return render_template('market.html', stock_market=stock_market)


@app.route('/share-info')
def plot():
    return render_template('plot.html')

@app.route('/get_stock_data')
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL')
    stock = yf.Ticker(ticker)
    data = stock.history(period='60d', interval='1d')

    if data.empty:
        return jsonify({"error": "No data found!"})

    result = {
        "dates": data.index.strftime("%Y-%m-%d").tolist(),
        "open": data["Open"].tolist(),
        "high": data["High"].tolist(),
        "low": data["Low"].tolist(),
        "close": data["Close"].tolist(),
        "ticker": ticker
    }
    return jsonify(result)

# Function to predict stock price using LSTM
def predict_stock_lstm(ticker, future_days=5):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y', interval='1d')

    if data.empty:
        return None

    df = data[['Close']].copy()

    # Normalize data
    scaler = MinMaxScaler()
    df['Scaled_Close'] = scaler.fit_transform(df[['Close']])

    # Prepare training data
    X_train, y_train = [], []
    window_size = 50  # Number of past days used to predict the next day

    for i in range(window_size, len(df) - future_days):
        X_train.append(df['Scaled_Close'].iloc[i - window_size:i].values)
        y_train.append(df['Scaled_Close'].iloc[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Prepare test data for future prediction
    last_window = df['Scaled_Close'].iloc[-window_size:].values.reshape(1, window_size, 1)
    future_scaled = []

    for _ in range(future_days):
        prediction = model.predict(last_window)[0][0]
        future_scaled.append(prediction)

        # Update window with new prediction
        last_window = np.roll(last_window, -1)
        last_window[0, -1, 0] = prediction

    future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).ravel()

    # Create DataFrame for future predictions
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
    predicted_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_prices})

    return df, predicted_df

@app.route('/predict_stock')
def predict_stock_route():
    ticker = request.args.get('ticker')

    historical_data, predicted_data = predict_stock_lstm(ticker)

    if historical_data is None:
        return jsonify({"error": "No data available for prediction!"})

    # Convert data to Plotly JSON format
    trace_actual = go.Scatter(
        x=historical_data.index, y=historical_data["Close"], 
        mode='lines', name="Actual Prices"
    )
    trace_predicted = go.Scatter(
        x=predicted_data["Date"], y=predicted_data["Predicted_Close"], 
        mode='lines+markers', name="Predicted Prices"
    )

    graph_data = [trace_actual, trace_predicted]
    graph_json = json.dumps(graph_data, cls=PlotlyJSONEncoder)

    return jsonify({"graph": graph_json, "ticker": ticker})

if __name__ == "__main__":
    app.run(debug=True)
