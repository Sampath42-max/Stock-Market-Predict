# 📈 StockPulse | AI Stock Market Intelligence & Machine Learning Forecasting Engine

A high-performance, modern web application for real-time stock market analysis, technical indicator tracking, and machine learning price predictions powered by **Flask**, **yfinance**, **scikit-learn**, and **Plotly**.

---

## ✨ Key Features

- ⚡ **Fast Machine Learning Forecast Engine**:
  - Predicts 5, 10, or 30-day stock price trajectories using a **Ridge Regression + Exponential Trend Momentum** ensemble.
  - Generates **95% Confidence Interval Upper/Lower Shading**, model accuracy evaluation scores (**RMSE**, **MAE**, **R² Fit Score**), and **Bullish / Bearish Sentiment Signals**.

- 🎨 **Modern Glassmorphism Dark Mode UI**:
  - High-tech dark aesthetics (`#080b11`) with glowing electric cyan accents, glass card blurs, and crisp typography (`Plus Jakarta Sans`).
  - Animated live stock ticker marquee and market status badge ("Market Open" / "Market Closed").

- 📊 **Interactive Technical Analytics Studio**:
  - Interactive Plotly charts supporting **Candlestick** and **Line / Area** view modes.
  - Multi-timeframe selection (`1M`, `6M`, `1Y`, `5Y`).
  - Technical indicator overlays: **Simple Moving Average (SMA 20 & SMA 50)**, **Exponential Moving Average (EMA 20)**, and **Relative Strength Index (RSI 14)**.

- 🚀 **Parallel Data Pipeline & 5-Min TTL Caching**:
  - Multi-threaded `ThreadPoolExecutor` yfinance data fetching with an in-memory TTL cache (`CACHE_TTL = 300s`).
  - Reduces multi-stock loading times from **30+ seconds down to ~1-2 seconds**.

- 🌍 **Multi-Market Ticker Coverage**:
  - **US Tech Leaders**: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `GOOGL`, `AMZN`, `META`, `PLTR`, `ARM`, `SMCI`, `COIN`, `UBER`
  - **Indian Stock Market (NSE)**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `MOTILALOFS.NS`
  - **ETFs & Cryptocurrencies**: `SPY`, `QQQ`, `BTC-USD`, `ETH-USD`

- 🌐 **Dual Compatibility (Flask Server & VS Code Live Server)**:
  - Works seamlessly via Flask (`http://127.0.0.1:5000`) or VS Code Live Server (`http://127.0.0.1:5500`) with CORS cross-origin bridge and offline fallback sample data.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.12, Flask, Flask-CORS, yfinance, scikit-learn, NumPy, Pandas, Plotly.
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphic Design System), JavaScript (ES6+), Plotly.js.

---

## 📁 Project Architecture

```text
Stock-Market-Predict/
│
├── flask_app.py              # Core Flask backend server, API endpoints & ML engine
├── templates/                # HTML Jinja templates
│   ├── index.html            # Home Dashboard, Ticker Marquee & Global Search
│   ├── market.html           # Market Overview Table with Gainers/Losers tabs & Sparklines
│   └── plot.html             # Interactive Stock Studio & AI Forecast Plotter
│
├── static/                   # Static Web Assets & Stylesheets
│   ├── style.css             # Glassmorphism base design system & dark palette
│   ├── market.css            # Table grid, sparkline & category tab styling
│   ├── plot.css              # Studio layout, chart controls & metrics sidebar
│   ├── script.js             # Main interactive JS, market status & Live Server bridge
│   ├── css/                  # Mirrored CSS directory
│   └── js/                   # Mirrored JS directory
│
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start & Setup Guide

### Prerequisites
Make sure you have **Python 3.10+** installed.

### 1. Install Dependencies
Open your terminal in the project directory and run:
```bash
pip install flask flask-cors yfinance plotly scikit-learn numpy pandas
```

### 2. Run the Application
Start the Flask web server:
```bash
python flask_app.py
```

### 3. Open in Browser
Visit **`http://127.0.0.1:5000`** in your browser to launch the StockPulse application.

---

## 📡 API Endpoints Documentation

| Endpoint | Method | Description | Example Query |
| :--- | :--- | :--- | :--- |
| `/api/market-summary` | `GET` | Returns 24h market metrics, close prices, percentage changes, volume, and sparklines. | `/api/market-summary` |
| `/api/get_stock_data` | `GET` | Returns OHLC historical prices and calculated SMA/EMA/RSI technical indicators. | `/api/get_stock_data?ticker=AAPL&period=6mo` |
| `/api/predict_stock` | `GET` | Generates ML price predictions, upper/lower confidence bounds, and RMSE metrics. | `/api/predict_stock?ticker=NVDA&future_days=5` |
| `/api/search` | `GET` | Searches matching stock tickers and validates ticker symbols. | `/api/search?q=AA` |

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
