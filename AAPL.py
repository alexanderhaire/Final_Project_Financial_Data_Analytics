import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
import time
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings

# Alpaca API credentials
API_KEY = 'API Key'
API_SECRET = 'Secret'
BASE_URL = 'https://paper-api.alpaca.markets'

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Function to get historical data from Yahoo Finance and preprocess
def get_historical_data(ticker, start_date="2010-01-01", end_date="2024-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data.dropna(inplace=True)
    return data

# Function to train the model using XGBoost with walk-forward validation
def train_model(data):
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data.dropna(inplace=True)

    features = ['Lag_1', 'Lag_2', '50_MA', '200_MA', 'RSI', 'Close']
    X = data[features]
    y = data['Target']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series split for walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)

    # XGBoost model
    xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

    # Grid search parameters
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    grid_search = GridSearchCV(xgb_model, param_grid, cv=tscv, n_jobs=-1)
    grid_search.fit(X_scaled, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    return best_model, scaler

# Function to get the latest data from Alpaca and preprocess it
def get_latest_data(ticker, timeframe='1Min'):
    try:
        clock = api.get_clock()
        if not clock.is_open:
            print("Market is currently closed.")
            return pd.DataFrame()

        bars = api.get_bars(ticker, timeframe=timeframe, limit=200).df
        if bars.empty:
            print("No data returned from Alpaca.")
            return pd.DataFrame()

        df = pd.DataFrame(bars.reset_index())
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)
        
        if 'Close' in df:
            df['50_MA'] = df['Close'].rolling(window=50).mean()
            df['200_MA'] = df['Close'].rolling(window=200).mean()
            delta = df['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        else:
            print("Required columns are missing.")
        
        df.dropna(inplace=True)
        return df
    
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return pd.DataFrame()

# Function to place an order
def place_order(symbol, qty, side, order_type='market'):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='gtc'
        )
        print(f"Order placed: {side} {qty} shares of {symbol}")
    except Exception as e:
        print(f"Error placing order: {e}")

# Function to run the strategy with machine learning predictions
def run_ml_strategy(ticker, quantity, model, scaler):
    df = get_latest_data(ticker)
    if df.empty:
        print("No data available for trading.")
        return
    
    lag_1 = df['Close'].shift(1).iloc[-1]
    lag_2 = df['Close'].shift(2).iloc[-1]
    ma_50 = df['50_MA'].iloc[-1]
    ma_200 = df['200_MA'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    current_price = df['Close'].iloc[-1]

    # Prepare the input for the model
    X_new = np.array([[lag_1, lag_2, ma_50, ma_200, rsi, current_price]])
    X_new_scaled = scaler.transform(X_new)

    prediction = model.predict(X_new_scaled)
    
    try:
        position = api.get_position(ticker)
        current_qty = int(position.qty)
    except Exception:
        current_qty = 0

    if prediction == 1 and current_qty == 0:
        place_order(ticker, quantity, 'buy')
    elif prediction == 0 and current_qty > 0:
        place_order(ticker, current_qty, 'sell')

# Main execution
ticker = "AAPL"
historical_data = get_historical_data(ticker)
model, scaler = train_model(historical_data)

while True:
    try:
        run_ml_strategy(ticker, 10, model, scaler)
        account = api.get_account()
        print(f"Portfolio Value: {account.portfolio_value}")
        time.sleep(60)
    except Exception as e:
        print(f"Error: {e}")
        break
