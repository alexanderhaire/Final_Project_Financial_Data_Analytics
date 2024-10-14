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
API_KEY = 'Your API'
API_SECRET = 'Your Secret API'
BASE_URL = 'https://paper-api.alpaca.markets'

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Function to get historical data from Yahoo Finance and preprocess
def get_historical_data(ticker, start_date="2010-01-01", end_date="2024-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Adding features
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Adding momentum indicator
    data['Momentum'] = data['Close'] - data['Close'].shift(10)

    # Add lagged features
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)

    # Drop any rows with missing values
    data.dropna(inplace=True)
    return data

# Function to get real-time data from Alpaca and preprocess it
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
            # Adding moving averages
            df['50_MA'] = df['Close'].rolling(window=50).mean()
            df['200_MA'] = df['Close'].rolling(window=200).mean()

            # Adding momentum indicator
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Adding RSI
            delta = df['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Adding lag features
            df['Lag_1'] = df['Close'].shift(1)
            df['Lag_2'] = df['Close'].shift(2)
            
            # Drop any missing values
            df.dropna(inplace=True)

        return df
    
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return pd.DataFrame()

# Function to place a limit order
def place_limit_order(symbol, qty, side, limit_price):
    try:
        limit_price = round(limit_price, 2)  # Round limit price to 2 decimal places
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='gtc',
            limit_price=str(limit_price)
        )
        print(f"Limit order placed: {side} {qty} shares of {symbol} at {limit_price}")
    except Exception as e:
        print(f"Error placing order: {e}")

# Function to retrain the model periodically
def retrain_model(ticker, scaler):
    print(f"Retraining model for {ticker}...")
    historical_data = get_historical_data(ticker)
    model, new_scaler = train_model([ticker])
    return model, new_scaler

# Function to train the model using XGBoost with walk-forward validation across multiple tickers
def train_model(tickers):
    combined_data = []
    for ticker in tickers:
        data = get_historical_data(ticker)
        data['Ticker'] = ticker  # Add a column for the ticker
        combined_data.append(data)

    all_data = pd.concat(combined_data)
    all_data['Target'] = np.where(all_data['Close'].shift(-1) > all_data['Close'], 1, 0)

    # Features used for the model
    features = ['Lag_1', 'Lag_2', '50_MA', '200_MA', 'RSI', 'Momentum', 'Close']
    X = all_data[features]
    y = all_data['Target']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series split for walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)

    # XGBoost model
    xgb_model = XGBClassifier(eval_metric='logloss')

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

# Function to run the HFT strategy with periodic retraining
def run_hft_strategy(tickers, quantity, model_dict, scaler_dict, retrain_interval=3600):
    iteration_count = 0

    while True:
        for ticker in tickers:
            print(f"Running HFT strategy for {ticker}...")
            df = get_latest_data(ticker)
            if df.empty:
                print(f"No data available for {ticker}.")
                continue
            
            # Ensure the features used for prediction exist in the latest data
            try:
                features = ['Lag_1', 'Lag_2', '50_MA', '200_MA', 'RSI', 'Momentum', 'Close']
                X_new = df[features].iloc[-1:].values  # Use the most recent data
            except KeyError as e:
                print(f"Error: {e}")
                continue

            # Scale the features
            X_new_scaled = scaler_dict[ticker].transform(X_new)

            # Make a prediction
            prediction = model_dict[ticker].predict(X_new_scaled)

            # Execute trade based on the prediction or quantity
            try:
                position = api.get_position(ticker)
                current_qty = int(position.qty)
            except Exception:
                current_qty = 0

            # Get the current price for placing limit orders
            current_price = df['Close'].iloc[-1]

            # Sell if quantity exceeds 200 or model predicts a decrease
            if current_qty > 300 or prediction == 0:
                limit_price = round(current_price * 0.99, 2)  # 1% below the current price
                print(f"Placing limit sell order for {ticker}.")
                place_limit_order(ticker, current_qty, 'sell', limit_price)
            else:
                # Buy if the model predicts an increase
                if prediction == 1:
                    limit_price = round(current_price * 1.01, 2)  # 1% above the current price
                    print(f"Placing limit buy order for {ticker}.")
                    place_limit_order(ticker, quantity, 'buy', limit_price)

        # Ensure retrain_interval is not zero, and handle the periodic retraining
        if retrain_interval > 0 and iteration_count % (retrain_interval // 60) == 0:
            print("Retraining models...")
            for ticker in tickers:
                model_dict[ticker], scaler_dict[ticker] = retrain_model(ticker, scaler_dict[ticker])

        account = api.get_account()
        print(f"Portfolio Value: {account.portfolio_value}")
        time.sleep(60)  # Adjust the sleep time as needed
        iteration_count += 1

# Main execution
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "VZ", "CSCO", "INTC", "CAT", "PFE"]
model_dict = {}
scaler_dict = {}

# Train models for each ticker
for ticker in tickers:
    print(f"Training model for {ticker}...")
    historical_data = get_historical_data(ticker)
    model, scaler = train_model([ticker])
    model_dict[ticker] = model
    scaler_dict[ticker] = scaler

# Run the HFT strategy with periodic retraining
run_hft_strategy(tickers, 1, model_dict, scaler_dict, retrain_interval=60) 
