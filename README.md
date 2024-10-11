

# Stock Trading Bot Using XGBoost and Alpaca API

This project implements an automated stock trading bot using machine learning models (XGBoost) and the Alpaca API. The bot is designed to make trading decisions based on stock market data, leveraging various technical indicators and time series analysis.

## Features
- Utilizes the Alpaca API to execute real-time trades.
- Fetches historical stock data using Yahoo Finance (yfinance).
- Calculates various technical indicators including:
  - 50-day Moving Average (50_MA)
  - 200-day Moving Average (200_MA)
  - Relative Strength Index (RSI)
- Applies a machine learning model (XGBoost) trained with walk-forward validation to predict stock movements.
- Automatically places buy and sell orders based on model predictions.
  
## Setup and Installation

**It is recommended to open and run this bot in a virtual environment** to avoid conflicts with dependencies on your current device.

1. **Set up a virtual environment:**
   ```bash
   python3 -m venv trading_bot_env
   source trading_bot_env/bin/activate  # On Windows, use `trading_bot_env\Scripts\activate`
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   (Ensure that your `requirements.txt` includes all necessary packages such as `alpaca-trade-api`, `pandas`, `numpy`, `yfinance`, `xgboost`, and `scikit-learn`.)

3. **Set up your Alpaca API credentials:**
   Replace `API_KEY` and `API_SECRET` in the script with your Alpaca API credentials.

4. **Run the bot:**
   ```bash
   python trading_bot.py
   ```

## How It Works

1. **Data Collection:**
   The bot fetches historical stock data using the Yahoo Finance API to calculate technical indicators and prepare the dataset for training the model.

2. **Model Training:**
   An XGBoost model is trained using walk-forward validation to predict whether the stock price will rise or fall. The model is tuned using GridSearchCV.

3. **Live Trading:**
   The bot checks the latest stock data from Alpaca in real-time and uses the trained model to predict market movements. It places buy or sell orders based on the predictions.

4. **Virtual Environment Recommendation:**
   It is highly recommended to use a virtual environment to isolate dependencies and prevent conflicts with existing packages on your device.

## Limitations and Considerations

- The bot currently supports a simple trading strategy based on daily stock data and common technical indicators. For more complex strategies, additional indicators or data sources may be integrated.
- Ensure that your Alpaca account is set to paper trading mode for testing purposes before deploying it in a live environment.
- The performance of the trading bot depends on the accuracy of the model, market conditions, and data reliability.

## License
This project is licensed under the MIT License.

---
