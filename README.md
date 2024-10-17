# README: Hybrid Trading Strategy with LSTM and XGBoost

## Overview

This Python script implements a **hybrid trading strategy** that uses both **deep learning (LSTM)** and **machine learning (XGBoost)** models to predict stock price movements and execute trades automatically. The strategy fetches **real-time stock data** from the **Alpaca API** and applies advanced technical indicators such as Bollinger Bands, MACD, and On-Balance Volume (OBV) for feature engineering. It executes trades for multiple stock tickers (e.g., AAPL, GOOGL, TSLA), leveraging the strengths of both LSTM for time-series analysis and XGBoost for non-linear relationships.

---

## Features

1. **Deep Learning (LSTM)**: Captures temporal dependencies in stock price data, making it suitable for time-series predictions.
2. **Machine Learning (XGBoost)**: Complements LSTM by handling tabular data and non-linear relationships.
3. **Hybrid Model**: Combines LSTM and XGBoost predictions to enhance accuracy.
4. **Technical Indicators**: Includes Bollinger Bands, MACD, OBV, and ATR for better market analysis.
5. **Real-Time Data**: Fetches stock market data in real time using Alpaca API, allowing the script to react dynamically to the latest market conditions.
6. **Automatic Trade Execution**: Uses limit orders for both long and short positions, minimizing slippage and optimizing trade execution.
7. **Periodic Model Retraining**: Retrains the models periodically to ensure the strategy adapts to changing market conditions.

---

## Requirements

- Python 3.x
- Required Python libraries:
  - `alpaca-trade-api`
  - `pandas`
  - `numpy`
  - `tensorflow` (for LSTM)
  - `xgboost`
  - `scikit-learn`
  - `yfinance`

You can install the necessary libraries using the following command:

```bash
pip install alpaca-trade-api pandas numpy tensorflow xgboost scikit-learn yfinance
```

---

## Setup

1. **Alpaca API Setup**:
   - Create an account at [Alpaca](https://alpaca.markets/) to get your **API Key** and **Secret Key**.
   - Insert your API credentials into the script:

   ```python
   API_KEY = "your_alpaca_api_key"
   SECRET_KEY = "your_alpaca_secret_key"
   ```

2. **Stock Ticker Configuration**:
   - Add the stock tickers you want to trade in the `tickers` list:

   ```python
   tickers = ['AAPL', 'GOOGL', 'TSLA']  # Add tickers of your choice
   ```

3. **Model Retraining Interval**:
   - You can adjust how frequently the model retrains itself by modifying the `retrain_interval` parameter (e.g., every 10 minutes):

   ```python
   retrain_interval = 60 * 10  # Retrains the model every 10 minutes
   ```

---

## How to Run

1. **Run the Script**:
   - Make sure all dependencies are installed, and the Alpaca API keys are configured correctly.
   - Run the script in your terminal or command prompt:

   ```bash
   python hybrid_trading_strategy.py
   ```

2. **Monitoring**:
   - The script will print the portfolio value at regular intervals. You can monitor the progress and stop the script at any time by using `Ctrl+C`.

---

## Key Functions

1. **`add_technical_indicators`**:
   - Adds technical indicators like Bollinger Bands, MACD, OBV, and ATR to the raw stock data, enriching the dataset for both the LSTM and XGBoost models.

2. **`create_lstm_model`**:
   - Defines the architecture of the LSTM model, including layers and dropout for reducing overfitting.

3. **`train_hybrid_model`**:
   - Trains both LSTM and XGBoost models using historical stock data and technical indicators as features.

4. **`get_latest_data`**:
   - Fetches real-time stock data from Alpaca and applies the same technical indicators used during model training.

5. **`run_hybrid_strategy`**:
   - Combines predictions from the LSTM and XGBoost models and executes buy/sell trades based on the averaged prediction.

6. **`place_limit_order`**:
   - Executes buy or sell trades with limit orders, optimizing trade execution to reduce slippage.

---

## Considerations

- **API Limits**: Alpaca API has rate limits. Ensure your script complies with these limits to avoid throttling.
- **Market Conditions**: This strategy is designed for real-time, high-frequency trading and may behave differently in low-volume or highly volatile markets.
- **Backtesting**: It is recommended to backtest your strategy with historical data before deploying it in live markets.

---

## Conclusion

This Python script provides a robust, hybrid trading solution that combines the predictive strengths of LSTM and XGBoost models. It adapts dynamically to changing market conditions by continuously retraining itself, making it a powerful tool for algorithmic trading in real-time stock markets. By leveraging advanced technical indicators and automating trade execution, it offers an effective way to engage in high-frequency trading strategies.

