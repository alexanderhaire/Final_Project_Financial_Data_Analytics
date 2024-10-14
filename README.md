# README: High-Frequency Trading Strategy using Alpaca API and XGBoost

## Overview
This project is an implementation of a High-Frequency Trading (HFT) strategy using Python, the Alpaca trade API, Yahoo Finance, and XGBoost. The goal of the project is to train a predictive model for selected stocks, execute trades based on real-time data, and periodically retrain the model to adapt to changing market conditions.

The script uses historical data to train models for predicting stock price movements and Alpaca API to execute trades automatically based on the model's predictions. This README provides an overview of the project's functionality, components, and how to use it.

## Prerequisites
To run this project, the following prerequisites must be installed:

1. **Python 3.6+**
2. **Python Libraries**: Install the required libraries using the command below:
   ```sh
   pip install alpaca-trade-api pandas numpy yfinance xgboost scikit-learn
   ```
3. **Alpaca Account**: You need an Alpaca account to trade using their API. Use the [Alpaca website](https://alpaca.markets) to create an account.
4. **API Credentials**: Obtain your API Key and Secret from Alpaca and replace the placeholders in the script.

## Structure of the Project

### Main Components
1. **API Credentials**: The script uses Alpaca API for trading. Replace the placeholders with your credentials:
   ```python
   API_KEY = 'API Key'
   API_SECRET = 'API Secret'
   BASE_URL = 'https://paper-api.alpaca.markets'
   ```
2. **Historical Data Collection**: The script uses Yahoo Finance to collect historical stock data for the selected tickers. Historical data is used to generate features like moving averages, RSI, and momentum.
   - **Function**: `get_historical_data(ticker, start_date, end_date)`
3. **Feature Engineering**: The following features are added to both historical and real-time data:
   - **Moving Averages (50 & 200 periods)**
   - **Relative Strength Index (RSI)**
   - **Momentum Indicator**
   - **Lag Features**
4. **Real-Time Data Collection**: Real-time data is collected using the Alpaca API for the selected tickers.
   - **Function**: `get_latest_data(ticker, timeframe)`
5. **Model Training**: The script trains an XGBoost classifier using walk-forward validation to predict stock price movement (up or down). The trained model predicts if the stock will rise or fall based on the features.
   - **Function**: `train_model(tickers)`
6. **Trading Logic**: The trained model is used to predict stock price movement. Based on the predictions, the script buys or sells stocks.
   - **Buy**: If the model predicts an increase.
   - **Sell**: If the model predicts a decrease or if current holdings exceed a specified limit.
7. **Retraining**: The model is periodically retrained based on new data to adapt to market conditions.
   - **Function**: `retrain_model(ticker, scaler)`
8. **Execution Loop**: The script loops continuously to retrieve the latest data, make predictions, and execute trades.
   - **Function**: `run_hft_strategy(tickers, quantity, model_dict, scaler_dict, retrain_interval)`

## Instructions for Use

1. **Set Up Alpaca Credentials**: Update the `API_KEY` and `API_SECRET` with your credentials. Ensure the `BASE_URL` is set to the correct Alpaca endpoint.
2. **Choose Stock Tickers**: Update the `tickers` list with the tickers of the stocks you want to trade. The default list is:
   ```python
   tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "VZ", "CSCO", "INTC", "CAT", "PFE"]
   ```
3. **Running the Script**:
   - Ensure the market is open.
   - Run the script to start the trading bot. The bot will fetch the latest data, make predictions, and execute trades.
4. **Adjust Retraining Interval**: The `retrain_interval` is set to 60 seconds by default, meaning the model will be retrained every minute. Adjust this value as needed for your requirements.
5. **Order Execution**: The script uses market orders to buy or sell the selected quantity (`quantity`). You can adjust the quantity parameter in the `run_hft_strategy()` function.

## Key Features

1. **Automated Data Collection**: Collects historical and real-time data for feature extraction and model training.
2. **Machine Learning Model**: Uses XGBoost, a powerful machine learning model, to predict price movement and generate trading signals.
3. **Order Execution**: Automatically places market buy or sell orders via Alpaca based on model predictions.
4. **Periodic Retraining**: The model is retrained periodically to ensure that it adapts to changing market conditions.
5. **Risk Management**: Includes logic to limit position size (e.g., selling if quantity exceeds 50 shares).

## Example Usage
To run the trading strategy, use the following steps:

1. **Training the Models**:
   The script will train individual models for each ticker before executing the trading loop.
   ```python
   model, scaler = train_model([ticker])
   model_dict[ticker] = model
   scaler_dict[ticker] = scaler
   ```

2. **Running the Strategy**:
   Start the trading strategy loop, which will execute trades and retrain the model periodically.
   ```python
   run_hft_strategy(tickers, 1, model_dict, scaler_dict, retrain_interval=60)  # Retrain every minute
   ```

## Limitations & Considerations

1. **Paper Trading**: The script uses the Alpaca paper trading API (`BASE_URL`) for testing purposes. Do not use this script in live trading until you have thoroughly tested it and ensured the trading strategy is reliable.
2. **Risk Management**: The current script has basic risk management, but it's essential to implement more advanced strategies for effective risk control, including stop losses, diversification, etc.
3. **Market Hours**: Ensure the script runs only during market hours. The `get_latest_data()` function checks whether the market is open before fetching data.
4. **Data Quality**: The script is dependent on the quality of data from Yahoo Finance and Alpaca. Inconsistent data can lead to incorrect predictions.
5. **Model Performance**: The XGBoost model may not always make correct predictions. Therefore, proper backtesting and parameter tuning are recommended.

## Future Improvements
1. **Hybrid Model Implementation**: Use hybrid models combining different algorithms for better predictive performance.
2. **Parallel Processing**: Use parallel processing for faster data collection, feature extraction, and model training.
3. **Reinforcement Learning**: Integrate reinforcement learning to improve trading decisions dynamically.
4. **Advanced Feature Engineering**: Use more sophisticated financial indicators like Bollinger Bands, MACD, etc., to improve the model's prediction capabilities.
5. **Position Sizing**: Implement advanced position sizing strategies based on risk metrics such as volatility.

## Disclaimer
This project is for educational purposes only. Trading in the stock market carries significant risk, and there is no guarantee of profitability. The author is not responsible for any financial losses incurred while using this code. Always consult with a financial advisor before engaging in trading activities.



