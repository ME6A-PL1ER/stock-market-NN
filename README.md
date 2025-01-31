# Neural Network Paper Trading Bot

This project implements a neural network-based paper trading bot that uses `yfinance` to fetch real-time XRP-USD data, trains a neural network to make buy/sell/hold decisions, and simulates paper trading with a starting balance of $1000. The bot also dynamically saves training data to an SQLite database.

## Features
- **Real-time Data Retrieval**: Fetches XRP-USD data every minute using `yfinance`.
- **Neural Network Training**: Uses TensorFlow to train a neural network for making trading decisions.
- **Paper Trading Simulation**: Simulates trading with a starting balance of $1000.
- **Dynamic Data Storage**: Saves training data to an SQLite database for each stock symbol.
- **Real-time Profit Visualization**: Displays a real-time profit graph and price graph.

## Requirements
- Python 3.x
- `yfinance`
- `tensorflow`
- `pandas`
- `matplotlib`
- `sqlite3`

Install the necessary packages using:
```bash
pip install yfinance tensorflow pandas matplotlib
```

## Usage
1. **Run the Script**: Execute the script to start the real-time trading simulation. (Will probably need to be run from terminal due to directory permissions).
2. **Real-time Display**: The script will update and display the profit graph and price graph every minute.
3. **Database Storage**: Training data is saved to an SQLite database (`trading_data.db`) dynamically for each stock symbol.

## Code Overview
1. **Data Retrieval**: Fetches XRP-USD data using `yfinance`.
2. **Data Preparation**: Prepares the data for training and prediction.
3. **Neural Network Model**: Defines and trains a neural network model using TensorFlow.
4. **Paper Trading Simulation**: Simulates trading based on model predictions.
5. **Real-time Update Loop**: Continuously updates data and re-evaluates trading decisions every minute.
6. **Database Storage**: Saves training data to an SQLite database for each stock symbol.

## Example
Run the script and watch the real-time profit and price graphs:
```bash
python main.py
```

## Future Improvements
- Enhance the neural network model for better performance.
- Add support for multiple stock symbols.
- Implement more advanced trading strategies.
