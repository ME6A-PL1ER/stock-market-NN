import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sqlite3
import time

# Prepare the model
window_size = 10
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(window_size,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Function to get the latest XRP data
def get_xrp_data():
    data = yf.download('XRP-USD', period='15d', interval='1m')
    data['returns'] = data['Close'].pct_change().dropna()
    return data

# Function to prepare data for training and prediction
def prepare_data(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data['returns'])):
        X.append(data['returns'][i-window_size:i])
        y.append(data['returns'][i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Function to save data to a database
def save_to_db(symbol, data):
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {symbol} (
            datetime TEXT,
            close REAL,
            returns REAL
        )
    """)
    for idx, row in data.iterrows():
        cursor.execute(f"""
            INSERT INTO {symbol} (datetime, close, returns) VALUES (?, ?, ?)
        """, (idx, row['Close'], row['returns']))
    conn.commit()
    conn.close()

# Training loop
data = get_xrp_data()
save_to_db('XRP-USD', data)
X, y = prepare_data(data, window_size)
model.fit(X, y, epochs=10, batch_size=32)

# Paper trading simulation
balance = 1000
positions = 0
profit_history = []
prices = data['Close'].values[-window_size:]

# Real-time trading loop
plt.ion()
fig, ax = plt.subplots()

while True:
    data = get_xrp_data()
    save_to_db('XRP-USD', data)
    X, y = prepare_data(data, window_size)
    prediction = model.predict(X[-1].reshape(1, -1))[0][0]
    current_price = data['Close'].values[-1]

    if prediction > 0 and balance > 0:
        positions = balance / current_price
        balance = 0
    elif prediction < 0 and positions > 0:
        balance = positions * current_price
        positions = 0
    
    profit_history.append(balance + positions * current_price)
    prices = np.append(prices, current_price)

    ax.clear()
    ax.plot(profit_history, label='Profit')
    ax.plot(prices, label='Price')
    ax.legend()
    plt.pause(60)  # Update every minute

plt.ioff()
plt.show()
