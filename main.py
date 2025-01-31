import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sqlite3
import time
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gym import spaces
import gym

# Neural network model for price prediction
class PricePredictor:
    def __init__(self, window_size):
        self.window_size = window_size
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.window_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X, y):
        self.model.fit(X, y, epochs=10, batch_size=32)
    
    def predict(self, X):
        return self.model.predict(X)

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

# Custom environment for RL agent with risk factor
class TradingEnv(gym.Env):
    def __init__(self, data, window_size, risk_factor):
        super(TradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.balance = 1000
        self.positions = 0
        self.risk_factor = risk_factor
        self.profit_history = []
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size + 1,), dtype=np.float32)
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = 1000
        self.positions = 0
        self.profit_history = []
        return self._next_observation()
    
    def _next_observation(self):
        return np.append(self.data['returns'].values[self.current_step - self.window_size:self.current_step], self.balance)
    
    def step(self, action):
        current_price = self.data['Close'].values[self.current_step]

        # Adjust the position size based on the risk factor
        position_size = self.balance * self.risk_factor
        
        if action == 1 and self.balance > 0:  # Buy
            self.positions = position_size / current_price
            self.balance -= position_size
        elif action == 2 and self.positions > 0:  # Sell
            self.balance += self.positions * current_price
            self.positions = 0
        
        self.profit_history.append(self.balance + self.positions * current_price)
        
        reward = self.profit_history[-1] - self.profit_history[-2] if len(self.profit_history) > 1 else 0
        self.current_step += 1
        done = self.current_step == len(self.data)
        
        return self._next_observation(), reward, done, {}
    
    def render(self, mode='human'):
        plt.plot(self.profit_history, label='Profit')
        plt.plot(self.data['Close'].values[:self.current_step], label='Price')
        plt.legend()
        plt.show()

# Initialize and train the supervised learning model
window_size = 10
price_predictor = PricePredictor(window_size)
data = get_xrp_data()
X, y = prepare_data(data, window_size)
price_predictor.train(X, y)

# Define risk factor (0 = least risk, 1 = most risk)
risk_factor = 0.5  # Adjust this value as needed

# Initialize and train the RL agent
env = DummyVecEnv([lambda: TradingEnv(data, window_size, risk_factor)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Real-time trading loop
plt.ion()
fig, ax = plt.subplots()

while True:
    data = get_xrp_data()
    X, y = prepare_data(data, window_size)
    price_predictions = price_predictor.predict(X)
    
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
    
    current_price = data['Close'].values[-1]
    balance = env.envs[0].balance
    positions = env.envs[0].positions
    profit_history = env.envs[0].profit_history

    ax.clear()
    ax.plot(profit_history, label='Profit')
    ax.plot(data['Close'].values[:len(profit_history)], label='Price')
    ax.legend()
    plt.pause(60)  # Update every minute

plt.ioff()
plt.show()
