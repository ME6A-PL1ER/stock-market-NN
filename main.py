import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
from gym import spaces
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os

# -----------------------------
# 🔹 Data Fetching Function
# -----------------------------
def get_historical_data():
    data = yf.download("BTC", interval="1h", period="7d")
    data['returns'] = data['Close'].pct_change().fillna(0)
    return data

def get_live_data():
    try:
        data = yf.download("BTC-USD", interval="1m", period="1d")
        if data.empty:
            return None
        data['returns'] = data['Close'].pct_change().fillna(0)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# -----------------------------
# 🔹 Custom Trading Environment
# -----------------------------
class TradingEnv(gym.Env):
    def __init__(self, data, window_size, risk_factor):
        super(TradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.balance = float(1000)
        self.positions = float(0)
        self.risk_factor = risk_factor
        self.profit_history = []
        self.last_price = None
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size + 1,), dtype=np.float32)

    def reset(self):
        self.current_step = self.window_size
        self.balance = 1000
        self.positions = 0
        self.profit_history = []
        return self._next_observation()

        self.current_price = None
        self.last_valid_step = window_size

    def _get_current_price(self):
        try:
            return float(self.data['Close'].iloc[self.current_step])
        except IndexError:
            return self.current_price

    def step(self, action):
        current_price = self._get_current_price()
        if current_price is None:
            return self._next_observation(), 0.0, True, {}
        
        self.current_price = current_price
        position_size = float(self.balance * self.risk_factor)

        if action == 1 and self.balance > 0:  # Buy
            self.positions = float(position_size / current_price)
            self.balance -= position_size
        elif action == 2 and self.positions > 0:  # Sell
            self.balance += float(self.positions * current_price)
            self.positions = float(0)

        profit = float(self.balance + (self.positions * current_price))
        self.profit_history.append(profit)
        reward = float(profit - self.profit_history[-2]) if len(self.profit_history) > 1 else 0.0
        
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        return self._next_observation(), reward, done, {}

    def _next_observation(self):
        try:
            obs = self.data['returns'].iloc[self.current_step - self.window_size:self.current_step].values
            return np.append(obs, self.balance)
        except:
            # Return last valid observation if data is not available
            return self._last_observation

    def render(self, mode='human'):
        plt.plot(self.profit_history, label='Profit')
        plt.plot(self.data['Close'].values[:self.current_step], label='Price')
        plt.legend()
        plt.show()

# -----------------------------
# 🔹 Deep Q-Network (DQN)
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# 🔹 RL Trading Agent (DQN)
# -----------------------------
class TradingAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = []
        self.gamma = 0.99

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        rewards = np.array([float(r) for r in rewards])

        states = np.stack(states)
        actions = np.array(actions)
        next_states = np.stack(next_states)
        dones = np.array(dones)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# -----------------------------
# 🔹 PyQt GUI
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent

        self.setWindowTitle("Trading Bot GUI")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.balance_label = QLabel("Balance: $1000")
        self.profit_label = QLabel("Profit: $0")
        self.reward_label = QLabel("Reward: $0")
        self.avg_profit_label = QLabel("Average Profit: $0")

        self.layout.addWidget(self.balance_label)
        self.layout.addWidget(self.profit_label)
        self.layout.addWidget(self.reward_label)
        self.layout.addWidget(self.avg_profit_label)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(1000)

    def update_gui(self):
        live_data = get_live_data()
        if live_data is not None and not live_data.empty:
            old_len = len(self.env.data)
            self.env.data = pd.concat([self.env.data, live_data]).drop_duplicates()
            if len(self.env.data) > old_len:
                # Only update if new data is available
                state = self.env._next_observation()
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                try:
                    balance = float(self.env.balance)
                    profit = float(self.env.profit_history[-1])
                    reward = float(reward)
                    avg_profit = float(np.mean(self.env.profit_history))

                    self.balance_label.setText(f"Balance: ${balance:.2f}")
                    self.profit_label.setText(f"Profit: ${profit:.2f}")
                    self.reward_label.setText(f"Reward: ${reward:.2f}")
                    self.avg_profit_label.setText(f"Average Profit: ${avg_profit:.2f}")

                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.plot([float(x) for x in self.env.profit_history], label='Profit')
                    ax.plot([float(x) for x in self.env.data['Close'].iloc[:self.env.current_step]], label='Price')
                    ax.legend()
                    self.canvas.draw()
                except Exception as e:
                    print(f"Error updating GUI: {e}")

# -----------------------------
# 🔹 Training Loop
# -----------------------------
class TradingThread(QThread):
    update_signal = pyqtSignal()

    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent

    def run(self):
        num_episodes = 100
        max_data_length = 1 * 24 * 60

        conn = sqlite3.connect('trading_data.db')
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                episode INTEGER,
                total_reward REAL,
                total_balance REAL
            )
        ''')
        conn.commit()

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.store_experience(state, action, reward, next_state, done)
                self.agent.train()
                state = next_state
                total_reward += reward

                if episode % 10 == 0:
                    live_data = get_live_data()
                    if live_data is not None:
                        self.env.data = pd.concat([self.env.data, live_data])
                        self.env.data = self.env.data.iloc[-max_data_length:]
                        self.env.reset()

                self.update_signal.emit()
                time.sleep(1)

            c.execute('INSERT INTO training_data (episode, total_reward, total_balance) VALUES (?, ?, ?)',
                      (episode + 1, total_reward, self.env.balance))
            conn.commit()

        conn.close()

# -----------------------------
# 🔹 Main Application
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window_size = 10
    risk_factor = 0.5
    historical_data = get_historical_data()

    if historical_data is None:
        exit()

    data = historical_data.copy()

    env = TradingEnv(data, window_size, risk_factor)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = TradingAgent(state_size, action_size)

    main_window = MainWindow(env, agent)
    main_window.show()

    trading_thread = TradingThread(env, agent)
    trading_thread.update_signal.connect(main_window.update_gui)
    trading_thread.start()

    sys.exit(app.exec_())