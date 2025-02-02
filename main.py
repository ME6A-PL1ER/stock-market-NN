import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from gym import spaces
import pandas as pd
import yfinance as yf
import time

# -----------------------------
# 🔹 Data Fetching Function
# -----------------------------
def get_historical_data():
    # Fetching past 7 days of 1-hour candle data
    data = yf.download("XRP-USD", interval="1h", period="7d")
    data['returns'] = data['Close'].pct_change().fillna(0)
    return data

def get_live_data():
    # Fetch the most recent 1 minute candle data
    data = yf.download("XRP-USD", interval="1m", period="1d")
    data['returns'] = data['Close'].pct_change().fillna(0)
    return data

# -----------------------------
# 🔹 Custom Trading Environment
# -----------------------------
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
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
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
        import matplotlib.pyplot as plt
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
# 🔹 RL Trading Agent
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
        self.gamma = 0.99  # Discount factor

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)  # Random action (exploration)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()  # Greedy action

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Ensure rewards are scalars
        rewards = np.array([float(r) for r in rewards])  # Convert all rewards to scalars

        # Convert lists of arrays to numpy arrays, making sure all elements have the same shape
        states = np.stack(states)
        actions = np.array(actions)
        next_states = np.stack(next_states)
        dones = np.array(dones)

        # Convert numpy arrays into tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss and optimize
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



# -----------------------------
# 🔹 Training Loop
# -----------------------------
window_size = 10
risk_factor = 0.5  # Adjust risk preference
historical_data = get_historical_data()

if historical_data is None:
    exit()

# Fetch 1-hour data initially
data = historical_data.copy()

# Initialize environment with historical data
env = TradingEnv(data, window_size, risk_factor)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = TradingAgent(state_size, action_size)

num_episodes = 100
max_data_length = 1 * 24 * 60 # Maximum data points to keep (e.g., 7 days of 1-minute data)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

        # Every 10 episodes, fetch new live data and append
        if episode % 10 == 0:
            live_data = get_live_data()
            if live_data is not None:
                env.data = pd.concat([env.data, live_data])  # Append new data
                env.data = env.data.iloc[-max_data_length:] # Keep only the last max_data_length rows
                env.reset() # This is important: reset the environment to start from the beginning of the new data

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

env.render()
