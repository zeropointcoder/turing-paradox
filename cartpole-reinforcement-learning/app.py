import streamlit as st
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Trainer:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cpu")

        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.buffer = ReplayBuffer()
        self.optimiser = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0

    def train_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q = self.model(torch.tensor(state, dtype=torch.float32))
                    action = int(torch.argmax(q))

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(self.buffer) > 64:
                self.learn()

        self.epsilon = max(0.05, self.epsilon * 0.995)
        self.target_model.load_state_dict(self.model.state_dict())
        return total_reward

    def learn(self, batch_size=64):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions).squeeze()
        next_q = self.target_model(next_states).max(1)[0]
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target.detach())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


class Evaluator:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def run(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = int(torch.argmax(self.model(torch.tensor(state, dtype=torch.float32))))
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

        return total_reward


def main():
    st.title("CartPole Reinforcement Learning")

    env = gym.make("CartPole-v1")
    trainer = Trainer(env)

    episodes = st.slider("Training episodes", 10, 300, 100)
    train_btn = st.button("Train agent")

    if train_btn:
        rewards = []
        progress = st.progress(0)

        for ep in range(episodes):
            reward = trainer.train_episode()
            rewards.append(reward)
            progress.progress((ep + 1) / episodes)

        evaluator = Evaluator(env, trainer.model)
        eval_score = evaluator.run()

        st.subheader("Results")
        st.line_chart(rewards, width=700)
        st.write(f"Evaluation reward: **{eval_score:.0f}**")


if __name__ == "__main__":
    main()
