import gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("Ant-v2")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        return mean, std

policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

def select_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    mean, std = policy(state)
    dist = torch.distributions.Normal(mean, std)
    dist = torch.distributions.Independent(dist, 1)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.detach().numpy(), log_prob

num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    log_probs = []
    done = False
    while not done:
        action, log_prob = select_action(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    loss = []
    for log_prob, R in zip(log_probs, returns):
        loss.append(-log_prob * R)
    loss = torch.stack(loss).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_reward = sum(rewards)
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}\tReward: {total_reward}")

env.close()
