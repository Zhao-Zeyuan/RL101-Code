import gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("CartPole-v1")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def select_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    probs = policy(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

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