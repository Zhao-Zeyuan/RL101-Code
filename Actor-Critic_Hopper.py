import gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("Hopper-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.v_fc1 = nn.Linear(state_dim, 256)
        self.v_fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)
    def forward(self, x):
        x_act = torch.tanh(self.fc1(x))
        x_act = torch.tanh(self.fc2(x_act))
        mean = self.mean(x_act)
        std = torch.exp(self.log_std)
        x_val = torch.tanh(self.v_fc1(x))
        x_val = torch.tanh(self.v_fc2(x_val))
        v = self.value(x_val)
        return mean, std, v

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
gamma = 0.99
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        mean, std, value = model(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        dist = torch.distributions.Independent(dist, 1)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        next_state, reward, done, _ = env.step(action.detach().numpy())
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        _, _, next_value = model(next_state_tensor)
        td_target = reward + gamma * next_value * (1 - int(done))
        advantage = td_target - value
        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state
        total_reward += reward
    if (episode+1) % 50 == 0:
        print(f"Episode {episode+1}\tReward: {total_reward}")
env.close()
