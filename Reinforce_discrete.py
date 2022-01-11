import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.prob = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        prob = T.softmax(self.prob(x), dim=-1)

        return prob

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class Reinforce:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir, gamma=0.99):
        self.gamma = gamma
        self.checkpoint_dir = ckpt_dir
        self.reward_memory = []
        self.log_prob_memory = []

        self.policy = PolicyNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=fc1_dim, fc2_dim=fc2_dim)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(device)
        probabilities = self.policy.forward(state)
        dist = Categorical(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_prob_memory.append(log_prob)

        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        # ---------------------------------------------------------------
        # 法一
        # ---------------------------------------------------------------
        G_list = []
        G_t = 0
        for item in self.reward_memory[::-1]:
            G_t = self.gamma * G_t + item
            G_list.append(G_t)
        G_list.reverse()
        G_tensor = T.tensor(G_list, dtype=T.float).to(device)

        # ---------------------------------------------------------------
        # 法二
        # ---------------------------------------------------------------
        # G = np.zeros_like(self.reward_memory, dtype=np.float64)
        # for t in range(len(self.reward_memory)):
        #     G_sum = 0
        #     discount = 1
        #     for k in range(t, len(self.reward_memory)):
        #         G_sum += self.reward_memory[k] * discount
        #         discount *= self.gamma
        #     G[t] = G_sum
        # G_tensor2 = T.tensor(G, dtype=T.float).to(device)

        loss = 0
        for g, log_prob in zip(G_tensor, self.log_prob_memory):
            loss += -g * log_prob

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        self.reward_memory.clear()
        self.log_prob_memory.clear()

    def save_models(self, episode):
        self.policy.save_checkpoint(self.checkpoint_dir + 'Reinforce_policy_{}.pth'.format(episode))
        print('Saved the policy network successfully!')

    def load_models(self, episode):
        self.policy.load_checkpoint(self.checkpoint_dir + 'Reinforce_policy_{}.pth'.format(episode))
        print('Loaded the policy network successfully!')
