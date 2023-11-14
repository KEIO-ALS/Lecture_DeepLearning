from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)

# Policy Network
class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            # SHARED FC1
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            # SHARED FC2
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )
        # MEAN FC
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )
        # STDDEV FC
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


# Building an agent
class REINFORCE:
    def __init__(self, obs_space_dims: int, action_space_dims: int, device: str):
        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        # Check if GPU is available
        self.device = device
        self.net = Policy_Network(obs_space_dims, action_space_dims).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    # 状態に対する行動をサンプリング
    def sample_action(self, state: np.ndarray) -> float:
        state = torch.tensor(np.array([state])).to(self.device)
        action_means, action_stddevs = self.net(state)

        # PolicyNetによって出力された「平均」と「標準偏差」を元に正規分布を作成
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        # 作成された正規分布から、アクションをサンプリング
        action = distrib.sample()
        # サンプリングした行動の対数確立を計算し、保存
        prob = distrib.log_prob(action)
        self.probs.append(prob)

        return action.cpu().numpy()
    
    # エピソード終了後にPolicyNetを更新する
    def update(self):
        running_g = 0
        gs = []

        # 割引累積報酬(gs)の導出
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        # あるエピソード内の各タイムステップにおける割引された報酬の合計
        deltas = torch.tensor(gs)

        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []


# 環境を準備
env = gym.make("InvertedPendulum-v4")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

total_num_episodes = int(1e4)
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

for seed in range(5):
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # agentの準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = REINFORCE(obs_space_dims, action_space_dims, device)

    reward_over_episodes = []

    for episode in tqdm(range(total_num_episodes)):
        obs, info = wrapped_env.reset(seed=seed)
        done = False
        while not done:
            # PolicyNetを用いてアクションを選択
            action = agent.sample_action(obs)
            # アクションの結果を獲得
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            # タスク失敗 or 制限時間到達　で終了
            done = terminated or truncated

        # 最新エピソードにおける報酬を保存
        reward_over_episodes.append(wrapped_env.return_queue[-1])
        # PolicyNetの学習等
        agent.update()

        if (episode+1) % 1000 == 0:
            # 最新エピソードにおける報酬平均を出力
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode+1, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)
    torch.save(agent.net.state_dict(), f'outputs/REINFORCE_for_InvertedPendulum_v4.agent.{seed}.pth')


# Plot learning curve
rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.savefig('outputs/REINFORCE_for_InvertedPendulum_v4.rewards.png')
plt.show()