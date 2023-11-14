from models import *

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import gymnasium as gym
import os

task_name = "__"
if not os.path.exists(f"outputs/{task_name}"):
    os.makedirs(f"outputs/{task_name}")

plt.rcParams["figure.figsize"] = (10, 5)

# 環境を準備
env = gym.make("InvertedPendulum-v4")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

total_num_episodes = int(1e2)
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
    torch.save(agent.net.state_dict(), f'outputs/{task_name}/REINFORCE_for_InvertedPendulum_v4.agent.{seed}.pth')


# Plot learning curve
rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.savefig(f'outputs/{task_name}/REINFORCE_for_InvertedPendulum_v4.rewards.png')
plt.show()