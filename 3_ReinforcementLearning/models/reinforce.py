import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

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