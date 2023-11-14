from models import REINFORCE
import torch
import gymnasium as gym

num_episode = sys.argv[1]

for i in range(5):
    model_path = f"outputs/{num_episode}_episode/REINFORCE_for_InvertedPendulum_v4.agent.{i}.pth"

    # エージェントのロードと初期化
    device = torch.device("cpu")
    obs_space_dims = 4  # InvertedPendulum-v4の観測空間の次元
    action_space_dims = 1  # InvertedPendulum-v4の行動空間の次元
    agent = REINFORCE(obs_space_dims, action_space_dims, device)
    agent.net.load_state_dict(torch.load(model_path, map_location=device))

    # 環境のセットアップ
    env = gym.make("InvertedPendulum-v4", render_mode="human")

    # エージェントのテスト実行
    obs, info = env.reset()
    done = False
    num_step = 0
    while not done:
        num_step += 1
        action = agent.sample_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated
    # 環境のクローズ
    env.close()
    print(f"terminated: {terminated}\ntruncated: {truncated}\nscore: {num_step}")