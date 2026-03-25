import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path

# ==========================================
# 1단계: 환경 설정 및 라이브러리 준비
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 결과 저장을 위한 경로 설정
IMAGES_PATH = Path() / "images" / "rl"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# ==========================================
# 2단계: 기준점 설정 (Manual Baseline Policy)
# ==========================================
def basic_policy(obs):
    """단순히 막대기의 각도에 따라 반대 방향으로 미는 규칙"""
    angle = obs[2]
    return 0 if angle < 0 else 1

# ==========================================
# 3단계: 정책 신경망 설계 (Policy Network)
# ==========================================
class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_hidden, n_outputs):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 각 행동을 취할 확률 분포 출력
        return F.softmax(self.fc2(x), dim=-1)

# ==========================================
# 4단계: 상호작용 및 데이터 수집 (Interaction)
# ==========================================
def play_one_step(env, obs, model):
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    probs = model(obs_tensor)
    
    # 확률 분포에 따라 행동 샘플링
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    
    next_obs, reward, done, truncated, info = env.step(action.item())
    return next_obs, reward, done, truncated, log_prob

def play_multiple_episodes(env, n_episodes, n_max_steps, model):
    all_rewards = []
    all_log_probs = []
    for episode in range(n_episodes):
        current_rewards = []
        current_log_probs = []
        obs, info = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, truncated, log_prob = play_one_step(env, obs, model)
            current_rewards.append(reward)
            current_log_probs.append(log_prob)
            if done or truncated:
                break
        all_rewards.append(current_rewards)
        all_log_probs.append(current_log_probs)
    return all_rewards, all_log_probs

# ==========================================
# 5단계: 보상 처리 - 할인 및 정규화 (Reward Processing)
# ==========================================
def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(dr - reward_mean) / (reward_std + 1e-7) for dr in all_discounted_rewards]

# ==========================================
# 6단계: 정책 경사 업데이트 (Optimization Loop)
# ==========================================
def train_policy_gradient():
    env = gym.make('CartPole-v1')
    n_iterations = 150
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_rate = 0.95
    
    model = PolicyNet(4, 5, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("학습을 시작합니다...")
    for iteration in range(n_iterations):
        all_rewards, all_log_probs = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model)
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        
        optimizer.zero_grad()
        loss_list = []
        for log_probs, final_rewards in zip(all_log_probs, all_final_rewards):
            for log_prob, reward in zip(log_probs, final_rewards):
                # 정책 경사 공식: -log(prob) * G
                loss_list.append(-log_prob * reward)
        
        loss = torch.stack(loss_list).sum()
        loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            avg_reward = sum(map(sum, all_rewards)) / n_episodes_per_update
            print(f'Iteration: {iteration}, Avg Reward: {avg_reward:.2f}')
    
    env.close()
    return model

# ==========================================
# 7단계: 결과 시각화 및 평가 (Evaluation)
# ==========================================
def render_and_save_animation(model, filename="cartpole_result.mp4"):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    frames = []
    
    for _ in range(200):
        frames.append(env.render())
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(obs_tensor)
            action = torch.argmax(probs).item() # 가장 높은 확률의 행동 선택
        
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()

    # 애니메이션 생성 및 저장
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def update(i):
        patch.set_data(frames[i])
        return [patch]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=40)
    # 주피터 환경이 아닐 경우 파일로 저장하거나 plt.show() 사용
    print(f"결과 영상을 저장 중입니다: {filename}")
    # anim.save(filename, writer='ffmpeg') # ffmpeg 설치 필요
    plt.show()

# 실행부
if __name__ == "__main__":
    trained_model = train_policy_gradient()
    render_and_save_animation(trained_model)