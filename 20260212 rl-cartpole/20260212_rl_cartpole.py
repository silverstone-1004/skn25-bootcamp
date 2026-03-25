#수정해야함

pip install gymnasium

import matplotlib.animation
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('animation', html='jshtml')

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from pathlib import Path

IMAGES_PATH = Path() / "images" / "rl"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=
fig_extension, dpi=resolution)


import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
envs = gym.envs.registry
envs["CartPole-v1"]
obs, info = env.reset(seed=42)

pip install "gymnasium[classic-control]


img = env.render()

plt.imshow(img)



def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render()
    plt.imshow(img)
    plt.axis("off")
    return img


plot_environment(env)
plt.show()


right_action = 1
env.step(right_action)



def basic_policy(obs):
    angle = obs[2]
    return 0   if angle < 0 else 1

totals = []
for episode in  range(500):
    episode_rewards = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        obs, reward, done, truncated, info = env.step(basic_policy(obs))
        episode_rewards += reward
        if  done or truncated:
            break
    totals.append(episode_rewards)


import numpy as np
print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = matplotlib.animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def show_one_episode(policy, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    for step in range(n_max_steps):
        frames.append(env.render())
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()
    return plot_animation(frames)

show_one_episode(basic_policy)



import torch.nn as nn 
class BasicPolicyNet(nn.Module):
    def __init__(self, obs_size, n_hidden, n_outputs):
        super(BasicPolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


env = gym.make('CartPole-v1')
obs, info = env.reset()
model = BasicPolicyNet(env.observation_space.shape[0], 5, 2).to(device)


obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)

probs = model(obs_tensor)

action = torch.multinomial(probs, num_samples=1).item()



def play_one_step(env, obs, model, loss_fn):
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    probs = model(obs_tensor)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    next_obs, reward, done, truncated, info = env.step(action.item())
    return next_obs, reward, done, truncated, log_prob


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_log_probs = []
    for episode in range(n_episodes):
        current_rewards = []
        current_log_probs = []
        obs, info = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, truncated, log_prob = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_log_probs.append(log_prob)
            if done or truncated:
                break
        all_rewards.append(current_rewards)
        all_log_probs.append(current_log_probs)
    return all_rewards, all_log_probs



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
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_rate = 0.95
learning_rate = 0.01

model = BasicPolicyNet(4, 5, 2).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



for iteration in range(n_iterations):
    all_rewards, all_log_probs = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, None)
    
    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
    
    optimizer.zero_grad()
    policy_loss = []
    for log_probs, final_rewards in zip(all_log_probs, all_final_rewards):
        for log_prob, reward in zip(log_probs, final_rewards):
            # Policy Gradient Loss: -log(prob) * return
            policy_loss.append(-log_prob * reward)
            
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
    
    if iteration % 10 == 0:
        avg_reward = sum(map(sum, all_rewards)) / n_episodes_per_update
        print(f'Iteration: {iteration}, Avg Reward: {avg_reward:.2f}')




from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import HTML

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval
    )
    plt.close()
    return anim

def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    # 렌더링을 위해 render_mode='rgb_array' 설정 필수
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    obs, info = env.reset(seed=seed)
    
    for step in range(n_max_steps):
        frames.append(env.render())
        
        # PyTorch 모델로 행동 결정
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            # DQN인 경우와 PolicyNet인 경우 분기 처리
            q_values = model(obs_tensor)
            if q_values.shape[-1] > 1: # Softmax 출력인 경우 (Policy Net)
                 action = torch.argmax(q_values).item()
            else: # Q-Value 출력인 경우 (DQN)
                 action = torch.argmax(q_values).item()
        
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()
    return frames

frames = render_policy_net(model)
anim = plot_animation(frames)
HTML(anim.to_jshtml())
