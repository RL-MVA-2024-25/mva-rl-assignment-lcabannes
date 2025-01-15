from datetime import datetime
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient  # Ensure this is your custom environment
from fast_env import FastHIVPatient
from collections import deque
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Settings:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    net_arch_pi: list[int] = None
    net_arch_vf: list[int] = None


def affine_schedule(y_1, y_0):
    """
    Returns a function that gradually goes from y_1 to y_0
    as 'progress_remaining' goes from 1.0 to 0.0.
    Adjust this as you wish.
    """
    def _schedule(progress_remaining):
        return y_0 + (y_1 - y_0) * progress_remaining

    return _schedule

def train_ppo():
    now = datetime.now()
    log_dir = f"./logs/{now.strftime('%Y-%m-%d_%H:%M:%S')}"

    nb_envs = 8
    vec_env = make_vec_env(
        lambda: TimeLimit(
            env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        n_envs=nb_envs,
        vec_env_cls=SubprocVecEnv,
    )

    vf_size = 1024
    pi_size = 1024
    settings = Settings(
        net_arch_pi=[pi_size for layer in range(6)], net_arch_vf=[vf_size for layer in range(6)]
    )
    dqn = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=affine_schedule(
            y_1=settings.learning_rate,  # progress_remaining=1.0
            y_0=settings.learning_rate / 3,  # progress_remaining=0.0
        ),
        batch_size=settings.batch_size,
        gamma=settings.gamma,
        tensorboard_log=log_dir,
        policy_kwargs={
            "activation_fn": nn.ReLU,
            "net_arch": settings.net_arch_vf,
        },
        verbose=1,
        device="cuda",
    ).learn(
        total_timesteps=5_000_000,
        callback=[
            CheckpointCallback(
                max(200_000 // nb_envs, 1_000),
                name_prefix="checkpoint",
                save_path=log_dir,
                verbose=1,
            )
        ],
    )

    dqn.save(f"{log_dir}/ppo_hiv_patient.zip")



def train_dqn():
    now = datetime.now()
    log_dir = f"./logs/{now.strftime('%Y-%m-%d_%H:%M:%S')}"

    nb_envs = 8
    vec_env = make_vec_env(
        lambda: TimeLimit(
            env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        n_envs=nb_envs,
        vec_env_cls=SubprocVecEnv,
    )

    vf_size = 1024
    pi_size = 1024
    settings = Settings(
        net_arch_pi=[pi_size for layer in range(6)], net_arch_vf=[vf_size for layer in range(6)]
    )
    dqn = DQN(
        "MlpPolicy",
        vec_env,
        target_update_interval=400,
        tau=0.01,
        gradient_steps=50, # 5,
        learning_rate=affine_schedule(
            y_1=settings.learning_rate,  # progress_remaining=1.0
            y_0=settings.learning_rate / 3,  # progress_remaining=0.0
        ),
        batch_size=settings.batch_size,
        gamma=settings.gamma,
        tensorboard_log=log_dir,
        policy_kwargs={
            "activation_fn": nn.ReLU,
            "net_arch": settings.net_arch_vf,
        },
        verbose=1,
        # device="cuda",
    ).learn(
        total_timesteps=5_000_000,
        callback=[
            CheckpointCallback(
                max(200_000 // nb_envs, 1_000),
                name_prefix="checkpoint",
                save_path=log_dir,
                verbose=1,
            )
        ],
    )

    dqn.save(f"{log_dir}/dqn_hiv_patient.zip")


class StackObsActionWrapper(gym.ObservationWrapper):
    def __init__(self, env, stack_size: int = 4):
        super().__init__(env)
        self.stack_size = stack_size
        self.obs_shape = self.observation_space.shape
        self.action_dim = self.action_space.n
        self.buffer_obs = deque(maxlen=stack_size)
        self.buffer_actions = deque(maxlen=stack_size)
        self.cur_step = 0

        # Update observation space to include stacked observations and actions
        obs_dim = np.prod(self.obs_shape)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(stack_size * (obs_dim + self.action_dim),), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buffer_obs = deque([obs] * self.stack_size, maxlen=self.stack_size)
        self.buffer_actions = deque([np.zeros(self.action_dim)] * self.stack_size, maxlen=self.stack_size)
        return self._get_stacked_state(), info

    def observation(self, obs):
        return self._get_stacked_state()

    def step(self, action):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        obs, reward, done, truncated, info = self.env.step(action)

        self.buffer_obs.append(obs)
        self.buffer_actions.append(one_hot_action)

        self.cur_step += 1
        if self.cur_step == 200:
            self.reset()

        return self._get_stacked_state(), reward, done, truncated, info

    def _get_stacked_state(self):
        stacked_obs = np.concatenate(self.buffer_obs, axis=None)
        stacked_actions = np.concatenate(self.buffer_actions, axis=None)
        return np.concatenate([stacked_obs, stacked_actions], axis=None)


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class special_DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        layers = []
        self.fc3 = nn.Linear(128, action_space)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ProjectAgent():

    def __init__(self):
        self.model = None
        
    def act(self, observation, use_random=False):
        # print(observation)
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def save(self, path):
        torch.save(self.state_dict(), "model.pth")

    def load(self):
        self.model = DQN.load("checkpoints/model.zip", device="cpu")
    
    def forward(self, obs):
        policy = self.actor(obs)
        value = self.critic(obs)
        return policy, value

def compute_advantages(rewards: list[float], values: list[float], dones: list[float], gamma: float = 0.99, lambd: float = 0.95):
    advantages = []
    gae = 0
    values.append(0)

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambd * (1 - dones[t]) * gae 
        advantages.append(gae)

    advantages = advantages[::-1]
    return advantages 

if __name__ == "__main__":

    train_ppo()
    train_dqn()
