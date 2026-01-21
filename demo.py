from __future__ import annotations
 
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional
 
import gymnasium as gym
import numpy as np
import torch
import tyro
from rsl_rl.modules import ActorCritic
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from tensordict import TensorDict
from torch import nn
 
 
class GymnasiumVecEnv:
    """Minimal VecEnv adapter for rsl_rl using gymnasium.vector.SyncVectorEnv."""
 
    def __init__(self, env_id: str, num_envs: int, seed: int, device: torch.device):
        def make_env(rank: int) -> Callable[[], gym.Env]:
            def _make() -> gym.Env:
                env = gym.make(env_id)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env
 
            return _make
 
        self.env = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])
        self.device = device
        self.num_envs = num_envs
        self._seeds = [seed + i for i in range(num_envs)]
        self.cfg = {"env_id": env_id, "num_envs": num_envs, "seed": seed}
 
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space
        if not isinstance(obs_space, gym.spaces.Box) or not isinstance(act_space, gym.spaces.Box):
            raise ValueError("This example only supports continuous Box spaces.")
        if len(obs_space.shape) != 1 or len(act_space.shape) != 1:
            raise ValueError("This example only supports 1D observation/action spaces.")
 
        self.num_obs = obs_space.shape[0]
        self.num_privileged_obs = 0
        self.num_actions = act_space.shape[0]
        self.max_episode_length = self.env.envs[0].spec.max_episode_steps or 1000
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._obs = None
        self._obs_td = None
        self.reset()
 
    def _as_tensordict(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({"obs": obs}, batch_size=[self.num_envs], device=obs.device)
 
    def reset(self):
        obs, _ = self.env.reset(seed=self._seeds)
        self._obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self._obs_td = self._as_tensordict(self._obs)
        self.episode_length_buf.zero_()
        return self._obs_td
 
    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        obs, rewards, terminated, truncated, infos = self.env.step(actions_np)
        dones = np.logical_or(terminated, truncated)
        self._obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self._obs_td = self._as_tensordict(self._obs)
        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.bool)
        infos_out = dict(infos) if isinstance(infos, dict) else {}
        infos_out["time_outs"] = torch.tensor(truncated, device=self.device, dtype=torch.bool)
        self.episode_length_buf += 1
        self.episode_length_buf[dones_t] = 0
        return self._obs_td, rewards_t, dones_t, infos_out
 
    def get_observations(self):
        return self._obs_td
 
    def get_privileged_observations(self):
        return None
 
 
def _activation_from_name(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError(f"Unsupported activation: {name}")
 
 
def _build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int], activation: str) -> nn.Sequential:
    act = _activation_from_name(activation)
    layers: List[nn.Module] = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)
 
 
class CustomActorCritic(ActorCritic):
    """Example Actor-Critic with easy-to-edit actor/critic definitions."""
 
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, List[str]],
        num_actions: int,
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        activation: str,
        init_noise_std: float,
        state_dependent_std: bool = False,
        **kwargs,
    ):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )
        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
        # Replace these two lines with your own nn.Module definitions as needed.
        self.actor = _build_mlp(num_actor_obs, actor_out_dim, actor_hidden_dims, activation)
        self.critic = _build_mlp(num_critic_obs, 1, critic_hidden_dims, activation)
 
 
@dataclass
class EnvConfig:
    env_id: str = "Hopper-v4"
    num_envs: int = 8
    seed: int = 1
 
 
@dataclass
class PolicyConfig:
    class_name: str = "CustomActorCritic"
    # init_noise_std: float = 1.0
    init_noise_std: float = 0.2
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
 
 
@dataclass
class AlgorithmConfig:
    class_name: str = "PPO"
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.0
    num_learning_epochs: int = 10
    num_mini_batches: int = 32
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 0.5
 
 
@dataclass
class RunnerConfig:
    num_steps_per_env: int = 256
    max_iterations: int = 1000
    save_interval: int = 50
 
 
@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    log_dir: str = "runs/rsl_rl_hopper"
    device: str = "auto"
 
 
def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
 
 
def _build_train_cfg(cfg: TrainConfig) -> dict:
    return {
        "seed": cfg.env.seed,
        "obs_groups": {
            "policy": ["obs"],
            "critic": ["obs"],
        },
        "policy": {
            "class_name": cfg.policy.class_name,
            "init_noise_std": cfg.policy.init_noise_std,
            "actor_hidden_dims": cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": cfg.policy.critic_hidden_dims,
            "activation": cfg.policy.activation,
        },
        "algorithm": {
            "class_name": cfg.algorithm.class_name,
            "value_loss_coef": cfg.algorithm.value_loss_coef,
            "use_clipped_value_loss": cfg.algorithm.use_clipped_value_loss,
            "clip_param": cfg.algorithm.clip_param,
            "entropy_coef": cfg.algorithm.entropy_coef,
            "num_learning_epochs": cfg.algorithm.num_learning_epochs,
            "num_mini_batches": cfg.algorithm.num_mini_batches,
            "learning_rate": cfg.algorithm.learning_rate,
            "schedule": cfg.algorithm.schedule,
            "gamma": cfg.algorithm.gamma,
            "lam": cfg.algorithm.lam,
            "desired_kl": cfg.algorithm.desired_kl,
            "max_grad_norm": cfg.algorithm.max_grad_norm,
            "rnd_cfg": None,
        },
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "max_iterations": cfg.runner.max_iterations,
    }
 
 
def main(cfg: TrainConfig):
    device = _resolve_device(cfg.device)
    random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
 
    env = GymnasiumVecEnv(cfg.env.env_id, cfg.env.num_envs, cfg.env.seed, device)
    train_cfg = _build_train_cfg(cfg)
 
    # Make CustomActorCritic discoverable by OnPolicyRunner.
    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as rsl_runner
 
    rsl_modules.CustomActorCritic = CustomActorCritic
    rsl_runner.CustomActorCritic = CustomActorCritic
 
    runner = OnPolicyRunner(
        env, train_cfg, log_dir=f"{cfg.log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}", device=device
    )
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)
 
 
if __name__ == "__main__":
    main(tyro.cli(TrainConfig))