from typing import List

from TACR.trajectory.trajectory import Trajectory


class TACRConfig:
    def __init__(
        self,
        lookback: int = 20,
        alpha: float = 0.9,
        pct_traj: float = 1.0,
        batch_size: int = 64,
        embed_dim: int = 128,
        critic_lr: float = 1e-6,
        actor_lr: float = 1e-6,
        weight_decay: float = 1e-4,
        warmup_steps: int = 10000,
        max_iterations: int = 10,
        n_steps_per_iteration: int = 4000,
        gamma=0.99,
        tau=0.005,
        state_dim: int = 2,
        action_dim: int = 1,
        embed_size: int = 128,
        max_episode_length: int = 8192,
        action_softmax: bool = True,
        action_softmax_dim: int = 2,
        train_traj_states: list = [],
        train_traj_lenghts: list = [],
        train_traj_returns: list = [],
        train_trajectories: List[Trajectory] = [],
        state_mean: float = 0.0,
        state_std: float = 1.0,
    ):
        self.lookback = lookback
        self.alpha = alpha
        self.pct_traj = pct_traj
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_iterations = max_iterations
        self.n_steps_per_iteration = n_steps_per_iteration

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_size = embed_size
        self.max_episode_length = max_episode_length
        self.action_softmax = action_softmax
        self.action_softmax_dim = action_softmax_dim
        self.gamma = gamma
        self.tau = tau

        self.train_traj_states = train_traj_states
        self.train_traj_lenghts = train_traj_lenghts
        self.train_traj_returns = train_traj_returns
        self.train_trajectories = train_trajectories

        self.state_mean = state_mean
        self.state_std = state_std
