import os
import sys

# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_of_parent_dir)

import gymnasium
from TACR.utils.utils import Batch
from TACR.tacr import TACR
import numpy as np
import torch
import random
from TACR.utils.log import TensorBoardHandler

class Trainer:
    def __init__(
        self, agent: TACR, env: gymnasium.Env
    ):
        """
        Initialize the Trainer object.

        Parameters:
        - agent (TACR): The agent to be trained.
        - env (gymnasium.Env): The environment in which the agent will be trained.

        This class is responsible for training an agent in a specified environment. It handles the training iterations,
        logging, and batch generation.
        """
        self.agent = agent
        self.env = env
        self.total_it = 0
        self.tb = TensorBoardHandler()

    def train_it(self, num_steps: int):
        """
        Train the agent for a specified number of steps.

        Parameters:
        - num_steps (int): Number of training steps to perform.

        Returns:
        - dict: A dictionary containing training metrics.

        This method performs training iterations, updates the agent, and logs metrics to TensorBoard.
        """
        train_metrics = {
            "critic_loss": [],
            "actor_loss": [],
            "mean_critic_predictions": [],
            "average_critic_estimate": [],
        }
        self.agent.actor.train()
        for _ in range(num_steps):
            self.total_it += 1
            metrics = self.agent.update(
                self.get_batch(self.agent.config.batch_size, self.agent.config.lookback)
            )

            # Log to TensorBoard
            self.tb.log_scalar("Train Critic Loss", metrics["critic_loss"], self.total_it)
            self.tb.log_scalar("Train Actor Loss", metrics["actor_loss"], self.total_it)
            self.tb.log_scalar(
                "Train Mean Critic Predictions (Predictions)",
                metrics["mean_critic_predictions"],
                self.total_it,
            )
            self.tb.log_scalar(
                "Train Average Critic Estimate (Q-values)",
                metrics["average_critic_estimate"],
                self.total_it,
            )

            train_metrics["critic_loss"].append(metrics["critic_loss"])
            train_metrics["actor_loss"].append(metrics["actor_loss"])
            train_metrics["mean_critic_predictions"].append(
                metrics["mean_critic_predictions"]
            )
            train_metrics["average_critic_estimate"].append(
                metrics["average_critic_estimate"]
            )

            # print(f"Total Steps: {self.total_it}, Train Loss: {train_loss}")

        return train_metrics
            


    def finish_training(self):
        """
        Close the TensorBoard handler after training.

        This method should be called after training to properly close the TensorBoard logging resources.
        """
        self.tb.close()

    def get_batch(self, batch_size: int, lookback: int) -> Batch:
        """
        Generate a batch of data for training.

        Parameters:
        - batch_size (int): Size of the batch to generate.
        - lookback (int): Number of past states to include in the batch.

        Returns:
        - Batch: A batch of data for training.

        This method creates a batch of data by sampling from the training trajectories. It handles state normalization,
        action padding, and timestep handling for the batch.
        """
        batch_inds = np.random.choice(
            np.arange(len(self.agent.config.train_traj_lenghts)),
            size=batch_size,
            replace=True,
        )

        s, next_s, next_a, next_r, a, r, d, dd, timesteps, n_timesteps, mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(batch_size):
            traj = self.agent.config.train_trajectories[int(batch_inds[i])]
            si = random.randint(0, traj.rewards.shape[0] - 1)

            # Get sequences from dataset
            s.append(
                traj.observations[si : si + lookback].reshape(
                    1, -1, self.agent.config.state_dim
                )
            )
            a.append(
                traj.actions[si : si + lookback].reshape(
                    1, -1, self.agent.config.action_dim
                )
            )
            r.append(traj.rewards[si : si + lookback].reshape(1, -1, 1))
            dd.append(traj.dones[si : si + lookback].reshape(1, -1, 1))

            if si >= traj.rewards.shape[0] - lookback:
                next_s.append(
                    np.append(
                        traj.observations[si + 1 : si + 1 + lookback],
                        traj.observations[traj.rewards.shape[0] - 1],
                    ).reshape(1, -1, self.agent.config.state_dim)
                )
                next_a.append(
                    np.append(
                        traj.actions[si + 1 : si + 1 + lookback],
                        traj.actions[traj.rewards.shape[0] - 1],
                    ).reshape(1, -1, self.agent.config.action_dim)
                )
                next_r.append(
                    np.append(
                        traj.rewards[si + 1 : si + 1 + lookback],
                        np.array([traj.rewards[traj.rewards.shape[0] - 1]]),
                    ).reshape(1, -1, 1)
                )
            else:
                next_s.append(
                    traj.observations[si + 1 : si + 1 + lookback].reshape(
                        1, -1, self.agent.config.state_dim
                    )
                )
                next_a.append(
                    traj.actions[si + 1 : si + 1 + lookback].reshape(
                        1, -1, self.agent.config.action_dim
                    )
                )
                next_r.append(
                    traj.rewards[si + 1 : si + 1 + lookback].reshape(1, -1, 1)
                )

            d.append(traj.dones[si : si + lookback].reshape(1, -1))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.agent.config.max_episode_length] = (
                self.agent.config.max_episode_length - 1
            )  # padding cutoff
            n_timesteps.append(
                np.arange(si + 1, si + 1 + s[-1].shape[1]).reshape(1, -1)
            )
            n_timesteps[-1][n_timesteps[-1] >= self.agent.config.max_episode_length] = (
                self.agent.config.max_episode_length - 1
            )  # padding cutoff

            # padding and state normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, lookback - tlen, self.agent.config.state_dim)), s[-1]],
                axis=1,
            )
            s[-1] = (s[-1] - self.agent.config.state_mean) / self.agent.config.state_std
            next_s[-1] = np.concatenate(
                [
                    np.zeros((1, lookback - tlen, self.agent.config.state_dim)),
                    next_s[-1],
                ],
                axis=1,
            )
            next_s[-1] = (next_s[-1] - self.agent.config.state_mean) / self.agent.config.state_std
            a[-1] = np.concatenate(
                [
                    np.ones((1, lookback - tlen, self.agent.config.action_dim)) * -10.0,
                    a[-1],
                ],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, lookback - tlen, 1)), r[-1]], axis=1)
            dd[-1] = np.concatenate([np.zeros((1, lookback - tlen, 1)), dd[-1]], axis=1)
            next_a[-1] = np.concatenate(
                [
                    np.ones((1, lookback - tlen, self.agent.config.action_dim)) * -10.0,
                    next_a[-1],
                ],
                axis=1,
            )
            next_r[-1] = np.concatenate(
                [np.zeros((1, lookback - tlen, 1)), next_r[-1]], axis=1
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, lookback - tlen)), timesteps[-1]], axis=1
            )
            n_timesteps[-1] = np.concatenate(
                [np.zeros((1, lookback - tlen)), n_timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, lookback - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=self.agent.device
        )
        next_s = torch.from_numpy(np.concatenate(next_s, axis=0)).to(
            dtype=torch.float32, device=self.agent.device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=self.agent.device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=self.agent.device
        )
        dd = torch.from_numpy(np.concatenate(dd, axis=0)).to(
            dtype=torch.float32, device=self.agent.device
        )
        next_a = torch.from_numpy(np.concatenate(next_a, axis=0)).to(
            dtype=torch.float32, device=self.agent.device
        )
        next_r = torch.from_numpy(np.concatenate(next_r, axis=0)).to(
            dtype=torch.float32, device=self.agent.device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=self.agent.device
        )
        n_timesteps = torch.from_numpy(np.concatenate(n_timesteps, axis=0)).to(
            dtype=torch.long, device=self.agent.device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(
            device=self.agent.device
        )
        
        # Convert the collected data into tensors and return as a Batch object
        batch = Batch(
            states=s,
            actions=a,
            rewards=r,
            dones=dd,
            next_state=next_s,
            next_actions=next_a,
            next_rewards=next_r,
            timesteps=timesteps,
            next_timesteps=n_timesteps,
            attention_mask=mask,
        )
        return batch
