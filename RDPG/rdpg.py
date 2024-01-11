import numpy as np
import torch
import torch.optim as optim

from networks.history import RecordedHistory
from networks.actor import ActorMLP
from buffer import RecurrentBatch
from networks.critic import CriticMLP
from utils import get_device, save_model, load_model, get_target_network, polyak_update
from noise import OrnsteinUhlenbeckNoise as noise


class RDPG:
    """
    Recurrent Deterministic Policy Gradient (RDPG) class for reinforcement learning.

    RDPG is an extension of the DDPG algorithm, incorporating recurrent neural networks
    (RNNs) to handle partially observable environments or environments with long-term dependencies.

    Attributes:
        input_dim (int): Dimensionality of the input space.
        action_dim (int): Dimensionality of the action space.
        hidden_dim (int): Size of the hidden layer in the neural networks.
        gamma (float): Discount factor for future rewards.
        lr (float): Learning rate for the optimizers.
        tau (float): Coefficient for Polyak averaging in updating target networks.
        action_noise (float): Standard deviation of the Gaussian noise added to the actions.
        device (torch.device): The device (CPU or GPU) to run the computations on.
        hidden (torch.Tensor): Hidden state for the recurrent neural network.

        actor_rh (RecordedHistory): Recurrent network to process observations for the actor.
        actor_rh_target (RecordedHistory): Target network for actor_rh.
        actor (ActorMLP): The actor network predicting actions.
        actor_target (ActorMLP): Target network for the actor.

        critic_rh (RecordedHistory): Recurrent network to process observations for the critic.
        critic_rh_target (RecordedHistory): Target network for critic_rh.
        critic (CriticMLP): The critic network estimating the value function.
        critic_target (CriticMLP): Target network for the critic.

        actor_optimizer (torch.optim.Adam): Optimizer for the actor network.
        actor_rh_optimizer (torch.optim.Adam): Optimizer for the actor's recorded history network.
        critic_optimizer (torch.optim.Adam): Optimizer for the critic network.
        critic_rh_optimizer (torch.optim.Adam): Optimizer for the critic's recorded history network.

    Methods:
        reset_hidden: Resets the hidden state of the RNNs.
        get_action: Computes and returns the action for a given observation.
        update: Performs a training step using a batch of data.
        save_actor: Saves the actor model and its recorded history.
        load_actor: Loads the actor model and its recorded history.

    RDPG is well-suited for tasks where the agent's observation at a single time step does not
    provide full information about the environment's state (i.e., partially observable environments).
    The recurrent component allows the agent to maintain internal states that help in capturing
    information over multiple time steps.
    """

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        lr=3e-4,
        tau=0.995,
        action_noise=0.1,
    ):
        """
        Initializes the RDPG agent.

        Args:
            input_dim (int): Dimensionality of the input space.
            action_dim (int): Dimensionality of the action space.
            hidden_dim (int, optional): Size of the hidden layer in the neural networks. Defaults to 256.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            lr (float, optional): Learning rate for the optimizers. Defaults to 3e-4.
            tau (float, optional): Coefficient for Polyak averaging in updating target networks. Defaults to 0.995.
            action_noise (float, optional): Standard deviation of the Gaussian noise added to the actions. Defaults to 0.1.

        This method initializes the RDPG agent by setting up the actor and critic networks
        along with their corresponding target networks. It also initializes the optimizers for
        these networks and sets the necessary hyperparameters.

        The `input_dim` and `action_dim` parameters define the shape of the input and output
        for the networks. The `hidden_dim` parameter is used to determine the size of the hidden
        layers in the actor and critic networks.

        The `gamma`, `lr`, and `tau` parameters are standard hyperparameters in reinforcement
        learning, controlling the discounting of future rewards, the learning rate of the
        optimizers, and the rate of updating the target networks, respectively.

        Additionally, the agent's device is set based on the availability of GPU, and
        initial hidden states for the recurrent networks are set to None.
        """
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.action_noise = action_noise

        self.device = get_device()

        self.hidden = None

        # rh = recorded history

        # Actor networks (local and target)
        self.actor_rh = RecordedHistory(input_dim, hidden_dim).to(self.device)
        self.actor_rh_target = get_target_network(self.actor_rh)
        self.actor = ActorMLP(hidden_dim, action_dim).to(self.device)
        self.actor_target = get_target_network(self.actor)

        # Critic networks (local and target)
        self.critic_rh = RecordedHistory(input_dim, hidden_dim).to(self.device)
        self.critic_rh_target = get_target_network(self.critic_rh)
        self.critic = CriticMLP(hidden_dim, action_dim).to(self.device)
        self.critic_target = get_target_network(self.critic)

        # Actor optimizer (local and target)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor_rh_optimizer = optim.Adam(self.actor_rh.parameters(), lr=self.lr)

        # Critic optimizer (local and target)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_rh_optimizer = optim.Adam(self.critic_rh.parameters(), lr=self.lr)

        # Initialize noise
        self.exp_mu = 0
        self.exp_theta = 0.05
        self.exp_sigma = 0.25
        self.noise = noise(
            size=action_dim, mu=self.exp_mu, theta=self.exp_theta, sigma=self.exp_sigma
        )

    def reset_hidden(self):
        """
        Resets the hidden state of the RDPG agent's recurrent networks.

        This is typically done at the beginning of each new episode during training or when the agent starts
        interacting with the environment. Clearing the hidden state is crucial for episodic tasks, as it ensures that the
        learning from one episode does not carry over to the next. This reset maintains the integrity of the episodic learning process,
        preventing information leakage between episodes.
        """
        self.hidden = None

    def get_action(self, observation, deterministic=True):
        """
        Computes and returns the action for a given observation using the actor network.

        Args:
            observation (array-like): The current observation from the environment.

        Returns:
            action (numpy.ndarray): The action determined by the actor network.
        """
        with torch.no_grad():
            # Dimension of observation is (1, 1, hidden_dim)
            observation = (
                torch.tensor(observation)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(get_device())
            )
            rh, self.hidden = self.actor_rh(observation, self.hidden)

            if torch.isnan(rh).any():
                print("ERROR: Actor RH has NaNs")
                return np.zeros(self.action_dim)
            
            action = self.actor(rh).view(-1).detach().cpu().numpy()

            # For deterministic policy, we don't need to sample from the distribution
            if deterministic:
                return action
            else:
                noise = self.noise.sample()
                action = action = np.clip(action * 0.2 + noise, -1, 1)

            return action

    def update(self, batch: RecurrentBatch):
        """
        Performs a training step using a batch of data.

        Args:
            batch (namedtuple): A batch of experiences, typically containing observations,
                                actions, rewards, next observations, and done flags.

        This method updates both the actor and critic networks using the provided batch of
        experiences. It involves several key steps:

        1. Processing the batch data through both the actor and critic recorded history networks
           to generate the required inputs for the policy and value function updates.

        2. Calculating the target values for the critic update using the target networks,
           which are more stable versions of the main networks. This is in line with the
           Temporal Difference (TD) learning approach and is crucial for stable training.

        3. Computing the loss for the critic network based on the difference (TD error)
           between the predicted and target values. The loss is then backpropagated to update
           the critic networks.

        4. Updating the actor network by maximizing the expected return, as estimated by the critic network.
           This involves computing a policy loss and performing backpropagation to update the actor networks.

        5. Utilizing Polyak averaging to update the target networks. This step ensures that the target networks
           change slowly, which contributes to the overall stability of the learning process.

        The method returns a dictionary containing metrics such as the critic loss and
        the average Q values, which can be useful for monitoring the training process.
        """
        # Get recorded history from actor and critic
        actor_rh, _ = self.actor_rh(batch.observations)
        critic_rh, _ = self.critic_rh(batch.observations)

        # Get recorded history from actor and critic target
        actor_rh_target, _ = self.actor_rh_target(batch.observations)
        critic_rh_target, _ = self.critic_rh_target(batch.observations)

        # Original sequence from the actor network
        # This gives us all actions/states from the start up to the second-to-last time step
        actor_rh_1_T = actor_rh[:, :-1, :]

        # Sequence from the target actor network
        # This gives us all actions/states from the second time step to the end
        actor_rh_2_Tplus1 = actor_rh_target[:, 1:, :]

        # Same as above, but for the critic network
        critic_rh_1_T = critic_rh[:, :-1, :]
        critic_rh_2_Tplus1 = critic_rh_target[:, 1:, :]

        # Get the predicted values from the critic network
        preds = self.critic(critic_rh_1_T, batch.actions)

        # Get the target values from the critic network
        # Note that we use the target actor network to get the actions
        # This is because we want to use the target actor network to get the next state
        # This is the same as the original DDPG paper
        next_actions = self.actor(actor_rh_2_Tplus1)
        targets = batch.rewards + self.gamma * (1 - batch.done) * self.critic_target(
            critic_rh_2_Tplus1, next_actions
        )

        # Compute the loss for the critic network
        # Similar to TD error
        critic_loss = (preds - targets).pow(2)
        # Mask out the loss for the padded time steps (i.e. where the mask is 0)
        critic_loss = (
            torch.mean(critic_loss * batch.mask)
            / batch.mask.sum()
            * np.prod(batch.mask.shape)
        )

        # Minimize the critic loss
        self.critic_rh_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Apply gradient clipping for critic_rh
        torch.nn.utils.clip_grad_norm_(self.critic_rh.parameters(), max_norm=5.0)
    
        self.critic_rh_optimizer.step()
        self.critic_optimizer.step()

        # Get the predicted actions from the actor network
        # Note that we use the actor recorded history
        actions = self.actor(actor_rh_1_T)
        Q_values = self.critic(critic_rh_1_T.detach(), actions)
        pi_loss = -Q_values
        # Mask out the loss for the padded time steps (i.e. where the mask is 0)
        pi_loss = (
            torch.mean(pi_loss * batch.mask)
            / batch.mask.sum()
            * np.prod(batch.mask.shape)
        )

        # Minimize the actor loss
        self.actor_rh_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        pi_loss.backward()

        # Apply gradient clipping for actor_rh
        torch.nn.utils.clip_grad_norm_(self.actor_rh.parameters(), max_norm=5.0)
  
        self.actor_rh_optimizer.step()
        self.actor_optimizer.step()

        # Update the target networks using Polyak averaging
        # This ensures that the target networks are slowly updated
        polyak_update(self.actor_rh, self.actor_rh_target, self.tau)
        polyak_update(self.actor, self.actor_target, self.tau)

        polyak_update(self.critic_rh, self.critic_rh_target, self.tau)
        polyak_update(self.critic, self.critic_target, self.tau)

        # Return metrics for logging
        return {
            "mean_critic_predictions": torch.mean(preds * batch.mask).item(),
            "critic_loss": critic_loss.item(),
            "average_critic_estimate": torch.mean(Q_values * batch.mask).item(),
            "actor_loss": pi_loss.item(),
        }

    def save_actor(self, path, name="actor"):
        """
        Saves the actor model and its recorded history to the specified path.

        Args:
            path (str): The directory path where the model should be saved.
            name (str, optional): The base name for the saved model files. Defaults to "actor".
        """
        save_model(self.actor, path, name)
        save_model(self.actor_rh, path, name + "_rh")

    def load_actor(self, path, name="actor"):
        """
        Loads the actor model and its recorded history from the specified path.

        Args:
            path (str): The directory path from where the model should be loaded.
            name (str, optional): The base name of the model files to be loaded. Defaults to "actor".
        """
        self.actor = load_model(self.actor, path, name)
        self.actor_rh = load_model(self.actor_rh, path, name + "_rh")

    def copy_network(self, rdpg):
        self.actor_rh.load_state_dict(rdpg.actor_rh.state_dict())
        self.actor_rh_target.load_state_dict(rdpg.actor_rh_target.state_dict())

        self.critic_rh.load_state_dict(rdpg.critic_rh.state_dict())
        self.critic_rh_target.load_state_dict(rdpg.critic_rh_target.state_dict())

        self.actor.load_state_dict(rdpg.actor.state_dict())
        self.actor_target.load_state_dict(rdpg.actor_target.state_dict())

        self.critic.load_state_dict(rdpg.critic.state_dict())
        self.critic_target.load_state_dict(rdpg.critic_target.state_dict())
