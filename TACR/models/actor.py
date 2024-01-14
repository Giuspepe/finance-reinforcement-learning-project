
import numpy as np
import torch
import torch.nn as nn
import transformers
from .gpt2 import GPT2Model

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, max_length=None, max_episode_length=8192, action_softmax=True):
        super(Actor, self).__init__()

        # Store arguments as object attributes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.max_episode_length = max_episode_length
        self.action_softmax = action_softmax

        # Configure GPT2 with the number of embeddings equal to the hidden size and vocab_size=1 since we don't use the vocabulary
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=5,
            n_head=1,
            n_inner=hidden_size*4,
            activation_function="relu",
            n_positions=1024
        )

        # Initialize GPT2
        self.transformer = GPT2Model(config)

        # Create a lookup table for each timestep
        self.embed_timestep_layer = nn.Embedding(self.max_episode_length, self.hidden_size)

        # Create embeddings for the MDP elements using Linear Layerss
        self.embed_return_layer = torch.nn.Linear(1, self.hidden_size)
        self.embed_state_layer = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action_layer = torch.nn.Linear(self.action_dim, self.hidden_size)
        
        # Create a Layer Normalization Layer
        self.embed_layernormalization = nn.LayerNorm(self.hidden_size)

        if action_softmax:
            # Create a Linear Layer followed by a Softmax to predict actions
            self.action_prediction_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.action_dim),
                nn.Softmax(dim=2)
            )
        else:
            self.action_prediction_layer = nn.Linear(self.hidden_size, self.action_dim) 


    def forward(self, states, actions, rewards, timesteps, attention_mask=None):
        # Get the batch size and sequence length
        batch_size, seq_length = states.shape[0], states.shape[1]

        # If no attention mask for GPT is given, assume all tokens can be attended to
        if attention_mask is None:
            # If an attention mask is given, then 1 if can be attended to and 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Embeddings for the MDP
        state_embeddings = self.embed_state_layer(states)
        action_embeddings = self.embed_action_layer(actions)
        returns_embeddings = self.embed_return_layer(rewards)
        time_embeddings = self.embed_timestep_layer(timesteps)

        # Positional embeddings (timesteps)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stacking MDP elements, so the sequence looks like
        # (r_1, x_1, a_1, r_2, x_2, a_2, ...)
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)

        # Applies Layer Normalization over the stacked inputs
        stacked_inputs = self.embed_layernormalization(stacked_inputs)

        # Stack the attention mask so it is compatible with the stacked inputs
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # Feed input embeddings to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )

        # Get the last hidden state
        x = transformer_outputs['last_hidden_state']
        

        # Reshape the last hidden state so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Get the action predictions
        action_preds = self.action_prediction_layer(x[:,1]) 
        return action_preds
    
    def get_action(self, states, actions, rewards, timesteps):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # Pad tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.action_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds = self.forward( states, actions, rewards, timesteps, attention_mask=attention_mask)

        return action_preds[0,-1]