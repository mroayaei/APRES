import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical
from utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class FCNetwork(nn.Module):
    def __init__(self, dims, out_layer=None):
        """
        Creates a network using ReLUs between layers and no activation at the end
        :param dims: tuple in the form of (100, 100, ..., 5). for dim sizes
        """
        super().__init__()
        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(nn.ReLU())
            mods.append(nn.Linear(h_sizes[i], h_sizes[i + 1]))

        if out_layer:
            mods.append(out_layer)

        self.layers = nn.Sequential(*mods)

    def forward(self, x):
        # Feedforward
        return self.layers(x)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

class RelevanceNetwork(nn.Module):
    """
    Predicts a scalar weight in (0,1) indicating how useful agent j's
    experiences are for agent i, given a joint observation.
    """
    
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # output in (0,1)
        )

    def forward(self, obs_i, obs_j):
        """
        obs_i: (batch, obs_dim) or (obs_dim)
        obs_j: (batch, obs_dim) or (obs_dim)
        Returns weight(s) in (0,1) of shape (batch, 1)
        """
        # Ensure both have same batch dimension
        if obs_i.dim() == 1:
            obs_i = obs_i.unsqueeze(0)
        if obs_j.dim() == 1:
            obs_j = obs_j.unsqueeze(0)
        x = torch.cat([obs_i, obs_j], dim=-1)
        return self.net(x)
class AdaptiveLambda(nn.Module):
    """
    Learns adaptive λ coefficient based on agent's hidden state and training progress.
    Outputs λ ∈ (0, 1) indicating how much to trust others.
    
    Input:
        hidden_state: (batch, hidden_dim) - from agent's policy network
        t: scalar or (1,) - normalized training progress (0 to 1)
    
    Output:
        λ: (batch, 1) - adaptive coefficient in (0, 1)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),  # +1 for timestep t
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # output in (0, 1)
        )
        
    def forward(self, hidden_state, t):
        """
        hidden_state: (batch, hidden_dim) - from agent's policy network
        t: float or tensor - normalized training progress (0 to 1)
        """
        # Ensure t is a tensor
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=hidden_state.device)
        t = t.view(1, 1).expand(hidden_state.size(0), 1)
        
        # Concatenate and forward
        x = torch.cat([hidden_state, t], dim=-1)
        return self.net(x)  # (batch, 1)
class Policy(nn.Module):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()

        obs_shape = obs_space.shape

        if base_kwargs is None:
            base_kwargs = {}

        self.base = MLPBase(obs_shape[0], **base_kwargs)

        num_outputs = action_space.n
        self.dist = Categorical(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def get_distribution(self, inputs, rnn_hxs, masks):
        """
        Returns the action distribution for given observations.
        Used for computing KL divergence between policies.
        """
        _, actor_features, _ = self.base(inputs, rnn_hxs, masks)
        return self.dist(actor_features)
class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs