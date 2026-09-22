import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import gym
from model import Policy, FCNetwork
from gym.spaces.utils import flatdim
from storage import RolloutStorage
from sacred import Ingredient

algorithm = Ingredient("algorithm")


@algorithm.config
def config():
    lr = 3e-4
    adam_eps = 0.001
    gamma = 0.99
    use_gae = False
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    use_proper_time_limits = True
    recurrent_policy = False
    use_linear_lr_decay = False

    seac_coef = 1.0
    # NEW:
    threshold_kl = 0.5      # max KL divergence to consider another agent
    threshold_is = 0.05     # min importance sampling weight to keep a transition

    num_processes = 4
    num_steps = 5

    device = "cpu"


class A2C:
    @algorithm.capture()
    def __init__(
        self,
        agent_id,
        obs_space,
        action_space,
        lr,
        adam_eps,
        recurrent_policy,
        num_steps,
        num_processes,
        device,
        
    ):
        self.agent_id = agent_id
        self.obs_size = flatdim(obs_space)
        self.action_size = flatdim(action_space)
        self.obs_space = obs_space
        self.action_space = action_space

        self.model = Policy(
            obs_space, action_space, base_kwargs={"recurrent": recurrent_policy},
        )

        self.storage = RolloutStorage(
            obs_space,
            action_space,
            self.model.recurrent_hidden_state_size,
            num_steps,
            num_processes,
        )

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr, eps=adam_eps)

        # self.intr_stats = RunningStats()
        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
        }

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @algorithm.capture
    def compute_returns(self, use_gae, gamma, gae_lambda, use_proper_time_limits):
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()

        self.storage.compute_returns(
            next_value, use_gae, gamma, gae_lambda, use_proper_time_limits,
        )

    
    
    @algorithm.capture
    def update(
        self,
        storages,          # list of RolloutStorage for all agents
        other_agents,      # list of A2C agents (same length as storages)
        value_loss_coef,
        entropy_coef,
        seac_coef,
        max_grad_norm,
        device,
        threshold_kl,      # from config, no default
        threshold_is,      # from config, no default
    ):
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()

        # ----- 1. Standard A2C loss from own experience -----
        values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
        )
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        advantages = self.storage.returns[:-1] - values
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()

        # ----- 2. Filter other agents by policy similarity (KL divergence) -----
        # Sample a minibatch of observations from current agent's storage
        n_kl_samples = min(32, num_steps * num_processes)
        obs_flat = self.storage.obs[:-1].view(-1, *obs_shape)
        indices = torch.randperm(obs_flat.size(0))[:n_kl_samples].to(device)
        obs_sample = obs_flat[indices]

        # Prepare dummy RNN states and masks (non-recurrent case)
        dummy_rnn = torch.zeros(obs_sample.size(0), self.model.recurrent_hidden_state_size).to(device)
        dummy_mask = torch.ones(obs_sample.size(0), 1).to(device)

        def kl_divergence(policy_i, policy_j, obs):
            with torch.no_grad():
                dist_i = policy_i.get_distribution(obs, dummy_rnn, dummy_mask)
                dist_j = policy_j.get_distribution(obs, dummy_rnn, dummy_mask)
                # Manual KL for categorical distributions (robust to custom Categorical)
                # Assumes .logits is available (logits shape: [batch, num_actions])
                log_probs_i = dist_i.logits.log_softmax(dim=-1)
                log_probs_j = dist_j.logits.log_softmax(dim=-1)
                probs_j = log_probs_j.exp()
                # KL(P||Q) = sum(P * log(P/Q))
                kl = (probs_j * (log_probs_j - log_probs_i)).sum(dim=-1).mean().item()
            return kl

        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
        useful_agents = []
        for oid in other_agent_ids:
            other_model = other_agents[oid].model
            kl = kl_divergence(self.model, other_model, obs_sample)
            if kl <= threshold_kl:
                useful_agents.append(oid)

        # ----- 3. Process each useful agent with importance sampling threshold -----
        seac_policy_loss = 0.0
        seac_value_loss = 0.0
        total_kept_transitions = 0
        all_kept_is = []   # collect all kept importance weights for logging

        for oid in useful_agents:
            other_storage = storages[oid]
            # Pre-computed log probs under other agent's own policy
            other_log_probs = other_storage.action_log_probs.view(num_steps, num_processes, 1)

            # Evaluate other agent's obs+actions under current agent's policy
            other_values, logp, _, _ = self.model.evaluate_actions(
                other_storage.obs[:-1].view(-1, *obs_shape),
                other_storage.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
                other_storage.masks[:-1].view(-1, 1),
                other_storage.actions.view(-1, action_shape),
            )
            other_values = other_values.view(num_steps, num_processes, 1)
            logp = logp.view(num_steps, num_processes, 1)
            other_advantage = other_storage.returns[:-1] - other_values

            # Importance sampling weight per timestep
            with torch.no_grad():
                is_weight = (logp.exp() / (other_log_probs.exp() + 1e-7)).detach()

            # Flatten over (steps, processes)
            is_flat = is_weight.view(-1)
            other_advantage_flat = other_advantage.view(-1, 1)
            logp_flat = logp.view(-1, 1)

            # Apply threshold
            mask = is_flat > threshold_is
            if mask.sum() == 0:
                continue

            kept_is = is_flat[mask]
            kept_advantage = other_advantage_flat[mask]
            kept_logp = logp_flat[mask]

            seac_policy_loss += (-kept_is * kept_logp * kept_advantage.detach()).mean()
            seac_value_loss += (kept_is * kept_advantage.pow(2)).mean()
            total_kept_transitions += mask.sum().item()
            all_kept_is.append(kept_is)

        if total_kept_transitions == 0:
            seac_policy_loss = torch.tensor(0.0, device=device)
            seac_value_loss = torch.tensor(0.0, device=device)
            avg_is = 0.0
        else:
            # Concatenate all kept IS weights and take mean
            avg_is = torch.cat(all_kept_is).mean().item()

        # ----- 4. Backward pass -----
        self.optimizer.zero_grad()
        (policy_loss
        + value_loss_coef * value_loss
        - entropy_coef * dist_entropy
        + seac_coef * seac_policy_loss
        + seac_coef * value_loss_coef * seac_value_loss).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "importance_sampling": avg_is,
            "seac_policy_loss": seac_coef * seac_policy_loss.item(),
            "seac_value_loss": seac_coef * value_loss_coef * seac_value_loss.item(),
            "num_useful_agents": len(useful_agents),
            "kept_transitions": total_kept_transitions,
        }