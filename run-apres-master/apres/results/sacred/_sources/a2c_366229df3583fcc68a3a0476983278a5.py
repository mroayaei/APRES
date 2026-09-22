import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import gym
from model import Policy, FCNetwork,RelevanceNetwork, AdaptiveLambda
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
    

    num_processes = 4
    num_steps = 5
    num_env_steps = 40_000_000  # ← ADD THIS (match your train.py)
    device = "cpu"
    # ===== AC-SEAC NEW PARAMETERS =====
    use_attention_weighting = True   # Enable attention weights (β)
    threshold_is = 0.0               # Importance sampling threshold (0 = disabled)
    top_k = 3                        # Number of top agents to select

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
        # ===== RELEVANCE NETWORK (β) - YOUR EXISTING ATTENTION =====
        self.relevance_net = RelevanceNetwork(self.obs_size)
        self.relevance_net.to(device)
        self.relevance_optimizer = optim.Adam(self.relevance_net.parameters(), lr=lr)

        # ===== ADAPTIVE LAMBDA NETWORK (λ) - NEW =====
        # hidden_dim=64 matches MLPBase hidden size
        self.adaptive_lambda = AdaptiveLambda(hidden_dim=64)
        self.adaptive_lambda.to(device)
        self.lambda_optimizer = optim.Adam(self.adaptive_lambda.parameters(), lr=lr)

        # self.intr_stats = RunningStats()
        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
            "relevance_net": self.relevance_net,
            "relevance_optimizer": self.relevance_optimizer,
            "adaptive_lambda": self.adaptive_lambda,      # NEW
            "lambda_optimizer": self.lambda_optimizer,    # NEW
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
        _storages,                     # list of all agents' storages
        value_loss_coef,
        entropy_coef,
        seac_coef,
        max_grad_norm,
        device,
        use_attention_weighting,
        threshold_is=0.0,
        training_step=0,               # ← from train.py (current update number)
        top_k=3,
        num_steps=None,                # ← captured from config
        num_processes=None,            # ← captured from config
        num_env_steps=None,            # ← captured from config
    ):
        # ============================================================
        # STEP 1: Compute total updates and normalized training progress
        # ============================================================
        total_updates = int(num_env_steps) // num_steps // num_processes
        t_normalized = training_step / max(total_updates, 1)
        
        # Optional: print progress occasionally
        if training_step % 100000 == 0 and training_step > 0:
            print(f"[Agent {self.agent_id}] Progress: {training_step}/{total_updates} ({t_normalized:.2%})")
        
        # ============================================================
        # STEP 2: Get shapes and dimensions
        # ============================================================
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps_storage, num_processes_storage, _ = self.storage.rewards.size()
        
        # ============================================================
        # STEP 3: Standard A2C loss from own experience
        # ============================================================
        values, action_log_probs, dist_entropy, hidden_state = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
        )
        values = values.view(num_steps_storage, num_processes_storage, 1)
        action_log_probs = action_log_probs.view(num_steps_storage, num_processes_storage, 1)
        advantages = self.storage.returns[:-1] - values
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()
        
        # Extract hidden state for adaptive lambda (use first timestep, first process)
        hidden_for_lambda = hidden_state.view(num_steps_storage, num_processes_storage, -1)[0, 0, :].unsqueeze(0)
        
        # ============================================================
        # STEP 4: Prepare observation batch for attention computation
        # ============================================================
        n_samples = min(num_steps_storage * num_processes_storage, 32)
        obs_flat = self.storage.obs[:-1].view(-1, *obs_shape)
        indices = torch.randperm(obs_flat.size(0))[:n_samples].to(device)
        obs_i_batch = obs_flat[indices]
        
        # ============================================================
        # STEP 5: Compute attention weights (β) for ALL other agents
        # ============================================================
        other_agent_ids = [x for x in range(len(_storages)) if x != self.agent_id]
        all_weights = []  # list of (agent_id, weight)
        
        if use_attention_weighting and len(other_agent_ids) > 0:
            for oid in other_agent_ids:
                # Get observations from other agent
                other_obs_flat = _storages[oid].obs[:-1].view(-1, *obs_shape)
                obs_j_batch = other_obs_flat[indices].to(device)
                
                # Compute attention weight
                with torch.no_grad():
                    weights = self.relevance_net(obs_i_batch, obs_j_batch)
                    weight = weights.mean().item()
                all_weights.append((oid, weight))
        else:
            # Fallback: uniform weights
            all_weights = [(oid, 1.0) for oid in other_agent_ids]
        
        # ============================================================
        # STEP 6: Select Top-K agents by attention weight
        # ============================================================
        all_weights.sort(key=lambda x: x[1], reverse=True)
        selected_agents = all_weights[:top_k]  # list of (oid, weight)
        
        # ============================================================
        # STEP 7: Compute adaptive lambda (λ) - when to trust others
        # ============================================================
        adaptive_lambda = self.adaptive_lambda(hidden_for_lambda, t_normalized).item()
        
        # ============================================================
        # STEP 8: Weighted SEAC loss from SELECTED agents only
        # ============================================================
        seac_policy_loss = 0.0
        seac_value_loss = 0.0
        
        for oid, beta_weight in selected_agents:
            other_storage = _storages[oid]
            
            # Get other agent's log probs (under its own policy)
            other_log_probs = other_storage.action_log_probs.view(num_steps_storage, num_processes_storage, 1)
            
            # Evaluate other agent's experiences under current agent's policy
            other_values, logp, _, _ = self.model.evaluate_actions(
                other_storage.obs[:-1].view(-1, *obs_shape),
                other_storage.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
                other_storage.masks[:-1].view(-1, 1),
                other_storage.actions.view(-1, action_shape),
            )
            other_values = other_values.view(num_steps_storage, num_processes_storage, 1)
            logp = logp.view(num_steps_storage, num_processes_storage, 1)
            other_advantage = other_storage.returns[:-1] - other_values
            
            # Importance sampling weight (corrects for off-policy actions)
            with torch.no_grad():
                is_weight = (logp.exp() / (other_log_probs.exp() + 1e-7)).detach()
            
            # Optional: filter by importance sampling threshold
            if threshold_is > 0:
                mask = is_weight > threshold_is
                if mask.sum() == 0:
                    continue
                is_weight = is_weight[mask]
                other_advantage = other_advantage[mask]
                logp = logp[mask]
            
            # Apply both β (attention) and λ (adaptive)
            seac_policy_loss += adaptive_lambda * beta_weight * (-is_weight * logp * other_advantage.detach()).mean()
            seac_value_loss += adaptive_lambda * beta_weight * (is_weight * other_advantage.pow(2)).mean()
        
        # Normalize by number of selected agents (K)
        num_selected = len(selected_agents)
        if num_selected > 0:
            seac_policy_loss = seac_policy_loss / num_selected
            seac_value_loss = seac_value_loss / num_selected
        
        # ============================================================
        # STEP 9: Total loss and backward pass
        # ============================================================
        total_loss = (policy_loss
                    + value_loss_coef * value_loss
                    - entropy_coef * dist_entropy
                    + seac_coef * seac_policy_loss
                    + seac_coef * value_loss_coef * seac_value_loss)
        
        # Zero all gradients
        self.optimizer.zero_grad()
        if use_attention_weighting:
            self.relevance_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        if use_attention_weighting:
            nn.utils.clip_grad_norm_(self.relevance_net.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.adaptive_lambda.parameters(), max_grad_norm)
        
        # Update all networks
        self.optimizer.step()
        if use_attention_weighting:
            self.relevance_optimizer.step()
        self.lambda_optimizer.step()
        
        # ============================================================
        # STEP 10: Logging
        # ============================================================
        if use_attention_weighting and selected_agents:
            avg_selected_weight = sum(w for _, w in selected_agents) / num_selected
        else:
            avg_selected_weight = 1.0
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "seac_policy_loss": seac_coef * seac_policy_loss.item(),
            "seac_value_loss": seac_coef * value_loss_coef * seac_value_loss.item(),
            "adaptive_lambda": adaptive_lambda,
            "avg_selected_weight": avg_selected_weight,
            "num_selected_agents": num_selected,
            "t_normalized": t_normalized,
        }