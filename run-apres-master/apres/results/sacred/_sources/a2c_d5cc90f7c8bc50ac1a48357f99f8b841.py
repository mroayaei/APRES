import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import gym
from model import Policy, FCNetwork, RelevanceNetwork, AdaptiveLambda
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

    num_processes = 4
    num_steps = 5
    num_env_steps = 40_000_000
    device = "cpu"
    
    # ===== AC-SEAC PARAMETERS =====
    use_attention_weighting = True
    threshold_is = 0.0
    top_k = 3


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
        
        # ===== RELEVANCE NETWORK (β) =====
        self.relevance_net = RelevanceNetwork(self.obs_size)
        self.relevance_net.to(device)
        self.relevance_optimizer = optim.Adam(self.relevance_net.parameters(), lr=lr)
        
        # ===== ADAPTIVE LAMBDA NETWORK (λ) =====
        hidden_dim = self.model.recurrent_hidden_state_size
        
        # FIXED: Remove duplicate AdaptiveLambda call
        self.adaptive_lambda = AdaptiveLambda(input_dim=hidden_dim)
        self.adaptive_lambda.to(device)
        self.lambda_optimizer = optim.Adam(self.adaptive_lambda.parameters(), lr=lr)

        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
            "relevance_net": self.relevance_net,
            "relevance_optimizer": self.relevance_optimizer,
            "adaptive_lambda": self.adaptive_lambda,
            "lambda_optimizer": self.lambda_optimizer,
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
        _storages,
        value_loss_coef,
        entropy_coef,
        seac_coef,
        max_grad_norm,
        device,
        use_attention_weighting,
        threshold_is=0.0,
        training_step=0,
        top_k=3,
        num_steps=None,
        num_processes=None,
        num_env_steps=None,
    ):
        # ============================================================
        # STEP 1: Compute total updates and normalized training progress
        # ============================================================
        total_updates = int(num_env_steps) // num_steps // num_processes
        t_normalized = training_step / max(total_updates, 1)
        
        
        # ============================================================
        # STEP 2: Get shapes and dimensions
        # ============================================================
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps_storage, num_processes_storage, _ = self.storage.rewards.size()
        
        # ============================================================
        # STEP 3: Standard A2C loss from own experience
        # ============================================================
        values, action_log_probs, dist_entropy, actor_features = self.model.evaluate_actions(
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
        
        # ============================================================
        # Extract hidden state for adaptive lambda
        # ============================================================
        # Take first sample for lambda
        if actor_features.dim() == 2:
            hidden_for_lambda = actor_features[0:1, :]  # (1, hidden_dim)
        else:
            hidden_for_lambda = actor_features.view(-1, actor_features.size(-1))[0:1, :]
        
        
        
        # ============================================================
        # STEP 4: Prepare observation batch for attention computation
        # ============================================================
        n_samples = min(num_steps_storage * num_processes_storage, 32)
        obs_flat = self.storage.obs[:-1].view(-1, *obs_shape)
        
        if obs_flat.size(0) == 0:
            n_samples = 1
            obs_flat = self.storage.obs[:-1].view(-1, *obs_shape)
        
        indices = torch.randperm(min(obs_flat.size(0), n_samples))[:n_samples].to(device)
        obs_i_batch = obs_flat[indices]
        
        # ============================================================
        # STEP 5: Compute attention weights (β) for ALL other agents
        # ============================================================
        other_agent_ids = [x for x in range(len(_storages)) if x != self.agent_id]
        all_weights = []
        
        if use_attention_weighting and len(other_agent_ids) > 0:
            for oid in other_agent_ids:
                other_obs_flat = _storages[oid].obs[:-1].view(-1, *obs_shape)
                
                if other_obs_flat.size(0) == 0:
                    all_weights.append((oid, 0.5))
                    continue
                
                other_indices = torch.randperm(min(other_obs_flat.size(0), n_samples))[:n_samples].to(device)
                obs_j_batch = other_obs_flat[other_indices]
                
                with torch.no_grad():
                    weights = self.relevance_net(obs_i_batch, obs_j_batch)
                    weight = weights.mean().item()
                all_weights.append((oid, weight))
        else:
            all_weights = [(oid, 1.0) for oid in other_agent_ids]
        
        # ============================================================
        # STEP 6: Select Top-K agents by attention weight
        # ============================================================
        all_weights.sort(key=lambda x: x[1], reverse=True)
        selected_agents = all_weights[:min(top_k, len(all_weights))]
        
        # ============================================================
        # STEP 7: Compute adaptive lambda (λ)
        # ============================================================
        # FIXED: Use the existing self.adaptive_lambda (don't recreate it!)
        adaptive_lambda = self.adaptive_lambda(hidden_for_lambda, t_normalized).item()
        
        # ADD YOUR DEBUG PRINT HERE (AFTER adaptive_lambda is defined)
        if training_step % 10000 == 0 and training_step > 0:
            print(f"Step {training_step}: λ = {adaptive_lambda:.4f}, t = {t_normalized:.4f}")
        
        # ============================================================
        # STEP 8: Weighted SEAC loss from SELECTED agents only
        # ============================================================
        seac_policy_loss = 0.0
        seac_value_loss = 0.0
        
        for oid, beta_weight in selected_agents:
            other_storage = _storages[oid]
            other_log_probs = other_storage.action_log_probs.view(num_steps_storage, num_processes_storage, 1)
            
            other_values, logp, _, _ = self.model.evaluate_actions(
                other_storage.obs[:-1].view(-1, *obs_shape),
                other_storage.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
                other_storage.masks[:-1].view(-1, 1),
                other_storage.actions.view(-1, action_shape),
            )
            other_values = other_values.view(num_steps_storage, num_processes_storage, 1)
            logp = logp.view(num_steps_storage, num_processes_storage, 1)
            other_advantage = other_storage.returns[:-1] - other_values
            
            with torch.no_grad():
                is_weight = (logp.exp() / (other_log_probs.exp() + 1e-7)).detach()
            
            if threshold_is > 0:
                mask = is_weight > threshold_is
                if mask.sum() == 0:
                    continue
                is_weight = is_weight[mask]
                other_advantage = other_advantage[mask]
                logp = logp[mask]
            
            seac_policy_loss += adaptive_lambda * beta_weight * (-is_weight * logp * other_advantage.detach()).mean()
            seac_value_loss += adaptive_lambda * beta_weight * (is_weight * other_advantage.pow(2)).mean()
        
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
        
        self.optimizer.zero_grad()
        if use_attention_weighting:
            self.relevance_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        
        total_loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        if use_attention_weighting:
            nn.utils.clip_grad_norm_(self.relevance_net.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.adaptive_lambda.parameters(), max_grad_norm)
        
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