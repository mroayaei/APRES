import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy.spatial.distance import hamming
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

    def compute_hamming_similarity(self, obs_i, obs_j):
        """
        Compute Hamming similarity between two observation batches.
        Returns a tensor of shape (batch_size,) with per-sample similarities.
        Assumes binary/discrete observations.
        """
        # Convert to numpy
        obs_i_np = obs_i.cpu().numpy()
        obs_j_np = obs_j.cpu().numpy()
        
        # Compute Hamming distance per sample
        similarities = []
        for a, b in zip(obs_i_np, obs_j_np):
            dist = hamming(a, b)
            similarities.append(1.0 - dist)  # 1 - distance = similarity
        
        # Return as tensor on the same device as input
        return torch.tensor(similarities, device=obs_i.device).float()

    @algorithm.capture
    def update(
        self,
        storages,
        value_loss_coef,
        entropy_coef,
        seac_coef,           # ← Will be replaced by lambda_val (dynamic)
        max_grad_norm,
        device,
        training_step=0,
        total_updates=1,
    ):
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()

        # ----- 1. Own A2C loss -----
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

        # ----- 2. Compute Hamming similarity weights (PER SAMPLE) -----
        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
        
        obs_flat = self.storage.obs[:-1].view(-1, *obs_shape)

        similarity_tensors = []
        for oid in other_agent_ids:
            other_obs_flat = storages[oid].obs[:-1].view(-1, *obs_shape)
            sim_tensor = self.compute_hamming_similarity(obs_flat, other_obs_flat)
            similarity_tensors.append((oid, sim_tensor))

        # ----- 3. Time-Based Decay: seac_coef becomes dynamic -----
        t_normalized = training_step / max(total_updates, 1)
        lambda_val = max(0.2, 1.0 - t_normalized)  # decays from 1.0 to 0.4
        
        # 🔑 seac_coef IS NOW lambda_val (dynamic)
        seac_coef = lambda_val

        # ----- 4. SEAC sum (β inside) -----
        seac_policy_sum = 0.0
        seac_value_sum = 0.0

        for oid, sim_tensor in similarity_tensors:
            
            if sim_tensor.mean().item() < 1e-6:
                continue

            beta = sim_tensor.view(num_steps, num_processes, 1)
            
            other_storage = storages[oid]
            other_log_probs = other_storage.action_log_probs.view(num_steps, num_processes, 1)

            other_values, logp, _, _ = self.model.evaluate_actions(
                other_storage.obs[:-1].view(-1, *obs_shape),
                other_storage.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
                other_storage.masks[:-1].view(-1, 1),
                other_storage.actions.view(-1, action_shape),
            )
            other_values = other_values.view(num_steps, num_processes, 1)
            logp = logp.view(num_steps, num_processes, 1)
            other_advantage = other_storage.returns[:-1] - other_values

            importance_sampling = (logp.exp() / (other_log_probs.exp() + 1e-7)).detach()

            seac_policy_sum += (beta * (-importance_sampling * logp * other_advantage.detach())).mean()
            seac_value_sum += (beta * (importance_sampling * other_advantage.pow(2))).mean()

        num_other = len(other_agent_ids)
        if num_other > 0:
            seac_policy_sum /= num_other
            seac_value_sum /= num_other

        # ----- 5. Apply seac_coef (now dynamic) OUTSIDE the sum -----
        seac_policy_loss = seac_coef * seac_policy_sum
        seac_value_loss = seac_coef * seac_value_sum
        
        # ----- 6. Total loss (seac_coef is now dynamic) -----
        total_loss = (policy_loss
                    + value_loss_coef * value_loss
                    - entropy_coef * dist_entropy
                    + seac_coef * seac_policy_loss      # ← seac_coef is dynamic
                    + seac_coef * value_loss_coef * seac_value_loss)  # ← seac_coef is dynamic

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

        # ----- 7. Logging (seac_coef is dynamic) -----
        avg_similarity = sum(sim.mean().item() for _, sim in similarity_tensors) / len(similarity_tensors) if similarity_tensors else 0.0

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "importance_sampling": importance_sampling.mean().item(),
            "seac_policy_loss": seac_coef * seac_policy_loss.item(),      # ← seac_coef is dynamic
            "seac_value_loss": seac_coef * value_loss_coef * seac_value_loss.item(),  # ← seac_coef is dynamic
            "avg_similarity": avg_similarity,
            "seac_coef": seac_coef,           # ← Dynamic (decays from 1.0 to 0.4)
            "t_normalized": t_normalized,
        }