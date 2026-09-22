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
    # def compute_hamming_similarity(self, obs_i, obs_j):
    #     """
    #     Compute Hamming similarity for binary observations.

    #     obs_i: tensor [batch, obs_dim]
    #     obs_j: tensor [batch, obs_dim]

    #     return:
    #     similarity [batch]
    #     """

    #     # Hamming distance:
    #     # number of different bits / vector length
        
    #     distance = (obs_i != obs_j).float().mean(dim=1)

    #     # similarity = 1 - distance
    #     similarity = 1.0 - distance

    #     return similarity
    @algorithm.capture
    def update(
        self,
        storages,
        value_loss_coef,
        entropy_coef,
        seac_coef,
        max_grad_norm,
        device,
        training_step=0,
        total_updates=1,
    ):

        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()

        values, action_log_probs, dist_entropy,_= self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size
            ),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
             return_dist=False
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.storage.returns[:-1] - values

        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()


        # calculate prediction loss for the OTHER actor
        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
         # ----- 3. Time-Based Decay: seac_coef becomes dynamic -----
        t_normalized = training_step / max(total_updates, 1)
        all =round(1/  len(storages) ,1)
        
        lambda_val = max(all, 1.0 - t_normalized)  # decays from 1.0 to 0.4
        # # 🔑 seac_coef IS NOW lambda_val (dynamic)
        seac_coef = lambda_val
        # 🔑 seac_coef IS NOW lambda_val (dynamic)
        # seac_coef = lambda_val
        seac_policy_loss = 0
        seac_value_loss = 0
        for oid in other_agent_ids:

            other_values, logp,_,_= self.model.evaluate_actions(
                storages[oid].obs[:-1].view(-1, *obs_shape),
                storages[oid]
                .recurrent_hidden_states[0]
                .view(-1, self.model.recurrent_hidden_state_size),
                storages[oid].masks[:-1].view(-1, 1),
                storages[oid].actions.view(-1, action_shape),
                 return_dist=False
            )
            other_values = other_values.view(num_steps, num_processes, 1)
            logp = logp.view(num_steps, num_processes, 1)
            # -----------------------------------
            # TD-error
            # -----------------------------------
            # gamma = 0.99
            # alpha = 0.5
            # top_ratio = 0.5
            # 100% -> 50% linearly over training
            top_ratio = 1.0 - 0.5 * t_normalized

            # Keep ratio within [0.5, 1.0]
            top_ratio = max(0.5, min(1.0, top_ratio))

            # td_error = (
            #     storages[oid].rewards
            #     + gamma * storages[oid].value_preds[1:]
            #     - storages[oid].value_preds[:-1]
            # )

            # td_priority = td_error.abs()

            # -----------------------------------
            # Advantage
            # -----------------------------------
            other_advantage = (
                storages[oid].returns[:-1]
                - other_values
            )
            # eps = 1e-6
            # alpha = 0.5
            priority = other_advantage.detach().abs()
            # norm_priority = priority / (priority.mean() + 1e-8)
            
            # priority = torch.relu(other_advantage.detach()) + eps
            # priority = (
            #     other_advantage.detach().abs() + eps
            # ) ** alpha
            

            # -----------------------------------
            # Combined priority
            # -----------------------------------
            # priority = (alpha * td_priority+ (1.0 - alpha) * adv_priority)
            
            priority_flat = priority.view(-1)
            
            num_keep = max(1,int(top_ratio * priority_flat.numel()))
           
            _, top_idx = torch.topk(priority_flat, num_keep)
          
            importance_sampling = (
                logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)
            ).detach()

            # Hamming similarity
            # =====================================================
            # others=storages[oid].value_preds[:-1].view(num_steps, num_processes, 1)
            # val_diff=torch.abs(other_values-others)
            # beta=torch.exp(-val_diff).detach()
            pol_diff = torch.abs( storages[oid].action_log_probs - logp )
            beta = torch.exp(- pol_diff).detach()
            
            
            # my_entropy = other_dist.entropy().view(num_steps, num_processes, 1).detach()  # Still your model
            # other_entropy = sender_dist.entropy().view(num_steps, num_processes, 1).detach()  # Their model

            # entropy_weight = torch.exp(my_entropy - other_entropy).clamp(0.5, 2.0).detach()
            # =====================================================
            # Select only prioritized transitions
            # =====================================================
            selected_advantage = other_advantage.view(-1)[top_idx]

            selected_logp = logp.view(-1)[top_idx]

            selected_is = importance_sampling.view(-1)[top_idx]

            selected_beta = beta.view(-1)[top_idx]
            
            # importance_sampling = 1.0
            
            seac_value_loss += (
                 selected_beta*
                 selected_is
                * selected_advantage.pow(2)
            ).mean()
            seac_policy_loss += (
                - selected_beta
                *selected_is
                * selected_logp
                * selected_advantage.detach()
               
            ).mean()

        self.optimizer.zero_grad()
        (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * dist_entropy
            + seac_coef * seac_policy_loss
            + seac_coef * value_loss_coef * seac_value_loss
        ).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "importance_sampling": importance_sampling.mean().item(),
            "seac_policy_loss": seac_coef * seac_policy_loss.item(),
            "seac_value_loss": seac_coef
            * value_loss_coef
            * seac_value_loss.item(),
            "seac_coef": seac_coef,           # ← Dynamic (decays from 1.0 to 0.4)
            "t_normalized": t_normalized,
        }
        #     seac_value_loss += (
        #             importance_sampling
        #         * other_advantage.pow(2)
        #         ).mean()
        #     seac_policy_loss += (
        #             -importance_sampling
        #         * logp
        #         * other_advantage.detach()
        #         ).mean()

        # self.optimizer.zero_grad()
        # (
        #     policy_loss
        #     + value_loss_coef * value_loss
        #     - entropy_coef * dist_entropy
        #     + seac_coef * seac_policy_loss
        #     + seac_coef * value_loss_coef * seac_value_loss
        # ).backward()

        # nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        # self.optimizer.step()

        # return {
        #     "policy_loss": policy_loss.item(),
        #     "value_loss": value_loss_coef * value_loss.item(),
        #     "dist_entropy": entropy_coef * dist_entropy.item(),
        #     "importance_sampling": importance_sampling.mean().item(),
        #     "seac_policy_loss": seac_coef * seac_policy_loss.item(),
        #     "seac_value_loss": seac_coef
        #     * value_loss_coef
        #     * seac_value_loss.item(),
        #     # "seac_coef": seac_coef,           # ← Dynamic (decays from 1.0 to 0.4)
        #     # "t_normalized": t_normalized,
        # }