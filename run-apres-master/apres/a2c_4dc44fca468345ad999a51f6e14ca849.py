import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import math
from math import sqrt
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.spatial.distance import cityblock

import gym
from model import Policy, FCNetwork
from gym.spaces.utils import flatdim
from storage import RolloutStorage
from sacred import Ingredient

CUDA_VISIBLE_DEVICES = 1
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

    # PPGA method
    def cosine_similarity_global(self, model1, model2):
        params1 = []
        params2 = []

        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2 and param1.shape == param2.shape:
                params1.append(param1.flatten())
                params2.append(param2.flatten())

        # Concatenate all parameters
        all_params1 = torch.cat(params1)
        all_params2 = torch.cat(params2)

        cos_sim = F.cosine_similarity(all_params1.unsqueeze(0), all_params2.unsqueeze(0))
        return cos_sim.item()

    

    @algorithm.capture
    def update(
            self,
            agents,
            storages,
            value_loss_coef,
            entropy_coef,
            max_grad_norm,
            device,
    ):
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        reward_shape = self.storage.rewards.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()
        # ,current_dist
        values, action_log_probs, dist_entropy, _, current_dist = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size
            ),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
            return_dist=True  # modified to return distribution mine
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        

        advantages = self.storage.returns[:-1] - values

        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()

        # calculate prediction loss for the OTHER actor
        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
       

        for oid in other_agent_ids:
            # Get distributions for other agent's experiences
            #    , other_dist
            other_values, logp, _, _, other_dist = self.model.evaluate_actions(
                storages[oid].obs[:-1].view(-1, *obs_shape),
                storages[oid]
                .recurrent_hidden_states[0]
                .view(-1, self.model.recurrent_hidden_state_size),
                storages[oid].masks[:-1].view(-1, 1),
                storages[oid].actions.view(-1, action_shape),
                return_dist=True  # Modified to return distribution mine
            )
            other_values = other_values.view(num_steps, num_processes, 1)
            logp = logp.view(num_steps, num_processes, 1)
            other_advantage = (
                    storages[oid].returns[:-1] - other_values
            )  

            coeff=[]
            for agent in agents:
                coeff.append(self.cosine_similarity_global(self.model,
                                                           agent.model))
            print(coeff)
            breakpoint()
           
        self.optimizer.zero_grad()
        grad = (policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * dist_entropy
                )
        
        (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * dist_entropy

        ).backward()
        # aggregation phase
        weighted_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        for agent in agents:
            if  agent.agent_id != self.agent_id:
                for (name, param), (_, agent_param) in zip(self.model.named_parameters(), agent.model.named_parameters()):

                        if param.grad is not None and agent_param.grad is not None:
                            
                            we=coeff[agent.agent_id]

                            weighted_grads[name] += we * agent_param.grad.clone()

        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.copy_(0.1*weighted_grads[name]+0.9*param.grad)

        
        

        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        

        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "grad":grad.item(),

        }

    
