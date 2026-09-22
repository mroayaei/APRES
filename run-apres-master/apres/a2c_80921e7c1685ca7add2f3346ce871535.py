import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
# mine
# from sklearn.metrics.pairwise import cosine_similarity
import math
from math import sqrt
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.spatial.distance import cityblock
# from sklearn.metrics import jaccard_score
# from sklearn.metrics import mean_squared_error
# # mine
# from network import SimilarityNetwork
import gym
from model import Policy, FCNetwork
from gym.spaces.utils import flatdim
from storage import RolloutStorage
from sacred import Ingredient

CUDA_VISIBLE_DEVICES = 1
algorithm = Ingredient("algorithm", save_git_info=False)


# b=[]

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

    # seac_coef = 1
    # seac_coef = torch.empty(5, 4, 1)
    num_processes = 4
    num_steps = 5
    device = "cpu"


class A2C:
    @algorithm.capture()
    def __init__(
            self,
            # g_policy,
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
        # self.global_net=g_policy
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
        # mine
        # # Privacy-preserving sharing parameters
        # self.sharing_strength = 0.3
        # self.performance_cache = {}  # Store performance estimates
        # self.previous_params = None  # For policy stability tracking
        # mine
        # self.intr_stats = RunningStats()
        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
        }

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    # mine
    # def landa(self):
    #     global b
    #     lan=b
    #     return lan

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

    # mine
    # def policy_similarity(self, model1, model2):
    #     """Calculate similarity between two policies without private data"""
    #     params1 = []
    #     params2 = []
    #     for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    #         if name1 == name2 and p1.shape == p2.shape:
    #             params1.append(p1.flatten())
    #             params2.append(p2.flatten())
    #
    #     if not params1:  # No compatible parameters
    #         return 0.0
    #
    #     params1 = torch.cat(params1)
    #     params2 = torch.cat(params2)
    #     return F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()
    #
    # def estimate_performance(self, agent):
    #     """Estimate performance without accessing private experiences"""
    #     if agent.agent_id in self.performance_cache:
    #         return self.performance_cache[agent.agent_id]
    #
    #     # Use policy stability as performance proxy
    #     if hasattr(agent, 'previous_params') and agent.previous_params is not None:
    #         total_change = 0.0
    #         param_count = 0
    #         current_params = list(agent.model.parameters())
    #
    #         for current, prev in zip(current_params, agent.previous_params):
    #             if current is not None and prev is not None:
    #                 change = torch.norm(current.data - prev).item()
    #                 total_change += change
    #                 param_count += 1
    #
    #         if param_count > 0:
    #             stability = 1.0 / (1.0 + total_change / param_count)
    #             return min(1.0, stability)
    #
    #     return 0.5  # Default estimate
    # mine

    # def cosine_similarity_global(self,model1, model2):
    #     params1 = []
    #     params2 = []

    #     for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
    #         if name1 == name2 and param1.shape == param2.shape:
    #             params1.append(param1.flatten())
    #             params2.append(param2.flatten())

    #     # Concatenate all parameters
    #     all_params1 = torch.cat(params1)
    #     all_params2 = torch.cat(params2)

    #     cos_sim = F.cosine_similarity(all_params1.unsqueeze(0), all_params2.unsqueeze(0))
    #     return cos_sim.item()
    # mine
    # def compute_entropy_importance(self, current_dist, other_dist, current_obs, other_obs):
    #     """
    #     Compute entropy-based importance sampling weights using the formula:
    #     H(π) = -Σ π(a|o) log π(a|o)

    #     Higher entropy → more exploration → higher weight
    #     Lower entropy → more exploitation → lower weight
    #     """
    #     # If you need the batch dimensions, reshape the probabilities:
    #     current_probs = current_dist.probs
    #     other_probs = other_dist.probs
    #     other_probs_reshaped=other_probs.view(5,4,-1)
    #     current_probs_reshaped=current_probs.view(5,4,-1)

    #     # Then compute entropy on the reshaped probabilities
    #     current_entropy = -torch.sum(current_probs_reshaped * torch.log(current_probs_reshaped + 1e-8), dim=-1)
    #     other_entropy = -torch.sum(other_probs_reshaped * torch.log(other_probs_reshaped + 1e-8), dim=-1)

    #     # Normalize entropies by maximum possible entropy
    #     max_entropy = torch.log(torch.tensor(self.action_size, dtype=torch.float32))
    #     current_entropy_norm = current_entropy / max_entropy
    #     other_entropy_norm = other_entropy / max_entropy

    #     # Entropy-based importance: weight by relative entropy difference
    #     # This encourages learning from experiences where our policy is uncertain
    #     entropy_importance = other_entropy_norm / (current_entropy_norm + 1e-8)

    #     return entropy_importance.unsqueeze(-1)

    @algorithm.capture
    def update(
            self,
            agents,
            storages,
            value_loss_coef,
            entropy_coef,
            # seac_coef,
            max_grad_norm,
            device,
    ):
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
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
        # tensor_mine=[]
        # for step in reversed(range(self.storage.rewards.size(0))):

        #     delta = (
        #                     self.storage.rewards[step]
        #                     + 0.99 * self.storage.value_preds[step + 1]
        #                     - self.storage.value_preds[step]
        #                 )
        #     tensor_mine.append(abs(delta))
        # result_mine=torch.stack(tensor_mine, dim=0).detach()

        advantages = self.storage.returns[:-1] - values

        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()

        # calculate prediction loss for the OTHER actor
        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
        # seac_policy_loss = 0
        # seac_value_loss = 0
        # hazf kardam baraye single

        # zarib=[]
        total_weight = []

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
            )  # or storages[oid].
            # # Use entropy-based importance sampling instead of standard importance sampling
            # entropy_importance =  self.compute_entropy_importance(
            #     current_dist,    # Pass the distribution object directly
            #     other_dist,      # Pass the distribution object directly
            #     self.storage.obs[:-1],
            #     storages[oid].obs[:-1]
            # )

            # zarib=[]
            # for agent in agents:
            #     zarib.append(self.cosine_similarity_global(self.model,
            #                                                agent.model))
            # tensor_other=[]
            # # td error
            # for step in reversed(range(storages[oid].rewards.size(0))):
            #     delta_other = (
            #                     storages[oid].rewards[step]
            #                     + 0.99 * storages[oid].value_preds[step + 1]
            #                     - storages[oid].value_preds[step]
            #                 )
            #     tensor_other.append(abs(delta_other))
            # result_other=torch.stack(tensor_other, dim=0).detach()

            # measure=(-(advantages.detach() * action_log_probs)).data+ (value_loss_coef * (advantages.pow(2))).data-( entropy_coef * dist_entropy).data

            # measure=abs(self.storage.returns[:-1]/storages[oid].returns[:-1])
            # measure=abs(self.storage.value_preds[:-1]/storages[oid].value_preds[:-1])
            # measure3=self.storage.returns[:-1]
            # measure4=abs(-(storages[oid].action_log_probs)*((storages[oid].returns[:-1] -storages[oid].value_preds[:-1]).detach()))
            # x=value_loss_coef*((storages[oid].returns[:-1] -storages[oid].value_preds[:-1]).pow(2))
            # measure6=storages[oid].returns[:-1]
            # measure7=storages[oid].rewards+0.99*(storages[oid].value_preds[:-1])
            # measure=(
            #     logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)
            # ).detach()
            # measure=abs(-(storages[oid].action_log_probs)*((storages[oid].returns[:-1] -storages[oid].value_preds[:-1]).detach()))
            # measure=value_loss_coef*((storages[oid].returns[:-1] -storages[oid].value_preds[:-1]).pow(2))
            # measure =measure1+measure2
            # measure=abs(storages[oid].action_log_probs.exp())
            # measure= -(advantages.detach() * action_log_probs)
            # akhari
            # x=-((storages[oid].returns[:-1] - storages[oid].value_preds[:-1]).detach()*storages[oid].action_log_probs )
            # measure=x.clone().detach().requires_grad_(True)

            # measure9=abs((-(storages[oid].returns[:-1].detach() * logp)).data+ (value_loss_coef * (storages[oid].returns[:-1].pow(2)))- (entropy_coef * dist_entropy).data)

            # jaccard similarity of mine
            # b=[]
            # for i in range(num_steps):
            #     a=[]
            #     for j in range(num_processes):

            #         A=storages[oid].obs[:-1][i][j]
            #         B=self.storage.obs[:-1][i][j]
            #         z=A[2:]
            #         d=B[2:]
            #         jaccard_index=jaccard_score(z,d)
            #         a.append([jaccard_index])
            #     b.append(a)
            # seac_coef=torch.FloatTensor(b)
            # my_dice/my_jaccard similarity of mine
            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

            # # global b
            b=[]
            for i in range(num_steps):
                a=[]
                for j in range(num_processes):
                    # A=storages[oid].obs[:-1][i][j]
                    # B=self.storage.obs[:-1][i][j]
                    # C=storages[oid].obs[:-1][i][j][7]
                    # D=self.storage.obs[:-1][i][j][7]
                    A=storages[oid].obs[:-1][i][j][2:]
                    B=self.storage.obs[:-1][i][j][2:]
                    # # x=storages[oid].obs[:-1][i][j][0:2]
                    # # y=self.storage.obs[:-1][i][j][0:2]
                    # intersection=np.logical_and(A, B)
                    # uni=np.logical_or(A,B)
                    # my_jaccard=intersection.sum()/uni.sum()
                    # my_dice=2. * intersection.sum() / (A.sum() + B.sum())
                    ham=1-(distance.hamming(A,B))
            
            
            #         # rect=math.log(1+math.exp(ham))
            #         # if ham>=0.5:
            #         #     a.append([1])
            #         # else:
            #         #     a.append([0])
            
            #         # if A==B and C==D:
            #         #     a.append([1])
            #         # else:
            #         #     a.append([0])
            
            #         # modell=SimilarityNetwork(len(A),64,1)
            #         # x1=A.unsqueeze(0)
            #         # x2=B.unsqueeze(0)
            #         # y_pred=modell(x1,x2)
            #         # y_target=measure[i][j].unsqueeze(1)
            
            #         # y_target=storages[oid].returns[:-1][i][j].unsqueeze(1)
            #         # print(storages[oid].action_log_probs.exp())
            #         # y_target=self.storage.returns[:-1][i][j].unsqueeze(1)
            #         # y_target=storages[oid].rewards[i][j].unsqueeze(1)
            #         # loss_fn   = nn.MSELoss()
            
            #         # opt = optim.Adam(modell.parameters(), lr=3e-4)
            #         # # my_loss = loss_fn(y_pred, y_target)
            #         # my_loss =y_target
            #         # opt.zero_grad()
            #         # my_loss.backward()
            #         # # retain_graph=True
            #         # opt.step()
            #         # a.append([y_pred.item()])
                    a.append([ham])
            
                b.append(a)
            
            seac_coef=torch.FloatTensor(b)
            # # Print model's state_dict
            # print("Model's state_dict:")
            # for param_tensor in self.model.state_dict():
            #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

            # print()

            # # Print optimizer's state_dict
            # print("Optimizer's state_dict:")
            # for var_name in self.model.optimizer.state_dict():
            #     print(var_name, "\t", self.model.optimizer.state_dict()[var_name])
            # breakpoint()
            # print("hi",result_mine)

            # b=[]
            # for i in range(num_steps):

            #     a=[]
            #     for j in range(num_processes):

            #         A=result_mine[:-1][i][j]
            #         B=result_other[:-1][i][j]
            #         weight=A/B
            #         a.append(weight)

            #     b.append(a)
            # print("hello",b)
            # breakpoint()
            # seac_coef=torch.FloatTensor(result_other)

            # mine
            importance_sampling =seac_coef * (
                logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)
            ).detach()
            # importance_sampling=entropy_importance.detach()

            # importance_sampling = (
            #         logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)
            # ).detach()

            # entropy mine
            # importance_sampling_new1 = (
            #     -torch.sum(logp.exp()*logp,dim=-1) / (-torch.sum((storages[oid].action_log_probs.exp()*storages[oid].action_log_probs + 1e-7),dim=-1))
            # ).detach()
            # importance_sampling=importance_sampling_new1.unsqueeze(-1)
            total_weight.append(importance_sampling.mean())
            # akhar
            # akhar
            # total_weight.append(importance_sampling_new1.mean())
            # total_weight.append(result_other.mean())
            # akhar
        total_weight.insert(self.agent_id, 1)
        
        # Normalize importance weights so they sum to 1 (among other agents only)
        # total_importance = sum(total_weight)
        # if total_importance > 0:
        #     norm_weights = [w / total_importance for w in total_weight]  # Now sums to 1
        # else:
        #     norm_weights = [1.0 / len(total_weight) for _ in total_weight]  # Fallback uniform

        # total_weight.insert(self.agent_id,result_mine.mean())

        # mine\\
        # global b
        # b=importance_sampling
        # importance_sampling = 1.0
        # seac_value_loss += (
        #     importance_sampling * other_advantage.pow(2)
        # ).mean()
        # seac_policy_loss += (
        #     -importance_sampling * logp * other_advantage.detach()
        # ).mean()

        # total_weight.insert(self.agent_id,result_mine.mean())

        # # agar kodesh tosh nabashe bayed injahazf beshe
        # a=result_mine.mean()
        # zarib=[a / x for x in total_weight]
        # zarib=total_weight
        # akhar
        # zarib = [x / sum(total_weight) for x in total_weight]

        # mine
        # zarib.append(storages[oid].rewards.mean())

        # mine

        self.optimizer.zero_grad()

        # (
        #     policy_loss
        #     + value_loss_coef * value_loss
        #     - entropy_coef * dist_entropy
        #     +  seac_policy_loss
        #     +  value_loss_coef * seac_value_loss
        # ).backward()

        # (
        #     policy_loss
        #     + value_loss_coef * value_loss
        #     - entropy_coef * dist_entropy
        #     + seac_coef * seac_policy_loss
        #     + seac_coef * value_loss_coef * seac_value_loss
        # ).backward()
        grad = (policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * dist_entropy
                )
        # single mine
        (
                policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * dist_entropy

        ).backward()
        # # chatgpt code for meta policy that worksfor all but not for hard(final best)
        # num_agents = len(agents)
        # all_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        # for agent in agents:
        #     if agent.agent_id != self.agent_id:

        #         for (name, agent_param) in agent.model.named_parameters():
        #             if  agent_param.grad is not None:
        #                 all_grads[name] += agent_param.grad.clone()

        # # میانگین گرفتن از گرادیان‌ها
        # for name in all_grads:
        #     all_grads[name] /= (num_agents -1)
        #     # all_grads[name] /= num_agents

        # alpha=0.1
        # # جایگزین کردن گرادیان این عامل با میانگین گرادیان‌ها
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:

        #         param.grad.copy_(alpha*all_grads[name]+(1-alpha)*param.grad)
        # new importance sampling code
        # # zarib.insert(self.agent_id,1)
        # # khodesh agar tosh nabashe inja ro az comment dar biyar
        # importance sampling ke natije behtari dashte
        # akhar
        weighted_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        for agent in agents:
            if agent.agent_id != self.agent_id:
                for (name, param), (_, agent_param) in zip(self.model.named_parameters(),
                                                           agent.model.named_parameters()):

                    if param.grad is not None and agent_param.grad is not None:
                        # nazar ostad
                        we = total_weight[agent.agent_id]
                        # we=norm_weights[agent.agent_id]
                        # we=zarib[agent.agent_id]

                        weighted_grads[name] += we * agent_param.grad.clone()

        # nazar ostad
        # جایگزین کردن گرادیان این عامل با میانگین گرادیان‌ها
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.copy_(weighted_grads[name] + param.grad)
        # # chatgpt code for weighted averaging gradient

        # # # ضریب گرادیان عامل اصلی
        # zarib.insert(self.agent_id,storages[self.agent_id].rewards.mean())

        # دیکشنری برای ذخیره مجموع گرادیان‌ها
        # weighted_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        # for agent in agents:
        #     if  agent.agent_id != self.agent_id:
        #         weight = zarib[agent.agent_id]

        #         # weight = total_weight[agent.agent_id]

        #         for (name, param), (_, agent_param) in zip(self.model.named_parameters(), agent.model.named_parameters()):
        #             if param.grad is not None and agent_param.grad is not None:
        #                 weighted_grads[name] += weight * agent_param.grad.clone()

        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         param.grad.copy_(weighted_grads[name])

        # self.my_grad(models=[self.model])
        # mine
        # Store self gradients safely
        # mine
        # self_grads = {}
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         self_grads[name] = param.grad.clone()
        # # PRIVACY-PRESERVING POLICY SHARING
        # shared_grads = {}
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         shared_grads[name] = torch.zeros_like(param.grad)
        #
        # total_weight = 0
        # my_perf = self.estimate_performance(self)
        #
        # for agent in agents:
        #     if agent.agent_id != self.agent_id:
        #         try:
        #             # Calculate weights without private data
        #             similarity = max(0.1, self.policy_similarity(self.model, agent.model))
        #             their_perf = self.estimate_performance(agent)
        #             perf_boost = max(0, their_perf - my_perf)
        #
        #             weight = self.sharing_strength * similarity * (1.0 + perf_boost)
        #
        #             # Share gradients (only model parameters, no experiences)
        #             for (name, param), (_, other_param) in zip(self.model.named_parameters(),
        #                                                        agent.model.named_parameters()):
        #                 if name in shared_grads and param.grad is not None and other_param.grad is not None:
        #                     # Only share compatible parameters
        #                     if param.shape == other_param.grad.shape:
        #                         shared_grads[name] += weight * other_param.grad.clone()
        #
        #             total_weight += weight
        #
        #         except Exception as e:
        #             print(f"Sharing failed with agent {agent.agent_id}: {e}")
        #             continue
        #
        # # Update performance cache and track parameters
        # self.performance_cache[self.agent_id] = my_perf
        # self.previous_params = [param.data.clone() for param in self.model.parameters()]
        #
        # # Safe gradient blending
        # if total_weight > 0:
        #     blend_ratio = min(0.5, total_weight)
        #     for name, param in self.model.named_parameters():
        #         if param.grad is not None and name in self_grads:
        #             if name in shared_grads and torch.norm(shared_grads[name]) > 1e-8:
        #                 try:
        #                     shared_norm = shared_grads[name] / total_weight
        #                     param.grad.copy_(blend_ratio * shared_norm + (1 - blend_ratio) * self_grads[name])
        #                 except Exception as e:
        #                     print(f"Gradient blending failed for {name}: {e}")
        #                     param.grad.copy_(self_grads[name])
        #             else:
        #                 param.grad.copy_(self_grads[name])
        # else:
        #     for name, param in self.model.named_parameters():
        #         if param.grad is not None and name in self_grads:
        #             param.grad.copy_(self_grads[name])
        # mine
        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        # alpha=0.5
        # for local_param, global_param in zip(self.model.parameters(), self.global_net.parameters()):
        # #     # global_param._grad = local_param.grad

        # #     # new
        #     if global_param.grad is None:
        #         global_param._grad = local_param.grad

        #     else:
        #         # global_param._grad = (1-alpha)*local_param.grad +alpha*global_param.grad
        #         global_param._grad = (local_param.grad +global_param.grad)/2
        #         local_param._grad=global_param._grad

        self.optimizer.step()
        # mine
        # return {
        #     "policy_loss": policy_loss.item(),
        #     "value_loss": value_loss_coef * value_loss.item(),
        #     "dist_entropy": entropy_coef * dist_entropy.item(),
        #     "importance_sampling": (seac_coef * importance_sampling).mean().item(),
        #     "seac_policy_loss":   seac_policy_loss.item(),
        #     "seac_value_loss":
        #      value_loss_coef
        #     * seac_value_loss.item(),
        #     "grad":grad.item(),
        # }
        # return {
        #     "policy_loss": policy_loss.item(),
        #     "value_loss": value_loss_coef * value_loss.item(),
        #     "dist_entropy": entropy_coef * dist_entropy.item(),
        #     "importance_sampling":  importance_sampling.mean().item(),
        #     "seac_policy_loss":   seac_coef * seac_policy_loss.item(),
        #     "seac_value_loss": seac_coef
        #     * value_loss_coef
        #     * seac_value_loss.item(),
        #     "grad":grad.item(),
        # }
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "grad": grad.item(),

        }
        # mine
        # return {
        #     "policy_loss": policy_loss.item(),
        #     "value_loss": value_loss.item() * 0.5,
        #     "privacy_shared": total_weight > 0,
        #     "shared_weight": total_weight,
        #     "grad": grad.item(),
        #     "agents_shared_with": len([a for a in agents if a.agent_id != self.agent_id])
        # }
    # @algorithm.capture
    # def my_grad(
    #         self,
    #         models,
    # ):
    #     other = [x for x in range(len(models)) if x != self.agent_id]

    #     for i in other:
    #         for local_param, share_param in zip(self.model.parameters(), models[i].parameters()):
    #             if share_param.grad is None:
    #                 pass
    #             else:
    #                 local_param._grad = local_param.grad +share_param.grad
    #     for self_param in (self.model.parameters()):
    #         if self_param.grad is None:
    #             pass
    #         else:
    #             self_param._grad=self_param._grad/len(models)
