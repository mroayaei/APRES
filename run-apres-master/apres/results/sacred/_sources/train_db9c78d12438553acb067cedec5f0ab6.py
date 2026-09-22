import glob
# import pickle
import csv
import logging
import os
import shutil
import time,copy
from collections import deque, OrderedDict
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import (  # noqa
    FileStorageObserver,
    MongoObserver,
    QueuedMongoObserver,
    QueueObserver,
)
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CosineSimilarity
import utils
from a2c import A2C, algorithm
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones
from model import Policy
from numpy.linalg import norm
import rware # noqa
import lbforaging # noqa
import torch.nn.functional as F
ex = Experiment(ingredients=[algorithm])
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(FileStorageObserver("./results/sacred"))

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


# mine
test_list_before = [
                OrderedDict([
    ('base.actor.0.weight', torch.zeros((64, 71))),
    ('base.actor.0.bias', torch.zeros(64)),
    ('base.actor.2.weight', torch.zeros((64, 64))),
    ('base.actor.2.bias', torch.zeros(64)),
    ('base.critic.0.weight', torch.zeros((64, 71))),
    ('base.critic.0.bias', torch.zeros(64)),
    ('base.critic.2.weight', torch.zeros((64, 64))),
    ('base.critic.2.bias', torch.zeros(64)),
    ('base.critic_linear.weight', torch.zeros((1, 64))),
    ('base.critic_linear.bias', torch.zeros(1)),
    ('dist.linear.weight', torch.zeros((5, 64))),
    ('dist.linear.bias', torch.zeros(5))
]),
OrderedDict([
    ('base.actor.0.weight', torch.zeros((64, 71))),
    ('base.actor.0.bias', torch.zeros(64)),
    ('base.actor.2.weight', torch.zeros((64, 64))),
    ('base.actor.2.bias', torch.zeros(64)),
    ('base.critic.0.weight', torch.zeros((64, 71))),
    ('base.critic.0.bias', torch.zeros(64)),
    ('base.critic.2.weight', torch.zeros((64, 64))),
    ('base.critic.2.bias', torch.zeros(64)),
    ('base.critic_linear.weight', torch.zeros((1, 64))),
    ('base.critic_linear.bias', torch.zeros(1)),
    ('dist.linear.weight', torch.zeros((5, 64))),
    ('dist.linear.bias', torch.zeros(5))
])
            ]


@ex.config
def config():
    env_name = None
    time_limit = None
    wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
    )
    dummy_vecenv = False
# 100e6
    num_env_steps = 10e6
    eval_dir = "./results/video/{id}"
    loss_dir = "./results/loss/{id}"
    save_dir = "./results/trained_models/{id}"
    log_interval = 2000
    save_interval = int(1e6)
    eval_interval = int(1e6)
    episodes_per_eval = 8
    # mine
    # my_update=1e6

for conf in glob.glob("configs/*.yaml"):
    name = f"{Path(conf).stem}"
    ex.add_named_config(name, conf)

def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


@ex.capture
def evaluate(
    agents,
    monitor_dir,
    episodes_per_eval,
    env_name,
    seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    algorithm,
    _log,
):
    device = algorithm["device"]

    eval_envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        episodes_per_eval,
        time_limit,
        wrappers,
        device,
        monitor_dir=monitor_dir,
    )

    n_obs = eval_envs.reset()
    n_recurrent_hidden_states = [
        torch.zeros(
            episodes_per_eval, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id], recurrent_hidden_states, masks
                    )
                    for agent, recurrent_hidden_states in zip(
                        agents, n_recurrent_hidden_states
                    )
                ]
            )

        # Obser reward and next obs
        n_obs, _, done, infos = eval_envs.step(n_action)

        n_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    info = _squash_info(all_infos)
    _log.info(
        f"Evaluation using {len(all_infos)} episodes: mean reward {info['episode_reward']:.5f}\n"
    )



@ex.automain
def main(
    _run,
    _log,
    num_env_steps,
    env_name,
    seed,
    algorithm,
    dummy_vecenv,
    time_limit,
    wrappers,
    save_dir,
    eval_dir,
    loss_dir,
    log_interval,
    save_interval,
    eval_interval,
    # my_update,#mine
):
    x=[]
    y=[]
    
    if loss_dir:
        loss_dir = path.expanduser(loss_dir.format(id=str(_run._id)))
        utils.cleanup_log_dir(loss_dir)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    eval_dir = path.expanduser(eval_dir.format(id=str(_run._id)))
    save_dir = path.expanduser(save_dir.format(id=str(_run._id)))

    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)
    envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        algorithm["device"],
    )
    # global_policy_net = Policy(envs.observation_space[0], envs.action_space[0],)
    # agents = [
    #     A2C(global_policy_net,i, osp, asp)
    #     for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    # ]
    agents = [
        A2C(i, osp, asp)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]
    

    obs = envs.reset()

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(algorithm["device"])

    start = time.time()
    num_updates = (
        int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
    )

    all_infos = deque(maxlen=10)

    for j in range(1, num_updates + 1):

        for step in range(algorithm["num_steps"]):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(n_action)
            # envs.envs[0].render()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            bad_masks = torch.FloatTensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ]
            )
            for i in range(len(agents)):
                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    n_action_log_prob[i],
                    n_value[i],
                    reward[:, i].unsqueeze(1),
                    masks,
                    bad_masks,
                )

            for info in infos:
                if info:
                    all_infos.append(info)            
            
        # for agent in agents:
        #     agent.my_grad([a.model for a in agents])
# value_loss, action_loss, dist_entropy = agent.update(rollouts)
        for agent in agents:
            agent.compute_returns()
        grads=[]
            
        for agent in agents:
            # loss = agent.update([a.storage for a in agents])
            loss = agent.update(agents,[a.storage for a in agents])
            grads.append(loss["grad"])
            
            for k, v in loss.items():
                if writer:
                    writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)
            
        

        for agent in agents:
            agent.storage.after_update()
        

        
        if j % log_interval == 0 and len(all_infos) > 1:
            # # # for agent in agents:
                
            # # Opening the file with append mode 
            # file = open("./results/chart.txt", "a") 
            
            # # Content to be added 
            # content = "\n"+str(agents[0].landa())
            
            # # Writing the file 
            # file.write(content) 
            
            # # Closing the opened file 
            # file.close()
            # mines
            
            squashed = _squash_info(all_infos)

            total_num_steps = (
                (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
            )
            end = time.time()
            _log.info(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
            )
            _log.info(
                f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
            )
            for k, v in squashed.items():
                _run.log_scalar(k, v, j)
            all_infos.clear()
            x.append(total_num_steps)
            y.append(round(squashed['episode_reward'].sum(), 3))
            #    man ezafe kardam
            # for param1, param2 in zip(agents[0].model.parameters(), agents[1].model.parameters()):
            # #     summed_param = (param1 + param2)/2
            # #     agents[0].model.state_dict()[param1.name] = summed_param.data
            # #     agents[1].model.state_dict()[param2.name] = summed_param.data
            #     temp=param2.grad
            #     param2._grad = param1.grad
            #     param1._grad=temp

            #     # agents[0].model.load_state_dict(agents[1].model.state_dict())
            #     # agents[1].model.load_state_dict(agents[0].model.state_dict())
            # torch.mean(agents[0].storage.returns[:-1]) >torch.mean(agents[1].storage.returns[:-1])

            # # policy sharing ke javab dad
            # if squashed['agent0/episode_reward'].sum()>squashed['agent1/episode_reward'].sum():
            #     agents[1].model.load_state_dict(agents[0].model.state_dict())
            # elif squashed['agent0/episode_reward'].sum()<squashed['agent1/episode_reward'].sum():
            #     agents[0].model.load_state_dict(agents[1].model.state_dict()) 
            
            #new policy sharing code that get max reward teta
            # new_dic={key :value for key , value in squashed.items() if key.startswith("agent")}
            # total_weight=sum(new_dic.values())
            
            # if total_weight==0.0:
            #     pass
            # else:
            #     for key in new_dic:
            #         new_dic[key]=new_dic[key]/total_weight
            #     for i in range(len(agents)):
            #         for key in agents[i].model.state_dict():
            #             agents[i].model.state_dict()[key] = list(new_dic.values())[i]*agents[i].model.state_dict()[key]

            #     for i in range(len(agents)):
                    
            #         for j in range(len(agents)):
            #             if j !=i :
            #                 for key in agents[i].model.state_dict():
            #                     agents[i].model.state_dict()[key]=agents[i].model.state_dict()[key]+agents[j].model.state_dict()[key]
            #         for key in agents[i].model.state_dict():
            #             agents[i].model.state_dict()[key]=agents[i].model.state_dict()[key]/len(agents)
            #cosine simi                      

            # share parameter every 20 m epoch
            # if j % my_update == 0 and len(all_infos) > 1:
            #     new_dic={key :value for key , value in squashed.items() if key.startswith("agent")}
            #     keymax=max(new_dic,key=lambda x: new_dic[x])         
            #     if squashed[keymax].sum()==0.0:
            #         pass
            #     else:
            #         for i in range(len(agents)):
            #             if  i != int(keymax[5]):
            #                 agents[i].model.load_state_dict(agents[int(keymax[5])].model.state_dict())
            
            
            # mine
            # test_list_after = [agents[0].model.state_dict(),agents[1].model.state_dict()]
            # def subtract(dict1, dict2):
            #     result = OrderedDict()
            #     temp={}
            #     for key in dict1:
            #         if key in dict2:
            #             result[key] = dict1[key] - dict2[key]
            #         else:
            #             raise KeyError(f"Key {key} not found in second OrderedDict.")
            #     grad1=grads[0]
            #     grad2=grads[1]
            #     avg=(grad1+grad2)/2
            #     for key,tensor in result.items():
            #         # agents[0].landa()[4].mean()
                    
            #         temp[key]=tensor*avg
            #     return temp

            # global test_list_before
            # result_list = []
            # for d1, d2 in zip(test_list_after,test_list_before):
            #     result_list.append(subtract(d1, d2))
                
                

            # def sum(dict1, dict2):
            #     result = OrderedDict()
            #     for key in dict1:
            #         if key in dict2:
            #             result[key] = dict1[key] + dict2[key]
            #         else:
            #             raise KeyError(f"Key {key} not found in second OrderedDict.")
            #     return result
            # result_list2 = []
            # new_test_list_after=[test_list_after[1],test_list_after[0]]
            # for d1, d2 in zip(result_list, new_test_list_after):
            #     result_list2.append(sum(d1, d2))
            
            # for i in range(len(agents)): 
            #     agents[i].model.load_state_dict(result_list2[i])
            
            # test_list_before=test_list_after
            # alpha ostad            
            # def summ(dict1,dict2):
            #     result=OrderedDict()
            #     for key in dict1:
            #         if key in dict2:
            #             result[key]=dict1[key]+dict2[key]
            #         else:
            #             raise KeyError(f"Key {key} not found in second OrderedDict.")
            #     return result

            # alpha=0.9
            # alpha_list_before=[]
            # for i in range(len(agents)):
            #     alpha_list_before.append(OrderedDict(
            #         (key,tensor*alpha)
            #         for (key,tensor) in agents[i].model.state_dict().items()
            #     ))
            # beta=0.1
            # beta_list_before=[]
            
            # for i in range(len(agents)):
            #     beta_list_before.append(OrderedDict(
            #         (key,tensor*beta)
            #         for (key,tensor) in agents[i].model.state_dict().items()
            #     ))
            
            # beta_list_after=[beta_list_before[1],beta_list_before[0]]
            # result_list2=[]
            # for d1,d2 in zip(alpha_list_before,beta_list_after):
            #     result_list2.append(summ(d1,d2))
            # agents[1].model.load_state_dict(result_list2[0])
            # agents[0].model.load_state_dict(result_list2[1])
            # average of parameter
            # avg_model = copy.deepcopy(agents[0].model)
            # for param in avg_model.state_dict():
            #     avg_model.state_dict()[param] = sum(agents[m].model.state_dict()[param] for m in range(len(agents))) / len(agents)
            # for i in range(len(agents)):
            #     agents[i].model.load_state_dict(avg_model.state_dict())


            
            
            
            


            
        if save_interval is not None and (
            j > 0 and j % save_interval == 0 or j == num_updates
        ):
            cur_save_dir = path.join(save_dir, f"u{j}")
            for agent in agents:
                save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                os.makedirs(save_at, exist_ok=True)
                agent.save(save_at)
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
            shutil.rmtree(cur_save_dir)
            _run.add_artifact(archive_name)

        if eval_interval is not None and (
            j > 0 and j % eval_interval == 0 or j == num_updates
        ):
            evaluate(
                agents, os.path.join(eval_dir, f"u{j}"),
            )
            videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
            for i, v in enumerate(videos):
                _run.add_artifact(v, f"u{j}.{i}.mp4")
        
        if j==num_updates:
            # Writing to file
            with open(f"./results/myfile{_run._id}.txt", "w") as file1:
                # Writing data to a file
                file1.write(str(x)+'\n')
                file1.write(str(y))
            # chat gpt wite file 
            # # first model
            # data_filename = f"./results/my_data_run{_run._id}.npz"
            # np.savez(data_filename, x=np.array(x), y=np.array(y))
            # second model
            # with open(f"./results/my_data_run{_run._id}.pkl", "wb") as f:
            #     pickle.dump((x, y), f)
           
            plt.plot(x, y)
            plt.xlabel('Environment Steps')
            plt.ylabel('Returns')
            plt.title('Rware')
            plt.savefig(f"./results/plot_run{_run._id}.png")   # optional
            plt.show()
    envs.close()


