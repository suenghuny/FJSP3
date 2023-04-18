import pandas as pd

from GDN import Agent
from functools import partial
import numpy as np

import sys
import os
import time
from Environment_Multi.FJSP_multi import RL_ENV
from cfg import get_cfg
import torch
import random

cfg = get_cfg()

vessl_on = cfg.vessl
if vessl_on == True:
    import vessl
    vessl.init()
else:
    from torch.utils.tensorboard import SummaryWriter




map_name1 = cfg.map_name
GNN = cfg.GNN
heterogenous = False

"""
Protoss
colossi : 200.0150.01.0
stalkers : 80.080.00.625
zealots : 100.050.00.5

Terran
medivacs  : 150.00.00.75
marauders : 125.00.00.5625
marines   : 45.00.00.375

Zerg
zergling : 35.00.00.375
hydralisk : 80.00.00.625
baneling : 30.00.00.375
spine crawler : 300.00.01.125`
"""




def train(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer, output_dir, vdn):

    env.reset()
    done = False
    episode_reward = 0
    step = 0
    losses = []
    epi_r = list()
    eval = False
    start = time.time()
    sum_learn = 0
    while not done:

        node_feature_machine, num_waiting_operations, edge_index_machine, status = env.get_heterogeneous_graph()
        n_node_feature_machine = np.array(node_feature_machine).shape[0]
        if GNN == 'GAT':
            agent.eval_check(eval = True)
            node_representation = agent.get_node_representation(
                                                                node_feature_machine,
                                                                num_waiting_operations,
                                                                edge_index_machine,
                                                                n_node_feature_machine,
                                                                mini_batch=False)  # 차원 : n_agents X n_representation_comm

        avail_action = env.get_avail_actions()
        action, utility_list = agent.sample_action(node_representation, avail_action, epsilon)
        env.last_action = np.array(torch.tensor(action).detach().tolist())
        reward, done, info = env.step(action, q_values = utility_list, vdn = vdn)
        reward /=200



        agent.buffer.memory(node_feature_machine,num_waiting_operations, edge_index_machine, info, reward, done, avail_action, status)
        episode_reward += reward
        # if env.env.now>0:
        #     print(env.env.now/episode_reward)

        t += 1
        if t % 10000 <= 0.1:
            if vessl_on == True:
                agent.save_model(e, t, epsilon, output_dir+"{}.pt".format(t))
            else:
                agent.save_model(e,t,epsilon, output_dir+"{}.pt".format(t))
        step += 1
        if (t % 5000 == 0) and (t >0):
            eval = True

        if e >= train_start:
            if epsilon >= min_epsilon:
                epsilon = epsilon - anneal_epsilon
            else:
                epsilon = min_epsilon

            st = time.time()
            agent.eval_check(eval = False)
            loss = agent.learn(regularizer=0, vdn = vdn)

            sum_learn += time.time()-st
            losses.append(loss.detach().item())

    print(agent.scheduler.get_last_lr()[0])

    print("Total reward in episode {} = {}, epsilon : {}, time_step : {}, episode_duration : {}, training_duration : {}".format(
                                                                                            e,
                                                                                            np.round(episode_reward, 3),
                                                                                            np.round(epsilon, 3),
                                                                                            t, np.round(time.time()-start, 3),
    sum_learn))
    return episode_reward, epsilon, t, eval

def main():
    torch.manual_seed(81)
    random.seed(81)
    np.random.seed(81)
    env1 = RL_ENV()
    hidden_size_obs = cfg.hidden_size_obs       # GAT 해당(action 및 node representation의 hidden_size)
    hidden_size_comm = cfg.hidden_size_comm
    hidden_size_Q = cfg.hidden_size_Q         # GAT 해당
    hidden_size_meta_path = cfg.hidden_size_meta_path # GAT 해당
    n_representation_job = cfg.n_representation_job  # GAT 해당
    n_representation_machine = cfg.n_representation_machine
    buffer_size = cfg.buffer_size
    batch_size = cfg.batch_size
    gamma = cfg.gamma
    learning_rate = cfg.lr
    n_multi_head = cfg.n_multi_head
    dropout = cfg.dropout
    num_episode = cfg.num_episode
    train_start = cfg.train_start
    epsilon = cfg.epsilon
    min_epsilon = cfg.min_epsilon
    anneal_steps = cfg.anneal_steps
    teleport_probability = cfg.teleport_probability
    gtn_beta = cfg.gtn_beta
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps
    vdn = cfg.vdn

    if vessl_on == True:
        output_dir = "/output/"
    else:
        output_dir = "../output_dir_random/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_dir = 'output_dir/logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    initializer = True
    # writer = SummaryWriter(log_dir,
    #                        comment="map_name_{}_GNN_{}_lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}_nrepresentationcomm_{}.csv".format(
    #                            map_name1, GNN, learning_rate, hidden_size_obs, hidden_size_Q, n_representation_obs,
    #                            n_representation_comm))

    # "n_agents": num_machines,
    # "obs_shape": 7 + num_jobs + max_ops_length,  # + self.n_agents,
    # "n_actions": len(ops_name_list) + 1

    agent1 = Agent(num_agent=env1.get_env_info()["n_agents"],
                   feature_size_job=env1.get_env_info()["job_feature_shape"],
                   feature_size_machine=env1.get_env_info()["machine_feature_shape"],
                   hidden_size_meta_path = hidden_size_meta_path,
                   hidden_size_obs=hidden_size_obs,
                   hidden_size_comm=hidden_size_comm,
                   hidden_size_Q=hidden_size_Q,
                   n_multi_head=n_multi_head,
                   n_representation_job=n_representation_job,
                   n_representation_machine=n_representation_machine,
                   dropout=dropout,
                   action_size=env1.get_env_info()["n_actions"],
                   buffer_size=buffer_size,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   gamma=gamma,
                   GNN=GNN,
                   teleport_probability = teleport_probability,
                   gtn_beta = gtn_beta)


    t = 0
    epi_r = []
    win_rates = []
    for e in range(num_episode):
        episode_reward, epsilon, t, eval = train(agent1, env1, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer, output_dir, vdn)

        initializer = False
        epi_r.append(episode_reward)
        #writer.add_scalar("episode_reward/train", episode_reward, e)


        if vessl_on == True:
            vessl.log(step = e, payload = {'reward' : np.mean(epi_r)})
            epi_r = []
            r_df= pd.DataFrame(epi_r)
            r_df.to_csv(output_dir+"reward.csv")
        else:
            r_df= pd.DataFrame(epi_r)
            r_df.to_csv(output_dir+"reward.csv")
        #
        # if eval == True:
        #     win_rate = evaluation(env1, agent1, 32)
        #     win_rates.append(win_rate)
        #     if vessl_on == True:
        #         vessl.log(step = t, payload = {'win_rate' : win_rate})
        #         wr_df = pd.DataFrame(win_rates)
        #         wr_df.to_csv(output_dir+"win_rate.csv")
        #     else:
        #         wr_df = pd.DataFrame(win_rates)
        #         wr_df.to_csv(output_dir+"win_rate.csv")






main()

