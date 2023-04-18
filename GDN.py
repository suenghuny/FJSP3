from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Categorical
import numpy as np

from GAT.model import GAT
from GAT.layers import device
from copy import deepcopy
from GTN.utils import _norm
from GTN.model_fastgtn import FastGTNs
from scipy.sparse import csr_matrix


class IQN(nn.Module):
    def __init__(self, state_size, action_size, batch_size, layer_size=256, N=32):
        super(IQN, self).__init__()
        self.input_shape = state_size
        self.batch_size = batch_size
        # print(state_size)
        self.action_size = action_size
        self.K = 32
        self.N = N
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(
            device)  # Starting from 0 as in the paper

        self.head = nn.Linear(self.input_shape, layer_size)  # cound be a cnn
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)

        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_1_bn = nn.BatchNorm1d(layer_size)

        self.ff_2 = nn.Linear(layer_size, 128)
        self.ff_2_bn = nn.BatchNorm1d(128)

        self.ff_3 = nn.Linear(128, 96)
        self.ff_3_bn = nn.BatchNorm1d(96)

        self.ff_4 = nn.Linear(96, 81)
        self.ff_4_bn = nn.BatchNorm1d(81)

        self.ff_5 = nn.Linear(81, 81)
        self.ff_5_bn = nn.BatchNorm1d(81)

        # self.ff_6 = nn.Linear(64, 64)
        # self.ff_6_bn = nn.BatchNorm1d(64)

        self.ff_6 = nn.Linear(81, action_size)

        torch.nn.init.xavier_uniform_(self.ff_1.weight)
        torch.nn.init.xavier_uniform_(self.ff_2.weight)
        torch.nn.init.xavier_uniform_(self.ff_3.weight)
        torch.nn.init.xavier_uniform_(self.ff_4.weight)
        torch.nn.init.xavier_uniform_(self.ff_5.weight)
        torch.nn.init.xavier_uniform_(self.ff_6.weight)
        #torch.nn.init.xavier_uniform_(self.ff_7.weight)

        # weight_init([self.head_1, self.ff_1])

    def calc_cos(self, batch_size):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, self.N).to(device).unsqueeze(-1)  # (batch_size, self.N, 1)
        cos = torch.cos(taus * self.pis)  # self.pis shape : 1,1,self.n_cos
        assert cos.shape == (batch_size, self.N, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(self, input, cos, mini_batch):
        N = self.N
        """
        Quantile Calculation depending on the number of tau
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        """
        if mini_batch == False:
            batch_size = 1
            input = input.unsqueeze(0)
        else:
            batch_size = self.batch_size

        # batch_size = input.shape[0]

        x = F.elu(self.head(input.to(device)))  # x의 shape는 batch_size, layer_size
        # cos, taus = self.calc_cos(batch_size)                                             # cos shape (batch, self.N, layer_size)
        cos = cos.view(batch_size * N, self.n_cos)
        cos_x = F.elu(self.cos_embedding(cos)).view(batch_size, N, self.layer_size)  # (batch, n_tau, layer)

        x = (x.unsqueeze(1) * cos_x).view(batch_size * N, self.layer_size)  # 이부분이 phsi * phi에 해당하는 부분
        x = self.ff_1(x)
        x = self.ff_1_bn(x)
        x = F.elu(x)
        x = self.ff_2(x)
        x = self.ff_2_bn(x)
        x = F.elu(x)
        x = self.ff_3(x)
        x = self.ff_3_bn(x)
        x = F.elu(x)

        x = self.ff_4(x)
        x = self.ff_4_bn(x)
        x = F.elu(x)

        x = self.ff_5(x)
        x = self.ff_5_bn(x)
        x = F.elu(x)

        out = self.ff_6(x)
        # x = self.ff_6_bn(x)
        # x = F.elu(x)
        #
        # out = self.ff_7(x)
        quantiles = out.view(batch_size, N, self.action_size)
        q = quantiles.mean(dim=1)
        return q


class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim=1)


class Network(nn.Module):
    def __init__(self, obs_and_action_size, hidden_size_q, action_size):
        super(Network, self).__init__()
        self.obs_and_action_size = obs_and_action_size
        self.fcn_1 = nn.Linear(obs_and_action_size, hidden_size_q + 10)
        self.fcn_1bn = nn.BatchNorm1d(hidden_size_q + 10)

        self.fcn_2 = nn.Linear(hidden_size_q + 10, hidden_size_q - 5)
        self.fcn_2bn = nn.BatchNorm1d(hidden_size_q - 5)

        self.fcn_3 = nn.Linear(hidden_size_q - 5, hidden_size_q - 20)
        self.fcn_3bn = nn.BatchNorm1d(hidden_size_q - 20)

        self.fcn_4 = nn.Linear(hidden_size_q - 20, hidden_size_q - 40)
        self.fcn_4bn = nn.BatchNorm1d(hidden_size_q - 40)

        self.fcn_5 = nn.Linear(hidden_size_q - 40, action_size)
        # self.fcn_5 = nn.Linear(int(hidden_size_q/8), action_size)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)
        torch.nn.init.xavier_uniform_(self.fcn_5.weight)
        # torch.nn.init.xavier_uniform_(self.fcn_5.weight)

    def forward(self, obs_and_action):
        if obs_and_action.dim() == 1:
            obs_and_action = obs_and_action.unsqueeze(0)
        # print(obs_and_action.dim())

        x = F.elu(self.fcn_1bn(self.fcn_1(obs_and_action)))
        x = F.elu(self.fcn_2bn(self.fcn_2(x)))
        x = F.elu(self.fcn_3bn(self.fcn_3(x)))
        x = F.elu(self.fcn_4bn(self.fcn_4(x)))
        q = self.fcn_5(x)
        # q = self.fcn_5(x)
        return q


class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_size, n_representation_obs, machine=False, n_agent=False):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size

        if machine == True:
            self.fcn_1bn = nn.BatchNorm1d(n_agent)
            self.fcn_2bn = nn.BatchNorm1d(n_agent)
            self.fcn_3bn = nn.BatchNorm1d(n_agent)

        else:
            self.fcn_1bn = nn.BatchNorm1d(n_agent)
            self.fcn_2bn = nn.BatchNorm1d(n_agent)
            self.fcn_3bn = nn.BatchNorm1d(n_agent)
        self.fcn_1 = nn.Linear(feature_size, hidden_size + 15)
        self.fcn_2 = nn.Linear(hidden_size + 15, hidden_size + 15)

        self.fcn_3 = nn.Linear(hidden_size + 15, hidden_size + 15)
        self.fcn_4 = nn.Linear(hidden_size + 15, n_representation_obs)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)

    def forward(self, node_feature, machine=False):
        if machine == False:

            if node_feature.dim() == 2:
                node_feature = node_feature.unsqueeze(0)
                # print(node_feature.shape)

        else:
            if node_feature.dim() == 2:
                node_feature = node_feature.unsqueeze(0)

        # print(node_feature.shape, machine)

        x = F.elu(self.fcn_1bn(self.fcn_1(node_feature)))
        x = F.elu(self.fcn_2bn(self.fcn_2(x)))
        x = F.elu(self.fcn_3bn(self.fcn_3(x)))
        node_representation = self.fcn_4(x)
        return node_representation


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, num_agent, action_size):
        self.buffer = deque()

        self.step_count_list = list()
        for _ in range(10):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.num_agent = num_agent
        self.agent_id = np.eye(self.num_agent).tolist()
        self.one_hot_actions = np.eye(action_size).tolist()
        self.batch_size = batch_size
        self.step_count = 0

    def pop(self):
        self.buffer.pop()

    def memory(self, node_feature_machine, num_waiting_operations, edge_index_machine, action, reward, done,
               avail_action, status):
        # self.buffer[0].append(node_feature_job)
        self.buffer[1].append(node_feature_machine)
        self.buffer[2].append(num_waiting_operations)
        self.buffer[3].append(edge_index_machine)
        self.buffer[4].append(action)
        self.buffer[5].append(reward)
        self.buffer[6].append(done)
        self.buffer[7].append(avail_action)
        self.buffer[8].append(status)
        self.buffer[9].append(np.sum(status))


        if self.step_count < self.buffer_size - 1:
            self.step_count_list.append(self.step_count)
            self.step_count += 1


    def generating_mini_batch(self, datas, batch_idx, cat):
        for s in batch_idx:
            if cat == 'node_feature_machine':
                yield datas[1][s]
            if cat == 'num_waiting_operations':
                yield datas[2][s]
            if cat == 'edge_index_machine':
                yield torch.sparse_coo_tensor(datas[3][s],
                                              torch.ones(torch.tensor(datas[3][s]).shape[1]),
                                              (self.num_agent, self.num_agent)).to_dense()
            if cat == 'action':
                yield datas[4][s]
            if cat == 'reward':
                yield datas[5][s]
            if cat == 'done':
                yield datas[6][s]
            if cat == 'node_feature_machine_next':
                yield datas[1][s + 1]
            if cat == 'num_waiting_operations_next':
                yield datas[2][s + 1]
            if cat == 'edge_index_machine_next':
                yield torch.sparse_coo_tensor(datas[3][s + 1],
                                              torch.ones(torch.tensor(datas[3][s + 1]).shape[1]),
                                              (self.num_agent, self.num_agent)).to_dense()
            if cat == 'avail_action_next':
                yield datas[7][s + 1]

            if cat == 'status':
                yield datas[8][s]
            if cat == 'status_next':
                yield datas[8][s+1]

    #def search_sample_space(self, sampled_batch_idx):




    def sample(self, vdn):
        step_count_list = self.step_count_list[:]
        step_count_list.pop()

        if vdn == False:
            priority_point = list(self.buffer[9])[:]
            priority_point.pop()
            one_ratio = priority_point.count(1)/len(priority_point)
            #print(step_count_list)

            if np.random.uniform(0, 1) <= one_ratio:
                #print(len(step_count_list), (np.array(priority_point)/np.sum(priority_point)).shape)
                sampled_batch_idx =np.random.choice(step_count_list, p = np.array(priority_point)/np.sum(priority_point), size = self.batch_size)
            else:
                sampled_batch_idx = np.random.choice(step_count_list, size=self.batch_size)
        else:
            sampled_batch_idx = np.random.choice(step_count_list, size=self.batch_size)

        node_feature_machine = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_machine')
        node_features_machine = list(node_feature_machine)

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)

        num_waiting_operations = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='num_waiting_operations')
        num_waiting_operations = list(num_waiting_operations)

        num_waiting_operations_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                                 cat='num_waiting_operations_next')
        num_waiting_operations_next = list(num_waiting_operations_next)

        edge_index_machine = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_machine')
        edge_indices_machine = list(edge_index_machine)

        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)

        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)

        #
        # node_feature_job_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_job_next')
        # node_features_job_next = list(node_feature_job_next)

        node_feature_machine_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                               cat='node_feature_machine_next')
        node_features_machine_next = list(node_feature_machine_next)

        # edge_index_job_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_job_next')
        # edge_indices_job_next = list(edge_index_job_next)

        edge_index_machine_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                             cat='edge_index_machine_next')
        edge_indices_machine_next = list(edge_index_machine_next)

        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        status = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status')
        status = list(status)


        status_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status_next')
        status_next = list(status_next)


        return node_features_machine, num_waiting_operations, edge_indices_machine, actions, rewards, dones, node_features_machine_next, num_waiting_operations_next, edge_indices_machine_next, avail_actions_next, status, status_next


class Agent:
    def __init__(self,
                 num_agent,
                 feature_size_job,
                 feature_size_machine,

                 hidden_size_obs,
                 hidden_size_comm,
                 hidden_size_Q,
                 hidden_size_meta_path,
                 n_multi_head,
                 n_representation_job,
                 n_representation_machine,

                 dropout,
                 action_size,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 gamma,
                 GNN,
                 teleport_probability,
                 gtn_beta):

        self.num_agent = num_agent
        self.feature_size_job = feature_size_job
        self.feature_size_machine = feature_size_machine
        self.hidden_size_meta_path = hidden_size_meta_path
        self.hidden_size_obs = hidden_size_obs
        self.hidden_size_comm = hidden_size_comm
        self.n_multi_head = n_multi_head
        self.teleport_probability = teleport_probability

        self.n_representation_job = n_representation_job
        self.n_representation_machine = n_representation_machine

        self.action_size = action_size

        self.dropout = dropout
        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()
        self.agent_index = [i for i in range(self.num_agent)]
        self.max_norm = 10
        self.VDN = VDN().to(device)
        self.VDN_target = VDN().to(device)

        self.VDN_target.load_state_dict(self.VDN.state_dict())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, self.num_agent, self.action_size)

        self.action_space = [i for i in range(self.action_size)]

        self.GNN = GNN
        if self.GNN == 'GAT':
            self.node_representation_job_obs = NodeEmbedding(feature_size=feature_size_job,
                                                             hidden_size=hidden_size_obs,
                                                             n_representation_obs=n_representation_job,
                                                             n_agent=self.num_agent).to(device)  # 수정사항

            self.node_representation = NodeEmbedding(feature_size=feature_size_machine, hidden_size=hidden_size_obs,
                                                     n_representation_obs=n_representation_machine, machine=True,
                                                     n_agent=self.num_agent).to(device)  # 수정사항

            self.func_job_obs = GAT(nfeat=n_representation_job,
                                    nhid=hidden_size_obs,
                                    nheads=n_multi_head,
                                    nclass=n_representation_job,
                                    dropout=dropout,
                                    alpha=0.2,
                                    mode='observation',
                                    n_node=self.num_agent,
                                    batch_size=self.batch_size,
                                    teleport_probability=self.teleport_probability).to(device)

            self.func_machine_comm = GAT(nfeat=n_representation_machine,
                                         nhid=hidden_size_comm,
                                         nheads=n_multi_head,
                                         nclass=n_representation_machine + 5,
                                         dropout=dropout,
                                         alpha=0.2,
                                         mode='observation',
                                         n_node=self.num_agent,
                                         batch_size=self.batch_size,
                                         teleport_probability=self.teleport_probability).to(device)  # 수정사항

            self.Q = IQN(n_representation_job + n_representation_machine + 5, self.action_size,
                         batch_size=self.batch_size).to(device)
            self.Q_tar = IQN(n_representation_job + n_representation_machine + 5, self.action_size,
                             batch_size=self.batch_size).to(device)
            # Network(n_representation_job+n_representation_machine+5, hidden_size_Q).to(device)
            # self.Q_tar = Network(n_representation_job+n_representation_machine+5, hidden_size_Q, self.action_size).to(device)
            self.Q_tar.load_state_dict(self.Q.state_dict())

            self.eval_params = list(self.VDN.parameters()) + \
                               list(self.Q.parameters()) + \
                               list(self.node_representation_job_obs.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.func_job_obs.parameters()) + \
                               list(self.func_machine_comm.parameters())
        if self.GNN == 'FastGTN':
            self.Q = Network(
                hidden_size_meta_path +
                n_representation_obs,
                hidden_size_Q).to(device)
            self.Q_tar = Network(hidden_size_meta_path + n_representation_obs, hidden_size_Q).to(device)
            self.action_representation = NodeEmbedding(feature_size=feature_size + 6 - 1, hidden_size=hidden_size_obs,
                                                       n_representation_obs=n_representation_obs).to(device)  # 수정사항

            # num_edge_type, feature_size, num_nodes, num_FastGTN_layers, hidden_size, num_channels, num_FastGT_layers)
            self.func_meta_path = FastGTNs(num_edge_type=5,
                                           feature_size=feature_size,
                                           num_nodes=self.num_nodes,
                                           num_FastGTN_layers=2,
                                           hidden_size=hidden_size_meta_path,
                                           num_channels=2,
                                           num_layers=2,
                                           teleport_probability=self.teleport_probability,
                                           gtn_beta=gtn_beta
                                           ).to(device)
            self.node_representation = NodeEmbedding(feature_size=feature_size - 1, hidden_size=hidden_size_obs,
                                                     n_representation_obs=n_representation_obs).to(device)  # 수정사항
            self.eval_params = list(self.VDN.parameters()) + \
                               list(self.Q.parameters()) + \
                               list(self.func_meta_path.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.action_representation.parameters())
        self.optimizer = optim.RMSprop(self.eval_params, lr=learning_rate)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=38000, gamma=0.1)
        self.time_check = [[], []]

    def save_model(self, e, t, epsilon, path):
        torch.save({
            'e': e,
            't': t,
            'epsilon': epsilon,
            'Q': self.Q.state_dict(),
            'Q_tar': self.Q_tar.state_dict(),
            'node_representation_job_obs': self.node_representation_job_obs.state_dict(),
            'node_representation': self.node_representation.state_dict(),
            'func_job_obs': self.func_job_obs.state_dict(),
            'func_machine_comm': self.func_machine_comm.state_dict(),
            'VDN': self.VDN.state_dict(),
            'VDN_target': self.VDN_target.state_dict(),
            'optimizer': self.optimizer.state_dict()}, "{}".format(path))

    def eval_check(self, eval):
        if eval == True:
            self.node_representation_job_obs.eval()
            self.node_representation.eval()
            self.func_job_obs.eval()
            self.func_machine_comm.eval()
            self.Q.eval()
            self.Q_tar.eval()
        else:
            self.node_representation_job_obs.train()
            self.node_representation.train()
            self.func_job_obs.train()
            self.func_machine_comm.train()
            self.Q.train()
            self.Q_tar.train()

    def load_model(self, path):
        checkpoint = torch.load(path)
        e = checkpoint["e"]
        t = checkpoint["t"]
        epsilon = checkpoint["epsilon"]
        self.Q.load_state_dict(checkpoint["Q"])
        self.Q_tar.load_state_dict(checkpoint["Q_tar"])
        self.node_representation_job_obs.load_state_dict(checkpoint["node_representation_job_obs"])
        self.node_representation.load_state_dict(checkpoint["node_representation"])
        self.func_job_obs.load_state_dict(checkpoint["func_job_obs"])
        self.func_machine_comm.load_state_dict(checkpoint["func_machine_comm"])
        self.VDN.load_state_dict(checkpoint["VDN"])
        self.VDN_target.load_state_dict(checkpoint["VDN_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.eval_params = list(self.VDN.parameters()) + \
                           list(self.Q.parameters()) + \
                           list(self.node_representation_job_obs.parameters()) + \
                           list(self.node_representation.parameters()) + \
                           list(self.func_job_obs.parameters()) + \
                           list(self.func_machine_comm.parameters())
        return e, t, epsilon

    def get_node_representation(self, node_feature_machine, num_waiting_operations, edge_index_machine,
                                n_node_features_machine, mini_batch=False):
        if self.GNN == 'GAT':
            if mini_batch == False:
                with torch.no_grad():

                    #workcenter_encodes = torch.tensor(node_feature_machine, dtype=torch.float).to(device)[:, -2:]

                    node_feature_machine = torch.tensor(node_feature_machine,
                                                        dtype=torch.float,
                                                        device=device)

                    num_waiting_operations = torch.tensor(num_waiting_operations,
                                                          dtype=torch.float,
                                                          device=device)


                    # num_waiting_operations = torch.stack(
                    #     [torch.cat([num_waiting_operations, torch.tensor(workcenter_encodes[i])]) for i in
                    #      range(self.num_agent)])
                    node_embedding_num_waiting_operations = self.node_representation_job_obs(num_waiting_operations)
                    node_embedding_machine_obs = self.node_representation(node_feature_machine, machine=True)
                    edge_index_machine = torch.tensor(edge_index_machine, dtype=torch.long, device=device)
                    node_representation = self.func_machine_comm(node_embedding_machine_obs, edge_index_machine,
                                                                 n_node_features_machine, mini_batch=mini_batch)
                    node_representation = torch.cat(
                        [node_embedding_num_waiting_operations.squeeze(0), node_representation], dim=1)
            else:

                #workcenter_encodes = torch.tensor(node_feature_machine, dtype=torch.float).to(device)[:, :, -2:]

                node_feature_machine = torch.tensor(node_feature_machine, dtype=torch.float).to(device)
                num_waiting_operations = torch.tensor(num_waiting_operations,
                                                      dtype=torch.float,
                                                      ).to(device)

                # num_waiting_operations = num_waiting_operations.unsqueeze(1).expand(
                #     [self.batch_size, self.num_agent, num_waiting_operations.shape[1]])

                #num_waiting_operations = torch.cat([num_waiting_operations, workcenter_encodes], dim=2)

                # print(num_waiting_operations.shape)

                node_embedding_num_waiting_operations = self.node_representation_job_obs(num_waiting_operations)

                node_embedding_machine_obs = self.node_representation(node_feature_machine)

                # edge_index_machine = torch.tensor(edge_index_machine, dtype=torch.long, device=device)

                node_representation = self.func_machine_comm(node_embedding_machine_obs, edge_index_machine,
                                                             n_node_features_machine,
                                                             mini_batch=mini_batch)
                node_representation = torch.cat([node_embedding_num_waiting_operations, node_representation], dim=2)

            """
            node_representation 
            - training 시        : batch_size X num_nodes X feature_size 
            - action sampling 시 : num_nodes X feature_size
            """
        if self.GNN == 'FastGTN':
            if mini_batch == False:
                with torch.no_grad():
                    node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
                    A = self.get_heterogeneous_adjacency_matrix(edge_index_enemy, edge_index_ally)
                    node_representation = self.func_meta_path(A, node_feature, num_nodes=self.num_nodes,
                                                              mini_batch=mini_batch)

            else:
                node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
                A = [self.get_heterogeneous_adjacency_matrix(edge_index_enemy[m], edge_index_ally[m]) for m in
                     range(self.batch_size)]
                node_representation = self.func_meta_path(A, node_feature, num_nodes=self.num_nodes,
                                                          mini_batch=mini_batch)
                # node_representation = torch.stack(node_representation, dim = 0).to(device)
        return node_representation

    def get_heterogeneous_adjacency_matrix(self, edge_index_enemy, edge_index_ally):
        A = []
        edge_index_enemy_transpose = deepcopy(edge_index_enemy)
        edge_index_enemy_transpose[1] = edge_index_enemy[0]
        edge_index_enemy_transpose[0] = edge_index_enemy[1]
        edge_index_ally_transpose = deepcopy(edge_index_ally)
        edge_index_ally_transpose[1] = edge_index_ally[0]
        edge_index_ally_transpose[0] = edge_index_ally[1]
        edges = [edge_index_enemy,
                 edge_index_enemy_transpose,
                 edge_index_ally,
                 edge_index_ally_transpose]
        for i, edge in enumerate(edges):
            edge = torch.tensor(edge, dtype=torch.long, device=device)
            value = torch.ones(edge.shape[1], dtype=torch.float, device=device)

            deg_inv_sqrt, deg_row, deg_col = _norm(edge.detach(),
                                                   self.num_nodes,
                                                   value.detach())  # row의 의미는 차원이 1이상인 node들의 index를 의미함

            value = deg_inv_sqrt[
                        deg_row] * value  # degree_matrix의 inverse 중에서 row에 해당되는(즉, node의 차원이 1이상인) node들만 골라서 value_tmp를 곱한다
            A.append((edge, value))

        edge = torch.stack((torch.arange(0, self.num_nodes), torch.arange(0, self.num_nodes))).type(
            torch.cuda.LongTensor)
        value = torch.ones(self.num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge, value))
        return A

    def cal_Q(self, obs, actions, avail_actions_next, agent_id, cos, vdn, target=False):

        """

        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size

        """
        if vdn == True:
            if target == False:

                obs_n = obs[:, agent_id]
                q = self.Q(obs_n, cos, mini_batch=True)
                actions = torch.tensor(actions, device=device).long()
                act_n = actions[:, agent_id].unsqueeze(1)  # action.shape : (batch_size, 1)
                q = torch.gather(q, 1, act_n).squeeze(1)  # q.shape :      (batch_size, 1)
                return q
            else:
                with torch.no_grad():

                    # cos, taus = self.Q_tar.calc_cos(self.batch_size)

                    obs_next = obs
                    obs_next = obs_next[:, agent_id]
                    q_tar = self.Q_tar(obs_next, cos, mini_batch=True)  # q.shape :      (batch_size, action_size, 1)
                    avail_actions_next = torch.tensor(avail_actions_next, device=device).bool()
                    mask = avail_actions_next[:, agent_id]
                    q_tar = q_tar.masked_fill(mask == 0, float('-inf'))
                    q_tar_max = torch.max(q_tar, dim=1)[0]
                    return q_tar_max
        else:
            if target == False:
                q = self.Q(obs, cos, mini_batch=True)
                # q.shape : batch_size, action_size
                actions = actions.long().to(device)
                q = torch.gather(q, 1, actions)
                return q.squeeze(1)
            else:
                obs_next = obs
                q_tar = self.Q(obs_next, cos, mini_batch=True)
                avail_actions_next = avail_actions_next.bool().to(device)
                mask = avail_actions_next
                q_tar = q_tar.masked_fill(mask == 0, float('-inf'))
                q_tar_max = torch.max(q_tar, dim=1)[0]
                return q_tar_max

    @torch.no_grad()
    def sample_action(self, node_representation, avail_action, epsilon):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        mask = torch.tensor(avail_action, device=device).bool()
        action = []
        utility = list()
        cos, taus = self.Q.calc_cos(1)

        for n in range(self.num_agent):
            obs = node_representation[n]
            Q = self.Q(obs, cos, mini_batch=False)
            Q = Q.masked_fill(mask[n, :] == 0, float('-inf'))

            greedy_u = torch.argmax(Q)

            # print(Q)

            mask_n = np.array(avail_action[n], dtype=np.float64)
            if np.random.uniform(0, 1) >= epsilon:
                u = greedy_u
                utility.append(Q[0][u].detach().item())
                action.append(u)
            else:
                u = np.random.choice(self.action_space, p=mask_n / np.sum(mask_n))
                utility.append(Q[0][u].detach().item())
                action.append(u)
        return action, utility

    def learn(self, regularizer, vdn = False):

        # import time
        # start = time.time()
        node_features_machine, num_waiting_operations, edge_indices_machine, actions, rewards, dones, node_features_machine_next, num_waiting_operations_next, edge_indices_machine_next, avail_actions_next, status, status_next = self.buffer.sample(vdn = vdn)
        # print("시간측정 1", time.time()-start)
        # dummy = [0] * self.feature_size_job

        # max_job_length = np.max([np.max([len(nfj) for nfj in node_features_job]), np.max([len(nfj) for nfj in node_features_job_next])])
        # import time
        # start = time.time()
        # for nfj in node_features_job:
        #     if len(nfj) <= max_job_length:
        #         for j in range(max_job_length- len(nfj)):
        #             nfj.append(dummy)
        #
        # for nfj in node_features_job_next:
        #     if len(nfj) <= max_job_length:
        #         for j in range(max_job_length- len(nfj)):
        #             nfj.append(dummy)

        # remain_duration = time.time() - start

        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """
        n_node_features_machine = torch.tensor(node_features_machine).shape[1]
        obs = self.get_node_representation(
            node_features_machine,
            num_waiting_operations,
            edge_indices_machine,
            n_node_features_machine,
            mini_batch=True)

        obs_next = self.get_node_representation(
            node_features_machine_next,
            num_waiting_operations_next,
            edge_indices_machine_next,
            n_node_features_machine,
            mini_batch=True)





        #print(status/status.sum(dim = 1, keepdims = True))
        if vdn == True:
            status = torch.tensor(status, device=device, dtype=torch.float)
            status_next = torch.tensor(status_next, device=device, dtype=torch.float)
            torch.where(status)
            dones = torch.tensor(dones, device=device, dtype=torch.float)
            rewards = torch.tensor(rewards, device=device, dtype=torch.float)
            cos, taus = self.Q.calc_cos(self.batch_size)
            q = [self.cal_Q(obs=obs,
                            actions=actions,
                            avail_actions_next=None,
                            agent_id=agent_id,
                            target=False,
                            cos=cos,vdn = vdn) for agent_id in range(self.num_agent)]
            q_tar = [self.cal_Q(obs=obs_next,
                                actions=None,
                                avail_actions_next=avail_actions_next,
                                agent_id=agent_id,
                                target=True,
                                cos=cos, vdn = vdn) for agent_id in range(self.num_agent)]
            q_tot = torch.stack(q, dim=1)
            q_tot_tar = torch.stack(q_tar, dim=1)

            q_tot = q_tot * status/status.sum(dim = 1, keepdims = True)
            q_tot_tar = q_tot_tar* status_next/status_next.sum(dim = 1, keepdims = True)
            q_tot = self.VDN(q_tot)
            q_tot_tar = self.VDN_target(q_tot_tar)
            td_target = rewards  + self.gamma * (1 - dones) * q_tot_tar
            loss1 = F.huber_loss(q_tot, td_target.detach())
            loss = loss1
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_params, 1)
            self.optimizer.step()
            tau = 1e-3
            for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        else:

            sampled_indexes = [np.random.choice(self.agent_index, p = np.array(sta)/np.sum(sta)) for sta in status]
            #print(obs.shape, torch.tensor(sampled_indexes, device = device).shape)
            #print(torch.tensor(sampled_indexes, device = device).unsqueeze(1).unsqueeze(1).long().shape)
            obs = torch.stack([obs[b, sampled_indexes[b], :] for b in range(self.batch_size)])
            obs_next = torch.stack([obs_next[b, sampled_indexes[b], :] for b in range(self.batch_size)])
            rewards = torch.gather(torch.tensor(rewards), 1, torch.tensor(sampled_indexes).long().unsqueeze(1)).to(device)
            actions = torch.gather(torch.tensor(actions), 1, torch.tensor(sampled_indexes).long().unsqueeze(1))
            #rewards = torch.tensor([rewards[b][sampled_indexes[b]] for b in range(self.batch_size)]).to(device)
            #print(rewards.shape, rewards_epi.shape)

            #actions = torch.tensor([actions[b][sampled_indexes[b]] for b in range(self.batch_size)])

            avail_actions_next = torch.tensor([avail_actions_next[b][sampled_indexes[b]] for b in range(self.batch_size)])
            cos, taus = self.Q.calc_cos(self.batch_size)


            # print(obs.shape)
            # print(rewards.shape)
            # print(actions.shape)
            # print(torch.tensor(avail_actions_next).shape)
            # torch.Size([32, 153])
            # torch.Size([32])
            # torch.Size([32])
            # torch.Size([32, 39])
            q_tot =self.cal_Q(obs=obs,
                       actions=actions,
                       avail_actions_next=None,
                       agent_id=None,
                       target=False,
                       vdn = False,
                       cos=cos)

            q_tot_tar =self.cal_Q(obs=obs_next,
                       actions=None,
                       avail_actions_next=avail_actions_next,
                       agent_id=None,
                       target=True,
                       vdn=False,
                       cos=cos)
            #print(rewards.shape, q_tot.shape, q_tot_tar.shape)
            # q_tot = q_tot * status/status.sum(dim = 1, keepdims = True)
            # q_tot_tar = q_tot_tar* status_next/status_next.sum(dim = 1, keepdims = True)
            """
            q_tot.shape : batch_size, num_agent
            rewards.shape : batch_size, num_agent
            """
            dones = torch.tensor(dones, device=device, dtype=torch.float)
            # q_tot = self.VDN(q_tot)
            # q_tot_tar = self.VDN_target(q_tot_tar)
            #print((1-dones).unsqueeze(1).shape, q_tot_tar.shape)

            td_target = rewards.squeeze(1) + self.gamma * (1-dones) * q_tot_tar

            #print(q_tot.shape, rewards.shape, q_tot_tar.shape)
            loss1 = F.huber_loss(q_tot, td_target.detach())
            loss = loss1
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_params, 1)
            self.optimizer.step()
            tau = 1e-3
            for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        # for target_param, local_param in zip(self.VDN_target.parameters(), self.VDN.parameters()):
        #     target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        return loss

