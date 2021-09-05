from copy import deepcopy
from torch.optim import Adam
import Utils
from Configurations import *
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time

best_hidden_dimension = 500
best_dropout = 0


class GCN_2017:
    def __init__(self):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout
        self.gcn_network = GCN_fixed_structure(nfeat=3, nhid=self.hidden_dimension, nclass=3,
                                               dropout=self.dropout_value, if_dropout=True, bias=True)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.gcn_network.cuda()

        self.optimizer = Adam(self.gcn_network.parameters(), lr=0.00001)
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def cr_gcm_n(self, global_positions, remain_list):
        start = time.perf_counter()
        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        remain_positions = np.array(remain_positions)
        num_remain = len(remain_list)
        # proposed
        # d_min = smallest_d_algorithm(deepcopy(remain_positions), num_remain, config_communication_range)
        # d_max = Utils.calculate_d_max(deepcopy(remain_positions))
        # A = Utils.make_A_matrix(remain_positions, num_remain, d_min + (d_max - d_min) * 0.25)
        #
        # D = Utils.make_D_matrix(A, num_remain)
        # L = D - A
        # A_norm = np.linalg.norm(A, ord=np.inf)
        # k0 = 1 / A_norm
        # K = 0.99 * k0
        # A_hat = np.eye(num_remain) - K * L

        # 2017
        A = Utils.make_A_matrix(remain_positions, len(remain_positions), config_communication_range)
        A_tilde = A + np.identity(len(A))
        D_tilde = Utils.make_D_matrix(A_tilde, len(remain_positions))

        D_tilde_sqrt = np.diag(D_tilde.diagonal() ** (-0.5))
        A_hat = D_tilde_sqrt.dot(A_tilde).dot(D_tilde_sqrt)

        remain_positions = torch.FloatTensor(remain_positions).type(self.FloatTensor)
        A_hat = torch.FloatTensor(A_hat).type(self.FloatTensor)
        best_final_positions = 0
        best_loss = 1000000000000
        for train_step in range(1000):

            final_positions = self.gcn_network(remain_positions, A_hat)

            final_positions = 0.5 * torch.Tensor(np.array([1000, 1000, 100])).type(self.FloatTensor) * final_positions

            # check if connected
            final_positions_ = final_positions.cpu().data.numpy()
            A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
            D = Utils.make_D_matrix(A, len(A))
            L = D - A
            flag, num = Utils.check_number_of_clusters(L, len(L))
            # loss
            temp_max = 0
            max_index = 0

            for j in range(len(final_positions)):
                if torch.norm(final_positions[j] - remain_positions[j]) > temp_max:
                    temp_max = torch.norm(final_positions[j] - remain_positions[j])
                    max_index = j
            loss = 1000 * (num - 1) + torch.norm(final_positions[max_index] - remain_positions[max_index])
            # loss_F = 1000 * (num - 1) + torch.norm(final_positions-F,p='fro')

            if loss.cpu().data.numpy() < best_loss:
                best_loss = deepcopy(loss.cpu().data.numpy())
                best_final_positions = deepcopy(final_positions.cpu().data.numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        speed = np.zeros((config_num_of_agents, 3))
        remain_positions_numpy = remain_positions.cpu().data.numpy()
        temp_max_distance = 0
        for i in range(num_remain):
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / np.linalg.norm(
                    best_final_positions[i] - remain_positions_numpy[i])
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > temp_max_distance:
                temp_max_distance = deepcopy(np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]))

        max_time = temp_max_distance / config_constant_speed
        # print(max_time)
        end = time.perf_counter()
        print(end-start)
        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_fixed_structure(nn.Module):
    def __init__(self, nfeat=3, nhid=5, nclass=3, dropout=0.5, if_dropout=True, bias=True):
        super(GCN_fixed_structure, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc3 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc4 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc5 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc6 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc7 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc8 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout
        self.training = if_dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = self.gc8(x, adj)
        return torch.tanh(x) + 1
