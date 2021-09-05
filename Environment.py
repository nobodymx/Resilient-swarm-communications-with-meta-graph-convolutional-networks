import numpy as np
from copy import deepcopy
from Configurations import *
import Utils


class Environment:
    def __init__(self):
        self.num_of_total_agents = config_num_of_agents

        self.initial_positions = deepcopy(config_initial_swarm_positions)
        self.environment_positions = deepcopy(self.initial_positions)
        self.environment_next_positions = deepcopy(self.environment_positions)

        self.num_of_remain_agents = config_num_of_agents
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_positions = deepcopy(self.environment_positions)
        self.max_destroy_num = config_maximum_destroy_num

    def reset(self):
        self.environment_positions = deepcopy(self.initial_positions)
        self.environment_next_positions = deepcopy(self.environment_positions)
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.num_of_remain_agents = config_num_of_agents
        self.make_remain_positions()

        return deepcopy(self.environment_positions)

    def check_if_connected_graph(self):
        self.make_remain_positions()
        A = Utils.make_A_matrix(self.remain_positions, self.num_of_remain_agents, config_communication_range)
        D = Utils.make_D_matrix(A, self.num_of_remain_agents)
        L = D - A
        connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, self.num_of_remain_agents)
        if connected_flag:
            return 1
        else:
            return 0

    def check_the_clusters(self):
        self.make_remain_positions()
        A = Utils.make_A_matrix(self.remain_positions, self.num_of_remain_agents, config_communication_range)
        D = Utils.make_D_matrix(A, self.num_of_remain_agents)
        L = D - A
        connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, self.num_of_remain_agents)
        return deepcopy(num_of_clusters)

    def next_state(self, actions):
        """
        :param actions: list
        :return:
        """
        for i in self.remain_list:
            delta_positions = actions[i] * config_constant_speed
            self.environment_next_positions[i] = self.environment_positions[i] + delta_positions
        return deepcopy(self.environment_next_positions)

    def update(self):
        self.environment_positions = deepcopy(self.environment_next_positions)

    def stochastic_destroy(self, mode=1, num_of_destroyed=10, real_destroy_list=[], destroy_center=np.array([0, 0, 0]),
                           destroy_range=200):
        destroy_list = []
        destroy_num = 0
        if self.num_of_remain_agents < config_minimum_remain_num:
            print("Warning: the number of remaining nodes is less than the minimal threshold")
            return deepcopy(destroy_num), deepcopy(destroy_list)
        else:
            if mode == 1:
                while True:
                    destroy_num = np.random.randint(0, self.max_destroy_num)
                    if destroy_num <= self.num_of_remain_agents:
                        break
                destroy_index = random.sample(range(0, self.num_of_remain_agents), destroy_num)
                for i in destroy_index:
                    destroy_list.append(self.remain_list[i])
                for i in destroy_list:
                    self.remain_list.remove(i)
                self.num_of_remain_agents -= destroy_num

            elif mode == 2:
                # stochastically destroy
                destroy_num = num_of_destroyed
                if destroy_num >= self.num_of_remain_agents:
                    print("ERROR: already destroy all nodes")
                    return deepcopy(destroy_num), deepcopy(destroy_list)
                destroy_index = random.sample(range(0, self.num_of_remain_agents), destroy_num)
                for i in destroy_index:
                    destroy_list.append(self.remain_list[i])
                for i in destroy_list:
                    self.remain_list.remove(i)
                self.num_of_remain_agents -= destroy_num

            elif mode == 4:
                # stochastically destroy
                destroy_num = len(real_destroy_list)
                if destroy_num >= self.num_of_remain_agents:
                    print("ERROR: already destroy all nodes")
                    return deepcopy(destroy_num), deepcopy(destroy_list)
                destroy_list = deepcopy(real_destroy_list)
                for i in destroy_list:
                    self.remain_list.remove(i)
                self.num_of_remain_agents -= destroy_num

            elif mode == 3:
                # destroy a certain range
                destroy_index = []
                for i in range(self.num_of_remain_agents):
                    if np.linalg.norm(self.remain_positions[i] - destroy_center) <= destroy_range:
                        destroy_num += 1
                        destroy_index.append(i)
                for i in destroy_index:
                    destroy_list.append(self.remain_list[i])
                for i in destroy_list:
                    self.remain_list.remove(i)
                self.num_of_remain_agents -= destroy_num

            # remain positions
            self.make_remain_positions()
            # for i in destroy_list:
            #     self.environment_positions[i] = np.array([-100000, -100000, -1000000], dtype=np.float64)
            #     self.environment_next_positions[i] = np.array([-100000, -100000, -100000], dtype=np.float64)

            return deepcopy(destroy_num), deepcopy(destroy_list)

    def make_remain_positions(self):
        self.remain_positions = []
        for i in self.remain_list:
            self.remain_positions.append(deepcopy(self.environment_positions[i]))
        self.remain_positions = np.array(self.remain_positions)
