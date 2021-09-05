from copy import deepcopy
import numpy as np
import Utils
from Configurations import *


class HERO:
    def __init__(self, initial_positions):
        self.init_positions = deepcopy(initial_positions)
        self.num_of_agents = len(self.init_positions)

        self.A = Utils.make_A_matrix(self.init_positions, self.num_of_agents, config_communication_range)
        self.mean = np.zeros((self.num_of_agents, 3))
        self.num_of_neighbors = np.zeros(self.num_of_agents)
        for i in range(self.num_of_agents):
            for j in range(self.num_of_agents):
                if i != j and self.A[i, j] == 1:
                    self.mean[i] += self.init_positions[j]
                    self.num_of_neighbors[i] += 1
            self.mean[i] = self.mean[i] / self.num_of_neighbors[i]

        self.changed_mean = deepcopy(self.mean)

    def hero(self, destroy_index, current_positions):
        speed = np.zeros((self.num_of_agents, 3))
        self.num_of_neighbors = np.zeros(self.num_of_agents)
        self.changed_mean = np.zeros((self.num_of_agents, 3))
        for i in range(self.num_of_agents):
            if i not in destroy_index:
                temp_flag = False
                temp_destroy_list = []
                for j in range(self.num_of_agents):
                    if i != j and self.A[i, j] == 1:
                        if j in destroy_index:
                            temp_destroy_list.append(deepcopy(j))
                            temp_flag = True
                        else:
                            self.changed_mean[i] += self.init_positions[j]
                            self.num_of_neighbors[i] += 1
                self.changed_mean[i] = self.changed_mean[i] / self.num_of_neighbors[i]
                if temp_flag:
                    target_position = 0
                    counter = 0
                    for k in temp_destroy_list:
                        target_position += self.init_positions[k]
                        counter += 1
                    target_position /= counter
                    if np.linalg.norm(
                            target_position - current_positions[i]) > 0.01:
                        speed[i] = (target_position - current_positions[i]) / np.linalg.norm(
                            target_position - current_positions[i])
                else:
                    if np.linalg.norm(
                            self.changed_mean[i] - self.mean[i]) > 0.01:
                        speed[i] = (self.changed_mean[i] - self.mean[i]) / np.linalg.norm(
                            self.changed_mean[i] - self.mean[i])
        self.mean = deepcopy(self.changed_mean)
        return deepcopy(speed)
