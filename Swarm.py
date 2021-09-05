import Utils
from copy import deepcopy
from Configurations import *
from Main_algorithm_GCN.GCO import GCO
from Main_algorithm_GCN.CR_MGC import CR_MGC
from Traditional_Algorithm.GCN_2017 import GCN_2017
from Traditional_Algorithm.Centering import centering_fly
from Traditional_Algorithm.SIDR import SIDR
from Traditional_Algorithm.CSDS import CSDS
from Traditional_Algorithm.HERO import HERO
import torch


class Swarm:
    def __init__(self, algorithm_mode=0, enable_csds=False, meta_param_use=False):
        self.initial_positions = deepcopy(config_initial_swarm_positions)
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.num_of_agents = config_num_of_agents
        self.max_destroy_num = config_maximum_destroy_num

        self.remain_positions = deepcopy(self.initial_positions)
        self.true_positions = deepcopy(self.initial_positions)

        self.database = [{"known_positions": deepcopy(self.initial_positions),
                          "existing_list": [i for i in range(config_num_of_agents)],
                          "connected": True,
                          "if_destroyed": False} for i in range(config_num_of_agents)]
        # 0 for CSDS, 1 for centering, 2 for SIDR, 3 for GCN_2017, 4 for CR-GCM, 5 for CR_GCM_N
        self.algorithm_mode = algorithm_mode

        self.gco = GCO()
        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, 3))
        self.max_time = 0

        self.cr_gcm = CR_MGC(use_meta=meta_param_use)
        self.gcn_2017 = GCN_2017()

        self.hero = HERO(self.initial_positions)

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, 3))

        if enable_csds:
            self.csds = CSDS(config_num_of_agents, self.initial_positions)
        self.best_final_positions = 0

        self.notice_destroy = False
        self.destination_positions = np.zeros((self.num_of_agents, 3))
        self.inertia_counter = 0
        self.inertia = 100
        self.if_finish = [True for i in range(self.num_of_agents)]

        self.time_consuming = []

    def destroy_happens(self, destroy_list, environment_positions):
        self.notice_destroy = True
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)
        # self.csds.notice_destroy(deepcopy(destroy_list))

    def update_true_positions(self, environment_positions):
        self.true_positions = deepcopy(environment_positions)

    def reset(self, change_algorithm_mode=False, algorithm_mode=0):
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.database = [{"known_positions": deepcopy(self.initial_positions),
                          "existing_list": [i for i in range(config_num_of_agents)],
                          "connected": True,
                          "if_destroyed": False} for i in range(config_num_of_agents)]
        self.positions = []
        self.mean_positions = []
        self.target_positions = []
        self.max_time = 0

        if change_algorithm_mode:
            self.algorithm_mode = algorithm_mode

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, 3))

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, 3))

    def take_actions(self):
        """
        take actions with global information (GI)
        :return: unit speed vectors
        """
        actions = np.zeros((self.num_of_agents, 3))
        max_time = 0
        self.make_remain_positions()
        flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(self.remain_positions), len(self.remain_list))
        if flag:
            # print("connected")
            return deepcopy(actions), max_time
        else:
            if self.algorithm_mode == 0:
                # CSDS
                actions_csds, max_time = self.csds.csds(deepcopy(self.true_positions), deepcopy(self.remain_list))

                for i in self.remain_list:
                    actions[i] = 0.05 * centering_fly(self.true_positions, self.remain_list, i) + 0.95 * actions_csds[i]

            elif self.algorithm_mode == 1:
                # HERO
                actions_hero = self.hero.hero(
                    Utils.difference_set([i for i in range(self.num_of_agents)], self.remain_list), self.true_positions)

                for i in self.remain_list:
                    actions[i] = 0.2 * centering_fly(self.true_positions, self.remain_list, i) + 0.8 * actions_hero[i]


            elif self.algorithm_mode == 2:
                # centering
                for i in self.remain_list:
                    actions[i] = centering_fly(self.true_positions, self.remain_list, i)

            elif self.algorithm_mode == 3:
                # SIDR
                actions = SIDR(self.true_positions, self.remain_list)


            elif self.algorithm_mode == 4:
                # GCN_2017
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(
                                self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        # else:
                        #     print("%d already finish" % self.remain_list[i])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.gcn_2017.cr_gcm_n(deepcopy(self.true_positions),
                                                                                     deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)
            elif self.algorithm_mode == 5:
                # proposed algorithm
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(
                                self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])

                        # else:
                        #     print("%d already finish" % self.remain_list[i])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.cr_gcm.cr_gcm(deepcopy(self.true_positions),
                                                                                 deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)
            else:
                print("No such algorithm")
        return deepcopy(actions), deepcopy(max_time)

    def make_remain_positions(self):
        self.remain_positions = []
        for i in self.remain_list:
            self.remain_positions.append(deepcopy(self.true_positions[i]))
        self.remain_positions = np.array(self.remain_positions)

    def check_if_finish(self, cluster_index):
        flag = True
        for i in range(len(cluster_index)):
            if not self.if_finish[self.remain_list[cluster_index[i]]]:
                flag = False
                break
        return flag

    def save_GCN(self, filename):
        torch.save(self.cr_gcm.gcn_network, filename)
