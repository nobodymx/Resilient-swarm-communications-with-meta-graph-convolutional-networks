import Utils
from copy import deepcopy
from Configurations import *
from Utils import Stack
from Main_algorithm_GCN.CR_MGC import CR_MGC
from Traditional_Algorithm.GCN_2017 import GCN_2017
from Traditional_Algorithm.Centering import centering_fly
from Traditional_Algorithm.SIDR import SIDR
from Traditional_Algorithm.CSDS import CSDS
from Traditional_Algorithm.HERO import HERO
import torch
import time


class Swarm:
    def __init__(self, algorithm_mode=0, meta_param_use=False):
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
        # 0 for CSDS, 1 for centering, 2 for SIDR, 3 for GCN_2017, 5 for CR_GCM_N
        self.algorithm_mode = algorithm_mode

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, 3))
        self.max_time = 0

        self.cr_mgc = CR_MGC(use_meta=meta_param_use)
        self.gcn_2017 = GCN_2017()

        self.hero = HERO(self.initial_positions)

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, 3))

        # self.csds = CSDS(config_num_of_agents, self.initial_positions)
        self.best_final_positions = 0

        self.notice_destroy = False
        self.destination_positions = np.zeros((self.num_of_agents, 3))
        self.inertia_counter = 0
        self.inertia = 100
        self.if_finish = [True for i in range(self.num_of_agents)]

        self.time_consuming = []

    def destroy_happens(self, destroy_list, environment_positions):
        self.notice_destroy = True
        self.inertia_counter = 0
        for i in destroy_list:
            self.remain_num -= 1
            self.remain_list.remove(i)
            self.database[i]["if_destroyed"] = True
            self.broadcast_destroy_information(i, environment_positions, destroy_list)

    def destroy_happens_GI_version(self, destroy_list, environment_positions):
        self.notice_destroy = True
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)

    # self.csds.notice_destroy(deepcopy(destroy_list))

    def update_true_positions(self, environment_positions):
        self.true_positions = deepcopy(environment_positions)

    def broadcast_destroy_information(self, No_destroy, environment_positions, destroy_list):
        """
        Broadcast according to the environment positions
        :param No_destroy:
        :param environment_positions:
        :return:
        """
        visited = np.zeros(config_num_of_agents)
        counter = 0
        stack = Stack()
        stack.push(No_destroy)
        visited[No_destroy] = 1
        counter += 1

        virtual_positions = []

        # for i in range(config_num_of_agents):
        #     if i in destroy_list and i != No_destroy:
        #         virtual_positions.append(deepcopy(np.array([-10000000.0, -10000000.0, -10000000.0])))
        #         # virtual_positions.append(deepcopy(environment_positions[i]))
        #     else:
        #         virtual_positions.append(deepcopy(environment_positions[i]))
        # second mechanism
        virtual_positions = deepcopy(environment_positions)

        virtual_positions = np.array(virtual_positions, dtype=np.float64)
        A = Utils.make_A_matrix(virtual_positions, config_num_of_agents, config_communication_range)

        while stack.length() != 0:
            current = stack.top_element()
            flag = True
            temp_counter = 0
            for i in range(config_num_of_agents):
                if A[current, i] == 1 and visited[temp_counter] == 0:
                    visited[temp_counter] = 1
                    counter += 1
                    stack.push(i)
                    flag = False
                    break
                temp_counter += 1
            if flag:
                stack.pop()

        for i in range(config_num_of_agents):
            if visited[i] == 1:
                self.database[i]["existing_list"].remove(No_destroy)

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

    def broadcast_next_position_information(self, environment_next_positions):
        """
        broadcast position information to other UAVs in the same cluster based no the topology formed by real positions
        :param environment_next_positions:
        :return:
        """
        A = Utils.make_A_matrix(environment_next_positions, config_num_of_agents, config_communication_range)
        for i in self.remain_list:
            self.database[i]["known_positions"][i] = deepcopy(environment_next_positions[i])
            visited = np.zeros(config_num_of_agents)
            stack = Stack()
            stack.push(i)
            visited[i] = 1

            while stack.length() != 0:
                current = stack.top_element()
                flag = True
                for j in range(config_num_of_agents):
                    if A[current, j] == 1 and visited[j] == 0:
                        visited[j] = 1
                        stack.push(j)
                        flag = False
                        break
                if flag:
                    stack.pop()

            for k in range(config_num_of_agents):
                if visited[k] == 1:
                    self.database[k]["known_positions"][i] = deepcopy(environment_next_positions[i])

    def broadcast_remain_list_information(self, environment_positions):
        """
        broadcast remain list information to other UAVs in the same cluster based no the topology formed by real positions
        :param environment_positions:
        :return:
        """
        num_of_remain = len(self.remain_list)
        remain_positions = []
        for i in self.remain_list:
            remain_positions.append(deepcopy(environment_positions[i]))
        remain_positions = np.array(remain_positions, dtype=np.float64)
        A = Utils.make_A_matrix(remain_positions, num_of_remain, config_communication_range)
        D = Utils.make_D_matrix(A, num_of_remain)
        L = D - A
        connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, num_of_remain)
        positions_with_clusters, cluster_index = Utils.split_the_positions_into_clusters_and_indexes(remain_positions,
                                                                                                     num_of_clusters,
                                                                                                     Utils.make_A_matrix(
                                                                                                         remain_positions,
                                                                                                         num_of_remain,
                                                                                                         config_communication_range))
        true_cluster_index = []
        for i in range(num_of_clusters):
            temp_true_index = []
            for j in cluster_index[i]:
                temp_true_index.append(deepcopy(self.remain_list[j]))
            true_cluster_index.append(deepcopy(temp_true_index))

        for i in range(num_of_clusters):
            temp_remain_list = self.database[true_cluster_index[i][0]]["existing_list"]
            for j in range(len(true_cluster_index[i]) - 1):
                # take intersections between remain lists in IDs
                temp_remain_list = Utils.intersection_set(deepcopy(temp_remain_list), deepcopy(
                    self.database[true_cluster_index[i][j + 1]]["existing_list"]))
            for k in range(len(true_cluster_index[i])):
                self.database[true_cluster_index[i][k]]["existing_list"] = deepcopy(temp_remain_list)

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
            print("connected")
            return deepcopy(actions), max_time
        else:
            if self.algorithm_mode == 0:
                # CSDS
                actions, max_time = self.csds.csds(deepcopy(self.true_positions), deepcopy(self.remain_list))

            elif self.algorithm_mode == 1:
                # centering
                for i in self.remain_list:
                    actions[i] = centering_fly(self.true_positions, self.remain_list, i)

            elif self.algorithm_mode == 2:
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
            elif self.algorithm_mode == 6:
                # CR-GCM-N
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
                    actions, max_time, best_final_positions = self.cr_mgc.cr_gcm(deepcopy(self.true_positions),
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

    def take_actions_GI_continuous_mode(self):
        """
        take actions with global information (GI)
        :return: unit speed vectors
        """
        actions = np.zeros((self.num_of_agents, 3))
        self.make_remain_positions()
        flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(self.remain_positions), len(self.remain_list))
        if flag:
            print("connected")
            return deepcopy(actions)
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
                # GCN2017
                if self.notice_destroy:
                    self.notice_destroy = False
                    actions, max_time, final_positions = self.gcn_2017.cr_gcm_n(deepcopy(self.true_positions),
                                                                                deepcopy(self.remain_list))
                    for i in range(len(self.remain_list)):
                        self.destination_positions[self.remain_list[i]] = deepcopy(final_positions[i])
                else:
                    for i in self.remain_list:
                        if np.linalg.norm(self.destination_positions[i] - self.true_positions[i]) >= 0.55:
                            actions[i] = deepcopy(
                                (self.destination_positions[i] - self.true_positions[i]) / np.linalg.norm(
                                    self.destination_positions[i] - self.true_positions[i]))

            elif self.algorithm_mode == 6:
                # CR-GCM-N
                if self.notice_destroy:
                    self.notice_destroy = False
                    actions, max_time, final_positions = self.cr_mgc.cr_gcm(deepcopy(self.true_positions),
                                                                            deepcopy(self.remain_list))
                    for i in range(len(self.remain_list)):
                        self.destination_positions[self.remain_list[i]] = deepcopy(final_positions[i])
                else:
                    for i in self.remain_list:
                        if np.linalg.norm(self.destination_positions[i] - self.true_positions[i]) >= 0.55:
                            actions[i] = deepcopy(
                                (self.destination_positions[i] - self.true_positions[i]) / np.linalg.norm(
                                    self.destination_positions[i] - self.true_positions[i]))
            else:
                print("No such algorithm")
        return deepcopy(actions)

    def take_actions_incomplete_information(self):
        """
        take actions with incomplete information in IDs (II)
        :return: unit speed vectors
        """
        actions = np.zeros((self.num_of_agents, 3))
        if self.algorithm_mode == 0:
            # CRS
            pass
        elif self.algorithm_mode == 1:
            # centering
            pass
        elif self.algorithm_mode == 2:
            # SIDR
            pass
        elif self.algorithm_mode == 3:
            # GCN-2017
            positions_with_clusters, cluster_index = Utils.split_the_positions_into_clusters_and_indexes(
                self.remain_positions, )
            for uav in self.remain_list:
                print("decision_for_UAV_%d" % uav)
                temp_actions, max_time = self.gcn_2017.cr_gcm_n(deepcopy(self.database[uav]["known_positions"]),
                                                                deepcopy(self.database[uav]["existing_list"]))
                actions[uav] = deepcopy(temp_actions[uav])

        elif self.algorithm_mode == 5:
            # CR-GCM-N
            # CR-GCM
            remain_positions = []
            for i in self.remain_list:
                remain_positions.append(deepcopy(self.true_positions[i]))
            remain_positions = np.array(remain_positions)

            A = Utils.make_A_matrix(remain_positions, len(self.remain_list), config_communication_range)
            flag, num_of_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))
            positions_with_clusters, cluster_index = Utils.split_the_positions_into_clusters_and_indexes(
                remain_positions, num_of_clusters, A)

            for cluster in range(num_of_clusters):
                temp_actions, final_positions, max_time = self.cr_gcm_n.cr_gcm_n(
                    deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["known_positions"]),
                    deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["existing_list"]))
                for i in cluster_index[cluster]:
                    actions[self.remain_list[i]] = temp_actions[i]

        else:
            print("No such algorithm")
        return deepcopy(actions)

    def take_actions_incomplete_information_continuous(self):
        """
        take actions with incomplete information in IDs (II)
        :return: unit speed vectors
        """
        record_time = False

        actions = np.zeros((self.num_of_agents, 3))
        if self.algorithm_mode == 0:
            # CSDS
            remain_positions = []
            for i in self.remain_list:
                remain_positions.append(deepcopy(self.true_positions[i]))
            remain_positions = np.array(remain_positions)
            A = Utils.make_A_matrix(remain_positions, len(self.remain_list), config_communication_range)
            flag, num_of_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))
            positions_with_clusters, cluster_index = Utils.split_the_positions_into_clusters_and_indexes(
                remain_positions, num_of_clusters, A)

            for cluster in range(num_of_clusters):
                temp_remain_positions = []
                for i in self.database[self.remain_list[cluster_index[cluster][0]]]["existing_list"]:
                    temp_remain_positions.append(
                        deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["known_positions"][i]))
                temp_remain_positions = np.array(temp_remain_positions)
                flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(temp_remain_positions),
                                                                     len(self.database[self.remain_list[
                                                                         cluster_index[cluster][0]]]["existing_list"]))
                if flag:
                    start = time.perf_counter()
                    print("connected")
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                    # return deepcopy(actions)
                elif num_of_clusters == 1:
                    start = time.perf_counter()
                    print("connected")
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                else:
                    start = time.perf_counter()
                    actions_csds, max_time = self.csds.csds(
                        self.database[self.remain_list[cluster_index[cluster][0]]]["known_positions"]
                        , self.database[self.remain_list[cluster_index[cluster][0]]]["existing_list"])
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                    for k in cluster_index[cluster]:
                        actions[self.remain_list[k]] = 0.05 * centering_fly(
                            self.database[self.remain_list[k]]["known_positions"],
                            self.database[self.remain_list[k]]["existing_list"], self.remain_list[k]) + 0.95 * \
                                                       actions_csds[self.remain_list[k]]

        elif self.algorithm_mode == 1:
            # hero
            remain_positions = []
            for i in self.remain_list:
                remain_positions.append(deepcopy(self.true_positions[i]))
            remain_positions = np.array(remain_positions)

            flag, num_of_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))
            if flag:
                start = time.perf_counter()
                print("connected")
                end = time.perf_counter()
                if not record_time:
                    self.time_consuming.append(deepcopy(end - start))
                    record_time = True
            else:
                for i in self.remain_list:
                    start = time.perf_counter()
                    actions_hero = self.hero.hero(
                        Utils.difference_set([i for i in range(self.num_of_agents)], self.database[i]["existing_list"]),
                        self.database[i]["known_positions"])
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                    actions[i] = 0.2 * centering_fly(self.database[i]["known_positions"],
                                                     self.database[i]["existing_list"], i) + 0.8 * actions_hero[i]

        elif self.algorithm_mode == 2:
            # centering
            remain_positions = []
            for i in self.remain_list:
                remain_positions.append(deepcopy(self.true_positions[i]))
            remain_positions = np.array(remain_positions)

            flag, num_of_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))
            if flag:
                start = time.perf_counter()
                print("connected")
                end = time.perf_counter()
                if not record_time:
                    self.time_consuming.append(deepcopy(end - start))
                    record_time = True

            else:
                for i in self.remain_list:
                    start = time.perf_counter()
                    actions[i] = centering_fly(self.database[i]["known_positions"], self.database[i]["existing_list"],
                                               i)
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
        elif self.algorithm_mode == 3:
            # SIDR
            remain_positions = []
            for i in self.remain_list:
                remain_positions.append(deepcopy(self.true_positions[i]))
            remain_positions = np.array(remain_positions)

            flag, num_of_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))

            if flag:
                start = time.perf_counter()
                print("connected")
                end = time.perf_counter()
                if not record_time:
                    self.time_consuming.append(deepcopy(end - start))
                    record_time = True
            else:
                for i in self.remain_list:
                    start = time.perf_counter()
                    temp_actions = SIDR(self.database[i]["known_positions"], self.database[i]["existing_list"])
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                    actions[i] = deepcopy(temp_actions[i])

        elif self.algorithm_mode == 4:
            # GCN-2017
            remain_positions = []
            for i in self.remain_list:
                remain_positions.append(deepcopy(self.true_positions[i]))
            remain_positions = np.array(remain_positions)

            A = Utils.make_A_matrix(remain_positions, len(self.remain_list), config_communication_range)
            flag, num_of_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))
            positions_with_clusters, cluster_index = Utils.split_the_positions_into_clusters_and_indexes(
                remain_positions, num_of_clusters, A)

            for cluster in range(num_of_clusters):
                temp_remain_positions = []
                for i in self.database[self.remain_list[cluster_index[cluster][0]]]["existing_list"]:
                    temp_remain_positions.append(
                        deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["known_positions"][i]))
                temp_remain_positions = np.array(temp_remain_positions)
                flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(temp_remain_positions),
                                                                     len(self.database[self.remain_list[
                                                                         cluster_index[cluster][0]]]["existing_list"]))

                if flag and self.check_if_finish(cluster_index[cluster]):
                    start = time.perf_counter()
                    print("connected")
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                    # return deepcopy(actions)
                elif num_of_clusters == 1:
                    start = time.perf_counter()
                    print("connected")
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                else:
                    start = time.perf_counter()
                    if self.notice_destroy or self.inertia_counter > self.inertia:

                        temp_actions, max_time, final_positions = self.gcn_2017.cr_gcm_n(
                            deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["known_positions"]),
                            deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["existing_list"]))
                        for i in cluster_index[cluster]:
                            actions[self.remain_list[i]] = temp_actions[i]
                            self.destination_positions[self.remain_list[i]] = final_positions[i]
                            self.if_finish[self.remain_list[i]] = False
                    else:

                        for i in cluster_index[cluster]:
                            if np.linalg.norm(self.destination_positions[self.remain_list[i]] - self.true_positions[
                                self.remain_list[i]]) >= 0.55:
                                actions[self.remain_list[i]] = (self.destination_positions[self.remain_list[i]] -
                                                                self.true_positions[
                                                                    self.remain_list[i]]) / np.linalg.norm(
                                    self.destination_positions[self.remain_list[i]] - self.true_positions[
                                        self.remain_list[i]])
                            else:
                                self.if_finish[self.remain_list[i]] = True
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
            if self.notice_destroy:
                self.notice_destroy = False
            if self.inertia_counter > self.inertia:
                self.inertia_counter = 0
            self.inertia_counter += 1


        elif self.algorithm_mode == 6:
            # CR-GCM-N
            # make the clusters
            remain_positions = []
            for i in self.remain_list:
                remain_positions.append(deepcopy(self.true_positions[i]))
            remain_positions = np.array(remain_positions)

            A = Utils.make_A_matrix(remain_positions, len(self.remain_list), config_communication_range)
            flag, num_of_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))
            positions_with_clusters, cluster_index = Utils.split_the_positions_into_clusters_and_indexes(
                remain_positions, num_of_clusters, A)

            for cluster in range(num_of_clusters):
                temp_remain_positions = []
                for i in self.database[self.remain_list[cluster_index[cluster][0]]]["existing_list"]:
                    temp_remain_positions.append(
                        deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["known_positions"][i]))
                temp_remain_positions = np.array(temp_remain_positions)
                flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(temp_remain_positions),
                                                                     len(self.database[self.remain_list[
                                                                         cluster_index[cluster][0]]]["existing_list"]))
                if flag and self.check_if_finish(cluster_index[cluster]):
                    start = time.perf_counter()
                    print("connected")
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                    # return deepcopy(actions)
                elif num_of_clusters == 1:
                    start = time.perf_counter()
                    print("connected")
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
                else:
                    start = time.perf_counter()
                    if self.notice_destroy or self.inertia_counter > self.inertia:

                        temp_actions, max_time, final_positions = self.cr_mgc.cr_gcm(
                            deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["known_positions"]),
                            deepcopy(self.database[self.remain_list[cluster_index[cluster][0]]]["existing_list"]))
                        for i in cluster_index[cluster]:
                            actions[self.remain_list[i]] = temp_actions[i]
                            self.destination_positions[self.remain_list[i]] = final_positions[i]
                            self.if_finish[self.remain_list[i]] = False
                    else:

                        for i in cluster_index[cluster]:
                            if np.linalg.norm(self.destination_positions[self.remain_list[i]] - self.true_positions[
                                self.remain_list[i]]) >= 0.55:
                                actions[self.remain_list[i]] = (self.destination_positions[self.remain_list[i]] -
                                                                self.true_positions[
                                                                    self.remain_list[i]]) / np.linalg.norm(
                                    self.destination_positions[self.remain_list[i]] - self.true_positions[
                                        self.remain_list[i]])
                            else:
                                self.if_finish[self.remain_list[i]] = True
                    end = time.perf_counter()
                    if not record_time:
                        self.time_consuming.append(deepcopy(end - start))
                        record_time = True
            if self.notice_destroy:
                self.notice_destroy = False
            if self.inertia_counter > self.inertia:
                self.inertia_counter = 0
            self.inertia_counter += 1


        else:
            print("No such algorithm")
        return deepcopy(actions)

    def check_if_finish(self, cluster_index):
        flag = True
        for i in range(len(cluster_index)):
            if not self.if_finish[self.remain_list[cluster_index[i]]]:
                flag = False
                break
        return flag

    def save_GCN(self, filename):
        torch.save(self.cr_mgc.gcn_network, filename)

    def save_time_consuming(self):
        if self.algorithm_mode == 0:
            self.time_consuming = pd.DataFrame(np.array(self.time_consuming))
            Utils.store_dataframe_to_excel(self.time_consuming, "Experiment_Fig/time_consuming/CSDS.xlsx")
        elif self.algorithm_mode == 1:
            self.time_consuming = pd.DataFrame(np.array(self.time_consuming))
            Utils.store_dataframe_to_excel(self.time_consuming, "Experiment_Fig/time_consuming/HERO.xlsx")
        elif self.algorithm_mode == 2:
            self.time_consuming = pd.DataFrame(np.array(self.time_consuming))
            Utils.store_dataframe_to_excel(self.time_consuming, "Experiment_Fig/time_consuming/CEN.xlsx")
        elif self.algorithm_mode == 3:
            self.time_consuming = pd.DataFrame(np.array(self.time_consuming))
            Utils.store_dataframe_to_excel(self.time_consuming, "Experiment_Fig/time_consuming/SIDR.xlsx")
        elif self.algorithm_mode == 4:
            self.time_consuming = pd.DataFrame(np.array(self.time_consuming))
            Utils.store_dataframe_to_excel(self.time_consuming, "Experiment_Fig/time_consuming/GCN_2017.xlsx")
        elif self.algorithm_mode == 6:
            self.time_consuming = pd.DataFrame(np.array(self.time_consuming))
            Utils.store_dataframe_to_excel(self.time_consuming, "Experiment_Fig/time_consuming/CR_GCM_N.xlsx")
