import numpy as np
from copy import deepcopy
import pandas as pd
from Configurations import *


def make_A_matrix(positions, num_of_agents, d):
    A = np.zeros((num_of_agents, num_of_agents))
    for i in range(num_of_agents):
        for j in range(i, num_of_agents):
            if j == i:
                A[i, j] = 0
            else:
                distance = np.linalg.norm(positions[i, :] - positions[j, :])
                if distance <= d:
                    A[i, j] = 1
                    A[j, i] = 1
    return deepcopy(A)


def make_D_matrix(A, num_of_agents):
    D = np.zeros((num_of_agents, num_of_agents))
    for i in range(num_of_agents):
        D[i, i] = np.sum(A[i])
    return deepcopy(D)


def check_number_of_clusters(L, num_of_agents):
    e_vals, e_vecs = np.linalg.eig(L)
    eig_0_counter = 0
    for i in range(num_of_agents):
        if e_vals[i] < 0.000001:
            eig_0_counter += 1
    if eig_0_counter == 1:
        return True, 1
    else:
        return False, eig_0_counter


def normalized_single_vector(speed):
    normalized_speed = speed / np.linalg.norm(speed)
    return deepcopy(normalized_speed)


def normalized_batch_vector(speed):
    normalized_speed = deepcopy(speed)
    for i in range(len(speed)):
        normalized_speed[i] = speed[i] / np.linalg.norm(speed[i])
    return deepcopy(normalized_speed)


def calculate_d_max(positions):
    d_max = 0
    for i in range(len(positions) - 1):
        for j in range(i + 1, len(positions)):
            if d_max < np.linalg.norm(positions[j] - positions[i]):
                d_max = deepcopy(np.linalg.norm(positions[j] - positions[i]))

    return deepcopy(d_max)


def store_dataframe_to_excel(data, filename, sheetname="None"):
    """
    :param data: receive dataframe datatype (not numpy)
    :param filename:
    :param sheetname:
    :return: storage flag
    """
    # data.to_excel(filename, sheet_name=sheetname)
    if isinstance(data, pd.DataFrame):
        try:
            data.to_excel(filename, sheet_name=sheetname)
            print("Store successful")
        except:
            print("Storage error")
    else:
        print("Data type error")


def calculate_norm(speeds):
    normalized_speed = np.zeros((len(speeds)))
    for i in range(len(normalized_speed)):
        normalized_speed[i] = np.linalg.norm(speeds[i])
    return deepcopy(normalized_speed)


def random_sampling():
    """
    :return: list with form [ , , , , ,]
    """
    list_ = []
    count = 0
    while count < config_sample_number:
        random_number = np.random.randint(0, config_buffer_capacity, 1).tolist()[0]
        if random_number not in list_:
            list_.append(random_number)
            count += 1
    return list_


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def normalize_positions(positions):
    positions = np.array(positions, dtype=np.float64)
    norm_positions = deepcopy(positions)
    norm_positions[:, 0] = 2 * (positions[:, 0] - 0.5 * config_width) / config_width
    norm_positions[:, 1] = 2 * (positions[:, 1] - 0.5 * config_length) / config_length
    norm_positions[:, 2] = 2 * (positions[:, 2] - 0.5 * config_height) / config_height
    return deepcopy(norm_positions)


def normalize_single_positions(positions):
    norm_positions = deepcopy(np.array(positions, dtype=np.float64))
    norm_positions[0] = float(2 * (positions[0] - 0.5 * config_width) / config_width)
    norm_positions[1] = float(2 * (positions[1] - 0.5 * config_length) / config_length)
    norm_positions[2] = float(2 * (positions[2] - 0.5 * config_height) / config_height)
    return deepcopy(norm_positions)


def check_if_a_connected_graph(remain_positions, remain_num):
    A = make_A_matrix(remain_positions, remain_num, config_communication_range)
    D = make_D_matrix(A, remain_num)
    L = D - A
    connected_flag, num_of_clusters = check_number_of_clusters(L, remain_num)
    return deepcopy(connected_flag), deepcopy(num_of_clusters)


def split_the_positions_into_clusters(positions, num_of_clusters, A):
    positions_with_clusters = []
    remain_list = [i for i in range(len(positions))]
    if num_of_clusters <= 1:
        return None
    else:
        for k in range(num_of_clusters):
            temp_positions = []

            visited = np.zeros(len(remain_list))
            counter = 0
            stack = Stack()
            stack.push(remain_list[0])
            visited[0] = 1
            counter += 1

            while stack.length() != 0:
                current = stack.top_element()
                flag = True
                temp_counter = 0
                for i in remain_list:
                    if A[current, i] == 1 and visited[temp_counter] == 0:
                        visited[temp_counter] = 1
                        counter += 1
                        stack.push(i)
                        flag = False
                        break
                    temp_counter += 1
                if flag:
                    stack.pop()

            visited_node = []
            for j in range(len(remain_list)):
                if visited[j] == 1:
                    visited_node.append(remain_list[j])
            for j in range(counter):
                remain_list.remove(visited_node[j])
                temp_positions.append(deepcopy(positions[visited_node[j]]))
            positions_with_clusters.append(deepcopy(np.array(temp_positions)))
        return deepcopy(positions_with_clusters)


class Stack:
    def __init__(self):
        self.memory = []

    def push(self, num):
        self.memory.append(num)

    def pop(self):
        if len(self.memory) == 0:
            return None
        else:
            temp = self.memory[-1]
            del self.memory[-1]
            return temp

    def length(self):
        return len(self.memory)

    def top_element(self):
        return self.memory[-1]


def split_the_positions_into_clusters_and_indexes(positions, num_of_clusters, A):
    positions_with_clusters = []
    cluster_index = []
    remain_list = [i for i in range(len(positions))]
    if num_of_clusters <= 1:
        positions_with_clusters.append(deepcopy(positions))
        cluster_index.append(remain_list)
        return deepcopy(positions_with_clusters), deepcopy(cluster_index)
    else:
        for k in range(num_of_clusters):
            temp_positions = []
            temp_index = []

            visited = np.zeros(len(remain_list))
            counter = 0
            stack = Stack()
            stack.push(remain_list[0])
            visited[0] = 1
            counter += 1

            while stack.length() != 0:
                current = stack.top_element()
                flag = True
                temp_counter = 0
                for i in remain_list:
                    if A[current, i] == 1 and visited[temp_counter] == 0:
                        visited[temp_counter] = 1
                        counter += 1
                        stack.push(i)
                        flag = False
                        break
                    temp_counter += 1
                if flag:
                    stack.pop()

            visited_node = []
            for j in range(len(remain_list)):
                if visited[j] == 1:
                    visited_node.append(remain_list[j])
                    temp_index.append(remain_list[j])
            for j in range(counter):
                remain_list.remove(visited_node[j])
                temp_positions.append(deepcopy(positions[visited_node[j]]))

            positions_with_clusters.append(deepcopy(np.array(temp_positions)))
            cluster_index.append(deepcopy(temp_index))
        return deepcopy(positions_with_clusters), deepcopy(cluster_index)


def intersection_set(listA, listB):
    intersection_list = [i for i in listA if i in listB]
    return deepcopy(intersection_list)


def difference_set(listA, listB):
    difference_list = [i for i in listA if i not in listB]
    return deepcopy(difference_list)


def union_set(listA, listB):
    union_set = [i for i in listA]
    for i in listB:
        if i not in listA:
            union_set.append(deepcopy(i))
    return deepcopy(union_set)
