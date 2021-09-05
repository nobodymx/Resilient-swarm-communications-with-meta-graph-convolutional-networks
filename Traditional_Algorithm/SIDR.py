import Utils
from Configurations import *
from copy import deepcopy


def SIDR(positions, remain_list):
    num_of_remain = len(remain_list)
    remain_positions = []
    for i in remain_list:
        remain_positions.append(deepcopy(positions[i]))
    remain_positions = np.array(remain_positions, dtype=np.float64)
    A = Utils.make_A_matrix(remain_positions, num_of_remain, config_communication_range)
    D = Utils.make_D_matrix(A, num_of_remain)
    L = D - A
    connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, num_of_remain)
    positions_with_clusters, cluster_index = Utils.split_the_positions_into_clusters_and_indexes(remain_positions,
                                                                                                 num_of_clusters,
                                                                                                 A)
    true_cluster_index = []
    for i in range(num_of_clusters):
        temp_true_index = []
        for j in cluster_index[i]:
            temp_true_index.append(deepcopy(remain_list[j]))
        true_cluster_index.append(deepcopy(temp_true_index))

    speed = np.zeros((config_num_of_agents, 3))
    trajectory_point = np.array([500, 500, 50])
    for num_c in range(len(true_cluster_index)):
        temp_positions = []
        for i in true_cluster_index[num_c]:
            temp_positions.append(positions[i])
        temp_positions = np.array(temp_positions)
        average_positions = np.mean(temp_positions)
        for uav in true_cluster_index[num_c]:
            if np.linalg.norm(trajectory_point - average_positions) > 0:
                speed[uav] = (trajectory_point - average_positions) / np.linalg.norm(trajectory_point - average_positions)
    return deepcopy(speed)
