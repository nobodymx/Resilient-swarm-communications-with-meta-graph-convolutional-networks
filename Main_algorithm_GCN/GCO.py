from Configurations import *
from copy import deepcopy
import Utils
from Main_algorithm_GCN.Smallest_d_algorithm import smallest_d_algorithm


class GCO:
    def __init__(self):
        self.K = config_K

    def gco(self, global_positions, remain_list, alpha=0.99, expansion_rate=0.25):
        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        remain_positions = np.array(remain_positions)

        final_positions, counter, storage_positions = self.graph_convolutional(remain_positions, len(remain_list),
                                                                               alpha=alpha,
                                                                               expansion_rate=expansion_rate)
        speed = np.zeros((config_num_of_agents, 3))
        for i in range(len(remain_list)):
            if np.linalg.norm(final_positions[i] - remain_positions[i]) > 0:
                speed[remain_list[i]] = (final_positions[i] - remain_positions[i]) / np.linalg.norm(
                    final_positions[i] - remain_positions[i])
        temp_max_distance = 0
        for i in range(len(remain_list)):
            if np.linalg.norm(final_positions[i] - remain_positions[i]) > temp_max_distance:
                temp_max_distance = deepcopy(np.linalg.norm(final_positions[i] - remain_positions[i]))
        max_time = temp_max_distance / config_constant_speed
        # print(max_time)
        return deepcopy(speed), deepcopy(final_positions), deepcopy(max_time), deepcopy(storage_positions)

    def graph_convolutional(self, remain_swarm_positions, num_remain, alpha=0.99, expansion_rate=0.25):
        """
        :param alpha:
        :param remain_swarm_positions:
        :param num_remain:
        :param expansion_rate:
        :return:
        """
        storage_positions = []
        storage_positions.append(deepcopy(remain_swarm_positions))

        d_min = smallest_d_algorithm(remain_swarm_positions, num_remain,
                                     config_communication_range)
        d_max = Utils.calculate_d_max(remain_swarm_positions)
        A_ = Utils.make_A_matrix(remain_swarm_positions, num_remain, d_min + (d_max - d_min) * expansion_rate)
        D_ = Utils.make_D_matrix(A_, num_remain)
        L_ = D_ - A_
        F = deepcopy(remain_swarm_positions)
        counter = 0

        A_norm = np.linalg.norm(A_, ord=np.inf)
        k0 = 1 / A_norm
        self.K = alpha * k0
        # print("K is %f" % self.K)

        A = Utils.make_A_matrix(F, num_remain, config_communication_range)
        D = Utils.make_D_matrix(A, num_remain)
        L = D - A
        connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, num_remain)
        if connected_flag:
            # print("remain one cluster")
            return deepcopy(F), deepcopy(counter), []
        else:
            # print("become into %d clusters" % num_of_clusters)
            while not connected_flag:
                F = np.dot((np.eye(num_remain) - self.K * L_), F)
                counter += 1

                storage_positions.append(deepcopy(F))
                if counter >= 10000:
                    break

                A = Utils.make_A_matrix(F, num_remain, config_communication_range)
                D = Utils.make_D_matrix(A, num_remain)
                L = D - A
                connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, num_remain)
                # print(connected_flag, num_of_clusters)
            # print("total %d times iterations" % counter)

            return deepcopy(F), deepcopy(counter), deepcopy(storage_positions)
