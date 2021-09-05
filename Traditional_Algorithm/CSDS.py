import numpy as np
import Utils
from copy import deepcopy


class CSDS:
    def __init__(self, total_num, global_positions):
        self.global_positions = deepcopy(global_positions)
        self.remain_positions = deepcopy(self.global_positions)

        self.total_num = total_num

        self.remain_list = [i for i in range(self.total_num)]
        self.remain_num = self.total_num

        self.destroy_happen = False
        self.num_cluster = 0
        self.speed = np.zeros((self.total_num, 3))
        self.prepare_speed = deepcopy(self.speed)
        self.destroy_list = []

        self.critical_set = []
        self.non_critical_set = []

        self.backup = []

        self.determine_the_backup()
        self.counter = 0
        self.backup_time = []

    def determine_the_backup(self):
        self.determine_critical_nodes()
        self.backup = []
        self.backup_time = []
        for i in self.critical_set:
            minimal_distance = 10000000
            index = 0
            for k in self.non_critical_set:
                if np.linalg.norm(self.global_positions[i] - self.global_positions[k]) < minimal_distance:
                    index = k
            self.backup.append(deepcopy(index))
            self.prepare_speed[index] = (self.global_positions[i] - self.global_positions[index]) / np.linalg.norm(
                self.global_positions[i] - self.global_positions[index])
            self.backup_time.append(deepcopy(np.linalg.norm(
                self.global_positions[i] - self.global_positions[index])))

    def csds(self, global_positions, remain_list):
        self.remain_positions = []
        self.global_positions = deepcopy(global_positions)
        speed = np.zeros((self.total_num, 3))
        for i in remain_list:
            self.remain_positions.append(deepcopy(global_positions[i]))
        self.remain_positions = np.array(self.remain_positions)
        self.remain_list = deepcopy(remain_list)
        self.remain_num = len(self.remain_list)
        flag, num_clusters = Utils.check_if_a_connected_graph(self.remain_positions, self.remain_num)
        self.num_cluster = deepcopy(num_clusters)
        if flag:
            return np.zeros((self.total_num, 3)),0
        else:
            if self.destroy_happen:
                self.destroy_happen = False
                self.counter = 0
                for destroy_index in self.destroy_list:
                    if destroy_index in self.non_critical_set:
                        pass
                    elif destroy_index in self.critical_set:
                        true_backup_index = np.argwhere(np.array(self.critical_set) == destroy_index)[0, 0]
                        speed[true_backup_index] = deepcopy(self.prepare_speed[true_backup_index])
                        if self.backup_time[true_backup_index] > self.counter:
                            self.counter = deepcopy(self.backup_time[true_backup_index])
                self.speed = deepcopy(speed)
                self.determine_the_backup()
                return deepcopy(speed), self.counter
            else:
                self.counter -= 1
                if self.counter < 0:
                    self.destroy_happen = True
                    self.determine_the_backup()
                return deepcopy(self.speed),0

    def determine_critical_nodes(self):
        self.critical_set = []
        self.non_critical_set = []
        remain_positions = []
        for i in self.remain_list:
            remain_positions.append(deepcopy(self.global_positions[i]))
        remain_positions = np.array(remain_positions)
        flag, overall_num_clusters = Utils.check_if_a_connected_graph(remain_positions, len(self.remain_list))

        for i in self.remain_list:
            temp_remain_positions = []
            for k in self.remain_list:
                if k != i:
                    temp_remain_positions.append(deepcopy(self.global_positions[k]))
            temp_remain_positions = np.array(temp_remain_positions)
            flag, num_clusters = Utils.check_if_a_connected_graph(temp_remain_positions, len(self.remain_list) - 1)
            if num_clusters == overall_num_clusters:
                self.non_critical_set.append(deepcopy(i))
            else:
                self.critical_set.append(deepcopy(i))

    def notice_destroy(self, destroy_list):
        self.destroy_happen = True
        self.destroy_list = deepcopy(destroy_list)
        self.counter = 0
        self.determine_the_backup()
