from copy import deepcopy
import numpy as np
import Utils
from operator import itemgetter


def smallest_d_algorithm(positions, num, d0):
    A = Utils.make_A_matrix(positions, num, d0)
    d_min = deepcopy(d0)
    unsorted_list = []
    dis = []
    sorted_list = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            unsorted_list.append(deepcopy({"start": i,
                                           "end": j,
                                           "distance": np.linalg.norm(positions[i] - positions[j])}))
            dis.append(deepcopy(np.linalg.norm(positions[i] - positions[j])))
    sorted_index = [index for index, value in sorted(enumerate(dis), key=itemgetter(1))]
    for i in range(len(sorted_index)):
        sorted_list.append(unsorted_list[sorted_index[i]])

    # find the threshold
    threshold_for_d0 = 0
    for i in range(len(sorted_index)):
        if sorted_list[i]["distance"] > d_min:
            threshold_for_d0 = deepcopy(i)
            break
    for i in range(threshold_for_d0, len(sorted_index)):
        A[sorted_list[i]["start"], sorted_list[i]["end"]] = 1
        A[sorted_list[i]["end"], sorted_list[i]["start"]] = 1
        D = Utils.make_D_matrix(A, num)
        L = D - A
        connected_flag, num_cluster = Utils.check_number_of_clusters(L, num)
        if connected_flag:
            d_min = deepcopy(sorted_list[i]["distance"])
            break
    return deepcopy(d_min)
