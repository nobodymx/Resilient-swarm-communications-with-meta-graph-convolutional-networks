from copy import deepcopy
import numpy as np


def centering_fly(node_global_positions, remain_list, index):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    remain_positions = []
    self_positions = node_global_positions[index]
    for i in remain_list:
        remain_positions.append(deepcopy(node_global_positions[i]))
    remain_positions = np.array(remain_positions)
    final_positions = np.mean(remain_positions, 0)
    flying_direction = (final_positions - self_positions)/np.linalg.norm(final_positions - self_positions)
    return deepcopy(flying_direction)
