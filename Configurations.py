import pandas as pd
import numpy as np
import random
import torch

"""
specify a certain GPU
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# random seed
np.random.seed(1)
random.seed(2)
torch.manual_seed(1)

config_initial_swarm_positions = pd.read_excel("Configurations/swarm_positions_200.xlsx")
config_initial_swarm_positions = config_initial_swarm_positions.values[:, 1:4]
config_initial_swarm_positions = np.array(config_initial_swarm_positions, dtype=np.float64)

# configurations on swarm
config_num_of_agents = 200
config_communication_range = 120

# configurations on environment
config_width = 1000.0
config_length = 1000.0
config_height = 100.0

config_constant_speed = 1

# configurations on destroy
config_maximum_destroy_num = 50
config_minimum_remain_num = 5

# configurations on meta learning
config_meta_training_epi = 500
# configurations on Graph Convolutional Network
config_K = 1 / 100
config_best_eta = 0.3
config_best_epsilon = 0.99

# configurations on one-off UEDs
config_num_destructed_UAVs = 100  # should be in the range of [1, num_of_UAVs-2]
config_normalize_positions = True

# configurations on training GCN
config_alpha_k = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.9, 0.95, 1, 1.5, 2, 3, 5]
config_gcn_repeat = 100
config_expension_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
config_d0_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

config_representation_step = 450

# configurations on continuous destroy setting 1
config_destroy_step_list_1 = [10, 90, 100, 131, 230, 310,  config_representation_step + 100]
config_destroy_mode_list_1 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# config_destroy_num_list_1 = [50, 8, 9, 25, 1, 0]
config_destroy_num_list_1 = [50, 8, 9, 7, 20, 1]
config_destroy_range_list_1 = [0, 0, 0, 0, 0, 0, 0, 0, 50, 10, 0]
config_destroy_center_list_1 = [None, None, None, None, None, None, None, None, np.array([300, 200, 50]),
                                np.array([600, 750, 50]), None]

# configurations on continuous destroy setting 2
config_destroy_step_list_2 = [3, 21, 40, 56, 70, 125, 145, 160, 176, 190, config_representation_step + 100]
config_destroy_mode_list_2 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_2 = [10, 20, 20, 30, 40, 10, 10, 30, 10, 10]
config_destroy_range_list_2 = [0, 0, 0, 0, 0, 0, 0, 0, 50, 10]
config_destroy_center_list_2 = [None, None, None, None, None, None, None, None, np.array([300, 200, 50]),
                                np.array([600, 750, 50]), None]

# configurations on continuous destroy setting 3
config_destroy_step_list_3 = [9, 15, 20, 56, 60, 70, 103, 156, 170, config_representation_step + 100]
config_destroy_mode_list_3 = [2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_3 = [10, 30, 15, 8, 50, 20, 10, 10, 10]
config_destroy_range_list_3 = [0, 0, 0, 0, 0, 0, 0, 0, 50, 10]
config_destroy_center_list_3 = [None, None, None, None, None, None, None, None, np.array([300, 200, 50]),
                                np.array([600, 750, 50]), None]

# configurations on continuous destroy setting 4
config_destroy_step_list_4 = [10, 51, 70, 91, 100, 120, 135, 150, 170, 198, 210, config_representation_step + 100]
config_destroy_mode_list_4 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_4 = [50, 8, 10, 3, 40, 5, 5, 30, 1, 0, 4]
config_destroy_range_list_4 = [10, 20, 30, 40, 30, 20, 40, 20, 50, 10, 10]
config_destroy_center_list_4 = [np.array([60, 70, 60]), np.array([500, 750, 30]), np.array([30, 75, 70]),
                                np.array([500, 500, 50]), np.array([100, 150, 20]), np.array([600, 750, 30]),
                                np.array([620, 70, 25]), np.array([160, 900, 50]), np.array([300, 200, 50]),
                                np.array([600, 750, 50]), np.array([100, 75, 50])]

# configurations on continuous destroy setting 5
config_destroy_step_list_7 = [250,251,  config_representation_step + 100]
config_destroy_mode_list_7 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_7 = [100,1]
config_destroy_range_list_7 = [20, 30, 10, 50, 50, 50, 20, 30, 50, 10]
config_destroy_center_list_7 = [np.array([60, 70, 60]), np.array([500, 750, 30]), np.array([30, 75, 70]),
                                np.array([500, 500, 50]), np.array([100, 150, 20]), np.array([600, 750, 30]),
                                np.array([620, 70, 25]), np.array([160, 900, 50]), np.array([300, 200, 50]),
                                np.array([600, 750, 50])]

# configurations on continuous destroy setting 6
config_destroy_step_list_6 = [140,160, config_representation_step + 100]
config_destroy_mode_list_6 = [2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_6 = [150, 2]
config_destroy_range_list_6 = [30, 20, 10, 20, 40, 50, 50, 30, 50, 10]
config_destroy_center_list_6 = [np.array([60, 70, 60]), np.array([500, 750, 30]), np.array([30, 75, 70]),
                                np.array([500, 500, 50]), np.array([100, 150, 20]), np.array([600, 750, 30]),
                                np.array([620, 70, 25]), np.array([160, 900, 50]), np.array([300, 200, 50]),
                                np.array([300, 200, 50]),
                                np.array([600, 750, 50])]

config_destroy_step_list_5 = [3, 21, 40, 56, 70, 125, 145, 160, 176, 190, config_representation_step + 100]
config_destroy_mode_list_5 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_5 = [10, 20, 20, 30, 40, 10, 10, 30, 10, 10]
config_destroy_range_list_5 = [20, 30, 10, 50, 50, 50, 20, 30, 50, 10]
config_destroy_center_list_5 = [np.array([60, 70, 60]), np.array([500, 750, 30]), np.array([30, 75, 70]),
                                np.array([500, 500, 50]), np.array([100, 150, 20]), np.array([600, 750, 30]),
                                np.array([620, 70, 25]), np.array([160, 900, 50]), np.array([300, 200, 50]),
                                np.array([600, 750, 50])]

config_destroy_step_list_8 = [11, 40, 80, 145, 220, 360,  config_representation_step + 100]
config_destroy_mode_list_8 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_8 = [30, 22, 91, 25, 10, 10]
config_destroy_range_list_8 = [0, 0, 0, 0, 0, 0, 0, 0, 50, 10, 0]
config_destroy_center_list_8 = [None, None, None, None, None, None, None, None, np.array([300, 200, 50]),
                                np.array([600, 750, 50]), None]

config_destroy_step_list_9 = [80, 90, 100, 130, 250, 320,  config_representation_step + 100]
config_destroy_mode_list_9 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_9 = [50, 10, 20, 30, 50, 10]
config_destroy_range_list_9 = [0, 0, 0, 0, 0, 0, 0, 0, 50, 10, 0]
config_destroy_center_list_9 = [None, None, None, None, None, None, None, None, np.array([300, 200, 50]),
                                np.array([600, 750, 50]), None]

config_destroy_step_list_10 = [20, 30, 70, 100, 130, 270,  config_representation_step + 100]
config_destroy_mode_list_10 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
config_destroy_num_list_10 = [50, 18, 19, 25, 10, 30]
config_destroy_range_list_10 = [0, 0, 0, 0, 0, 0, 0, 0, 50, 10, 0]
config_destroy_center_list_10 = [None, None, None, None, None, None, None, None, np.array([300, 200, 50]),
                                np.array([600, 750, 50]), None]

config_single_destroy_number = [10 * i for i in range(1, 20)]

config_destroy_step_list = [config_destroy_step_list_1, config_destroy_step_list_2, config_destroy_step_list_3,
                            config_destroy_step_list_4, config_destroy_step_list_5, config_destroy_step_list_6, config_destroy_step_list_7,
                            config_destroy_step_list_8,config_destroy_step_list_9,config_destroy_step_list_10]
config_destroy_mode_list = [config_destroy_mode_list_1, config_destroy_mode_list_2, config_destroy_mode_list_3,
                            config_destroy_mode_list_4, config_destroy_mode_list_5, config_destroy_mode_list_6,config_destroy_mode_list_7,
                            config_destroy_mode_list_8,config_destroy_mode_list_9,config_destroy_mode_list_10]
config_destroy_num_list = [config_destroy_num_list_1, config_destroy_num_list_2, config_destroy_num_list_3,
                           config_destroy_num_list_4, config_destroy_num_list_5, config_destroy_num_list_6,config_destroy_num_list_7,
                           config_destroy_num_list_8,config_destroy_num_list_9,config_destroy_num_list_10]
config_destroy_range_list = [config_destroy_range_list_1, config_destroy_range_list_2, config_destroy_range_list_3,
                             config_destroy_range_list_4, config_destroy_range_list_5, config_destroy_range_list_6,config_destroy_range_list_7,
                             config_destroy_range_list_8,config_destroy_range_list_9,config_destroy_range_list_10]
config_destroy_center_list = [config_destroy_center_list_1, config_destroy_center_list_2, config_destroy_center_list_3,
                              config_destroy_center_list_4, config_destroy_center_list_5, config_destroy_center_list_6,config_destroy_center_list_7,
                              config_destroy_center_list_8,config_destroy_center_list_9,config_destroy_center_list_10]
