from Environment import Environment
from Swarm_general import Swarm
from copy import deepcopy
from Configurations import *
import Utils
import time

# determine if use meta learning param
"""
Note: if use trained meta param, you need to down the trained meta parameters from 
       https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ or 
       https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing
       the size of meta parameter file is pretty large (about 1.2GB)
       otherwise, you could run the Meta-learning_all.py file to train the meta parameter on your own machine
"""
meta_param_use = True

config_algorithm_mode = 6
"""
    algorithm mode: 0 for CSDS
                    1 for HERO
                    2 for CEN
                    3 for SIDR
                    4 for GCN-2017
                    5 for CR-MGC (proposed algorithm)
"""
config_continuous_mode = 0
print("algorithm_mode:%d" % config_algorithm_mode)

environment = Environment()
swarm = Swarm(algorithm_mode=config_algorithm_mode, meta_param_use=meta_param_use)

num_cluster_list = []

# storage
storage_remain_list = []
storage_destroy_list = []
storage_positions = []
storage_connection_states = []
storage_remain_connectivity_matrix = []


environment_positions = environment.reset()
swarm.reset()

store_destroy_list = np.load("Configurations/continuous_destroy_index_list.npy", allow_pickle=True)
store_destroy_list = store_destroy_list.tolist()

storage_remain_list.append(deepcopy(swarm.remain_list))
storage_destroy_list.append([])
storage_positions.append(deepcopy(swarm.true_positions))
storage_connection_states.append(True)
storage_remain_connectivity_matrix.append(
    deepcopy(Utils.make_A_matrix(swarm.true_positions, config_num_of_agents, config_communication_range)))


destroy_counter = 0
for step in range(config_representation_step):
    print("episode step %d" % step)
    if step == config_destroy_step_list[config_continuous_mode][destroy_counter]:
        print("destroy %d -- mode %d num %d " % (
            destroy_counter, config_destroy_mode_list[config_continuous_mode][destroy_counter], config_destroy_num_list[config_continuous_mode][
                destroy_counter]))
        destroy_num, destroy_list = environment.stochastic_destroy(mode=4,
                                                                   num_of_destroyed=config_destroy_num_list[config_continuous_mode][
                                                                       destroy_counter],
                                                                   real_destroy_list=store_destroy_list[
                                                                       destroy_counter],
                                                                   destroy_center=config_destroy_center_list[config_continuous_mode][
                                                                       destroy_counter],
                                                                   destroy_range=config_destroy_range_list[config_continuous_mode][
                                                                       destroy_counter])
        # tell swarm destroy happen
        swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))
        destroy_counter += 1

    actions = swarm.take_actions_incomplete_information_continuous()
    environment_next_positions = environment.next_state(deepcopy(actions))
    swarm.update_true_positions(environment_next_positions)

    num_cluster_list.append(environment.check_the_clusters())
    print("---num of clusters %d" % environment.check_the_clusters())

    # update
    environment.update()
    swarm.broadcast_next_position_information(deepcopy(environment_next_positions))
    swarm.broadcast_remain_list_information(deepcopy(environment_next_positions))
    environment_positions = deepcopy(environment_next_positions)

# connection_steps_list_ = pd.DataFrame(np.array(num_cluster_list))
# if config_algorithm_mode == 0:
#     Utils.store_dataframe_to_excel(connection_steps_list_,
#                                    "Experiment_Fig/Experiment_5/CSDS_num_of_cluster_continuous_II.xlsx",
#                                    sheetname="CSDS")
# elif config_algorithm_mode == 1:
#     Utils.store_dataframe_to_excel(connection_steps_list_,
#                                    "Experiment_Fig/Experiment_5/HERO_num_of_cluster_continuous_II.xlsx",
#                                    sheetname="HERO")
# elif config_algorithm_mode == 2:
#     Utils.store_dataframe_to_excel(connection_steps_list_,
#                                    "Experiment_Fig/Experiment_5/CEN_num_of_cluster_continuous_II.xlsx",
#                                    sheetname="CEN")
# if config_algorithm_mode == 3:
#     Utils.store_dataframe_to_excel(connection_steps_list_,
#                                    "Experiment_Fig/Experiment_5/SIDR_num_of_cluster_continuous_II.xlsx",
#                                    sheetname="SIDR")
# elif config_algorithm_mode == 4:
#     Utils.store_dataframe_to_excel(connection_steps_list_,
#                                    "Experiment_Fig/Experiment_5/GCN_2017_num_of_cluster_continuous_II.xlsx",
#                                    sheetname="GCN_2017")
# elif config_algorithm_mode == 6:
#     Utils.store_dataframe_to_excel(connection_steps_list_,
#                                    "Experiment_Fig/Experiment_5/CR_GCM_N_num_of_cluster_continuous_II.xlsx",
#                                    sheetname="CR_GCM")
# swarm.save_time_consuming()