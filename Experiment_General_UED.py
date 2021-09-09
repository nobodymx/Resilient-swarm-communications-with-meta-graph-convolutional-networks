from Environment import Environment
from Swarm_general import Swarm
from copy import deepcopy
from Configurations import *
import Utils
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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

# if draw the gif
"""
Note: if true, it may take a little long time
"""
draw_gif = False

"""
if the global information can be known by UAVs
if not, UAVs will use monitoring mechanism
"""
config_global_info = True

config_algorithm_mode = 6
"""
    algorithm mode: 0 for CSDS
                    1 for HERO
                    2 for CEN
                    3 for SIDR
                    4 for GCN-2017
                    6 for CR-MGCM (proposed algorithm)
"""
algorithm_mode = {0: "CSDS",
                  1: "HERO",
                  2: "CEN",
                  3: "SIDR",
                  4: "GCN_2017",
                  6: "CR-MGCM (proposed algorithm)"}
config_continuous_mode = 0
print("algorithm_mode:%d" % config_algorithm_mode)

print("SCC problem under one-off UEDs Starts...")
print("------------------------------")
print("Algorithm: %s" % (algorithm_mode[config_algorithm_mode]))


environment = Environment()
swarm = Swarm(algorithm_mode=config_algorithm_mode, meta_param_use=meta_param_use)

# storage
storage_num_cluster_list = []
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
storage_connection_states.append("True")
storage_num_cluster_list.append(1)
storage_remain_connectivity_matrix.append(
    deepcopy(Utils.make_A_matrix(swarm.true_positions, config_num_of_agents, config_communication_range)))

destroy_counter = 0
Utils.store_dataframe_to_excel(pd.DataFrame(storage_positions[0]),
                               "Experiment_Fig/general_UEDs_2/positions/positions_%d.xlsx" % (0))
Utils.store_dataframe_to_excel(pd.DataFrame(storage_remain_connectivity_matrix[0]),
                               "Experiment_Fig/general_UEDs_2/connectivity_matrix/connectivity_matrix_%d.xlsx" % (
                                   0))
"""
the demo configurations: mode=4, config_continuous_mode=0, with real_destroy_list
"""
for step in range(config_representation_step):
    print("episode step %d" % step)
    if step == config_destroy_step_list[config_continuous_mode][destroy_counter]:
        print("destroy %d -- mode %d num %d " % (
            destroy_counter, config_destroy_mode_list[config_continuous_mode][destroy_counter],
            config_destroy_num_list[config_continuous_mode][
                destroy_counter]))
        destroy_num, destroy_list = environment.stochastic_destroy(mode=4,
                                                                   num_of_destroyed=
                                                                   config_destroy_num_list[config_continuous_mode][
                                                                       destroy_counter],
                                                                   real_destroy_list=store_destroy_list[
                                                                      destroy_counter],
                                                                   destroy_center=
                                                                   config_destroy_center_list[config_continuous_mode][
                                                                       destroy_counter],
                                                                   destroy_range=
                                                                   config_destroy_range_list[config_continuous_mode][
                                                                       destroy_counter])
        # store destroy counter
        storage_destroy_list.append(deepcopy(destroy_list))
        # tell swarm destroy happen
        if config_global_info:
            swarm.destroy_happens_GI_version(deepcopy(destroy_list), deepcopy(environment_positions))
        else:
            swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))
        destroy_counter += 1
    else:
        storage_destroy_list.append([])
    storage_remain_list.append(deepcopy(swarm.remain_list))
    if config_global_info:
        actions = swarm.take_actions_GI_continuous_mode()
    else:
        actions = swarm.take_actions_incomplete_information_continuous()
    environment_next_positions = environment.next_state(deepcopy(actions))
    swarm.update_true_positions(environment_next_positions)

    temp_num_clusters = environment.check_the_clusters()
    storage_num_cluster_list.append(temp_num_clusters)
    print("---num of clusters %d" % temp_num_clusters)
    # storage
    storage_positions.append(deepcopy(environment_next_positions))
    if temp_num_clusters == 1:
        storage_connection_states.append("True")
    else:
        storage_connection_states.append("False")

    remain_positions = []
    for i in swarm.remain_list:
        remain_positions.append(deepcopy(environment_next_positions[i]))
    remain_positions = np.array(remain_positions)
    storage_remain_connectivity_matrix.append(
        deepcopy(Utils.make_A_matrix(remain_positions, len(swarm.remain_list), config_communication_range)))

    # update
    environment.update()
    # update IDBs using monitoring mechanisms
    if not config_global_info:
        swarm.broadcast_next_position_information(deepcopy(environment_next_positions))
        swarm.broadcast_remain_list_information(deepcopy(environment_next_positions))
    environment_positions = deepcopy(environment_next_positions)
    # store to excel
    Utils.store_dataframe_to_excel(pd.DataFrame(storage_positions[step + 1]),
                                   "Experiment_Fig/general_UEDs_2/positions/positions_%d.xlsx" % (step + 1))
    Utils.store_dataframe_to_excel(pd.DataFrame(storage_connection_states),
                                   "Experiment_Fig/general_UEDs_2/connection_states.xlsx")
    Utils.store_dataframe_to_excel(pd.DataFrame(storage_remain_connectivity_matrix[step + 1]),
                                   "Experiment_Fig/general_UEDs_2/connectivity_matrix/connectivity_matrix_%d.xlsx" % (
                                               step + 1))
    Utils.store_dataframe_to_excel(pd.DataFrame(storage_num_cluster_list),
                                   "Experiment_Fig/general_UEDs_2/num_cluster_list.xlsx")
    Utils.store_dataframe_to_excel(pd.DataFrame(storage_destroy_list),
                                   "Experiment_Fig/general_UEDs_2/destroy_list.xlsx")
    Utils.store_dataframe_to_excel(pd.DataFrame(storage_remain_list),
                                   "Experiment_Fig/general_UEDs_2/remain_list.xlsx")

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


if draw_gif:
    def update(frame):
        ax = Axes3D(fig)
        for i in range(config_num_of_agents):
            if i in storage_remain_list[frame]:
                ax.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1],
                           storage_positions[frame][i, 2],
                           s=50, c='g')

                x = [storage_positions[frame][i, 0],
                     config_initial_swarm_positions[i, 0]]
                y = [storage_positions[frame][i, 1],
                     config_initial_swarm_positions[i, 1]]
                z = [storage_positions[frame][i, 2],
                     config_initial_swarm_positions[i, 2]]
                ax.plot(x, y, z, c='blue')
            else:
                if frame <= 30:
                    # red = int(50 - (50 - 10) * (frame / trajectory_step))
                    # green = int(230 - (230 - 30) * (i / trajectory_step))
                    # blue = int(50 - (50 - 10) * (i / trajectory_step))
                    # c = str(red) + ',' + str(green) + ',' + str(blue)
                    # c = RGB_to_Hex(c)
                    ax.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1],
                               storage_positions[frame][i, 2],
                               s=50, c='r')
                # ax.text(storage_positions[frame][i, 0] + 1, storage_positions[frame][i, 1] + 1,
                #         storage_positions[frame][i, 2] + 1,
                #         'Destroyed', c='r')
        ax.text(5, 5, 0, 'CLEC = %f' % 120, c='blue')
        ax.text(5, 5, -15, 'time steps = %d' % (frame - 10), c='b')
        ax.text(5, 5, -30, 'number_of_clusters = %d' % storage_num_cluster_list[frame], c='b')
        if frame >= 11:
            ax.text(5, 5, -45, 'destroy %d UAVs randomly... ' % config_num_destructed_UAVs, c='r')

        if storage_connection_states[frame]:
            ax.text(5, 5, 15, 'Connected...', c='g')
        else:
            ax.text(5, 5, 15, 'Unconnected...', c='r')
        ax.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
        ax.set_ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
        ax.set_xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})

        for i in range(len(storage_remain_list[frame])):
            for j in range(i, len(storage_remain_list[frame])):
                if storage_remain_connectivity_matrix[frame][i, j] == 1:
                    x = [storage_positions[frame][storage_remain_list[frame][i], 0],
                         storage_positions[frame][storage_remain_list[frame][j], 0]]
                    y = [storage_positions[frame][storage_remain_list[frame][i], 1],
                         storage_positions[frame][storage_remain_list[frame][j], 1]]
                    z = [storage_positions[frame][storage_remain_list[frame][i], 2],
                         storage_positions[frame][storage_remain_list[frame][j], 2]]
                    ax.plot(x, y, z, c='lightsteelblue')
        ax.set_xlim(0, config_width)
        ax.set_ylim(0, config_length)
        ax.set_zlim(-50, 150)
        print("finish frame %d ..." % frame)


    print("=======================================")
    print("Plotting the dynamic trajectory...")
    fig = plt.figure()
    frame = np.linspace(0, config_representation_step, config_representation_step + 1).astype(int)
    ani = animation.FuncAnimation(fig, update, frames=frame, interval=90, repeat_delay=10)
    ani.save("video/general_destruct.gif", writer='pillow', bitrate=2048, dpi=500)
    plt.show()
