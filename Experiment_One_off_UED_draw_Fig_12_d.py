from Environment import Environment
from Swarm import Swarm
from copy import deepcopy
from Configurations import *
import Utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from Drawing.Draw_Static import *
from Main_algorithm_GCN.GCO import GCO

# should first set if use meta learning param
"""
Note: if you use trained meta param, you need to down the trained meta parameters from 
       https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ or 
       https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing
       the size of meta parameter file is pretty large (about 1.2GB).
       otherwise, you could run the Meta-learning_all.py file to train the meta parameter on your own machine
"""
meta_param_use = True
"""
    algorithm mode: 0 for CSDS
                    1 for HERO
                    2 for CEN
                    3 for SIDR
                    4 for GCN-2017
                    5 for CR-MGC (proposed algorithm)
"""
algorithm_mode = {0: "CSDS",
                  1: "HERO",
                  2: "CEN",
                  3: "SIDR",
                  4: "GCN_2017",
                  5: "CR-MGC (proposed algorithm)"}

destroy_list = pd.read_excel("Experiment_Fig/one_off_UEDs/destroy_list.xlsx")
destroy_list = destroy_list.to_numpy()[:, 1]

# storage
cluster_list = []
for algorithm_mode_num in range(6):
    print("=======================================")
    print("algorithm: %s" % (algorithm_mode[algorithm_mode_num]))

    environment = Environment()
    if algorithm_mode_num == 0:
        swarm = Swarm(algorithm_mode=algorithm_mode_num, enable_csds=True, meta_param_use=meta_param_use)
    else:
        swarm = Swarm(algorithm_mode=algorithm_mode_num)
    environment_positions = environment.reset()
    swarm.reset()
    cluster_list.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for step in range(450):
        # destroy at time step 0
        if step == 0:
            print("=======================================")
            print("destroy %d -- mode %d num %d " % (0, 2, 100))
            destroy_num, destroy_list = environment.stochastic_destroy(mode=4, real_destroy_list=destroy_list)
            print("destroy 100 nodes \n destroy index list :")
            print(destroy_list)
            swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

        actions, max_time = swarm.take_actions()
        environment_next_positions = environment.next_state(deepcopy(actions))
        swarm.update_true_positions(environment_next_positions)

        temp_cluster = environment.check_the_clusters()
        cluster_list[algorithm_mode_num].append(deepcopy(temp_cluster))
        print("---------------------------------------")
        if temp_cluster == 1:
            print("step %d ---num of clusters %d -- connected" % (step, environment.check_the_clusters()))
        else:
            print("step %d ---num of clusters %d -- disconnected" % (step, environment.check_the_clusters()))

        # update
        environment.update()
        environment_positions = deepcopy(environment_next_positions)

# draw Fig. 12(d)
x = [i for i in range(-20, 450)]
destroy_num_list = []
for i in range(470):
    if i == 20:
        destroy_num_list.append(100)
    else:
        destroy_num_list.append(0)
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time Steps", family='Times New Roman', fontsize=16)
ax1.set_ylabel("Number of clusters", family='Times New Roman', fontsize=16)
ax2 = ax1.twinx()
ax2.set_ylabel("Number of disrupted Nodes", family='Times New Roman', fontsize=16)

cen, = ax1.plot(x, cluster_list[2], c="tomato", linewidth=2.5, marker='s', ms=0, label="CEN", linestyle="--")
sidr, = ax1.plot(x, cluster_list[3], c="g", linewidth=2.5, marker='1', ms=0, label="SIDR", linestyle="-.")
csds, = ax1.plot(x, cluster_list[0], c="m", linewidth=2.5, marker='*', ms=0, label="CSDS", linestyle=":")
hero, = ax1.plot(x, cluster_list[1], c="#1E90FF", linewidth=2.5, marker='s', ms=0, label="HERO", linestyle="-.")
gcn_2017, = ax1.plot(x, cluster_list[4], c="#858540", linewidth=2.5, marker='s', ms=0, label="HERO",
                     linestyle="--")
gcn, = ax1.plot(x, cluster_list[5], c="mediumblue", linewidth=2.5, marker='s', ms=0, label="CR_MCM", linestyle="-")

ax1.set_ylim([-1, 6])
ax1.set_xlim([-20.5, 450.5])

num = ax2.bar(x, destroy_num_list, width=15, fc='crimson', label="Disrupted number")
ax2.set_ylim([0, 450])

# legend = plt.legend(handles=[cen, gcn, sidr, csds, hero, num], prop=font_, loc='upper right')

plt.savefig("Experiment_Fig/one_off_UEDs/Fig_4(d).png", dpi=1000)
plt.show()
