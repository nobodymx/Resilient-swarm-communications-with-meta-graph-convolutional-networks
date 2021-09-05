"""
GIF drawing function
Note: should input the
        storage_remain_list,
        storage_positions,
        num_remain,
        storage_connection_states
    to this file
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from Configurations import *

# some input variable
storage_remain_list = []
storage_positions = []
num_remain = 0
storage_connection_states = []
storage_connectivity_matrix = []

print("Plotting the results...")
fig = plt.figure()


def update(frame):
    ax = Axes3D(fig)
    for i in range(config_num_of_agents):
        if i in storage_remain_list[frame]:
            ax.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1], storage_positions[frame][i, 2],
                       s=50, c='g')
        else:
            ax.scatter(storage_positions[i, 0], storage_positions[i, 1],
                       storage_positions[i, 2],
                       s=50, c='r')
            ax.text(storage_positions[i - num_remain, 0] + 1, storage_positions[i - num_remain, 1] + 1,
                    storage_positions[i - num_remain, 2] + 1,
                    'Destroyed', c='r')
    ax.text(5, 5, 5, 'distance = %f' % 120, c='blue')
    ax.text(5, 5, -3, 'time steps = %d' % frame, c='b')
    if storage_connection_states[frame]:
        ax.text(5, 5, 15, 'Connected...', c='g')
    else:
        ax.text(5, 5, 15, 'Unconnected...', c='r')
    ax.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})

    for i in range(len(storage_positions[frame])):
        for j in range(i, len(storage_positions[frame])):
            if storage_connectivity_matrix[frame][i, j] == 1:
                x = [storage_positions[frame][i, 0], storage_positions[frame][j, 0]]
                y = [storage_positions[frame][i, 1], storage_positions[frame][j, 1]]
                z = [storage_positions[frame][i, 2], storage_positions[frame][j, 2]]
                ax.plot(x, y, z, c='lightsteelblue')
    ax.set_xlim(0, config_width)
    ax.set_ylim(0, config_length)
    ax.set_zlim(-10, config_height)


frame = np.linspace(0, len(storage_positions) - 1, len(storage_positions)).astype(int)
ani = animation.FuncAnimation(fig, update, frames=frame, interval=200, repeat_delay=10)
ani.save("fake_mode_2_training_epi_%d.gif", writer='pillow')
plt.savefig("b.png")
plt.show()
