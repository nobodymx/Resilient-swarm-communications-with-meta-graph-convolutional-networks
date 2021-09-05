from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Drawing.Color_Material_Library import common_color_list, color_dict
import numpy as np
from copy import deepcopy


def draw_once(remain_num, storage_positions, storage_connectivity_matrix,
              save_path='Experiment_Fig/gcn_only_final_positions.png'):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(remain_num):
        ax.scatter(storage_positions[i, 0], storage_positions[i, 1], storage_positions[i, 2],
                   s=30, c='g')

    ax.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})

    for i in range(len(storage_positions)):
        for j in range(i, len(storage_positions)):
            if storage_connectivity_matrix[i, j] >= 1:
                x = [storage_positions[i, 0], storage_positions[j, 0]]
                y = [storage_positions[i, 1], storage_positions[j, 1]]
                z = [storage_positions[i, 2], storage_positions[j, 2]]
                ax.plot(x, y, z, c='lightsteelblue')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(-50, 150)
    plt.savefig(save_path, dpi=500)
    plt.show()


def draw_once_two_nodes(remain_num, storage_positions, storage_positions_2):
    fig = plt.figure()
    ax = Axes3D(fig)
    max_distance = 0
    max_distance_index = 0
    for i in range(remain_num):
        if max_distance < np.linalg.norm(storage_positions[i] - storage_positions_2[i]):
            max_distance = np.linalg.norm(storage_positions[i] - storage_positions_2[i])
            max_distance_index = deepcopy(i)
    print(max_distance)
    for i in range(remain_num):
        ax.scatter(storage_positions[i, 0], storage_positions[i, 1], storage_positions[i, 2],
                   s=30, c='g')
        ax.scatter(storage_positions_2[i, 0], storage_positions_2[i, 1], storage_positions_2[i, 2],
                   s=30, c='b')
        x = [storage_positions[i, 0], storage_positions_2[i, 0]]
        y = [storage_positions[i, 1], storage_positions_2[i, 1]]
        z = [storage_positions[i, 2], storage_positions_2[i, 2]]
        if i == max_distance_index:
            ax.plot(x, y, z, c='red', linestyle='--')
        else:
            ax.plot(x, y, z, c='lightsteelblue', linestyle='--')

    ax.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(-50, 150)
    plt.savefig('Experiment_Fig/one_off_UEDs/Fig_12(c).png', dpi=1000)
    plt.show()


def draw_pic_with_destroyed(num_of_remain, num_of_destroy, remain_positions, positions_of_exists_with_clusters,
                            positions_of_destroyed,
                            num_of_clusters, A):
    """
    :param num_of_remain:
    :param num_of_destroy:
    :param positions_of_exists_with_clusters:  list [  np, , , , ,]
    :param positions_of_destroyed: np
    :param num_of_clusters:
    :return:
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(num_of_destroy):
        ax.scatter(positions_of_destroyed[i, 0], positions_of_destroyed[i, 1], positions_of_destroyed[i, 2],
                   s=30, c='r')

    ax.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})

    for i in range(num_of_clusters):
        for j in range(len(positions_of_exists_with_clusters[i])):
            ax.scatter(positions_of_exists_with_clusters[i][j, 0], positions_of_exists_with_clusters[i][j, 1],
                       positions_of_exists_with_clusters[i][j, 2],
                       s=30, c=common_color_list[i % len(common_color_list)])

    for i in range(num_of_remain):
        for j in range(i, num_of_remain):
            if A[i, j] >= 1:
                x = [remain_positions[i, 0], remain_positions[j, 0]]
                y = [remain_positions[i, 1], remain_positions[j, 1]]
                z = [remain_positions[i, 2], remain_positions[j, 2]]
                ax.plot(x, y, z, c='lightsteelblue')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(-50, 150)
    plt.savefig('Experiment_Fig/one_off_UEDs/Fig_12(a)', dpi=500)
    plt.show()


def draw_approximate_pic(num_of_remain, positions_trajectory):
    trajectory_step = len(positions_trajectory)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(trajectory_step):
        red = int(50 - (50 - 10) * (i / trajectory_step))
        green = int(230 - (230 - 30) * (i / trajectory_step))
        blue = int(50 - (50 - 10) * (i / trajectory_step))
        c = str(red) + ',' + str(green) + ',' + str(blue)
        c = RGB_to_Hex(c)

        for j in range(num_of_remain):
            ax.scatter(positions_trajectory[i][j, 0], positions_trajectory[i][j, 1], positions_trajectory[i][j, 2],
                       s=30, c=c)
            if i > 0:
                x = [positions_trajectory[i - 1][j, 0], positions_trajectory[i][j, 0]]
                y = [positions_trajectory[i - 1][j, 1], positions_trajectory[i][j, 1]]
                z = [positions_trajectory[i - 1][j, 2], positions_trajectory[i][j, 2]]
                # ax.plot(x, y, z, c='b')
                ax.quiver(x[0], y[0], z[0], x[1] - x[0], y[1] - y[0], z[1] - z[0], normalize=False)

    ax.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(-50, 150)
    plt.savefig('Experiment_Fig/one_off_UEDs/Fig_12(b).png', dpi=500)
    plt.show()


def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color
