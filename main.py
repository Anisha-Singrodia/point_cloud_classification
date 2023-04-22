import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from argparse import ArgumentParser


def plot_point_cloud(rows, cols, height, num):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rows, cols, height)
    major_ticks = np.arange(0, 40, 10)
    minor_ticks = np.arange(0, 40, 5)
    
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_zticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_zticks(major_ticks)

    ax.grid(which='both')
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.001)
    ax.grid(True)

    # plt.show()
    plt.savefig('point_cloud_'+ str(num) +'.png')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--split-ratio', type=float, default=0.8)

    parser.add_argument('--dataset-split', type=str, default='default',
                        choices=['default', 'compositional', 'holdout'])
    parser.add_argument('--dataset-type', type=str, default='expert',
                        choices=['random', 'medium',
                                 'expert', 'medium-replay-subsampled'])
    parser.add_argument('--data-seed', type=int, default=0)
    parser.add_argument('--use-task-list-path', default=False,
                        action='store_true')

    parser.add_argument('--algo', type=str, default='bc',
                        choices=['bc', 'bcq', 'bear', 'cql', 'iql'])

    parser.add_argument('--robot', type=str, default='IIWA',
                        choices=['IIWA', 'Jaco', 'Kinova3', 'Panda'])
    parser.add_argument('--object', type=str, default='Box',
                        choices=['Box', 'HollowBox', 'Plate', 'Dumbbell'])
    parser.add_argument('--obstacle', type=str, default='None',
                        choices=['None', 'ObjectWall', 'ObjectDoor', 'GoalWall'])
    parser.add_argument('--objective', type=str, default='Push',
                        choices=['Push', 'PickPlace', 'Shelf', 'Trashcan'])

    parser.add_argument('--hparam-id', type=int, default=-1)

    args = parser.parse_args()
    return args

def main(split_ratio):
    print(split_ratio)
    N = 100
    M = 40
    input = np.zeros((N, M, M, M))
    test_split_num = 5*M*M*M//100
    # total points = 40*40*40: 5% = 3200
    # need to generate 3200 random points between 0 to 40, exclusive

    for i in range(N):
        np.random.seed(0)
        occupied_cells = np.random.randint(0, 40, size=(3200, 3))
        input[i, occupied_cells] = 1
        occ_list = list(map(tuple, occupied_cells))
        
        oset = set(occ_list)
        while len(oset) < test_split_num:
            diff = test_split_num - len(oset)
            add_occ_cells = np.random.randint(0, 40, size=(diff, 3))
            add_occ_list = list(map(tuple, add_occ_cells))
            oset.update(add_occ_list)
            input[i, add_occ_cells] = 1
        # occ_cells_array = np.array(list(map(list, oset)))
        # plot_point_cloud(list(occ_cells_array[:, 0]), list(occ_cells_array[:, 1]), list(occ_cells_array[:, 2]), i)
    K = 5
    labels = np.zeros((N))
    for i in range(K):
        labels[20*i:(20*i)+20] = i
    print(labels)



if __name__ == "__main__":
    args = parse_args()
    split_ratio = args.split_ratio
    main(split_ratio)


