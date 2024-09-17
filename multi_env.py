from scipy import spatial
from skimage import io
import numpy as np
import time
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import copy
from skimage.measure import block_reduce

from sensor import *
from parameter import *

class Env():
    def __init__(self, map_index, plot=False, test=False, num_agents=2):
        
        self.test = test
        if self.test:
            self.map_dir = f'DungeonMaps/easy'
        else:
            self.map_dir = f'DungeonMaps/train'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.ground_truth, self.start_position = self.import_ground_truth(self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth) # (480, 640)
        
        self.num_agents = num_agents
        self.robot_positions = self.generate_start_positions(num_agents)
        self.robot_beliefs = [np.ones(self.ground_truth_size) * 127 for _ in range(num_agents)]  # unexplored 127
        
        self.finish_percent = 0.985
        self.resolution = 4
        self.sensor_range = 80
        self.old_robot_beliefs = [copy.deepcopy(belief) for belief in self.robot_beliefs]

        self.plot = plot
        self.frame_files = []
        if self.plot:
            # initialize the routes
            self.xPoints = [[pos[0]] for pos in self.robot_positions]
            self.yPoints = [[pos[1]] for pos in self.robot_positions]

        self.travel_dists = [0 for _ in range(num_agents)]
        self.explored_rate = 0
        self.route_nodes = [[pos.copy()] for pos in self.robot_positions]
        self.frontiers = None
        self.downsampled_belief = None

    def generate_start_positions(self, num_agents):
        positions = [self.start_position.copy()]
        for _ in range(1, num_agents):
            new_pos = self.find_new_start_position(positions)
            positions.append(new_pos)
        return positions

    def find_new_start_position(self, existing_positions, max_attempts=100):
        free_cells = self.free_cells()
        for _ in range(max_attempts):
            candidate = free_cells[np.random.randint(len(free_cells))]
            if all(np.linalg.norm(candidate - pos) > 20 for pos in existing_positions):
                return candidate
        
        # If we couldn't find a position after max_attempts, just return a random free cell
        return free_cells[np.random.randint(len(free_cells))]

    def begin(self):
        for i in range(self.num_agents):
            self.robot_beliefs[i] = self.update_robot_belief(self.robot_positions[i], self.sensor_range, self.robot_beliefs[i], self.ground_truth)
        
        combined_belief = np.min(self.robot_beliefs, axis=0)
        self.downsampled_belief = block_reduce(combined_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)

        self.frontiers = self.find_frontier()
        
        self.old_robot_beliefs = [copy.deepcopy(belief) for belief in self.robot_beliefs]

    def step(self, next_node_coords):
        for i in range(self.num_agents):
            dist = np.linalg.norm(next_node_coords[i] - self.robot_positions[i])
            self.travel_dists[i] += dist
            self.robot_positions[i] = next_node_coords[i]
            self.route_nodes[i].append(self.robot_positions[i])
            self.robot_beliefs[i] = self.update_robot_belief(self.robot_positions[i], self.sensor_range, self.robot_beliefs[i], self.ground_truth)

        combined_belief = np.min(self.robot_beliefs, axis=0)
        self.downsampled_belief = block_reduce(combined_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)

        frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()

        if self.plot:
            for i in range(self.num_agents):
                self.xPoints[i].append(self.robot_positions[i][0])
                self.yPoints[i].append(self.robot_positions[i][1])

        self.frontiers = frontiers

        done = self.check_done()

        return done

    def import_ground_truth(self, map_index):
        # occupied 1, free 255, unexplored 127
        ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        robot_location = np.nonzero(ground_truth == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location

    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief

    def check_done(self):
        done = False
        if np.sum(self.ground_truth == 255) - np.sum(np.min(self.robot_beliefs, axis=0) == 255) <= 250:
            done = True
        return done

    def evaluate_exploration_rate(self):
        combined_belief = np.min(self.robot_beliefs, axis=0)
        rate = np.sum(combined_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def calculate_new_free_area(self):
        old_free_area = np.min(self.old_robot_beliefs, axis=0) == 255
        current_free_area = np.min(self.robot_beliefs, axis=0) == 255

        new_free_area = (current_free_area.astype(np.int) - old_free_area.astype(np.int)) * 255

        return new_free_area, np.sum(old_free_area)

    def calculate_utility_along_path(self, path, nodes_list):
        observable_frontiers = []
        for index in path:
            observable_frontiers += nodes_list[index].observable_frontiers
        np_observable_frontiers = np.array(observable_frontiers).reshape(-1,2)
        unique_frontiers = np.unique(np_observable_frontiers[:, 0] + np_observable_frontiers[:, 1]*1j)

        return unique_frontiers.shape[0]

    def calculate_node_gain_over_path(self, node_index, path, nodes_list):
        observable_frontiers = []
        for index in path:
            observable_frontiers += nodes_list[index].observable_frontiers
        np_observable_frontiers = np.array(observable_frontiers).reshape(-1,2)
        pre_unique_frontiers = np.unique(np_observable_frontiers[:, 0] + np_observable_frontiers[:, 1]*1j)
        observable_frontiers += nodes_list[node_index].observable_frontiers
        np_observable_frontiers = np.array(observable_frontiers).reshape(-1,2)
        unique_frontiers = np.unique(np_observable_frontiers[:, 0] + np_observable_frontiers[:, 1]*1j)

        return unique_frontiers.shape[0] - pre_unique_frontiers.shape[0]

    def calculate_dist_path(self, path, node_list):
        dist = 0
        start = path[0]
        end = path[-1]
        for index in path:
            if index == end:
                break
            dist += np.linalg.norm(node_list[start].coords - node_list[index].coords)
            start = index
        return dist

    # def find_frontier(self):
    #     y_len = self.downsampled_belief.shape[0]
    #     x_len = self.downsampled_belief.shape[1]
    #     mapping = self.downsampled_belief.copy()
    #     belief = self.downsampled_belief.copy()
    #     # 0-1 unknown area map
    #     mapping = (mapping == 127) * 1
    #     mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
    #     fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
    #               mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
    #                                                                                                   2:] + \
    #               mapping[:y_len][:, :x_len]
    #     ind_free = np.where(belief.ravel(order='F') == 255)[0]
    #     ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
    #     ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
    #     ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
    #     ind_to = np.intersect1d(ind_free, ind_fron)

    #     map_x = x_len
    #     map_y = y_len
    #     x = np.linspace(0, map_x - 1, map_x)
    #     y = np.linspace(0, map_y - 1, map_y)
    #     t1, t2 = np.meshgrid(x, y)
    #     points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

    #     f = points[ind_to]
    #     f = f.astype(int)

    #     f = f * self.resolution

    #     return f


    def find_frontier(self):
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        # 0-1 unknown area map
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        f = points[ind_to]
        f = f.astype(int)

        f = f * self.resolution

        return f

    def plot_env(self, n, path, step, planned_routes=None):
        plt.switch_backend('agg')
        plt.cla()
        combined_belief = np.min(self.robot_beliefs, axis=0)
        plt.imshow(combined_belief, cmap='gray')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        if planned_routes:
            for agent_id, agent_routes in enumerate(planned_routes):
                for p in agent_routes:
                    planned_x, planned_y = [], []
                    for coords in p:
                        planned_x.append(coords[0])
                        planned_y.append(coords[1])
                    plt.plot(planned_x, planned_y, c=['r', 'g', 'b', 'y'][agent_id % 4], linewidth=2, zorder=2)
        for i in range(self.num_agents):
            plt.plot(self.xPoints[i], self.yPoints[i], ['b', 'g', 'r', 'y'][i % 4], linewidth=2)
            plt.plot(self.robot_positions[i][0], self.robot_positions[i][1], 'mo', markersize=8)
        for i, start_pos in enumerate(self.robot_positions):
            plt.plot(start_pos[0], start_pos[1], 'co', markersize=8)
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        plt.suptitle('Explored ratio: {:.4g}  Travel distances: {}'.format(self.explored_rate, [round(d, 2) for d in self.travel_dists]))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=150))
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
