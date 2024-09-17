
import imageio
import csv
import os
import copy
import numpy as np
import random
import shapely.geometry
import matplotlib.pyplot as plt
from NBVP_env import Env
from test_parameter import *

gifs_path = f'results/f'

class MultiagentCollisionChecker:
    @staticmethod
    def closest_distance_between_lines(start1, end1, start2, end2):
        segment1 = end1 - start1
        segment2 = end2 - start2
        segment1_normalized = segment1 / np.linalg.norm(segment1)
        segment2_normalized = segment2 / np.linalg.norm(segment2)
        
        cross = np.cross(segment1_normalized, segment2_normalized)
        denominator = np.linalg.norm(cross)**2
        
        if denominator != 0:
            # Lines are not parallel
            t = start2 - start1
            numerator1 = np.dot(np.cross(t, segment2_normalized), cross)
            numerator2 = np.dot(np.cross(t, segment1_normalized), cross)
            t1 = numerator1 / denominator
            t2 = numerator2 / denominator
            
            solution1 = start1 + segment1_normalized * t1
            solution2 = start2 + segment2_normalized * t2
            
            # Clamp results to line segments if necessary
            if t1 < 0:
                solution1 = start1
            elif t1 > np.linalg.norm(segment1):
                solution1 = end1
            
            if t2 < 0:
                solution2 = start2
            elif t2 > np.linalg.norm(segment2):
                solution2 = end2
            
            return np.linalg.norm(solution1 - solution2)
        
        # Parallel lines
        d0 = np.dot(segment1_normalized, start2 - start1)
        d = np.linalg.norm((d0 * segment1_normalized + start1) - start2)
        
        # Check for overlapping lines
        d1 = np.dot(segment1_normalized, end2 - start1)
        if d0 <= 0 and 0 >= d1:
            # segment2 before segment1
            if abs(d0) < abs(d1):
                return np.linalg.norm(start2 - start1)
            return np.linalg.norm(end2 - start1)
        elif d0 >= np.linalg.norm(segment1) and np.linalg.norm(segment1) <= d1:
            # segment2 after segment1
            if abs(d0) < abs(d1):
                return np.linalg.norm(start2 - end1)
            return np.linalg.norm(end2 - end1)
        
        return d

def check_collision(start, end, robot_belief, other_agents_positions):
    print(f"Checking collision from {start} to {end}")
    collision = False
    line = shapely.geometry.LineString([start, end])

    sortx = np.sort([start[0], end[0]])
    sorty = np.sort([start[1], end[1]])

    robot_belief = robot_belief[sorty[0]:sorty[1] + 1, sortx[0]:sortx[1] + 1]

    occupied_area_index = np.where(robot_belief == 1)
    occupied_area_coords = np.asarray(
            [occupied_area_index[1] + sortx[0], occupied_area_index[0] + sorty[0]]).T
    unexplored_area_index = np.where(robot_belief == 127)
    unexplored_area_coords = np.asarray(
            [unexplored_area_index[1] + sortx[0], unexplored_area_index[0] + sorty[0]]).T
    unfree_area_coords = occupied_area_coords

    for i in range(unfree_area_coords.shape[0]):
        coords = ([(unfree_area_coords[i][0] -5, unfree_area_coords[i][1] -5),
               (unfree_area_coords[i][0] + 5, unfree_area_coords[i][1] -5),
               (unfree_area_coords[i][0] - 5, unfree_area_coords[i][1] + 5),
               (unfree_area_coords[i][0] + 5, unfree_area_coords[i][1] + 5)])
        obstacle = shapely.geometry.Polygon(coords)
        if abs(end[0] - unfree_area_coords[i][0]) <= 8 and abs(end[1] - unfree_area_coords[i][1]) <= 8:
            collision = True
        if not collision:
            collision = line.intersects(obstacle)
        if collision:
            print(f"Collision detected with obstacle at {unfree_area_coords[i]}")
            break

    if not collision:
        unfree_area_coords = unexplored_area_coords
        for i in range(unfree_area_coords.shape[0]):
            coords = ([(unfree_area_coords[i][0], unfree_area_coords[i][1]),
                       (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1]),
                       (unfree_area_coords[i][0], unfree_area_coords[i][1]),
                       (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1] + 1)])
            obstacle = shapely.geometry.Polygon(coords)
            collision = line.intersects(obstacle)
            if collision:
                print(f"Collision detected with unexplored area at {unfree_area_coords[i]}")
                break

    # Check collision with other agents using the MultiagentCollisionChecker
    for agent_pos in other_agents_positions:
        distance = MultiagentCollisionChecker.closest_distance_between_lines(
            np.array(start), np.array(end), 
            np.array(agent_pos), np.array(agent_pos)
        )
        if distance < 8:  # Assuming a buffer of 5 units
            print(f"Collision detected with another agent at {agent_pos}")
            collision = True
            break

    return collision


class Vertex:
    def __init__(self, id, parent_id, coords, env, agent_id):
        self.id = id
        self.parent_id = parent_id
        self.env = env
        self.coords = coords
        self.agent_id = agent_id
        self.observable_frontiers = []
        self.initialize_observable_frontiers()
        self.gain = 0
        self.branch_index = []
        self.branch_coords = []

    def initialize_observable_frontiers(self):
     print(f"Initializing observable frontiers for vertex at {self.coords}")
     dist_list = np.linalg.norm(self.env.frontiers - self.coords, axis=-1)
     frontiers_in_range = self.env.frontiers[dist_list < self.env.sensor_range - 10]
    
     other_agents_positions = [
        pos for i, pos in enumerate(self.env.robot_positions) if i != self.agent_id
     ]
    
     for point in frontiers_in_range:
        collision = check_collision(self.coords, point, self.env.robot_beliefs[self.agent_id], other_agents_positions)
        if not collision:
            self.observable_frontiers.append(point)
    
     print(f"Observable frontiers: {self.observable_frontiers}")


class Tree:
    def __init__(self, current_coords, env, agent_id):
        print(f"Creating new tree at {current_coords}")
        self.initial = 0
        self.vertices = dict()
        self.vertices_indices = []
        self.env = env
        self.agent_id = agent_id

        vertex = Vertex(self.initial, -1, current_coords, self.env, self.agent_id)
        self.add_vertex(vertex)

    def add_vertex(self, vertex):
        print(f"Adding vertex with ID {vertex.id} at {vertex.coords}")
        self.vertices[vertex.id] = vertex
        self.vertices_indices.append(vertex.id)
        vertex.branch_index, vertex.branch_coords = self.extract_branch(vertex)
        visible = self.env.calculate_utility_along_path(vertex.branch_index, self.vertices)
        assert visible <= len(self.env.frontiers)
        dist = self.env.calculate_dist_path(vertex.branch_index, self.vertices)
        vertex.gain = visible * np.exp(-10 * dist / 640)
        print(f"Vertex gain: {vertex.gain}")

    def initialize_tree_from_path(self, path, env):
     print(f"Initializing tree from path: {path}")
     for i, coords in enumerate(path[2:]):
        vertex = Vertex(i+1, i, coords, env, self.agent_id)
        self.add_vertex(vertex)


    def extract_branch(self, vertex):
        branch_index = [vertex.id]
        while vertex.parent_id != -1:
            vertex = self.vertices[vertex.parent_id]
            branch_index.append(vertex.id)
            assert vertex.id != vertex.parent_id
            if vertex.parent_id == -1:
                break
        branch_coords = [self.vertices[index].coords for index in branch_index]

        print(f"Extracted branch: {branch_coords}")
        return branch_index, branch_coords

class NBVP_worker:
    def __init__(self, metaAgentID, global_step, num_agents=2, save_image=False):
        print(f"Initializing NBVP worker with {num_agents} agents at step {global_step}")
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.step_length = 30
        self.num_agents = num_agents
        self.pre_best_paths = [[] for _ in range(num_agents)]
        self.planned_paths = [[] for _ in range(num_agents)]

        self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)

    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        print(f"Free area: {free.shape[0]} cells")
        return free

    def find_next_best_viewpoints(self, agent_id):
        max_iter_steps = 100
        tree = Tree(self.env.robot_positions[agent_id], self.env, agent_id)
        self.planned_paths[agent_id] = []
        if self.pre_best_paths[agent_id]:
            print(f"Initializing tree from previous best path for agent {agent_id}")
            tree.initialize_tree_from_path(self.pre_best_paths[agent_id], self.env)
        g_best = 0
        best_route = []
        free_area = self.free_area(self.env.robot_beliefs[agent_id])
        frontiers = self.env.frontiers
        indices_to_sample = list(range(free_area.shape[0]))
        frontiers_to_sample = list(range(len(frontiers)))
        
        other_agents_positions = [pos for i, pos in enumerate(self.env.robot_positions) if i != agent_id]

        i = 0
        while i <= max_iter_steps or g_best == 0:
            i += 1
            dice = np.random.random(1)
            print(f"Iteration {i}, Dice roll: {dice}")
            
            if dice > 0.2:
                if not indices_to_sample:
                    print("No indices to sample, skipping iteration.")
                    continue
                sample_index = random.sample(indices_to_sample, 1)
                sample_coords = free_area[sample_index][0]
            else:
                if not frontiers_to_sample:
                    print("No frontiers to sample, skipping iteration.")
                    continue
                sample_index = random.sample(frontiers_to_sample, 1)
                sample_coords = frontiers[sample_index][0]
            
            vertices_coords = np.array([node.coords for node in tree.vertices.values()]).reshape(-1, 2)
            dist_list = np.linalg.norm(sample_coords - vertices_coords, axis=-1)
            nearest_vertex_index = np.argmin(dist_list)
            nearest_vertex_coords = vertices_coords[nearest_vertex_index]

            dist = dist_list[nearest_vertex_index]
            if dist > self.step_length:
                cos = (sample_coords[0] - nearest_vertex_coords[0]) / dist
                sin = (sample_coords[1] - nearest_vertex_coords[1]) / dist

                x = int(nearest_vertex_coords[0] + cos * self.step_length)
                y = int(nearest_vertex_coords[1] + sin * self.step_length)
                new_vertex_coords = np.array([x, y])
            else:
                new_vertex_coords = sample_coords

            collision = check_collision(nearest_vertex_coords, new_vertex_coords, self.env.robot_beliefs[agent_id], other_agents_positions)
            if collision:
                print(f"Collision detected for vertex at {new_vertex_coords}, skipping.")
                continue

            new_vertex_index = len(tree.vertices)
            vertex = Vertex(new_vertex_index, nearest_vertex_index, new_vertex_coords, self.env, agent_id)
            tree.add_vertex(vertex)
            route = vertex.branch_coords[::-1]

            self.planned_paths[agent_id].append(route)

            if vertex.gain > g_best:
                g_best = vertex.gain
                best_route = route

        if len(best_route) > 1:
            return best_route[1], self.planned_paths[agent_id]
        else:
            return best_route[0], self.planned_paths[agent_id]

    def run_episode(self, currEpisode):
        perf_metrics = dict()
        print(f"Running episode {currEpisode}")
        done = False
        self.env.begin()
        i = 0

        while not done:
            i += 1
            next_node_coords = []
            planned_routes = []

            for agent_id in range(self.num_agents):
                next_node, planned_route = self.find_next_best_viewpoints(agent_id)
                next_node_coords.append(next_node)
                planned_routes.append(planned_route)

            done = self.env.step(next_node_coords)

            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, planned_routes)

            # if SAVE_TRAJECTORY:
            #     if not os.path.exists(trajectory_path):
            #         os.makedirs(trajectory_path)
            #     csv_filename = f'results/trajectory/nbvp_trajectory_result_multi_agent.csv'
            #     new_file = False if os.path.exists(csv_filename) else True
            #     field_names = ['dist', 'area']
            #     with open(csv_filename, 'a') as csvfile:
            #         writer = csv.writer(csvfile)
            #         if new_file:
            #             writer.writerow(field_names)
            #         csv_data = np.array([sum(self.env.travel_dists), np.sum(self.env.robot_beliefs[0] == 255)]).reshape(1, -1)
            #         writer.writerows(csv_data)

            if done:
                perf_metrics['travel_dist'] = sum(self.env.travel_dists)
                perf_metrics['explored_rate'] = self.env.explored_rate
                perf_metrics['success_rate'] = True
                perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
                break

        # if SAVE_LENGTH:
        #     if not os.path.exists(length_path):
        #         os.makedirs(length_path)
        #     csv_filename = f'results/length/NBVP_length_result_multi_agent.csv'
        #     new_file = False if os.path.exists(csv_filename) else True
        #     field_names = ['dist']
        #     with open(csv_filename, 'a') as csvfile:
        #         writer = csv.writer(csvfile)
        #         if new_file:
        #             writer.writerow(field_names)
        #         csv_data = np.array([sum(self.env.travel_dists)]).reshape(-1,1)
        #         writer.writerows(csv_data)

        if self.save_image:
            path1 = gifs_path
            self.make_gif(path1, currEpisode)

        return perf_metrics

    def work(self, currEpisode):
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

    def make_gif(self, path, n):
        print(f"Creating GIF for episode {n}")
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('GIF complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

if __name__ == "__main__":
    total_episode = 40
    total_dist = 0
    num_agents = 2  # Specify the number of agents here
    
    for i in range(total_episode):
        print(f"Starting episode {i+1}")
        worker = NBVP_worker(metaAgentID=0, global_step=i, num_agents=num_agents, save_image=SAVE_GIFS)
        performance = worker.run_episode(i)
        total_dist += performance["travel_dist"]
        mean_dist = total_dist / (i + 1)
        print(f"Episode {i+1}/{total_episode}, Mean distance: {mean_dist:.2f}")

    print(f"\nFinal mean distance over {total_episode} episodes: {mean_dist:.2f}")