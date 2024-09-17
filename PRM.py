import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import shapely.geometry
import random
import time
from Graph import Graph, dijkstra, to_array, a_star


class PRM():
    def __init__(self, sample_size, k_size, plot=False):
        self.sample_size = sample_size
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        self.map_x = 640
        self.map_y = 480
        self.uniform_points = self.generate_uniform_points()

    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []
        self.y = []

    def edge_clear(self, coords):
        node_index = str(self.find_index_from_coords(self.node_coords, coords))
        self.graph.clear_edge(node_index)

    def generate_test_graph(self, robot_location, robot_belief):
        self.edge_clear_all_nodes()
        free_area = self.free_area(robot_belief)

        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]

        node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords))
        self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

        robot_location_index = 0
        return self.node_coords, self.graph.edges, robot_location_index

    def update_test_graph(self, robot_location, robot_belief, old_robot_belief):
        new_free_area = self.free_area((robot_belief - old_robot_belief > 0) * 255)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_node_coords = self.uniform_points[candidate_indices]
        self.node_coords = np.concatenate((self.node_coords, new_node_coords))

        # old_node_to_update = []
        # for coords in new_node_coords:
        #     neighbor_indices = self.find_k_neighbor(coords, self.node_coords, robot_belief)
        #     old_node_to_update += neighbor_indices
        # old_node_to_update = set(old_node_to_update)
        # for index in old_node_to_update:
        #     coords = self.node_coords[index]
        #     self.edge_clear(coords)
        #     self.find_k_neighbor(coords, self.node_coords, robot_belief)

        self.edge_clear_all_nodes()
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

        robot_location_index = np.argmin(np.linalg.norm(self.node_coords - robot_location.reshape(1, 2), axis=1))

        return self.node_coords, self.graph.edges, robot_location_index, new_node_coords

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, 30).round().astype(int)
        y = np.linspace(0, self.map_y - 1, 30).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_k_neighbor(self, coords, node_coords, robot_belief):
        dist_list = np.linalg.norm((coords-node_coords), axis=-1)
        sorted_index = np.argsort(dist_list)
        k = 0
        neighbor_index_list = []
        while k < self.k_size and k< node_coords.shape[0]:
            neighbor_index = sorted_index[k]
            neighbor_index_list.append(neighbor_index)
            dist = dist_list[k]
            start = coords
            end = node_coords[neighbor_index]
            if not self.check_collision(start, end, robot_belief):
                a = str(self.find_index_from_coords(node_coords, start))
                b = str(neighbor_index)
                self.graph.add_node(a)
                self.graph.add_edge(a, b, dist)

                if self.plot:
                    self.x.append([start[0], end[0]])
                    self.y.append([start[1], end[1]])
            k += 1
        return neighbor_index_list

    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][:]]):
                start = p
                end = neighbour
                if not self.check_collision(start, end, robot_belief):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j])

                    if self.plot:
                        self.x.append([p[0], neighbour[0]])
                        self.y.append([p[1], neighbour[1]])
                        # x = [p[0], neighbour[0]]
                        # y = [p[1], neighbour[1]]
                        # plt.plot(x, y, c='tan', alpha=0.4, zorder=1)

    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]

    def check_collision(self, start, end, robot_belief):
        collision = False
        line = shapely.geometry.LineString([start, end])

        sortx = np.sort([start[0], end[0]])
        sorty = np.sort([start[1], end[1]])

        # print(robot_belief.shape)
        robot_belief = robot_belief[sorty[0]:sorty[1]+1, sortx[0]:sortx[1]+1]

        occupied_area_index = np.where(robot_belief == 1)
        occupied_area_coords = np.asarray([occupied_area_index[1]+sortx[0], occupied_area_index[0]+sorty[0]]).T
        unexplored_area_index = np.where(robot_belief == 127)
        unexplored_area_coords = np.asarray([unexplored_area_index[1]+sortx[0], unexplored_area_index[0]+sorty[0]]).T
        unfree_area_coords = np.concatenate((occupied_area_coords, unexplored_area_coords))

        # obstacles = []
        for i in range(unfree_area_coords.shape[0]):
            coords = ([(unfree_area_coords[i][0], unfree_area_coords[i][1]),
                       (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1]),
                       (unfree_area_coords[i][0], unfree_area_coords[i][1] + 1),
                       (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1] + 1)])
            obstacle = shapely.geometry.Polygon(coords)
            #obstacles.append(obstacle)
        # if obstacles != []:
            #all_obstacles = shapely.geometry.MultiPolygon(obstacles)
            # print(obstacle.is_valid)
            collision = line.intersects(obstacle)
            if collision:
                break

        return collision

    def find_shortest_path(self, current, destination, node_coords):
        self.startNode = str(self.find_index_from_coords(node_coords, current))
        self.endNode = str(self.find_index_from_coords(node_coords, destination))
        # t1 = time.time()
        # dist, prev = dijkstra(self.graph, self.startNode)
        # t2 = time. time()
        # pathToEnd = to_array(prev, self.endNode)
        # distance = dist[self.endNode]
        # distance = 0 if distance is None else distance
        route, dist = a_star(int(self.startNode), int(self.endNode), self.node_coords, self.graph)
        if self.startNode != self.endNode:
            assert route != []
        # t3 = time.time()
        # print(t2-t1, t3-t2)
        route = list(map(str, route))
        return dist, route



