This project was part of research done during research internship in NUS Singapore. 
It focusses of implementing a next best view planner for multi agent exploration in a 2d environment with limited field of view sensors.

The code has been developed using the following research publications for reference as cited below - 

A. Bircher, M. Kamel, K. Alexis, H. Oleynikova and R. Siegwart, "Receding Horizon "Next-Best-View" Planner for 3D Exploration," 2016 IEEE International Conference on Robotics and Automation (ICRA), Stockholm, Sweden, 2016, pp. 1462-1468, doi: 10.1109/ICRA.2016.7487281.
keywords: {Vehicles;Robot sensing systems;Space exploration;Planning;Navigation;Three-dimensional displays}, 


OVERVIEW

1. Collision Checking

The check_collision function is responsible for detecting collisions between an agent and obstacles or other agents in the environment.
Mathematics and Logic:

    Line Intersection: The core logic for collision detection uses line intersection checks. The path between the agent's current position (start) and the target position (end) is treated as a line segment:
    line=shapely.geometry.LineString([start,end])
    line=shapely.geometry.LineString([start,end])

    Bounding Box Clipping: Before checking for collisions, the function crops the robot_belief grid to the bounding box around start and end. This speeds up collision checking by limiting it to the relevant region.

    Occupied and Unexplored Areas: The function identifies cells marked as occupied (value 1) and unexplored (value 127) within this region:
    occupied_area_coords={(x,y)∣robot_belief[y,x]=1}
    occupied_area_coords={(x,y)∣robot_belief[y,x]=1}
    unexplored_area_coords={(x,y)∣robot_belief[y,x]=127}
    unexplored_area_coords={(x,y)∣robot_belief[y,x]=127}

    Collision with Occupied Cells: Each occupied cell is treated as an obstacle, and a small buffer area is created around it. The function checks if the line intersects any of these buffered areas using:
    collision=line.intersects(obstacle)
    collision=line.intersects(obstacle)

    where obstacle is a polygon around the occupied cell.

    Collision with Unexplored Cells: Similarly, a small area around unexplored cells is treated as a potential obstacle, and intersection checks are performed.

    Agent-Agent Collision: The function also checks for potential collisions with other agents using the MultiagentCollisionChecker class. The method closest_distance_between_lines calculates the shortest distance between the paths of two agents:
    distance=closest_distance_between_lines(start1,end1,start2,end2)
    distance=closest_distance_between_lines(start1​,end1​,start2​,end2​)

2. Vertex Initialization and Gain Calculation

The Vertex class represents a node in the exploration tree. It holds information about the current position (coords), observable frontiers, and the exploration gain.
Mathematics and Logic:

    Observable Frontiers: The function initialize_observable_frontiers identifies potential frontiers (boundary between explored and unexplored areas) within the sensor range that are not occluded:
    dist_list=∥env.frontiers−coords∥
    dist_list=∥env.frontiers−coords∥
    frontiers_in_range={point∣dist_list<env.sensor_range−10}
    frontiers_in_range={point∣dist_list<env.sensor_range−10}

    For each frontier point, a collision check is performed to ensure the path to the frontier is clear:
    collision=check_collision(coords,point,robot_belief)
    collision=check_collision(coords,point,robot_belief)

    Gain Calculation: The gain of a vertex is based on the number of observable frontiers (visible) and the distance traveled (dist). It's calculated using:
    vertex.gain=visible×exp⁡(−10×dist640)
    vertex.gain=visible×exp(−10×640dist​)

    This equation balances exploration (number of frontiers) against the cost of moving (distance). The exponential decay factor ensures that longer paths contribute less gain.

3. Tree Expansion

The Tree class represents the exploration tree for an agent. It starts from the agent's current position and grows by adding vertices.
Mathematics and Logic:

    Tree Initialization: The tree starts with a single vertex at the agent's current position:
    tree.vertices={initial:Vertex}
    tree.vertices={initial:Vertex}

    Branch Extraction: The method extract_branch traces back the path from a vertex to the root:
    branch_index=[vertex.id]
    branch_index=[vertex.id]

    Sampling for Expansion: During tree expansion, the algorithm samples points either from the free area or the frontiers:
    dice∼U(0,1)
    dice∼U(0,1)
    sample_coords={random sample from free areaif dice>0.2random sample from frontiersotherwise
    sample_coords={random sample from free arearandom sample from frontiers​if dice>0.2otherwise​

    Nearest Vertex: The closest existing vertex in the tree to the sampled point is found:
    dist_list=∥sample_coords−vertices_coords∥
    dist_list=∥sample_coords−vertices_coords∥
    nearest_vertex_index=arg⁡min⁡(dist_list)
    nearest_vertex_index=argmin(dist_list)

    New Vertex Addition: A new vertex is created and added to the tree if there is no collision:
    new_vertex_coords={sample_coordsif dist≤step_lengthintermediate point along lineotherwise
    new_vertex_coords={sample_coordsintermediate point along line​if dist≤step_lengthotherwise​

4. Next Best Viewpoint (NBV) Planning

The method find_next_best_viewpoints finds the next position for an agent to move to by expanding the tree.
Mathematics and Logic:

    Tree Expansion: The tree is expanded iteratively. The goal is to find a path with the highest gain:
    gbest=max⁡(vertex.gain)
    gbest​=max(vertex.gain)

    Iterative Process: The process runs for a fixed number of iterations or until a path with a positive gain is found:
    while i≤max_iter_steps or gbest=0:
    while i≤max_iter_steps or gbest​=0:

    Return: The function returns the coordinates of the next best node and the planned paths:
    best_route[1],self.planned_paths[agent_id]
    best_route[1],self.planned_paths[agent_id]



RESULTS


![0_1_samples](https://github.com/user-attachments/assets/f95466a7-4d85-4f7b-8c0c-dfa7518bd852)

![0_6_samples](https://github.com/user-attachments/assets/24dd26d0-ff61-455b-97ae-e6817025ac14)

![0_17_samples](https://github.com/user-attachments/assets/193012e0-098c-46c5-b345-e11590edb562)

![0_24_samples](https://github.com/user-attachments/assets/5c453464-e785-4098-8280-bdba760df63f)


![0_explored_rate_0 9968_length_4416](https://github.com/user-attachments/assets/48edbab6-2a81-4091-80a4-4d3587e7a362)
