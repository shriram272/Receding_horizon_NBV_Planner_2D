import numpy as np
import copy

def collision_check(x0, y0, x1, y1, ground_truth, robot_belief, robot_position, horizontal_fov_angle):
    def is_in_horizontal_fov(point, robot_position, horizontal_fov_angle):
        direction_vector = point - robot_position
        angle = np.arctan2(direction_vector[1], direction_vector[0])  # Angle in radians

        # Assuming robot's forward direction is along the positive x-axis
        robot_direction_angle = 0  # Adjust if robot's default direction is different

        relative_angle = np.abs(angle - robot_direction_angle)

        # Normalize angle to the range [-π, π]
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

        # Check if the relative angle is within the horizontal FoV
        horizontal_fov_rad = np.deg2rad(horizontal_fov_angle / 2.0)
        return np.abs(relative_angle) <= horizontal_fov_rad

    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 10

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        if not is_in_horizontal_fov(np.array([x, y]), robot_position, horizontal_fov_angle):
            break

        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k != 1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_belief


def sensor_work(robot_position, sensor_range, robot_belief, ground_truth, horizontal_fov_angle):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief, robot_position, horizontal_fov_angle)
        sensor_angle += sensor_angle_inc
    return robot_belief


def unexplored_area_check(x0, y0, x1, y1, current_belief, robot_position, horizontal_fov_angle):
    def is_in_horizontal_fov(point, robot_position, horizontal_fov_angle):
        direction_vector = point - robot_position
        angle = np.arctan2(direction_vector[1], direction_vector[0])  # Angle in radians

        # Assuming robot's forward direction is along the positive x-axis
        robot_direction_angle = 0  # Adjust if robot's default direction is different

        relative_angle = np.abs(angle - robot_direction_angle)

        # Normalize angle to the range [-π, π]
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

        # Check if the relative angle is within the horizontal FoV
        horizontal_fov_rad = np.deg2rad(horizontal_fov_angle / 2.0)
        return np.abs(relative_angle) <= horizontal_fov_rad

    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < current_belief.shape[1] and 0 <= y < current_belief.shape[0]:
        if not is_in_horizontal_fov(np.array([x, y]), robot_position, horizontal_fov_angle):
            break

        k = current_belief.item(y, x)
        if x == x1 and y == y1:
            break

        if k == 1:
            break

        if k == 127:
            current_belief.itemset((y, x), 0)
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return current_belief


def calculate_utility(waypoint_position, sensor_range, robot_belief, horizontal_fov_angle):
    sensor_angle_inc = 5 / 180 * np.pi
    sensor_angle = 0
    x0 = waypoint_position[0]
    y0 = waypoint_position[1]
    current_belief = copy.deepcopy(robot_belief)
    
    # Calculate the start and end angles based on the horizontal FOV
    start_angle = -horizontal_fov_angle / 2
    end_angle = horizontal_fov_angle / 2
    
    while start_angle <= sensor_angle <= end_angle:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        current_belief = unexplored_area_check(x0, y0, x1, y1, current_belief, waypoint_position, horizontal_fov_angle)
        sensor_angle += sensor_angle_inc
    
    utility = np.sum(robot_belief == 127) - np.sum(current_belief == 127)
    return utility