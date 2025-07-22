import heapq
import math
import numpy as np

class AStarPlanner:
    """
    A* path planner for grid-based maps.
    """
    def __init__(self, start, goal, grid_map):
        self.grid_map = grid_map
        self.width = grid_map.width
        self.height = grid_map.height
        self.resolution = grid_map.resolution
        self.center_x = grid_map.center_x
        self.center_y = grid_map.center_y
        self.origin_x = self.center_x - (self.width * self.resolution) / 2.0
        self.origin_y = self.center_y - (self.height * self.resolution) / 2.0

        self.start = start
        self.goal = goal
        self.expanded_nodes = 0  # For logging

    def plan(self):
        sx, sy, valid_s = self.grid_map.get_xy_index(self.start[0], self.start[1])
        gx, gy, valid_g = self.grid_map.get_xy_index(self.goal[0], self.goal[1])
        if not valid_s or not valid_g:
            return None

        start_idx = (sx, sy)
        goal_idx = (gx, gy)

        open_heap = []
        heapq.heappush(open_heap, (0 + self.heuristic(start_idx, goal_idx), 0, start_idx))
        came_from = {}
        cost_so_far = {start_idx: 0}


        while open_heap:
            _, cost, current = heapq.heappop(open_heap)
            self.expanded_nodes += 1

            if current == goal_idx:
                return self._reconstruct_path(came_from, start_idx, goal_idx)

            for neighbor in self._get_neighbors(current):
                #====================== Change cost function here ======================#
                base_cost = cost_so_far[current] + self._move_cost(current, neighbor)
                add_penalty = self._alignment_penalty(current, neighbor, start_idx, goal_idx)

                new_cost = base_cost + add_penalty
                #=======================================================================#

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_idx)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
        return None

    def heuristic(self, a, b):
        return 4*math.hypot((a[0] - b[0]), (a[1] - b[1])) # Euclidian Distance

    def _move_cost(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx and dy:
            return self.resolution * math.sqrt(2)
        return self.resolution

    def _get_neighbors(self, idx):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            x2 = idx[0] + dx
            y2 = idx[1] + dy
            if 0 <= x2 < self.width and 0 <= y2 < self.height:
                if not self.grid_map.check_occupancy(x2, y2):
                    neighbors.append((x2, y2))
        return neighbors

    def _idx_to_pos(self, idx):
        x = self.origin_x + (idx[0] + 0.5) * self.resolution 
        y = self.origin_y + (idx[1] + 0.5) * self.resolution
        return [x, y]

    def _reconstruct_path(self, came_from, start_idx, goal_idx):
        path = []
        current = goal_idx
        while current != start_idx:
            path.append(self._idx_to_pos(current))
            current = came_from.get(current, start_idx)
        path.append(self._idx_to_pos(start_idx))
        path.reverse()
        return path

#=========================== define penalties ================================#
    
    def _alignment_penalty(self, current, neighbor, start, goal):
        p_rate = 1 # Hyperparameter

        move_vec = np.array([neighbor[0] - start[0], neighbor[1] - start[1]])
        goal_vec = np.array([goal[0] - start[0], goal[1] - start[1]])

        mag1 = np.linalg.norm(move_vec)
        mag2 = np.linalg.norm(goal_vec)
        if mag1 == 0 or mag2 == 0:
            return 0

        cos = np.dot(move_vec, goal_vec) / (mag1 * mag2)
        ang_diff = math.acos(np.clip(cos, -1.0, 1.0))

        return ang_diff * self.resolution * p_rate


