import heapq
import math
import numpy as np

class AStarAnglePlanner:
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
        self.expanded_nodes = 0
        self.expanded_coords = []

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
            self.expanded_coords.append(current)

            if current == goal_idx:
                return self._reconstruct_path(came_from, start_idx, goal_idx)

            for neighbor in self._get_neighbors(current):
                new_cost = cost_so_far[current] + self._move_cost(current, neighbor)
                add_penalty = self._alignment_penalty(neighbor, start_idx, goal_idx)
                new_cost += add_penalty

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_idx)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
        return None

    def get_expanded_set(self):
        return self.expanded_coords

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    def _move_cost(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return self.resolution * (math.sqrt(2) if dx and dy else 1)

    def _get_neighbors(self, idx):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(idx[0] + dx, idx[1] + dy)
                for dx, dy in directions
                if 0 <= idx[0] + dx < self.width and 0 <= idx[1] + dy < self.height and
                not self.grid_map.check_occupancy(idx[0] + dx, idx[1] + dy)]

    def _idx_to_pos(self, idx):
        return [self.origin_x + (idx[0] + 0.5) * self.resolution,
                self.origin_y + (idx[1] + 0.5) * self.resolution]

    def _reconstruct_path(self, came_from, start_idx, goal_idx):
        path = []
        current = goal_idx
        while current != start_idx:
            path.append(self._idx_to_pos(current))
            current = came_from.get(current, start_idx)
        path.append(self._idx_to_pos(start_idx))
        path.reverse()
        return path

    def _alignment_penalty(self, neighbor, start, goal):
        p_rate = 0.3
        move_vec = np.array([neighbor[0] - start[0], neighbor[1] - start[1]])
        goal_vec = np.array([goal[0] - start[0], goal[1] - start[1]])
        mag1, mag2 = np.linalg.norm(move_vec), np.linalg.norm(goal_vec)
        if mag1 == 0 or mag2 == 0:
            return 0
        cos = np.dot(move_vec, goal_vec) / (mag1 * mag2)
        ang_diff = math.acos(np.clip(cos, -1.0, 1.0))
        return ang_diff * self.resolution * p_rate


class AStarDistancePlanner:
    def __init__(self, start_px, goal_points, grid_map):
        self.grid_map = grid_map
        self.width = grid_map.width
        self.height = grid_map.height
        self.resolution = grid_map.resolution
        self.center_x = grid_map.center_x
        self.center_y = grid_map.center_y
        self.origin_x = self.center_x - (self.width * self.resolution) / 2.0
        self.origin_y = self.center_y - (self.height * self.resolution) / 2.0

        self.start = start_px
        self.goal = goal_points
        self.expanded_nodes = 0
        self.expanded_coords = []

    def plan(self):
        sx, sy, valid_s = self.grid_map.get_xy_index(self.start[0], self.start[1])
        gx, gy, valid_g = self.grid_map.get_xy_index(self.goal[-1][0], self.goal[-1][1])
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
            self.expanded_coords.append(current)

            if current == goal_idx:
                return self._reconstruct_path(came_from, start_idx, goal_idx)

            for neighbor in self._get_neighbors(current):
                new_cost = cost_so_far[current] + self._move_cost(current, neighbor)
                add_penalty = self._distance_penalty(start_idx, neighbor)
                new_cost += add_penalty

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_idx)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
        return None

    def get_expanded_set(self):
        return self.expanded_coords

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    def _move_cost(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return self.resolution * (math.sqrt(2) if dx and dy else 1)

    def _get_neighbors(self, idx):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(idx[0] + dx, idx[1] + dy)
                for dx, dy in directions
                if 0 <= idx[0] + dx < self.width and 0 <= idx[1] + dy < self.height and
                not self.grid_map.check_occupancy(idx[0] + dx, idx[1] + dy)]

    def _idx_to_pos(self, idx):
        return [self.origin_x + (idx[0] + 0.5) * self.resolution,
                self.origin_y + (idx[1] + 0.5) * self.resolution]

    def _reconstruct_path(self, came_from, start_idx, goal_idx):
        path = []
        current = goal_idx
        while current != start_idx:
            path.append(self._idx_to_pos(current))
            current = came_from.get(current, start_idx)
        path.append(self._idx_to_pos(start_idx))
        path.reverse()
        return path

    def _distance_penalty(self, start, neighbor):
        if len(self.goal) == 1:
            prev_goal = start
        else:
            prev_goal = self.goal[-2]
        curr_goal = self.goal[-1]
        x = (neighbor[0] + 0.5) * self.resolution
        y = (neighbor[1] + 0.5) * self.resolution
        dx1, dy1 = x - prev_goal[0], y - prev_goal[1]
        dx2, dy2 = curr_goal[0] - x, curr_goal[1] - y
        dist1, dist2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
        p_rate = math.hypot(dx1 + dx2, dy1 + dy2)
        return (dist1 + dist2) / p_rate if p_rate != 0 else 0


class CentralAStarPlanner:
    def __init__(self, start, goal, grid_map):
        self.grid_map = grid_map
        self.width = grid_map.width
        self.height = grid_map.height
        self.resolution = grid_map.resolution
        self.start = start
        self.goal = goal
        self.expanded_coords = []

    def get_expanded_set(self):
        return self.expanded_coords

    def plan(self):
        sx, sy, valid_s = self.grid_map.get_xy_index(self.start[0], self.start[1])
        gx, gy, valid_g = self.grid_map.get_xy_index(self.goal[0], self.goal[1])
        if not valid_s or not valid_g:
            return None

        start_idx = (sx, sy)
        goal_idx = (gx, gy)

        successors = self._build_successors()
        reversed_successors = self._invert_graph(successors)
        costs = self.compute_costs(successors)
        counts_from_goal, _ = self.count_paths(goal_idx, reversed_successors, costs)
        counts_from_source, _ = self.count_paths(start_idx, successors, costs)
        path = self.generate_path(start_idx, goal_idx, counts_from_source, counts_from_goal, successors)
        return [self._idx_to_pos(p) for p in path]

    def _build_successors(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        successors = {}
        for x in range(self.width):
            for y in range(self.height):
                if self.grid_map.check_occupancy(x, y):
                    continue
                current = (x, y)
                neighbors = []
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if not self.grid_map.check_occupancy(nx, ny):
                            neighbors.append((nx, ny))
                successors[current] = neighbors
        return successors

    def _invert_graph(self, graph):
        inverted = {node: [] for node in graph}
        for node in graph:
            for succ in graph[node]:
                if succ in inverted:
                    inverted[succ].append(node)
        return inverted

    def compute_costs(self, successors):
        costs = {}
        for node, neighbors in successors.items():
            for succ in neighbors:
                dx, dy = succ[0] - node[0], succ[1] - node[1]
                costs[(node, succ)] = math.sqrt(2) if abs(dx) == 1 and abs(dy) == 1 else 1
        return costs

    def count_paths(self, source, successors, costs):
        counts = {}
        costs_map = {}
        queue = [(0, source)]
        counts[source] = 1
        costs_map[source] = 0
        while queue:
            _, vertex = heapq.heappop(queue)
            self.expanded_coords.append(vertex)
            for succ in successors.get(vertex, []):
                new_cost = costs_map[vertex] + costs.get((vertex, succ), 1)
                if succ not in costs_map or new_cost < costs_map[succ]:
                    costs_map[succ] = new_cost
                    heapq.heappush(queue, (new_cost, succ))
                counts[succ] = counts.get(succ, 0) + counts[vertex]
        return counts, None

    def generate_path(self, source, goal, counts_from_source, counts_from_goal, successors):
        path = [source]
        while path[-1] != goal:
            current = path[-1]
            best_succ = None
            max_count = -1
            for succ in successors.get(current, []):
                count = counts_from_source.get(succ, 0) * counts_from_goal.get(succ, 0)
                if count > max_count:
                    best_succ = succ
                    max_count = count
            if best_succ is None:
                break
            path.append(best_succ)
        return path

    def _idx_to_pos(self, idx):
        return [(idx[0] + 0.5) * self.resolution,
            (idx[1] + 0.5) * self.resolution
        ]
    
class LazyThetaStarPlanner:
    def __init__(self, start, goal, grid_map):
        self.start = start
        self.goal = goal
        self.grid_map = grid_map
        self.resolution = grid_map.resolution
        self.width = grid_map.width
        self.height = grid_map.height
        self.origin_x = grid_map.center_x - (self.width * self.resolution) / 2.0
        self.origin_y = grid_map.center_y - (self.height * self.resolution) / 2.0
        self.expanded_nodes = 0
        self.expanded_coords = []

    def plan(self):
        sx, sy, valid_s = self.grid_map.get_xy_index(self.start[0], self.start[1])
        gx, gy, valid_g = self.grid_map.get_xy_index(self.goal[0], self.goal[1])
        if not valid_s or not valid_g:
            return None

        start_idx = (sx, sy)
        goal_idx = (gx, gy)

        g_cost = {start_idx: 0}
        parent = {start_idx: start_idx}
        open_list = [(self.heuristic(start_idx, goal_idx), start_idx)]
        visited = set()

        while open_list:
            _, current = heapq.heappop(open_list)
            self.expanded_nodes += 1
            self.expanded_coords.append(current)

            if not self._line_of_sight(parent[current], current):
                for neighbor in self._get_neighbors(current):
                    tentative_g = g_cost[current] + self._euclidean(current, neighbor)
                    if tentative_g < g_cost.get(neighbor, float('inf')):
                        g_cost[neighbor] = tentative_g
                        parent[neighbor] = current
                        heapq.heappush(open_list, (tentative_g + self.heuristic(neighbor, goal_idx), neighbor))
                continue

            if current == goal_idx:
                return self._reconstruct_path(parent, start_idx, goal_idx)

            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                if neighbor not in g_cost:
                    g_cost[neighbor] = float('inf')
                    parent[neighbor] = None
                tentative_g = g_cost[parent[current]] + self._euclidean(parent[current], neighbor)
                if tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    parent[neighbor] = parent[current]
                    heapq.heappush(open_list, (tentative_g + self.heuristic(neighbor, goal_idx), neighbor))
            visited.add(current)
        return None

    def get_expanded_set(self):
        return self.expanded_coords

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _euclidean(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _line_of_sight(self, a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = 1 if x1 > x0 else -1, 1 if y1 > y0 else -1
        err = dx - dy
        while True:
            if self.grid_map.check_occupancy(x0, y0):
                return False
            if (x0, y0) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    def _get_neighbors(self, idx):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(idx[0] + dx, idx[1] + dy)
                for dx, dy in directions
                if 0 <= idx[0] + dx < self.width and 0 <= idx[1] + dy < self.height and
                not self.grid_map.check_occupancy(idx[0] + dx, idx[1] + dy)]

    def _idx_to_pos(self, idx):
        return [self.origin_x + (idx[0] + 0.5) * self.resolution,
                self.origin_y + (idx[1] + 0.5) * self.resolution]

    def _reconstruct_path(self, parent, start_idx, goal_idx):
        path = []
        current = goal_idx
        while current != start_idx:
            path.append(self._idx_to_pos(current))
            current = parent[current]
        path.append(self._idx_to_pos(start_idx))
        path.reverse()
        return path


