import numpy as np

class GridMap:
    def __init__(self, grid, resolution=1.0):
        """
        :param grid: 2D numpy array (0: obstacle, 1: free)
        :param resolution: size of one grid cell (e.g., in meters)
        """
        self.grid = grid
        self.height, self.width = grid.shape
        self.resolution = resolution
        self.center_x = self.width * resolution / 2.0
        self.center_y = self.height * resolution / 2.0

    def check_occupancy(self, x_idx, y_idx):
        """
        Check if cell at (x_idx, y_idx) is occupied.
        Returns True if occupied (i.e., obstacle).
        """
        if 0 <= x_idx < self.width and 0 <= y_idx < self.height:
            return self.grid[y_idx, x_idx] == 0  # 0 = obstacle
        return True  # Outside bounds = obstacle

    def get_xy_index(self, x, y):
        """
        Convert world coordinates (x, y) to grid indices.
        Returns (x_idx, y_idx, valid).
        """
        x_idx = int(x // self.resolution)
        y_idx = int(y // self.resolution)
        valid = (0 <= x_idx < self.width) and (0 <= y_idx < self.height)
        return x_idx, y_idx, valid
