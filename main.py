import cv2
import numpy as np
import time
import os
from planner import AStarPlanner
from grid_map import GridMap

# === Load binary map ===
map_path = os.path.expanduser("~/a_star_on_map/project/map.png")
img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
assert img is not None, "map.png failed to load. Check the file path."
_, binary_map = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
grid = (binary_map == 255).astype(np.uint8)  # 1: free, 0: obstacle

# Base image to reset later
base_img = binary_map.copy()

# Global state
points = []
reset_requested = False

# === Mouse callback ===
def mouse_callback(event, x, y, flags, param):
    global points, reset_requested
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        reset_requested = True
        print("Right-click detected. Resetting points and view...")

cv2.namedWindow("Map")
cv2.setMouseCallback("Map", mouse_callback)

# === Main interactive loop ===
while True:
    # === 1. Show base map until 2 points are selected ===
    current_img = base_img.copy()
    while len(points) < 2 and not reset_requested:
        vis = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
        for pt in points:
            cv2.circle(vis, pt, 6, (0, 0, 255), -1)
        cv2.imshow("Map", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pressed. Exiting.")
            cv2.destroyAllWindows()
            exit()

    if reset_requested:
        points.clear()
        reset_requested = False
        continue

    # === 2. Run A* when 2 points are selected ===
    start_px, goal_px = points
    resolution = 1  # meters per pixel
    grid_map = GridMap(grid, resolution)
    planner = AStarPlanner(start_px, goal_px, grid_map)

    print("Running A* path planning...")
    start_time = time.time()
    path = planner.plan()
    end_time = time.time()

    if path is None:
        print("No path found.")
        exit()
    else:
        print("----------------------------------")
        print(f"Path found.")
        print(f"Time taken: {end_time - start_time:.4f} sec")
        print(f"Path length: {len(path)} nodes ({len(path) * resolution:.2f} m)")
        print(f"Expanded nodes: {planner.expanded_nodes} nodes")
        print("----------------------------------")


        # === 3. Draw path ===
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(1, len(path)):
            pt1 = tuple(map(int, path[i - 1]))
            pt2 = tuple(map(int, path[i]))
            cv2.line(img_color, pt1, pt2, (255, 0, 0), 1)

        # Start/goal points
        cv2.circle(img_color, start_px, 6, (0, 255, 0), -1)
        cv2.circle(img_color, goal_px, 6, (0, 0, 255), -1)

        # === 4. Show result until right-click to restart ===
        print("Viewing result. Right-click to restart, or ESC to quit.")
        while True:
            cv2.imshow("Map", img_color)
            key = cv2.waitKey(1)
            if reset_requested:
                points.clear()
                reset_requested = False
                break
            elif key == 27:
                cv2.destroyAllWindows()
                exit()
