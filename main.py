import cv2
import numpy as np
import time
import os
from planner import AStarPlanner
from grid_map import GridMap

# === Load binary map ===
map_path = os.path.expanduser("~/a_star_algorithm/map.png")
img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
assert img is not None, "map.png failed to load. Check the file path."
_, binary_map = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
grid = (binary_map == 255).astype(np.uint8)  # 1: free, 0: obstacle

base_img = binary_map.copy()

# Global state
goal_points = []
path_segments = []
start_px = None
initial_start_px = None
reset_requested = False

def mouse_callback(event, x, y, flags, param):
    global start_px, initial_start_px, goal_points, reset_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_px is None:
            start_px = (x, y)
            initial_start_px = (x, y)
            print(f"Start set: ({x}, {y})")
        else:
            goal_points.append((x, y))
            print(f"New goal added: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        reset_requested = True
        print("Right-click detected. Resetting.")

cv2.namedWindow("Map")
cv2.setMouseCallback("Map", mouse_callback)

while True:
    current_img = base_img.copy()
    vis = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)

    # === 1. Show base map until at least start point is set ===
    while start_px is None and not reset_requested:
        cv2.imshow("Map", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pressed. Exiting.")
            cv2.destroyAllWindows()
            exit()

    if reset_requested:
        start_px = None
        initial_start_px = None
        goal_points.clear()
        path_segments.clear()
        reset_requested = False
        continue

    # === 2. If new goal point added, plan from last start to new goal ===
    if len(goal_points) > len(path_segments):
        goal_px = goal_points[-1]
        resolution = 1  # meters per pixel
        grid_map = GridMap(grid, resolution)
        planner = AStarPlanner(start_px, goal_px, grid_map)

        print("Running A* path planning...")
        start_time = time.time()
        path = planner.plan()
        end_time = time.time()

        if path is None:
            print("No path found.")
            goal_points.pop()  # Remove invalid goal
            continue

        print("-----------------------------------")
        print(f"Path found.")
        print(f"-Time taken: {end_time - start_time:.4f} sec")
        print(f"-Path length: {len(path)} nodes ({len(path) * resolution:.2f} m)")
        print(f"-Expanded nodes: {planner.expanded_nodes} nodes")
        print("-----------------------------------")

        path_segments.append(path)
        start_px = goal_px  # Update current position for next planning

    # === 3. Draw all paths ===
    for i, path in enumerate(path_segments):
        color = (0, 0, 255) if i == len(path_segments) - 1 else (255, 0, 0)
        for j in range(1, len(path)):
            pt1 = tuple(map(int, path[j - 1]))
            pt2 = tuple(map(int, path[j]))
            cv2.line(vis, pt1, pt2, color, 1)

    # === 4. Draw start and goal points with numbering ===
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.4;  thickness = 1; offset = (5, -5)

    if initial_start_px is not None:
        cv2.circle(vis, initial_start_px, 5, (0, 0, 255), -1)
        text_pos = (initial_start_px[0] + offset[0], initial_start_px[1] + offset[1])
        cv2.putText(vis, "0", text_pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    for i, pt in enumerate(goal_points):
        cv2.circle(vis, pt, 5, (0, 0, 255), -1)
        text = str(i + 1)
        text_pos = (pt[0] + offset[0], pt[1] + offset[1])
        cv2.putText(vis, text, text_pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # === 5. Show result, wait for new goal or right-click ===
    cv2.imshow("Map", vis)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        exit()
