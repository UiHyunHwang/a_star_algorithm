import cv2
import numpy as np
import time
import os
from planners import AStarAnglePlanner, AStarDistancePlanner, CentralAStarPlanner, LazyThetaStarPlanner
from grid_map import GridMap

# === Load binary map ===
map_path = os.path.expanduser("~/a_star_on_map/a_star_algorithm/map.png")
img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
assert img is not None, "map.png failed to load. Check the file path."
_, binary_map = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
grid = (binary_map == 255).astype(np.uint8)
base_img = binary_map.copy()

# === Global state ===
goal_points = []
path_segments = []
start_px = None
initial_start_px = None
reset_requested = False
terminate_flag = False  
use_mode2 = False
new_goal_input_ready = False
pending_goal = None
add_goal_prompt_pending = False
all_expanded_nodes = []

cv2.namedWindow("Map")

# === Visualization ===
def draw_paths(vis):
    for i, path in enumerate(path_segments):
        color = (0, 0, 255) if i == len(path_segments) - 1 else (255, 0, 0)
        for j in range(1, len(path)):
            pt1 = tuple(map(int, path[j - 1]))
            pt2 = tuple(map(int, path[j]))
            cv2.line(vis, pt1, pt2, color, 1)

def draw_points(vis):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    offset = (5, -5)
    if initial_start_px is not None:
        cv2.circle(vis, initial_start_px, 5, (0, 0, 255), -1)
        pos = (initial_start_px[0] + offset[0], initial_start_px[1] + offset[1])
        cv2.putText(vis, "0", pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    for i, pt in enumerate(goal_points):
        cv2.circle(vis, pt, 5, (0, 0, 255), -1)
        pos = (pt[0] + offset[0], pt[1] + offset[1])
        cv2.putText(vis, str(i + 1), pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def draw_expanded_nodes(vis):
    for node in all_expanded_nodes:
        cv2.circle(vis, node, 1, (200, 200, 200), -1)  # light gray

# === Mouse callback (mode1) ===
def mouse_callback(event, x, y, flags, param):
    global start_px, initial_start_px, goal_points, reset_requested
    if use_mode2:
        return
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

# === Terminal input (mode2) ===
def terminal_goal_input():
    global start_px, initial_start_px, new_goal_input_ready, pending_goal, add_goal_prompt_pending
    while not terminate_flag:
        try:
            start_px = None
            initial_start_px = None
            goal_points.clear()
            path_segments.clear()
            all_expanded_nodes.clear()

            sx_sy = input("Enter start (x y): ").strip().split()
            if len(sx_sy) != 2:
                raise ValueError
            sx, sy = map(int, sx_sy)
            start_px = (sx, sy)
            initial_start_px = (sx, sy)
        except:
            print("Invalid input. Aborting.")
            return

        while True:
            try:
                gx_gy = input("Enter goal (x y): ").strip().split()
                if len(gx_gy) != 2:
                    raise ValueError
                gx, gy = map(int, gx_gy)
                pending_goal = (gx, gy)
                new_goal_input_ready = True

                while new_goal_input_ready:
                    time.sleep(0.05)
                while not add_goal_prompt_pending:
                    time.sleep(0.05)

                while True:
                    cont = input("Add another goal? (y/n): ").strip().lower()
                    if cont == 'y':
                        add_goal_prompt_pending = False
                        break
                    elif cont == 'n':
                        add_goal_prompt_pending = False
                        break
                    else:
                        print("Please enter 'y' or 'n'.")

                if cont == 'n':
                    break
            except Exception as e:
                print("Error:", e)
                print("Invalid coordinate input. Skipping.")
                continue

# === Main loop ===
def main():
    global use_mode2, reset_requested
    global start_px, initial_start_px, goal_points, path_segments
    global new_goal_input_ready, pending_goal, add_goal_prompt_pending
    global all_expanded_nodes

    planner_type = input("Select planner (1: Angle, 2: Distance, 3: Central, 4: LazyTheta*): ").strip()
    mode = input("Select mode (1: mouse click, 2: terminal input): ").strip()
    use_mode2 = (mode == "2")

    if use_mode2:
        import threading
        threading.Thread(target=terminal_goal_input, daemon=True).start()

    cv2.setMouseCallback("Map", mouse_callback)

    while True:
        vis = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR)
        draw_expanded_nodes(vis)
        draw_paths(vis)
        draw_points(vis)
        cv2.imshow("Map", vis)

        key = cv2.waitKey(30)
        if key == 27:
            terminate_flag = True
            time.sleep(0.1)
            break

        if not use_mode2:
            if reset_requested:
                start_px = None
                initial_start_px = None
                goal_points.clear()
                path_segments.clear()
                all_expanded_nodes.clear()
                reset_requested = False
                all_expanded_nodes.clear()
                continue

            if start_px is not None and len(goal_points) > len(path_segments):
                goal_px = goal_points[-1]
                resolution = 1
                grid_map = GridMap(grid, resolution)

                if planner_type == "2":
                    planner = AStarDistancePlanner(start_px, goal_points, grid_map)
                else:
                    planner_class = {
                        "1": AStarAnglePlanner,
                        "3": CentralAStarPlanner,
                        "4": LazyThetaStarPlanner,
                    }.get(planner_type, AStarAnglePlanner)
                    planner = planner_class(start_px, goal_px, grid_map)

                print("Planning...")
                start_time = time.time()
                path = planner.plan()
                end_time = time.time()

                if path is None:
                    print("No path found.")
                    goal_points.pop()
                    continue

                print("-----------------------------------")
                print("Path found.")
                print(f"- Time taken: {end_time - start_time:.4f} sec")
                print(f"- Path length: {len(path)} nodes")
                print(f"- Expanded nodes: {planner.expanded_nodes}")
                print("-----------------------------------")

                path_segments.append(path)
                all_expanded_nodes.extend(planner.get_expanded_set())
                start_px = goal_px

        elif use_mode2 and new_goal_input_ready and pending_goal is not None:
            goal_points.append(pending_goal)
            goal_px = pending_goal
            pending_goal = None
            new_goal_input_ready = False

            resolution = 1
            grid_map = GridMap(grid, resolution)

            if planner_type == "2":
                planner = AStarDistancePlanner(start_px, goal_points, grid_map)
            else:
                planner_class = {
                    "1": AStarAnglePlanner,
                    "3": CentralAStarPlanner,
                    "4": LazyThetaStarPlanner,
                }.get(planner_type, AStarAnglePlanner)
                planner = planner_class(start_px, goal_px, grid_map)

            print("Planning...")
            start_time = time.time()
            path = planner.plan()
            end_time = time.time()

            if path is None:
                print("No path found.")
                goal_points.pop()
            else:
                print("-----------------------------------")
                print("Path found.")
                print(f"- Time taken: {end_time - start_time:.4f} sec")
                print(f"- Path length: {len(path)} nodes")
                print(f"- Expanded nodes: {planner.expanded_nodes}")
                print("-----------------------------------")

                path_segments.append(path)
                all_expanded_nodes.extend(planner.get_expanded_set())
                start_px = goal_px

            add_goal_prompt_pending = True

    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    main()
