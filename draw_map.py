import cv2
import numpy as np
import os

# Configuration
canvas_size = (512, 512)
brush_radius = 5
output_path = os.path.expanduser("~/a_star_on_map/project/map.png")


# Create a white canvas (255 = white)
canvas = np.ones(canvas_size, dtype=np.uint8) * 255
drawing = False
prev_point = None  # Previous point for smooth drawing


# Mouse callback function for drawing
def draw_obstacle(event, x, y, flags, param):
    global drawing, prev_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if prev_point is not None:
            cv2.line(canvas, prev_point, (x, y), 0, thickness=brush_radius * 2)
            prev_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_point = None

# Create window and set mouse callback 
cv2.namedWindow("Draw Map")
cv2.setMouseCallback("Draw Map", draw_obstacle)

print("Draw obstacles with the mouse. Press ESC to save and exit.")

while True:
    cv2.imshow("Draw Map", canvas)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key pressed
        break

cv2.destroyAllWindows()


# Ensure output directory exists and save the image
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, canvas)
print(f"Map saved to {output_path}")
