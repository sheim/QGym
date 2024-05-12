import cv2
import os
from gym import LEGGED_GYM_ROOT_DIR
import re


def extract_iteration(filename):
    # Use regular expression to find the numbers following 'it'
    match = re.search(r"it(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # Return 0 or some default value if 'it' followed by digits is not found


image_folder = (
    os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph")
    + "/20240512_153648"
)
video_name = "video.avi"
img_names = os.listdir(image_folder)
# Sort the list using the custom key function
img_names = sorted(img_names, key=extract_iteration)
images = [img for img in img_names if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
