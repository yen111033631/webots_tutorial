from controller import Supervisor
import cv2
import numpy as np


def get_image_with_camera(cam_node):
    # Get the image from the camera
    img = cam_node.getImage()

    # Decode the image data into a NumPy array
    img_np = np.frombuffer(img, dtype=np.uint8)

    # Reshape the NumPy array to get the image in the correct shape (height, width, channels)
    img_np = img_np.reshape((cam_node.getHeight(), cam_node.getWidth(), 4))

    # Extract the RGB channels (assuming 4 channels, where the fourth channel is often an alpha channel)
    img_rgb = img_np[:, :, :3]
    
    return img_rgb


robot = Supervisor()  # create Supervisor instance

timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

i = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    # Get the image from the camera
    img_rgb = get_image_with_camera(camera)

    # Now you can use OpenCV to perform various image processing tasks
    # For example, displaying the image
    cv2.imshow("cam_node Image", img_rgb)
    cv2.waitKey(1)  # This line is necessary for the OpenCV window to update

    i += 1


