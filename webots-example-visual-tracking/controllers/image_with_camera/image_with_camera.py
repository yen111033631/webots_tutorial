"""image_with_camera controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
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


# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
print(timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

CamLeft = robot.getDevice("CamLeft")
CamLeft.enable(timestep)

w, h = CamLeft.getWidth(), CamLeft.getHeight()
print(w, h)



i = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    # Get the image from the camera
    img_rgb = get_image_with_camera(CamLeft)

    # Now you can use OpenCV to perform various image processing tasks
    # For example, displaying the image
    cv2.imshow("CamLeft Image", img_rgb)
    cv2.waitKey(1)  # This line is necessary for the OpenCV window to update



    # if i == 1:
    #     print(type(img_rgb))
    #     print(len(img_rgb))
    #     print(len(img_rgb) / w / h)


    #     print(img_rgb[:10])
    #     int_values = [int(byte) for byte in img_rgb]
        # print(int_values)


    
    # cv2.imshow("img", img)
    # cv2.waitKey(1)
    i += 1