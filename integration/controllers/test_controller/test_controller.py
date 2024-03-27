"""test_controller controller."""

import argparse
import datetime
# import gym
import numpy as np
import itertools
# import torch
# from robot_controller import DRV90_Robot
from robot_controller import DRV90_Robot
from torch.utils.tensorboard import SummaryWriter
# from replay_memory import ReplayMemory
import random
# import utils
import pandas as pd
import os

class Object:
    def __init__(self, DEF_name):
        self.name = DEF_name
        self.node = armChain_down.supervisor.getFromDef(DEF_name)
        self.translation_field = self.node.getField('translation')
        self.position = self.node.getPosition()
        self.position_candididate = [[-0.23, 0.5, 0.24],
                                     [0.42, 0.5, 0.24],
                                     [0.02, 0.45, 0.86],
                                     [0.02, 0.35, 0.6],
                                     [0.02, -0.44095, 0.16704],
                                     [0.195, -0.22661, 0.38138],
                                     [0.37871, -0.62161, 0.62638],
                                     [0.20724, -0.57161, 0.71638]]
        self.new_position = None

    def read_position(self):
        self.position = self.node.getPosition()
    
    def random_choose_one_position(self):
            random_index = random.randint(0, len(self.position_candididate) - 1)
            self.new_position = self.position_candididate[random_index]

    def setP(self, position):
            self.translation_field.setSFVec3f(position)

#################################################################

def calculate_distance(a_position, b_position):
    assert len(a_position) == len(b_position), "Error: length not the same, cant calculate"
    
    tem = 0
    for i in range(len(a_position)):
        tem += (a_position[i] - b_position[i]) ** 2
    
    distance = tem ** 0.5

    return distance

def move_end_point_to_ball():
    ball.read_position()
    print(ball.position)

    target_position = ball.position.copy()
    target_position[-1] = target_position[-1] + 0.06

    armChain_down.movP(target_position)
    end_position = armChain_down.find_endposition()

    distance = calculate_distance(target_position, end_position)

    P = armChain_down.getCurP()
    A = armChain_down.currAngles()
    J = armChain_down.getCurJ()

    if i % 2 == 0: 
        print("target position: ", [round(x, 5) for x in target_position])
        print("fp", [round(x, 5) for x in end_position])
        print("gp", [round(x, 5) for x in P])
        print("ca", [round(x, 5) for x in A])
        print("gj", [round(x, 5) for x in J])
        print("distance", distance)
        print("---" * 10)
    
    if distance <= 1e-4:
        ball.random_choose_one_position()
        ball.setP(ball.new_position)


def move_joint():
    # armChain_down.arm_joint_move(30, 1)
    armChain_down.motors[1].setPosition(0.1)
    pass

###############################################################
if __name__ == "__main__":

    ### init
    armChain_down = DRV90_Robot("DRV90.urdf", end_down=False)
    target_position = [0.25, 0, 0.45]
    armChain_down.movP(target_position)

    # Set the random seed for reproducibility
    random_seed = 525
    random.seed(random_seed)

    ### ball
    ball = Object("ball")
    
    # Main loop:
    i = 0
    while armChain_down.supervisor.step(armChain_down.timestep) != -1:
        move_end_point_to_ball()
        # move_joint()
        i += 1
