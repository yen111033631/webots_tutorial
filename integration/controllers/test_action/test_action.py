import argparse
import datetime
# import gym
import numpy as np
import itertools
# import torch
# from robot_controller import DRV90_Robot
from robot_controller_copy import DRV90_Robot
# from sac import SAC
from torch.utils.tensorboard import SummaryWriter
# from replay_memory import ReplayMemory
import random
# import utils
import pandas as pd
import os

def list_add(list1, list2):
    list3 = []
    for i in range(len(list1)):
        list3.append(list1[i] + list2[i])
    return list3

def world2arm(world_position):
    arm_position = [world_position[0], -world_position[2], -world_position[1]]
    return arm_position

def arm2world(arm_position):
    world_position = [arm_position[0], -arm_position[2], -arm_position[1]]
    return world_position


armChain_down = DRV90_Robot("DRV90ASS_5axis.urdf",end_down=True)
# armChain_down = DRV90_Robot("DRV90_fixJ4.urdf",end_down=True)

print("timestep: ", armChain_down.timestep)

### cube
cube_node = armChain_down.supervisor.getFromDef('cube')
position = cube_node.getPosition()
print(position)
print('cube position: %f %f %f\n' %(position[0], position[1], position[2]))

### start position
current_position = armChain_down.find_endposition()
print("current_position:", end=" ")
print([round(x, 5) for x in current_position])

### end position
target_position = position.copy()
print('target_position: %f %f %f\n' %(target_position[0], target_position[1], target_position[2]))



i = 0
while armChain_down.supervisor.step(armChain_down.timestep) != 1:
    # armChain_down.position_move(0.5,0.0,0.3)
    # armChain_down.position_move(*target_position)
    armChain_down.movP(target_position)
    end_position = armChain_down.find_endposition()
    if i % 2 == 0: 
        print("target position: ", [round(x, 5) for x in target_position])
        print("now position: ", [round(x, 5) for x in end_position])
        print("---" * 10)

    i += 1