import argparse
import datetime
# import gym
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
import itertools
# import torch
from robot_controller import DRV90_Robot
from pyModbusTCP.client import ModbusClient
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import random
import utils
import pandas as pd
import os
from ikpy.chain import Chain
from time import sleep
from PIL import ImageGrab
import PIL

# from torchvision import transforms

# armChain_down = DRV90_Robot("drv90l.urdf", end_down=False)
armChain_down = DRV90_Robot("DRV90.urdf", end_down=False)
# armChain_down = DRV90_Robot("fixHeight.urdf", end_down=False)
# armChain_down = DRV90_Robot("fixJ1_TEST.urdf", end_down=False)
# armChain_down = DRV90_Robot("DRV90ASS_456fix_axis.urdf", end_down=False)
# armChain_down = DRV90_Robot("DRV90ASS_fix_axis.urdf", end_down=False)
# armChain_down = DRV90_Robot("DRV90ASS_5axis.urdf")

# wood2 = armChain_down.supervisor.getFromSolid('cube')
# fake_arm = armChain_down.supervisor.getFromDef('fake_arm')
# target_wood = wood2.getField("translation")
# fakearm_move = fake_arm.getField("translation")


def closeRobot(c):
    c.open()
    c.write_multiple_registers((0x1001), intL2DRA(4))
    c.close()
    print("stop DRA!")


def connectRobot(SERVER_HOST="192.168.1.1", SERVER_PORT=502):
    # SERVER_HOST = "169.254.194.1"
    # SERVER_HOST = "localhost"
    # SERVER_PORT = 502

    c = ModbusClient()

    # uncomment this line to see debug message
    # c.debug(True)

    # define modbus server host, port
    c.host(SERVER_HOST)
    c.port(SERVER_PORT)
    c.unit_id(2)
    c.open()
    # if c.is_open():
    #     # read 10 registers at address 0, store result in regs list
    #     print("connect success")
    #     reg = c.read_holding_registers(0x1000, 1)
    #     # print("reg", reg)
    #     if (reg[0] == 1):
    #         print("DRA is runing now")
    #     else:
    #         print("DRA is closed, please run DRA")
    # else:
    #     print("failed to connect")
    return c

def DRA2intL(n):
    a, b = int(n[0]), int(n[1])
    # print(a, b)
    t = (b << 16) + a
    return t if t < (2**31) else (t - 2**32)

def intL2DRA(i):
    if(i<0):
        return intL2DRA( i + (2**32))
    else:
        return [int(i % (2**16)), int(i // (2**16))] # a, b = i % (2**16), i // (2**16) #(i >> 16)

def OutputCSV(SAMPLE_List, num = 0):
    Result = savePath + '/RLTest_'+str(num)+'.csv'
    df_SAMPLE = pd.DataFrame.from_dict(SAMPLE_List).rename(columns = {0 : "rr", 1 : "x", 2 : "y", 3 : "z", 4 :
        "x_final", 5 : "y_final", 6 : "z_final", 7 : "origin", 8 : "final", 9 : "x_move", 10 : "y_move", 11 : "z_move",
         12 : "done"})
    df_SAMPLE.to_csv(Result, index = False)


def randomPick(Noise = False):
    global distance

    rr=random.randint(10,11)
    rr = 11

    noise_x, noise_y, noise_z, noise_r = 0.0002, 0.0002, 0.0002, 1
    if(rr%2==0):
        noise_x *= -1
        noise_y *= -1
        noise_z *= -1

    img1,img2,GT_POS,GT_ROT=utils.loadData(rr)
    # img1 = PIL.Image.open('../main/dataset/Dataset_can_256/'+img1)
    #
    # img1=np.array(PIL.ImageOps.grayscale(img1))
    #
    # pil_to_tensor1 = transforms.ToTensor()(img1).unsqueeze_(0)

    RL_POS_long = utils.initial_state(GT_POS, GT_ROT)

    if Noise:
        RL_POS_short_noise = [round(noise_x + RL_POS_long[0], 4), round(noise_y + RL_POS_long[1], 4),
                         round(noise_z + RL_POS_long[2], 4), round(noise_r * RL_POS_long[3], 4)]
        RL_ROT_short_noise = RL_POS_long[3]
        distance = round(abs(GT_POS[0] - RL_POS_short_noise[0]) * 1000 + abs(GT_POS[2] - RL_POS_short_noise[2]) * 1000, 4)
        return rr, GT_POS, RL_POS_short_noise

    else:
        RL_POS_short = [RL_POS_long[0], RL_POS_long[1], RL_POS_long[2]+0.05, RL_POS_long[3]]
        RL_ROT_short = RL_POS_long[3]
        distance = abs(GT_POS[0] - RL_POS_short[0]) * 1000 + abs(GT_POS[1] - RL_POS_short[1]) * 1000 + abs(GT_POS[2] - RL_POS_short[2]) * 1000
        return rr, GT_POS, RL_POS_short

def respawnRobot(wood,GT_POS,RL_POS_short):
    wood.setSFVec3f([GT_POS[0],GT_POS[1],GT_POS[2]])
    # print(RL_POS_short)
    err = armChain_down.movP(RL_POS_short)
    # print('aaa',err)
    # err = armChain_down.position_move(RL_POS_short[0], RL_POS_short[1], RL_POS_short[2])
    # print("respawn",err)
    # fakearm_move.setSFVec3f([RL_POS_short[0], RL_POS_short[1], RL_POS_short[2]])

def reset(targetWood):

    rr, GT_POS, RL_POS_short = randomPick(Noise = False)
    # print(rr, GT_POS, RL_POS_short)
    respawnRobot(targetWood, GT_POS, RL_POS_short)

    armChain_down.supervisor.simulationResetPhysics()

    return rr, [RL_POS_short[0], RL_POS_short[1], RL_POS_short[2]]

def get_arm_pos():
    Now_POS = armChain_down.getCurP()
    # Now_POS = armChain_down.find_endposition()
    return [Now_POS[0], Now_POS[1], Now_POS[2]]

def step(selectedAction):
    global init_dis
    Now_POS = get_arm_pos()
    # Now_POS = fakearm_move.getSFVec3f()
    if selectedAction == 0:
         Now_POS[0] -= 0.0001
    elif selectedAction == 1:
         Now_POS[0] += 0.0001
    elif selectedAction == 2:
         Now_POS[1] -= 0.0001
    elif selectedAction == 3:
         Now_POS[1] += 0.0001
    elif selectedAction == 4:
         Now_POS[2] -= 0.0001
    elif selectedAction == 5:
         Now_POS[2] += 0.0001
    elif selectedAction == 6:
        Target_POS = target_wood.getSFVec3f()     
        init_dis = sum = abs(Target_POS[0] - Now_POS[0]) * 1000 + abs(Target_POS[1] - Now_POS[1]) * 1000 + abs(Target_POS[2] - Now_POS[2]) * 1000
        return Now_POS, 0, 0, init_dis

    armChain_down.movP(Now_POS)
    # err = armChain_down.position_move(Now_POS[0], Now_POS[1], Now_POS[2])
    # fakearm_move.setSFVec3f([Now_POS[0], Now_POS[1], Now_POS[2]])
    Target_POS = target_wood.getSFVec3f()
    sum = abs(Target_POS[0] - Now_POS[0]) * 1000 + abs(Target_POS[1] - Now_POS[1]) * 1000 + abs(
        Target_POS[2] - Now_POS[2]) * 1000
    
    reward = (sum / init_dis)*-10
    # reward = avg_dis * -1.25 + 100
    
    done = False
    # reward = -sum
    if sum <= 0.2:
        reward = 10
        done = True

    return Now_POS, reward, done, sum

def solved(done, i_episode):
    solve_index = i_episode % solve_size
    
    if done:
        solve[solve_index] = 1
    else:
        solve[solve_index] = 0
    
    if sum(solve) > 21:
    #if sum(solve) > 16:
        agent.save_model(args.env_name, "Final" + str(int(i_episode/1000)) + "_" + str(sum(solve)))
        return True
    elif sum(solve) > 18 :
        agent.save_model(args.env_name, str(int(i_episode/1000)) + "_" + str(sum(solve)))
    
    return False

def angleCompensation(compensateAngle, setAngle, currAngle):
    for i in range(6):
        compensateAngle[i] = compensateAngle[i] + (setAngle[i]- currAngle[i])*0.1
    return compensateAngle


# Parameter

args.automatic_entropy_tuning = True
observation_space = 3
action_space = 6

# Agent
# agent = SAC(observation_space, action_space, args)
# agent.load_model('models/sac_actor_9pos_rewardchange_initchange_noend_Final631','models/sac_critic_9pos_rewardchange_initchange_noend_Final631')

# Tesnorboar
# savePath = os.getcwd() + '/runs/'
# writer = SummaryWriter(savePath)

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0
max_episode_steps = 200
solve_size = 20
solve = [0]*solve_size
pos = []

cover = armChain_down.supervisor.getFromDef('endpoint')
coverField = cover.getField("translation")
p = coverField.getSFVec3f()

endpoint = armChain_down.supervisor.getFromDef('endpoint')
endpointfield = endpoint.getField("translation")

recursive = range(-180, 180, 5)
setAngle = [0.000000, 30.000000, 30.000000, 0.000000, 0.000000, 0.000000]

# c = connectRobot()
armChain_down.supervisor.simulationSetMode(2)

while armChain_down.supervisor.step(armChain_down.timestep) != 1:
    p = coverField.getSFVec3f()
    # print("p", p)
    # armChain_down.movP(p)
    # HCurJ = armChain_down.getCurJ()
    # HCurP = armChain_down.getCurP()
    # H = [HCurP[0] * 1000, HCurP[1] * 1000, HCurP[2] * 1000, -180.0, -90.0 + HCurJ[1] + HCurJ[2] + HCurJ[4], HCurJ[0]]
    # print("Position", H)
    position = []
    count = 0
    # for i in np.arange(-0.3, 0.4, 0.02):
    #     print(i)
    #     for j in np.arange(0.1, 0.8, 0.02):
    armChain_down.movP([0.379, 0.4, 0.8])
    CurJ = armChain_down.getCurJ()
    CurP = armChain_down.getCurP()
    P = [CurP[0] * 1000, CurP[1] * 1000, CurP[2] * 1000, -180.0, -90.0 + CurJ[1] + CurJ[2] + CurJ[4], CurJ[0]]
    position.append(P)
    # armChain_down.supervisor.exportImage("/Users/peter3354152/Desktop/Sim_IMG/" + str(count) + ".png", 100)
    im = ImageGrab.grab()
    im.save("/Users/peter3354152/Desktop/Sim_IMG/" + str(count) + ".png", quality = 100)
    count += 1
    # print(armChain_down.movJ([30.0,30.0,30.0,90.0,0.0,0.0]))
    # print("angle", CurJ)
    print("Position", P)
    break
    # df = pd.DataFrame(position)
    # df.to_excel("/Users/peter3354152/Desktop/position.xlsx")



    # c = connectRobot()
    # for i in range(6):
        # print(DRA2intL(intL2DRA(angle[i]*10**6))/10**6)
        # print(intL2DRA(angle[i]*10**6))
        # c.write_multiple_registers(0x1010+2*i, intL2DRA(CurJ[i]*10**6))
        # c.write_multiple_registers(0x1010 + 2 * i, intL2DRA(P[i] * 10 ** 6))

        # print(c.read_holding_registers(0x1000+2*i,2))
        # jDegree[i] = DRA2intL(c.read_holding_registers(0x1010+2*i,2))/10**6
        # print(c.read_holding_registers(0x1000+i+1,1))

    # err0 = armChain_down.arm_joint_move(jointAngles)
    # err1 = armChain_down.arm_joint_move(jDegree)
    # print("WBT", armChain_down.currAngles())

    # err0 = 0
    # print("err0", err0,"err1", err1)


'''
for i_episode in range(5000):
    episode_reward = 0
    episode_steps = 0
    done = False
    rr, state = reset(target_wood)
    init_state, _, _, init_dis = step(6)
    # print(done)

    while not done and episode_steps < max_episode_steps and armChain_down.supervisor.step(armChain_down.timestep) != 1:
        action = agent.select_action(state)  # Sample action from policy
        # print(len(memory), action)
        # print(action)
        next_state, reward, done, dis= step(np.argmax(action)) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        
        # done = False
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == max_episode_steps else float(not done)

        # memory.push(state, action, reward, next_state, mask) # Append transition to memory
        # memory.push(state, action, reward, next_state, mask)

        state = next_state
        
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    pos.append(
        [rr, init_state[0], init_state[1], init_state[2], state[0], state[1], state[2], init_dis,
         dis, round(abs(init_state[0]-state[0])*10000), round(abs(init_state[1]-state[1])*10000), 
         round(abs(init_state[2]-state[2])*10000), bool(done)])

OutputCSV(pos, int(i_episode / 100 - 1))
'''

# Environment Variables:
# PYTHONUNBUFFERED=1;PATH=C:\Users\user\AppData\Local\Programs\Webots\lib\controller\\;C:\Users\user\AppData\Local\Programs\Webots\msys64\mingw64\bin\\;C:\Users\user\AppData\Local\Programs\Webots\msys64\mingw64\bin\cpp

