import argparse
import datetime

import cv2
# import gym
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
# import torch
from robot_controller import DRV90_Robot
# from pyModbusTCP.client import ModbusClient
from DDPG_i import DDPG
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
import torch
from itertools import count
from Train_Chart import *
from torchvision.transforms import transforms
import PIL.Image as Image




# armChain_down = DRV90_Robot("drv90l.urdf", end_down=False)
armChain_down = DRV90_Robot("DRV90.urdf", end_down=False)
# armChain_down = DRV90_Robot("fixHeight.urdf", end_down=False)
# armChain_down = DRV90_Robot("fixJ1_TEST.urdf", end_down=False)
# armChain_down = DRV90_Robot("DRV90ASS_456fix_axis.urdf", end_down=False)
# armChain_down = DRV90_Robot("DRV90ASS_fix_axis.urdf", end_down=False)
# armChain_down = DRV90_Robot("DRV90ASS_5axis.urdf")

wood2 = armChain_down.supervisor.getFromDef('cube')
# fake_arm = armChain_down.supervisor.getFromDef('fake_arm')
target_wood = wood2.getField("translation")
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
    return t if t < (2 ** 31) else (t - 2 ** 32)


def intL2DRA(i):
    if (i < 0):
        return intL2DRA(i + (2 ** 32))
    else:
        return [int(i % (2 ** 16)), int(i // (2 ** 16))]  # a, b = i % (2**16), i // (2**16) #(i >> 16)


def OutputCSV(SAMPLE_List, num=0):
    Result = savePath + '/RLTest_' + str(num) + '.csv'
    df_SAMPLE = pd.DataFrame.from_dict(SAMPLE_List).rename(columns={0: "rr", 1: "x", 2: "y", 3: "z", 4:
        "x_final", 5: "y_final", 6: "z_final", 7: "origin", 8: "final", 9: "x_move", 10: "y_move", 11: "z_move",
                                                                    12: "done"})
    df_SAMPLE.to_csv(Result, index=False)


def randomPick(Noise=False):
    global distance

    rr = random.randint(10, 11)
    rr = 11

    noise_x, noise_y, noise_z, noise_r = 0.0002, 0.0002, 0.0002, 1
    if (rr % 2 == 0):
        noise_x *= -1
        noise_y *= -1
        noise_z *= -1

    img1, img2, GT_POS, GT_ROT = utils.loadData(rr)
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
        RL_POS_short = [RL_POS_long[0], RL_POS_long[1], RL_POS_long[2] + 0.05, RL_POS_long[3]]
        RL_ROT_short = RL_POS_long[3]
        distance = abs(GT_POS[0] - RL_POS_short[0]) * 1000 + abs(GT_POS[1] - RL_POS_short[1]) * 1000 + abs(GT_POS[2] - RL_POS_short[2]) * 1000
        return rr, GT_POS, RL_POS_short


def respawnRobot(wood, GT_POS, RL_POS_short):
    wood.setSFVec3f([GT_POS[0], GT_POS[1], GT_POS[2]])
    # print(RL_POS_short)
    err = armChain_down.movP(RL_POS_short)
    # print('aaa',err)
    # err = armChain_down.position_move(RL_POS_short[0], RL_POS_short[1], RL_POS_short[2])
    # print("respawn",err)
    # fakearm_move.setSFVec3f([RL_POS_short[0], RL_POS_short[1], RL_POS_short[2]])



def reset(targetWood):
    rr, GT_POS, RL_POS_short = randomPick(Noise=False)
    # print(rr, GT_POS, RL_POS_short)
    respawnRobot(targetWood, GT_POS, RL_POS_short)

    armChain_down.supervisor.simulationResetPhysics()

    return rr, [RL_POS_short[0], RL_POS_short[1], RL_POS_short[2]]


def get_arm_pos():
    Now_POSS = armChain_down.getCurP()
    # Now_POS = armChain_down.find_endposition()
    return Now_POSS


def step(selectedAction):
    global init_dis
    Now_POS = get_arm_pos()
    # Now_POS = fakearm_move.getSFVec3f()
    if selectedAction == 0:
        Now_POS[0] -= 0.02
    elif selectedAction == 1:
        Now_POS[0] += 0.02
    elif selectedAction == 2:
        Now_POS[1] -= 0.02
    elif selectedAction == 3:
        Now_POS[1] += 0.02
    elif selectedAction == 4:
        Now_POS[2] -= 0.02
    elif selectedAction == 5:
        Now_POS[2] += 0.02
    elif selectedAction == 6:
        Target_POS = target_wood.getSFVec3f()
        init_dis = abs(Target_POS[0] - Now_POS[0]) * 1000 + abs(Target_POS[1] - Now_POS[1]) * 1000 + abs(Target_POS[2] - Now_POS[2]) * 1000
        return [Now_POS[0],Now_POS[1],Now_POS[2],Target_POS[0],Target_POS[1],Target_POS[2]], 0, 0, init_dis
    try:
        a = armChain_down.movP(Now_POS)
        if a == 1:
            print("stuck!!")
            print(a)
            done = True
            reward = -100
            return [Now_POS[0],Now_POS[1],Now_POS[2],Target_POS[0],Target_POS[1],Target_POS[2]], reward, done, sum
    except:
        print("Raise error!!!")
        done = True
        reward = -100
        Target_POS = target_wood.getSFVec3f()
        sum = abs(Target_POS[0] - Now_POS[0]) * 1000 + abs(Target_POS[1] - Now_POS[1]) * 1000 + abs(Target_POS[2] - Now_POS[2]) * 1000
        return  [Now_POS[0],Now_POS[1],Now_POS[2],Target_POS[0],Target_POS[1],Target_POS[2]], reward, done, sum

    Target_POS = target_wood.getSFVec3f()
    sum = abs(Target_POS[0] - Now_POS[0]) * 1000 + abs(Target_POS[1] - Now_POS[1]) * 1000 + abs(Target_POS[2] - Now_POS[2]) * 1000

    reward = (sum / init_dis) * -10
    # reward = 1 / (np.sqrt((Target_POS[0] - Now_POS[0]) ** 2 + (Target_POS[1] - Now_POS[1]) ** 2 +(Target_POS[2] - Now_POS[2]) ** 2))
    # reward = avg_dis * -1.25 + 100

    done = False
    # reward = -sum
    if sum <= 0.2:
        reward = 100
        done = True
    if Now_POS[2] < 0:
        reward = -100

    return [Now_POS[0],Now_POS[1],Now_POS[2],Target_POS[0],Target_POS[1],Target_POS[2]], reward, done, sum

def step_Continious(act):
    test = False
    global init_dis
    Now_POS = get_arm_pos()
    Target_POS = target_wood.getSFVec3f()
    before_move = np.sqrt((Target_POS[0] - Now_POS[0])**2 + (Target_POS[1] - Now_POS[1])**2 + (Target_POS[2] - Now_POS[2])**2)
    if act[0] == 6:
        init_dis = abs(Target_POS[0] - Now_POS[0]) * 1000 + abs(Target_POS[1] - Now_POS[1]) * 1000 + abs(
            Target_POS[2] - Now_POS[2]) * 1000
        armChain_down.supervisor.exportImage("/Users/peter3354152/Desktop/image_test/image//intial_state.png", 100)
        # a = cv2.imread("/Users/peter3354152/Desktop/image_test/image/intial_state.png")
        a = Image.open("/Users/peter3354152/Desktop/image_test/image/intial_state.png", mode="r")
        a = a.convert('RGB')
        # aa = cv2.resize(a, (640,480), interpolation = cv2.INTER_CUBIC)
        transform = transforms.Compose([transforms.PILToTensor(),transforms.Resize((416,416))])
        aa = transform(a)
        print(aa.shape)
        # return [Now_POS[0], Now_POS[1], Now_POS[2], Target_POS[0], Target_POS[1], Target_POS[2]], 0, 0, init_dis
        return aa, 0, 0, init_dis
    else:
    # Now_POS = fakearm_move.getSFVec3f()
        Now_POS[0] += abs(act[0])

        Now_POS[1] += act[1]

        Now_POS[2] += act[2]


    a = armChain_down.movP(Now_POS)
    after_move = np.sqrt((Target_POS[0] - Now_POS[0]) ** 2 + (Target_POS[1] - Now_POS[1]) ** 2 + (Target_POS[2] - Now_POS[2]) ** 2)
    # print("Now position",Now_POS)
        # print(a)
        # reward = -100
        # return [Now_POS[0], Now_POS[1], Now_POS[2], Target_POS[0], Target_POS[1],
        #         Target_POS[2]], reward, done, sum

    armChain_down.supervisor.exportImage("/Users/peter3354152/Desktop/image_test/image/state.png",100)
    s = Image.open("/Users/peter3354152/Desktop/image_test/image/state.png")
    s = s.convert('RGB')
    # aa = cv2.resize(a, (640,480), interpolation = cv2.INTER_CUBIC)
    transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((416, 416))])
    s = transform(s)

    sum = abs(Target_POS[0] - Now_POS[0]) * 1000 + abs(Target_POS[1] - Now_POS[1]) * 1000 + abs(Target_POS[2] - Now_POS[2]) * 1000

    # reward = (sum / init_dis) * -10
    # reward = 1 / (np.sqrt((Target_POS[0] - Now_POS[0]) ** 2 + (Target_POS[1] - Now_POS[1]) ** 2 +(Target_POS[2] - Now_POS[2]) ** 2)) ** 2
    reward = 1 / (np.sqrt((Target_POS[0] - Now_POS[0]) ** 2 + (Target_POS[1] - Now_POS[1]) ** 2 + (Target_POS[2] - Now_POS[2]) ** 2))**2
    # reward = avg_dis * -1.25 + 100
    # statee = [Now_POS[0], Now_POS[1], Now_POS[2], Target_POS[0], Target_POS[1], Target_POS[2]]
    statee = s
    done = False
    if a == 1:
        print("stuck!!")
        done = True
        test = False
        return statee, reward, done, after_move, test
    elif Now_POS[2] < 0 or Now_POS[0] < 0.35 or Now_POS[0] > 0.6 or Now_POS[2] > 1.0:
        reward = -200
        print("Crush!!!")
        done = True
        test = False
        return statee, reward, done, after_move, test
        # return s, reward, done, sum
    elif abs(Target_POS[0] - Now_POS[0]) < 0.1 and abs(Target_POS[1] - Now_POS[1]) < 0.1 and abs(Target_POS[2] - Now_POS[2]) < 0.1:
        reward = 2000
        done = True
        print("done!!!!!!!!!!!!!!!!!!")
        return statee, reward, done, after_move, test
        # return s, reward, done, sum
    if (after_move - before_move) < 0:
        test = True
    # print((test))

    return statee, reward, done, after_move, test
    # return s, reward, done, sum


def solved(done, i_episode):
    solve_index = i_episode % solve_size

    if done:
        solve[solve_index] = 1
    else:
        solve[solve_index] = 0

    if sum(solve) > 21:
        # if sum(solve) > 16:
        agent.save_model(args.env_name, "Final" + str(int(i_episode / 1000)) + "_" + str(sum(solve)))
        return True
    elif sum(solve) > 18:
        agent.save_model(args.env_name, str(int(i_episode / 1000)) + "_" + str(sum(solve)))

    return False


def angleCompensation(compensateAngle, setAngle, currAngle):
    for i in range(6):
        compensateAngle[i] = compensateAngle[i] + (setAngle[i] - currAngle[i]) * 0.1
    return compensateAngle

def main():
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    success_test = []
    real_test = []
    if mode == 'test':
        agent.load()
        for i in range(test_iteration):
            target_wood.setSFVec3f([0.45, 0.381, 0.03])
            armChain_down.supervisor.simulationResetPhysics()
            armChain_down.movP([0.4, 0, 0.46])
            init_state, _, _, init_dis = step_Continious([6, 6, 6])
            state = init_state
            print("Test start")
            for t in range(30):
                action = agent.select_action(state)
                next_state, reward, done, dis, test = step_Continious(action)
                if test == True:
                    success_test.append(1)
                else:
                    success_test.append(0)
                ep_r += reward
                state = [state[0]*1000,state[1]*1000,state[2]*1000,-180,0,0]
                print(state)
                real_test.append((state))
                rt = pd.DataFrame(real_test)
                rt.to_csv("/home/neaf-2080/code/Pan/integration/controllers/bad_real_test.csv")
                if done:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
        success = 0
        for i in success_test:
            if i == 1:
                success += 1
        print("---------success_rate:",success/len(success_test))
        print("success:", success)
        print("times:", len(success_test))
    elif mode == 'train':
        if load: agent.load()
        total_step = 0
        reward_plot = []
        Done_plot = []
        loss_chart = []
        critic_chart = []
        dis_chart = []
        for i in range(max_episode):
            total_reward = 0
            stepp =0
            dis = 0
            done = False
            # state = env.reset()
            # rr, state = reset(target_wood)
            target_wood.setSFVec3f([0.45,0.381,0.03])
            armChain_down.supervisor.simulationResetPhysics()
            armChain_down.movP([0.4, 0, 0.46])
            init_state, _, _, init_dis = step_Continious([6,6,6])
            state = init_state
            print("===================episode:", i+1, "===================")
            for t in range(30):
                # print("Current State: ",state)
                action = agent.select_action(state)
                print("action:",action)
                # action = (action + np.random.normal(0, 0.1, size=3)).clip(-0.04, 0.04)
                # print("action", action)
                #####add noise######
                next_state, reward, done, disss, test = step_Continious(action)
                # print("next_state:",next_state)
                dis = float(dis)
                # reward -= t*20
                agent.replay_buffer.push((state, next_state, action, reward, np.float32(done)))
                state = next_state
                if done:
                    total_reward += reward
                    Done_plot.append(1)
                    print("done")
                    break
                stepp += 1
                total_reward += reward
            dis += disss
            reward_plot.append([i,total_reward])
            dis_chart.append([i,dis])
            # print("dis:",dis_chart)
            # print("rew:",reward_plot)
            print("final distance:", dis)
            a, c = agent.update()
            loss_chart.append([i, a.item()])
            critic_chart.append([i,c.item()])

            path = "/Users/peter3354152/Desktop/image_test/"
            if i % 100 == 0:
                df = pd.DataFrame(reward_plot)
                dfl = pd.DataFrame(loss_chart)
                dfc = pd.DataFrame(critic_chart)
                diss = pd.DataFrame(dis_chart)

                df.to_csv(path +"0612_reward/reward_"+ str(i) + ".csv")
                dfl.to_csv(path +"0612_reward/Actor_loss_"+ str(i) + ".csv")
                dfc.to_csv(path +"0612_reward/Critic_loss_"+ str(i) + ".csv")
                diss.to_csv(path +"0612_reward/Distance_"+ str(i) + ".csv")

                plot_line_chart(path +"0612_reward/reward_"+ str(i) + ".csv",
                                path+"0612_reward/reward_"+ str(i) + ".png")
                plot_loss_chart(path +"0612_reward/Actor_loss_"+ str(i) + ".csv",
                                path +"0612_reward/Actor_loss_" + str(i) + ".png","Actor Loss")
                plot_loss_chart(path +"0612_reward/Critic_loss_" + str(i) + ".csv",
                                path +"0612_reward/Critic_loss_" + str(i) + ".png",
                                "Critic Loss")
                plot_dis_chart(path +"0612_reward/Distance_" + str(i) + ".csv",
                                path +"0612_reward/Distance_" + str(i) + ".png")

            total_step += stepp+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))

            # if i % log_interval == 0:
            agent.save(path+"0612_reward/0612_reward_")
    else:
        raise NameError("mode wrong!!!")

# Parameter
mode ='train' # mode = 'train' or 'test'
tau =0.005 # target smoothing coefficient #float
target_update_interval =1 #int
test_iteration =10 #imt
file_name = "0612_reward"
learning_rate =1e-4 #float
gamma =0.99  # discounted factor #float
capacity=1000000 # replay buffer size #int
batch_size=100 # mini batch size #int
seed=False #bool
random_seed=9527 #int


# optional parameters
sample_frequency=2000 #int
log_interval=50 #int
load=False # load model #bool
exploration_noise=0.1 #float
max_episode=100000 # num of games #int
print_log=5 #int
update_iteration=10 #int

device = 'cpu'

# Random Seed
# if seed:
#    env.seed(random_seed)
#    torch.manual_seed(random_seed)
#    np.random.seed(random_seed)

state_dim = (3,416,416)
action_dim = 3
# max_action = float(env.action_space.high[0])
max_action = float(0.04)
min_Val = torch.tensor(1e-7).float().to(device) # min value


if __name__ == '__main__':
    while armChain_down.supervisor.step(armChain_down.timestep) != 1:
        main()
        break