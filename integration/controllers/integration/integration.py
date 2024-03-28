import argparse
import datetime
# import gym
import numpy as np
import itertools
# import torch
from robot_controller import DRV90_Robot
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import random
import utils
import pandas as pd
import os
# import PIL
# from torchvision import transforms
# np.seterr(divide='ignore',invalid='ignore')
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# parser.add_argument('--env-name', default="HalfCheetah-v2",
#                     help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--env_name', default="9pos_sparse_LfD_40",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default= ((1000 * 1000 * 200) + 1) , metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=40, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', default = True, action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

armChain_down = DRV90_Robot("DRV90ASS_5axis.urdf",end_down=True)
wood2 = armChain_down.supervisor.getFromDef('fixDRV90')

# fake_arm = armChain_down.supervisor.getFromDef('fake_arm')
target_wood = wood2.getField("translation")
# fakearm_move = fake_arm.getField("translation")

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
        RL_POS_short = [RL_POS_long[0], RL_POS_long[1], RL_POS_long[2] , RL_POS_long[3]]
        RL_ROT_short = RL_POS_long[3]
        distance = abs(GT_POS[0] - RL_POS_short[0]) * 1000 + abs(GT_POS[1] - RL_POS_short[1]) * 1000 + abs(GT_POS[2] - RL_POS_short[2]) * 1000
        return rr, GT_POS, RL_POS_short

def respawnRobot(wood,GT_POS,RL_POS_short):
    wood.setSFVec3f([GT_POS[0],GT_POS[1],GT_POS[2]])
    # print(RL_POS_short)
    err = armChain_down.position_move(RL_POS_short[0], RL_POS_short[1], RL_POS_short[2])
    # print("respawn",err)
    # fakearm_move.setSFVec3f([RL_POS_short[0], RL_POS_short[1], RL_POS_short[2]])

def reset(targetWood):

    rr, GT_POS, RL_POS_short = randomPick(Noise = False)
    respawnRobot(targetWood, GT_POS, RL_POS_short)
    
    
    
    armChain_down.supervisor.simulationResetPhysics() 

   
    return rr, [RL_POS_short[0], RL_POS_short[1], RL_POS_short[2]]

def get_arm_pos():
    Now_POS =  armChain_down.find_endposition()
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

    err = armChain_down.position_move(Now_POS[0], Now_POS[1], Now_POS[2])
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

# Parameter
args.automatic_entropy_tuning = True
observation_space = 3
action_space = 6

# Agent
# agent = SAC(observation_space, action_space, args)
# agent.load_model('models/sac_actor_9pos_rewardchange_initchange_noend_Final631','models/sac_critic_9pos_rewardchange_initchange_noend_Final631')

#Tesnorboar
# savePath = os.getcwd() + '/runs/'
# writer = SummaryWriter(savePath)

# Memory
# memory = ReplayMemory(args.replay_size)

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

while armChain_down.supervisor.step(armChain_down.timestep) != 1:
    # armChain_down.position_move(0.5,0.0,0.3)
    # armChain_down.position_move(0.4, 0.0, 0.3)
    # armChain_down.position_move(0.2, 0.0, 0.3)
    armChain_down.position_move(0.3, -0.3, 0.49)
    # armChain_down.position_move(0.06, 0.0, -0.6)
    armChain_down.supervisor.exportImage("D:\\OneDrive_NTHU\\OneDrive - NTHU\\NTHU\\NEAF\\research\\webot\\integration\\1.jpg", 100)
    print(armChain_down.find_endposition())
    
    

# for i_episode in range(5000):
#     episode_reward = 0
#     episode_steps = 0
#     done = False
#     rr, state = reset(target_wood)
#     init_state, _, _, init_dis = step(6)
    
#     while not done and  episode_steps < max_episode_steps and armChain_down.supervisor.step(armChain_down.timestep) != 1:
#         action = agent.select_action(state)  # Sample action from policy
#         # print(len(memory), action)
#         # print(action)
#         next_state, reward, done, dis= step(np.argmax(action)) # Step
#         episode_steps += 1
#         total_numsteps += 1
#         episode_reward += reward
        
#         # done = False
#         # Ignore the "done" signal if it comes from hitting the time horizon.
#         # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
#         mask = 1 if episode_steps == max_episode_steps else float(not done)

#         # memory.push(state, action, reward, next_state, mask) # Append transition to memory
#         # memory.push(state, action, reward, next_state, mask)

#         state = next_state
        
#     print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

#     pos.append(
#         [rr, init_state[0], init_state[1], init_state[2], state[0], state[1], state[2], init_dis,
#          dis, round(abs(init_state[0]-state[0])*10000), round(abs(init_state[1]-state[1])*10000), 
#          round(abs(init_state[2]-state[2])*10000), bool(done)])

# OutputCSV(pos, int(i_episode / 100 - 1))
