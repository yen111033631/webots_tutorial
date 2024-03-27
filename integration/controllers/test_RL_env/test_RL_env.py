"""test_RL_env controller."""

"""train_RL_env controller."""

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
from controller import Supervisor
from itertools import count
import cv2
from DQN import DQN
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import getpass
from datetime import datetime
import socket

def unzip_list(key_list):
    result = []
    for sublist in key_list:
        result.extend(sublist)
    
    return result

def save_reward_as_csv(List, log_dir=""):
    df = pd.DataFrame(List)
    df = df.set_axis(['duration', 'reward'], axis=1)
    df.to_csv(f"{log_dir}/monitor.csv")

def set_log_dir(model_name="DQN", env_id="ball_discrete"):
    def create_text_file(log_dir):
        # 獲取當前機器名稱
        hostname = socket.gethostname()        
        # 獲取當前使用者名稱
        current_user = getpass.getuser()
        # 獲取當前執行腳本的路徑
        current_path = os.path.dirname(os.path.abspath(__file__))
        # 獲取當前時間
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 指定要創建的文本文件路徑
        txt_file_path = os.path.join(log_dir, "log.txt")

        # 建立文本文件並寫入內容
        with open(txt_file_path, "w") as txt_file:
            txt_file.write(f"current machine: {hostname}\n")
            txt_file.write(f"current path: {current_path}\n")
            txt_file.write(f"current time: {current_time}\n")

        print(f"{txt_file_path} create file")

    model_path = f"./logs/models/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    env_listdir = os.listdir(model_path)

    num = 0
    for env_dir in env_listdir:
        if env_id in env_dir:
            now_num = int(env_dir[-3:]) 
            num = now_num if now_num > num else num
    
    # 獲取當前機器名稱
    hostname = socket.gethostname()  

    log_dir = f"{model_path}/{env_id}_{hostname}_{(num + 1):03d}"
    os.makedirs(log_dir, exist_ok=True)

    create_text_file(log_dir)
    return log_dir

def add_end_time(log_dir, num_episodes):
    # 指定要創建的文本文件路徑
    txt_file_path = os.path.join(log_dir, "log.txt")
    
    # 獲取當前時間
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 建立文本文件並寫入內容
    with open(txt_file_path, "a") as txt_file:
        txt_file.write(f"end time: {current_time}\n")
        txt_file.write(f"num_episodes: {num_episodes}\n")
    
class env_ball:
    def __init__(self, seed):
        self.seed = seed
        self.observation = observation()
        self.info = "ball" 
        self.ball = Object("ball")
        self.max_num_step = 100
        self.constant_displacement = 0.05
        self.env_bounds = [[-0.27, 0.48],
                          [-0.74, 0.74],
                          [0.04, 1]
                          ]
        
        self.cam = Camera("camera")

        # TODO observation space
        # TODO action space

    def reset(self, seed):
        # init env
        random.seed(self.seed + seed) 
        self.ball.random_choose_one_position()
        self.ball.setP(self.ball.new_position)

        # init arm
        target_position = [0.3, 0, 0.55]
        armChain_down.movP(target_position)

        # set observation and state
        self.get_observation()
        self.get_state()

        # reset step 
        self.num_step = 0

        return self.state, self.info
    
    def get_observation(self):
        # set each observation
        self.observation.arm = {"joint_angle" : np.asarray(armChain_down.getCurJ()),
                                "end_point" : np.asarray(armChain_down.getCurP())}

        self.observation.interact_object = self.ball.position
        
        # self.frame = self.cam.get_frame()

        # self.cam.show_frame() # TODO add cam infomation

        return None
    
    def get_state(self):
        # set state 
        ## joint angle: 6 (not used now)
        ## end point: 3
        ## ball position: 3
        ## relationship vector: 3
        ## relationship distance: 1
        ## total: 16
        
        
        # FIXME now is only arm value and ball value
        # state_list = list(self.observation.arm.values())
        state_list = list([self.observation.arm["end_point"]])
        state_list.append(self.observation.interact_object)

        # relationship between arm end point and ball 
        vector = np.asarray(self.observation.interact_object)-np.asarray(self.observation.arm["end_point"])
        self.calculate_distance()
        relationship = [*(vector.tolist()), self.distance]
        state_list.append(relationship)
        

        self.state = unzip_list(state_list)

        return self.state

    def step(self, action):

        # execute action
        self.execute_action(action)


        # get next state
        self.get_observation()
        self.get_state()

        # calculate distance
        self.calculate_distance()

        # check if task is terminated or truncated
        terminated = False
        truncated = False
        
        # terminated
        if self.distance <= 0.1 or not(self.is_good):
            terminated = True

        # truncated
        if self.num_step >= self.max_num_step:
            truncated = True

        # calculate reward   
        if self.is_good:
            self.reward = -(self.distance * 10)  # FIXME reward design
        else:
            self.reward = -500

        return self.state, self.reward, terminated, truncated, self.info
    
    def execute_action(self, action):
        """
        first one: discrete
        choose +-(x, y, z) direction with constant displacement
        """        

        # get now observation
        self.get_observation()

        # turn action into displacement
        
        assert 0 <= action <= 5, "action index not between 0-5"
        displacement = 0
        if action == 0:
            displacement = np.asarray([self.constant_displacement, 0, 0])
        elif action == 1:
            displacement = np.asarray([-self.constant_displacement, 0, 0])
        elif action == 2:
            displacement = np.asarray([0, self.constant_displacement, 0])
        elif action == 3:
            displacement = np.asarray([0, -self.constant_displacement, 0])
        elif action == 4:
            displacement = np.asarray([0, 0, self.constant_displacement])
        elif action == 5:
            displacement = np.asarray([0, 0, -self.constant_displacement])

        # add action into next position
        now_arm_position = np.asarray(self.observation.arm["end_point"])
        next_position = now_arm_position + displacement
        # print("next_position", next_position)

        # check position in boundary (both arm and env)
        ik = armChain_down.get_IK_angle(next_position)
        
        is_ik_good = armChain_down.check_ik_in_boundarys(ik)
        is_position_good = self.check_position_in_boundary(next_position)
        self.is_good = is_ik_good and is_position_good
        
        # execute action
        if self.is_good:
            # print("good action")
            armChain_down.movP(next_position)
        else:
            print("-" * 5)
            print("bad action")
            print("next_position", next_position)
            print("joint boundary (is_ik_good)", is_ik_good)
            print("env boundary (is_position_good)", is_position_good)
            print("-" * 5)
            pass

        self.num_step += 1
        return None
    
    def calculate_distance(self):
        a_position = self.observation.arm["end_point"]
        b_position = self.observation.interact_object

        assert len(a_position) == len(b_position), "Error: length not the same, cant calculate"
        
        tem = 0
        for i in range(len(a_position)):
            tem += (a_position[i] - b_position[i]) ** 2
        
        self.distance = tem ** 0.5

        return self.distance
    
    def check_position_in_boundary(self, position):
        """
        bounds and position should be same type (angle or radian)
        """
        is_ok = True
        assert len(self.env_bounds) == len(position), "len(self.env_bounds) should == len(position)"
        for i, bound in enumerate(self.env_bounds):
            if not(bound[0] < position[i] < bound[1]):
                is_ok = False
        
        if is_ok:
            joint3_position = armChain_down.getCurP_Joint(3)
            if joint3_position[0] <= self.env_bounds[0][0] + 0.1:
                is_ok = False

        return is_ok

class observation:
    def __init__(self) -> None:
        self.arm = None
        self.cam = None
        self.interact_object = None

class Camera:
    def __init__(self, DEF_name) -> None:
        self.name = DEF_name
        self.cam_node = armChain_down.supervisor.getDevice(DEF_name)
        self.cam_node.enable(armChain_down.timestep)
        self.frame = None
    
    def get_frame(self):
        self.frame = np.frombuffer(self.cam_node.getImage(), dtype=np.uint8).reshape((self.cam_node.getHeight(), self.cam_node.getWidth(), 4))
        return self.frame
    
    def show_frame(self):
        cv2.imshow("cam_node image", self.get_frame())
        cv2.waitKey(1) 
        return None

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

    def get_position(self):
        self.position = self.node.getPosition()
        return self.position
    
    def random_choose_one_position(self):
            random_index = random.randint(0, len(self.position_candididate) - 1)
            self.new_position = self.position_candididate[random_index]

    def setP(self, position):
            self.translation_field.setSFVec3f(position)
            self.position = position
            # self.get_position()

###################################################################
def RL_process():
    
    log_dir = set_log_dir(model_name="DQN",
                          env_id="ball_discrete")
    
    n_observations = 10
    n_actions = 6
    
    dqn = DQN(env, 
              n_actions = n_actions,
              n_observations = n_observations)
    
    if dqn.is_cuda:
        num_episodes = int(1e5)
    else:
        num_episodes = 50
    
    num_episodes = 10
    all_total_rewards = []
    for i_episode in range(1, 1+num_episodes):
        print("===" * 10)
        state, info = env.reset(seed = SEED + i_episode)
        state = dqn.process_state(state)
        
        total_reward = 0
        
        for t in count():
            # print(i_episode, env.num_step, end=" ")
            
            ## RL
            # action = random.randint(0, 5)
            action = dqn.select_action(state)
            # # print("action", action)
            # print("action", action.item())
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_cpu = reward
            reward = dqn.process_reward(reward)
            done = terminated or truncated

            # next_state, reward, terminated, truncated, info = env.step(action)
            # print("next_state", observation)
            # print("reward", reward_cpu)
            # print("distance", env.distance)
            # print("terminated", terminated)
            # print("truncated", truncated)
            # print("--" * 10)
            
            if terminated:
                next_state = None
            else:
                next_state = dqn.process_state(observation)
                
            total_reward += reward_cpu
            
            # Store the transition in memory
            dqn.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            dqn.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            dqn.update_weights()

            if done:
                # dqn.episode_durations.append(total_reward)
                all_total_rewards.append([t, total_reward])
                # dqn.plot_durations()
                print(f"episode: {i_episode} \tdurations: {t}\t\trewards: {total_reward}")
                break
    
        if i_episode % 10 == 0:
            dqn.save(log_dir)
            save_reward_as_csv(all_total_rewards, log_dir)
            print("model saved!")
            
    
    add_end_time(log_dir, num_episodes)
    print('Complete')
    # dqn.plot_durations(show_result=True)
    # plt.ioff()
    # plt.show()

def RL_process_random():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
        
    num_episodes = 1000
    
    
    all_total_rewards = []
    for i_episode in range(1, 1+num_episodes):
        print("===" * 10)
        state, info = env.reset(seed=525+i_episode)
        
        for t in count():
            print(i_episode, env.num_step, end=" ")
            
            ## RL
            action = random.randint(0, 5)

            observation, reward, terminated, truncated, _ = env.step(action)
            # observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_cpu = reward
            # reward = dqn.process_reward(reward)
            done = terminated or truncated

            print("reward", reward)
            
            if done:
                print("env done")
                break

    
    print('Complete')
    # dqn.plot_durations(show_result=True)
    # plt.ioff()
    # plt.show()



#############################
def test():
    ob, info = env.reset(seed=525)
    # env.get_observation()
    # print(env.observation)
    # env.get_observation()
    # print(env.observation)
    target_position = [0.1946429, -0.00136273, 0.75214104]
    armChain_down.movP(target_position)
    print("joint 3", armChain_down.getCurP_Joint(3))
###################################################################

if __name__ == "__main__":
    
    SEED = 525

    ### init arm
    armChain_down = DRV90_Robot("DRV90.urdf", end_down=False)

    ### init env
    env = env_ball(seed=SEED)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Main loop:
    # i = 0
    while armChain_down.supervisor.step(armChain_down.timestep) != -1:
        RL_process()        
        break
        # RL_process_random()        
        # break

        # test()
        # break
    
        # i += 1







