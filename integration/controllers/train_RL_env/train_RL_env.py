"""train_RL_env controller."""

import os, math, random, socket
from datetime import datetime
from itertools import count
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from DQN import DQN
from DQN_Atari import DQNAgent
from robot_controller_env import DRV90_Robot, env_ball

# ------------------------------------------------------------------
# random seed 
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

setup_seed(525)
# ------------------------------------------------------------------
# save training info 
def save_txt(log_dir, i_frame, i_episode, mean_reward, last_big_mean_reward):
    # 指定要創建的文本文件路徑
    txt_file_path = os.path.join(log_dir, "log.txt")
    # 建立文本文件並寫入內容
    with open(txt_file_path, "a") as txt_file:
        txt_file.write(f"i_frame:\t{i_frame},\ti_episode:\t{i_episode},\tmean reward:\t{round(mean_reward, 2)},\tis_saved_model:\t{mean_reward > last_big_mean_reward}\n")
    
# ------------------------------------------------------------------
# set log file
class Log_file:
    def __init__(self, model_name="DQN", env_id="ball_discrete", num_episodes=0, is_save=True):
        self.model_name = model_name
        self.env_id = env_id
        self.num_episodes = num_episodes
        self.is_save = is_save
        
    def __enter__(self):
        if self.is_save:
            self.log_dir = self.set_log_dir(self.model_name, self.env_id)
        else:
            self.log_dir = None
        return self.log_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_save:
            self.add_end_time(self.log_dir, self.num_episodes)        
        if exc_type != None:
            print("type: ", exc_type)
            print("val: ", exc_val)
            print("tb: ", exc_tb)
    

    def set_log_dir(self, model_name="DQN", env_id="ball_discrete"):
        def create_text_file(log_dir):
            # 獲取當前機器名稱
            hostname = socket.gethostname()        
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
                txt_file.write(f"-----------------------------------\n")
                txt_file.write(f"num_episodes: {num_episodes}\n")
                txt_file.write(f"num_frames: {num_frames}\n")
                txt_file.write(f"-----------------------------------\n")
                txt_file.write(f"DQN hyperparameter\n")
                txt_file.write(f"LR:\t\t\t\t{LR}\n")
                txt_file.write(f"TAU:\t\t\t{TAU}\n")
                txt_file.write(f"MAX_BUFF:\t\t{MAX_BUFF}\n")
                txt_file.write(f"EPS_DECAY:\t\t{EPS_DECAY}\n")                
                txt_file.write(f"BATCH_SIZE:\t\t{BATCH_SIZE}\n")
                txt_file.write(f"-----------------------------------\n")
                txt_file.write(f"env hyperparameter\n")
                txt_file.write(f"reward_slope:\t\t\t{reward_slope}\n")
                txt_file.write(f"big_negative_reward:\t{big_negative_reward}\n")
                txt_file.write(f"-----------------------------------\n")

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
        
        current_time = datetime.now().strftime("%b%d_H%H_M%M_S%S")

        log_dir = f"{model_path}/{current_time}_{env_id}_{hostname}_{(num + 1):03d}"
        os.makedirs(log_dir, exist_ok=True)

        create_text_file(log_dir)
        return log_dir

    def add_end_time(self, log_dir, num_episodes):
        # 指定要創建的文本文件路徑
        txt_file_path = os.path.join(log_dir, "log.txt")
        
        # 獲取當前時間
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 建立文本文件並寫入內容
        with open(txt_file_path, "a") as txt_file:
            txt_file.write(f"end time: {current_time}\n")

###################################################################
def train_RL_process(log_dir, model_name="DQN", env_id="ball_discrete", num_episodes=10):
    def save_reward_as_csv(List, log_dir=""):
        if is_save:
            df = pd.DataFrame(List)
            df = df.set_axis(['duration', 'reward'], axis=1)
            df.to_csv(f"{log_dir}/monitor.csv")
    
    print("train process")
    
    n_observations = 10
    n_actions = 6
    
    dqn = DQN(env, 
              n_actions = n_actions,
              n_observations = n_observations)
    
    num_episodes = num_episodes
    all_total_rewards_T = []
    all_total_rewards = []
    for i_episode in range(1, 1+num_episodes):
        print("===" * 10)
        state, info = env.reset(seed = SEED + i_episode)
        state = dqn.process_state(state)
        
        total_reward = 0
        
        for t in count():
            # print(i_episode, env.num_step, end=" ")
            
            ## RL
            action = dqn.select_action(state)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_cpu = reward
            reward = dqn.process_reward(reward)
            done = terminated or truncated
            
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
                all_total_rewards_T.append([t, total_reward])
                all_total_rewards.append(total_reward)
                # dqn.plot_durations()
                print(f"episode: {i_episode} \tdurations: {t}\t\trewards: {total_reward}")
                break
    
        if i_episode % 10 == 0:
            if math.log10(i_episode).is_integer() or (i_episode % 10000 == 0):
                dqn.save(log_dir, i_episode, int(sum(all_total_rewards[-5:])/ 5))
                print("model saved!")
            save_reward_as_csv(all_total_rewards_T, log_dir)

    print('Complete')

def train_RL_process_cam(log_dir, num_episodes=10):
    print("train process")    
    
    # ------------------------------------------------------------------------------------
    # agent
    agent = DQNAgent(env = env, 
                     EPS_DECAY = EPS_DECAY,
                     TAU = TAU,
                     BATCH_SIZE = BATCH_SIZE,
                     LR = LR,
                     MAX_BUFF = MAX_BUFF,
                     EPS_START = EPS_START,
                     EPS_END = EPS_END)
    # ------------------------------------------------------------------------------------
    
    all_total_rewards = []
    all_durations = []
    i_frame = 0
    last_big_mean_reward = float('-inf')
    print_interval = 100
    
    # ------------------------------------------------------------------------------------
    # tensorboard
    if is_save:
        summary_writer = SummaryWriter(log_dir=log_dir)
    # ------------------------------------------------------------------------------------
    # training loop
    for i_episode in range(1, 1+num_episodes):
        # --------------------------------------------------------------------------------
        # env reset 
        print("===" * 10)
        frame, info = env.reset(seed = SEED + i_episode, is_random=is_random)
        state_tentor = agent.observe(frame)
        # print(state_tentor)
        # print(state_tentor.shape)
        
        total_reward = 0
        
        # --------------------------------------------------------------------------------
        # each episode 
        for t in count():
            # ----------------------------------------------------------------------------
            # select action 
            action = agent.act(state_tentor)
            
            # ----------------------------------------------------------------------------
            # execute action             
            next_frame, reward, terminated, truncated, _ = env.step(action)
            i_frame += 1
            done = terminated or truncated
            total_reward += reward
            
            # ----------------------------------------------------------------------------
            # save trajectory into replay buffer 
            agent.memory_buffer.push(frame, action, reward, next_frame, done)
            frame = next_frame
            
            # ----------------------------------------------------------------------------
            # Perform one step of the optimization
            loss = agent.learn_from_experience()
            
            # ----------------------------------------------------------------------------            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            agent.update_weights()
            
            # ----------------------------------------------------------------------------            
            # if terminated
            if terminated:
                state_tentor = None
            else:
                state_tentor = agent.observe(next_frame)
            
            # ----------------------------------------------------------------------------            
            # print or save some infomation
            if i_frame % print_interval == 0:
                # ------------------------------------------------------------------------
                # print some info 
                mean_reward = np.mean(all_total_rewards[-10:])
                mean_duration = np.mean(all_durations[-10:])
                print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i_frame, round(mean_reward, 2), loss, agent.epsilon, i_episode))
                
                # ------------------------------------------------------------------------
                # save some information
                if is_save:
                    summary_writer.add_scalar("Temporal Difference Loss", loss, i_frame)
                    summary_writer.add_scalar("Mean Reward", mean_reward, i_frame)
                    summary_writer.add_scalar("Epsilon", agent.epsilon, i_frame)
                    summary_writer.add_scalar("Mean Durations", mean_duration, i_frame)
                    
                    # save into txt 
                    save_txt(log_dir, i_frame, i_episode, mean_reward, last_big_mean_reward)           
                    print("##############")
                    if mean_reward > last_big_mean_reward:           
                        agent.save(log_dir) 
                        print("model saved!")
                        last_big_mean_reward = mean_reward
                    else:
                        print("model not saved!")
                # ------------------------------------------------------------------------                        
            # ----------------------------------------------------------------------------
            # if episode ends
            if done:
                all_total_rewards.append(total_reward)
                all_durations.append(t)                
                print(f"episode: {i_episode} \tframes: {i_frame} \tdurations: {t}\trewards: {round(total_reward, 2)}")
                break
            # ----------------------------------------------------------------------------
            
        if i_frame >= num_frames:
            break
# ===================================================
# test model
def test_RL_process():
    print("test process")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
    n_observations = 10
    n_actions = 6
    
    model_dir = "./logs/models/DQN/ball_discrete_neaf2080_007/good_model_EPI1000_R-43.pt"  
    model = torch.load(model_dir)
    
    model.eval()
    print(model)
    
    num_episodes = 100
    all_total_rewards_T = []
    for i_episode in range(1, 1+num_episodes):
        print("===" * 10)
        state, info = env.reset(seed = SEED + i_episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # state = dqn.process_state(state)
        
        total_reward = 0
        
        for t in count():
    #         # print(i_episode, env.num_step, end=" ")
            
            ## RL
            action = model(state).max(1)[1].view(1, 1)
            # action = dqn.select_action(state)
            
    #         # # print("action", action)
    #         # print("action", action.item())
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_cpu = reward
            # reward = dqn.process_reward(reward)
            done = terminated or truncated
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                pass
                
            total_reward += reward_cpu
            
            # Move to the next state
            state = next_state

            if done:
    #             # dqn.episode_durations.append(total_reward)
                all_total_rewards_T.append([t, total_reward])
    #             # dqn.plot_durations()
                print(f"episode: {i_episode} \tdurations: {t}\t\trewards: {total_reward}")
                break
    
    #     if i_episode % 10 == 0:
    #         dqn.save(log_dir)
    #         save_reward_as_csv(all_total_rewards_T, log_dir)
    #         print("model saved!")
            
    
    # # add_end_time(log_dir, num_episodes)
    # print('Complete')
    # # dqn.plot_durations(show_result=True)
    # # plt.ioff()
    # # plt.show()

# ===================================================
# not used 
def RL_process_random():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
        
    num_episodes = 1000
    
    
    all_total_rewards_T = []
    for i_episode in range(1, 1+num_episodes):
        print("===" * 10)
        state, info = env.reset(seed=525+i_episode)
        print(state.shape)
        print(state[0].shape)
        # 將 state 轉換為 PyTorch tensor
        state_tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else state

        # 使用 unsqueeze 方法進行轉換
        state_transformed = state_tensor.unsqueeze(0).permute(0, 3, 1, 2)

        # 輸出轉換後的形狀
        print(state_transformed.shape)  # 應該輸出 [1, 4, 128, 128]
        
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

# ===================================================
if __name__ == "__main__":
    # -----------------------------------------------
    # env setting
    is_training = True
    state_type = "gray"      # ["numerical", "RGB", "gray"]
    SEED = 525
    model_name = "DQN_cam"
    num_episodes = int(5e5)
    num_frames = int(1e6)
    is_save = True
    is_random = True       # target is random or not
    is_going_to_target = False
    object_name = "cube"    # ["ball", "cube"]
    # -----------------------------------------------
    # env hyperparameter
    big_negative_reward = -1000
    reward_slope = -10    
    # -----------------------------------------------
    # DQN hyperparameter
    EPS_DECAY = int(3e5)
    TAU = 0.005
    BATCH_SIZE = 128
    LR = 1e-4
    MAX_BUFF = 100000
    EPS_START = 0.9
    EPS_END = 0.05   
    # -----------------------------------------------
    # init env
    env = env_ball(seed=SEED, 
                   state_type=state_type, 
                   object_name=object_name, 
                   is_going_to_target=is_going_to_target,
                   reward_slope=reward_slope,
                   big_negative_reward=big_negative_reward)    
    #------------------------------------------------
    # Main loop
    while env.arm.supervisor.step(env.arm.timestep) != -1:
    
        with Log_file(model_name=model_name, env_id=env.info, num_episodes=num_episodes, is_save=is_save) as log_dir:
            # train_RL_process(log_dir, num_episodes=num_episodes)
            train_RL_process_cam(log_dir, num_episodes=num_episodes)
        break
    #------------------------------------------------
    
        # test_RL_process(model_dir)
        # break
    
        # RL_process_random()
        # break

        # test()
        # break