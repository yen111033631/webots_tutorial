import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128 # 512 
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
# if GPU is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Q_Network(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Q_Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN():
    def __init__(self, env, n_actions, n_observations) -> None:
        random.seed(env.seed) 
        torch.manual_seed(env.seed)
        
        self.env = env
        self.n_actions = n_actions
        # n_actions = env.action_space.n
        # n_observations = env.observation_space.shape[0]   
        self.is_cuda = torch.cuda.is_available() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device", self.device)
        
        self.policy_net = Q_Network(n_observations, n_actions).to(self.device)
        self.target_net = Q_Network(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        pass
    
    def process_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
    def process_reward(self, reward):
        return torch.tensor([reward], device=self.device)        

    def select_action(self, state):
        # global self.steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randint(0, self.n_actions-1)]], device=self.device, dtype=torch.long)
        
    def update_weights(self):
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            self.target_net.load_state_dict(target_net_state_dict)
        
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # print(transitions)
        # print(zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # print("BATCH_SIZE", BATCH_SIZE)
        # print("device", self.device)
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def save(self, log_dir, num_episode, avg_reward):
        torch.save(self.target_net, f"{log_dir}/good_model_EPI{num_episode}_R{avg_reward}.pt")
        torch.save(self.target_net.state_dict(), f"{log_dir}/good_model_state_dict_EPI{num_episode}_R{avg_reward}.pt")

##########################################
# -> copy from here

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
print(is_ipython)
if is_ipython:
    from IPython import display

plt.ion()

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    task = "CartPole"
    
    dqn = DQN(env)

    if dqn.is_cuda:
        num_episodes = 1000
        # num_episodes = 1
    else:
        num_episodes = 50
        
    all_total_rewards = []
    for i_episode in range(1, 1+num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        # print(state)
        env.render()        
        state = dqn.process_state(state)
        
        
                
        total_reward = 0
        
        for t in count():
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
                dqn.episode_durations.append(t + 1)
                all_total_rewards.append(total_reward)
                dqn.plot_durations()
                print(f"episode: {i_episode} \tdurations: {t}\t\trewards: {total_reward}")
                
                break

        if i_episode%50 == 0:
            torch.save(dqn.target_net, f"./checkpoint/DQN_{task}_{i_episode}_{int(sum(all_total_rewards[-10:])/10)}.pt")
            # torch.save(target_net.state_dict(), f"/home/neaf2080/code/yen/RL/DQN/checkpoint/DQN_{task}_{i_episode}_{int(sum(all_total_rewards[-10:])/10)}_state_dict.pt")
            print(f"{i_episode},  save model")

    print('Complete')
    dqn.plot_durations(show_result=True)
    plt.ioff()
    plt.show()