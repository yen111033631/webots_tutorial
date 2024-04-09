from itertools import count
import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

script_name = os.path.basename(__file__)
device = 'cpu'
directory = './exp' + script_name + "Webots" +'./'
update_iteration=20

# Parameter
mode ='train' # mode = 'train' or 'test'
tau =0.005 # target smoothing coefficient #float
target_update_interval =1 #int
test_iteration =10 #imt

learning_rate =1e-4 #float
gamma =0.99  # discounted factor #float
capacity=100000 # replay buffer size #int
batch_size=1 # mini batch size #int
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

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    # def __init__(self, state_dim, action_dim, max_action):
    #     super(Actor, self).__init__()
    #
    #     self.l1 = nn.Linear(state_dim, 400)
    #     self.l2 = nn.Linear(400, 300)
    #     self.l3 = nn.Linear(300, action_dim)
    #
    #     self.max_action = max_action
    #
    # def forward(self, x):
    #     x = F.relu(self.l1(x))
    #     x = F.relu(self.l2(x))
    #     x = self.max_action * torch.tanh(self.l3(x))
    #     return x
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) * self.max_action
        return x



class Critic(nn.Module):
    # def __init__(self, state_dim, action_dim):
    #     super(Critic, self).__init__()
    #
    #     self.l1 = nn.Linear(state_dim + action_dim, 400)
    #     self.l2 = nn.Linear(400 , 300)
    #     self.l3 = nn.Linear(300, 1)
    #
    # def forward(self, x, u):
    #     x = F.relu(self.l1(torch.cat([x, u], 1)))
    #     x = F.relu(self.l2(x))
    #     x = self.l3(x)
    #     return x
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 9 * 9 + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, u):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        xu = torch.cat([x, u], 1)
        x = F.relu(self.fc1(xu))
        x = self.fc2(x)
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = state.float()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action = np.clip(action, -0.04, 0.04)
        return action

    def update(self):
        for it in range(update_iteration):
            print("gg:",it)
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
        return actor_loss, critic_loss

    def save(self,path):
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic.state_dict(), path + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load("/Users/peter3354152/Desktop/image_test/"+"0612_reward_" + 'actor.pth'))
        self.critic.load_state_dict(torch.load("/Users/peter3354152/Desktop/image_test/"+"0612_reward/0612_reward_" + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")




