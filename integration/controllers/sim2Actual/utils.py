import math
import torch
import pandas as pd
import random

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def loadData(rr):
    # label = pd.read_csv('/home/neaf-2080/code/Haw/Dataset/Dataset_can_256/gt.csv').to_numpy()
    label = pd.read_csv('/Users/peter3354152/Desktop/碩論/台達手臂/integration/gtFix.csv').to_numpy()
    return label[rr][0],label[rr][1],[label[rr][2],label[rr][3],label[rr][4]],label[rr][6]

def initial_state(gt_pos, gt_rot):
    rl_pos_x = gt_pos[0] + 0.002 + 0.0001*(random.randint(-1,1))#+ 0.0025 * random.randint(0,2) - 0.0025
    rl_pos_y = gt_pos[1] + 0.005
    rl_pos_z = gt_pos[2] - 0.001 + 0.0001*(random.randint(-1,1))#+ 0.0025 * random.randint(0,2) - 0.0025
    rl_rot = gt_rot
    return [rl_pos_x, rl_pos_y, rl_pos_z, rl_rot]
