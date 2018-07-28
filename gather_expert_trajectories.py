import os
import sys
import pdb
import json
import torch
import random 
import argparse
import torchvision
import tensorboardX
import torch.optim as optim
import torchvision.utils as vutils

sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))

from utils import *
from State import *
from models import *
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from DataLoader import DataLoaderSimple as DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path', type=str, default='../data/SUN360/data.h5')
    parser.add_argument('--h5_path_unseen', type=str, default='')
    parser.add_argument('--mask_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--explorationBaseFactor', type=float, default=0)
    parser.add_argument('--init', type=str, default='xavier')
    # Agent options
    parser.add_argument('--dataset', type=int, default=0, help='[ 0: SUN360 | 1: ModelNet ]')
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--actOnElev', type=str2bool, default=True)
    parser.add_argument('--actOnAzim', type=str2bool, default=False)
    parser.add_argument('--actOnTime', type=str2bool, default=True)
    parser.add_argument('--knownElev', type=str2bool, default=True)
    parser.add_argument('--knownAzim', type=str2bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=True)
    parser.add_argument('--memorize_views', type=str2bool, default=True)
    parser.add_argument('--mean_subtract', type=str2bool, default=True)
    parser.add_argument('--actorType', type=str, default='actor', help='[ actor | random | greedyLookAhead ]')
    parser.add_argument('--baselineType', type=str, default='average', help='[ average | critic ]') 
    parser.add_argument('--act_full_obs', type=str2bool, default=False)
    parser.add_argument('--critic_full_obs', type=str2bool, default=False)
    # Environment options
    parser.add_argument('--T', type=int, default=4, help='Number of allowed steps / views')
    parser.add_argument('--M', type=int, default=8, help='Number of azimuths')
    parser.add_argument('--N', type=int, default=4, help='Number of elevations')
    parser.add_argument('--delta_M', type=int, default=5, help='Number of movable azimuths')
    parser.add_argument('--delta_N', type=int, default=3, help='Number of movable elevations')
    parser.add_argument('--wrap_azimuth', type=str2bool, default=True, help='wrap around the azimuths when rotating?')
    parser.add_argument('--wrap_elevation', type=str2bool, default=False, help='wrap around the elevations when rotating?')
    parser.add_argument('--reward_scale', type=float, default=1e-2, help='scaling for rewards')
    parser.add_argument('--reward_scale_expert', type=float, default=1e-4, help='scaling for expert rewards if used')
    parser.add_argument('--save_path', type=str, default='', help='Path to directory to save some sample results')
    parser.add_argument('--expert_trajectories', type=str2bool, default=False, help='Get expert trajectories for supervised policy learning')
    parser.add_argument('--utility_h5_path', type=str, default='', help='Stored utility maps from one-view expert to obtain expert trajectories')
    parser.add_argument('--supervised_scale', type=float, default=1e-2)
    parser.add_argument('--start_views_json', type=str, default='', help='adversarial starts for evaluation')

    opts = parser.parse_args()
    opts.debug = False
    opts.A = opts.delta_M * opts.delta_N
    opts.P = opts.delta_M * opts.N
    opts.expert_rewards = False
    opts.shuffle = False

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    if opts.dataset == 0:
        if opts.mean_subtract:
            opts.mean = [119.16, 107.68, 95.12]
            opts.std = [61.88, 61.72, 67.24]
        else:
            opts.mean = [0, 0, 0]
            opts.std = [1, 1, 1]
        opts.num_channels = 3
    elif opts.dataset == 1:
        if opts.mean_subtract:
            opts.mean = [193.0162338615919]
            opts.std = [37.716024486312811]
        else:
            opts.mean = [0]
            opts.std = [0]
        opts.num_channels = 1
    else:
        raise ValueError('Dataset %d does not exist!'%(opts.dataset))

    loader = DataLoader(opts)
    agent = Agent(opts, mode='eval')
    loaded_state = torch.load(opts.model_path)
    agent.policy.load_state_dict(loaded_state['state_dict'])
    train_trajectories = []
    train_trajectories = get_all_trajectories(loader, agent, 'train', opts)
    val_trajectories = get_all_trajectories(loader, agent, 'val', opts)
    torch.save({'train': train_trajectories, 'val': val_trajectories}, open(os.path.join(opts.save_path, 'expert_trajectories.t7'), 'w')) 
