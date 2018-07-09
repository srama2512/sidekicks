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
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path', type=str, default='../data/SUN360/data.h5')
    parser.add_argument('--h5_path_unseen', type=str, default='')
    parser.add_argument('--mask_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=str2bool, default=False)
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
    parser.add_argument('--greedy', type=str2bool, default=False)
    parser.add_argument('--memorize_views', type=str2bool, default=True)
    parser.add_argument('--mean_subtract', type=str2bool, default=True)
    parser.add_argument('--actorType', type=str, default='actor', help='[ actor | random | greedyLookAhead | demo_sidekick ]')
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
    parser.add_argument('--start_view', type=int, default=0, help='[0 - random starts, 1 - middle start, 2 - all views, 3 - adversarial]')
    parser.add_argument('--save_path', type=str, default='', help='Path to directory to save some sample results')
    parser.add_argument('--utility_h5_path', type=str, default='', help='Stored utility maps from one-view expert to obtain expert trajectories')
    parser.add_argument('--start_views_json', type=str, default='', help='adversarial starts for evaluation')
    parser.add_argument('--eval_val', type=str2bool, default=False, help='Evaluate on validation set?')

    opts = parser.parse_args()
    opts.debug = False
    opts.A = opts.delta_M * opts.delta_N
    opts.P = opts.delta_M * opts.N
    opts.reward_scale = 1
    opts.expert_rewards = False
    opts.reward_scale_expert = 1e-4
    opts.expert_trajectories = False
    opts.trajectories_type = 'utility_maps'
    opts.supervised_scale = 1e-2
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    if opts.actorType == 'demo_sidekick':
        from DataLoader import DataLoaderExpertPolicy as DataLoader
    else:
        from DataLoader import DataLoaderSimple as DataLoader

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

    if opts.start_view != 3:
        if opts.eval_val:
            val_err, val_std, val_std_err, _ = evaluate(loader, agent, 'val', opts)
        else:
            test_err, test_std, test_std_err, decoded_images = evaluate(loader, agent, 'test', opts)
            if opts.dataset == 1:
                if opts.h5_path_unseen != '':
                    test_unseen_err, test_unseen_std, test_unseen_std_err, decoded_images_unseen = evaluate(loader, agent, 'test_unseen', opts)

    else:
        if opts.eval_val:
            val_err, val_std, val_std_err, _ = evaluate(loader, agent, 'val', opts)
        else:
            test_err, test_std, test_std_err, decoded_images = evaluate_adversarial(loader, agent, 'test', opts)
            if opts.dataset == 1:
                if opts.h5_path_unseen != '':
                    test_unseen_err, test_unseen_std, test_unseen_std_err, decoded_images_unseen = evaluate_adversarial(loader, agent, 'test_unseen', opts)

    if not opts.eval_val:
        writer = SummaryWriter(log_dir=opts.save_path)
        rng_choices = random.sample(range(loader.test_count//opts.batch_size), 10)
        for choice in rng_choices:
            for pano_count in range(decoded_images[choice].size(0)):
                x = vutils.make_grid(decoded_images[choice][pano_count], padding=5, normalize=True, scale_each=True, nrow=opts.T+1, pad_value=1.0)
                writer.add_image('Test batch #%d, image #%d'%(choice, pano_count), x, 0)
        if opts.dataset == 1:
            if opts.h5_path_unseen != '':
                rng_choices = random.sample(range(loader.test_unseen_count//opts.batch_size), 10)
                for choice in rng_choices:
                    for pano_count in range(decoded_images_unseen[choice].size(0)):
                        x = vutils.make_grid(decoded_images_unseen[choice][pano_count], padding=5, normalize=True, scale_each=True, nrow=opts.T+1, pad_value=1.0)
                        writer.add_image('Test unseen batch #%d, image #%d'%(choice, pano_count), x, 0)
    if opts.eval_val:
        print('Val mean(x1000): %6.3f | std(x1000): %6.3f | std err(x1000): %6.3f'%(val_err*1000, val_std*1000, val_std_err*1000))
    else:
        print('Test mean(x1000): %6.3f | std(x1000): %6.3f | std err(x1000): %6.3f'%(test_err*1000, test_std*1000, test_std_err*1000))
        if opts.dataset == 1:
            if opts.h5_path_unseen != '':
                print('Test unseen mean (x1000): %6.3f | std(x1000): %6.3f | std err(x1000): %6.3f'%(test_unseen_err*1000, test_unseen_std*1000, test_unseen_std_err*1000))

        writer.close()
