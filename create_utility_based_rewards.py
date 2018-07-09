import os
import sys
import pdb

import json
import h5py
import torch
import random
import argparse
import torchvision
import tensorboardX
import torch.optim as optim
import torchvision.utils as vutils

sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))

from utils import *
from models import *
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from DataLoader import DataLoaderSimple as DataLoader
        
def get_utility_maps(loader, agent, split, opts):
    """
    get_utility_maps - computes the MSE error for using one view of panorama P and
    generating each view of P. This is used to generate a scoring map which defines 
    the utility (0-1) of picking a view V for reconstructing each of the views of P.  
    Outputs:
        true_images      : list of BxNxMxCx32x32 arrays (each element of list corresponds to one batch)
        utility_images   : list of BxNxMxNxMxCx32x32 arrays, each MxNxCx32x32 image is a set of constant 
                           images containing utility map corresponding to each location in NxM panorama
        utility_matrices : list of BxNxMxNxM arrays, contains utility maps corresponding to each
                           location in NxM panorama
    """
    depleted = False
    agent.policy.eval()
    true_images = []
    utility_images = []
    utility_matrices = []
    
    while not depleted:
        if split == 'train':
            pano, depleted = loader.next_batch_train()
        if split == 'val':
            pano, depleted = loader.next_batch_val()
        if split == 'test':
            pano, _, depleted = loader.next_batch_test()
        if split == 'test_unseen':
            pano, _, depleted = loader.next_batch_test_unseen()

        curr_err = 0
        batch_size = pano.shape[0]

        N = pano.shape[1]
        M = pano.shape[2]
        C = pano.shape[3]
        H = 8
        W = 8

        # Compute the performance with the initial state 
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)
        # Scores images are stored as BxNxMx3x32x32 images with all values in an image proportional to
        # the assigned score. 
        utility_image = np.zeros((batch_size, N, M, N, M, C, H, W))
        # Scores matrices are stored as BxNxM matrices with one value corresponding to one view. 
        utility_matrix = np.zeros((batch_size, N, M, N, M))

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, None, start_idx, opts)
                _, rec_errs, _, _,  decoded, _, _, _, _ = agent.gather_trajectory(state, {'greedy': opts.greedy, 'memorize_views': opts.memorize_views})

                rec_errs_per_view = np.reshape((state.views_prepro_shifted - decoded.data.cpu().numpy())**2, (batch_size, N, M, -1)).sum(axis=3)  
                rec_errs = rec_errs[0].data.cpu().numpy()
                if opts.debug:
                    assert((rec_errs - np.reshape(rec_errs_per_view, (batch_size, -1)).sum(axis=1)/state.total_pixels).sum() <= 1e-6)
                # Rotate the reconstruction errors by the starting view to get original orientation
                rec_errs_per_view = np.roll(rec_errs_per_view, j, axis=2)
                for k in range(batch_size):
                    utility_matrix[k, i, j] = 1/(rec_errs_per_view[k, :, :]*1000.0 + 1e-8)
        
        # Rescale utility by normalizing over the utilities of taking any view @ a particular view
        max_v = np.max(np.max(utility_matrix, axis=2), axis=1)
        min_v = np.min(np.min(utility_matrix, axis=2), axis=1)
        # expanding the max and min to span over all views
        max_v = np.repeat(np.repeat(max_v[:, np.newaxis, np.newaxis, :, :], repeats=N, axis=1), repeats=M, axis=2)
        min_v = np.repeat(np.repeat(min_v[:, np.newaxis, np.newaxis, :, :], repeats=N, axis=1), repeats=M, axis=2)
        utility_matrix -= min_v 
        utility_matrix /= (max_v-min_v + 1e-8)
        
        if opts.debug:
            assert((utility_matrix >= 0).all())
    
        if opts.threshold_maps:
            utility_matrix[utility_matrix > 0.5] = 1
            utility_matrix[utility_matrix <= 0.5] = 0

        utility_image = np.repeat(np.repeat(np.repeat(utility_matrix[:, :, :, :, :, np.newaxis, np.newaxis, np.newaxis], repeats=C, axis=5), repeats=H, axis=6), repeats=W, axis=7)
        true_images.append(pano)
        utility_images.append(utility_image)
        utility_matrices.append(utility_matrix)

    return true_images, utility_images, utility_matrices 

def main(opts):
    # Set number of actions
    opts.A = opts.delta_M * opts.delta_N
    # Set random seeds 
    set_random_seeds(opts.seed)

    if opts.dataset == 0:
        if opts.mean_subtract:
            opts.mean = [119.16, 107.68, 95.12]
            opts.std = [61.88, 61.72, 67.24]
        else:
            opts.mean = [0, 0, 0]
            opts.std = [0, 0, 0]
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
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=opts.save_path_vis)

    loader = DataLoader(opts)
    agent = Agent(opts, mode='eval')
    loaded_state = torch.load(opts.load_model)
    agent.policy.load_state_dict(loaded_state['state_dict'])

    h5file = h5py.File(opts.save_path_h5, 'w')
    
    all_splits = ['train', 'val', 'test']
    if opts.dataset == 1:
        all_splits.append('test_unseen')

    for split in all_splits:
        true_images, utility_images, utility_matrices = get_utility_maps(loader, agent, split, opts)
        reward_matrices = []
        for i in range(len(true_images)):
            shape = true_images[i].shape
            reward_matrix = np.zeros((shape[0], opts.N, opts.M))
            for j in range(shape[0]):
                optimal_views, utility_value = get_submodular_views(utility_matrices[i][j], 4) 
                for k in optimal_views:
                    for itera in [a_val % opts.N for a_val in range(k[0]-opts.nms_nbd, k[0]+opts.nms_nbd+1)]:
                        for iterb in [b_val % opts.M for b_val in range(k[1]-opts.nms_nbd, k[1]+opts.nms_nbd+1)]:
                            reward_matrix[j, itera, iterb] += 255.0/4.0**(max(abs(k[0]-itera), abs(k[1]-iterb)))
            reward_matrix = np.minimum(reward_matrix, 255.0)
            reward_matrices.append(reward_matrix)

        if opts.debug:
            num_batches = len(true_images)
            assert(len(utility_images) == num_batches)
            assert(len(utility_matrices) == num_batches)
            for i in range(num_batches):
                batch_size = true_images[i].shape[0]
                assert(utility_images[i].shape == (batch_size, opts.N, opts.M, opts.N, opts.M, opts.num_channels, 8, 8))
                assert(utility_matrices[i].shape == (batch_size, opts.N, opts.M, opts.N, opts.M))
        
        if split == 'val':
            images_count = 0
            # Iterate through the different batches
            for i in range(len(true_images)):
                shape = true_images[i].shape
                true_images[i] = np.reshape(true_images[i].transpose(0, 3, 1, 4, 2, 5), (shape[0], 1, shape[3], shape[1]*shape[4], shape[2]*shape[5]))/255.0
                utility_images_normal = np.reshape(utility_images[i].transpose(0, 1, 2, 5, 3, 6, 4, 7), (shape[0], opts.N*opts.M, opts.num_channels, opts.N*8, opts.M*8))
                for j in range(shape[0]):    
                    x = vutils.make_grid(torch.Tensor(utility_images_normal[j]), padding=3, normalize=False, scale_each=False, nrow=opts.M)
                    images_count += 1
                    writer.add_image('Panorama #%5.3d utility'%(images_count), x, 0)
                    # ---- Apply submodularity based greedy algorithm to get near-optimal views ----
                    optimal_views, utility_value = get_submodular_views(utility_matrices[i][j], 4) 
                    optimal_views_images = np.zeros((opts.N, opts.M, opts.num_channels, 32, 32))
                    # Convert the scores into images for visualization
                    for k in optimal_views:
                        optimal_views_images[k[0], k[1]] = 1.0
                    optimal_views_images = np.reshape(optimal_views_images.transpose(2, 0, 3, 1, 4), (1, opts.num_channels, opts.N*32, opts.M*32))
                    # Get the reward image computed based on optimal_views
                    reward_image = np.repeat(np.repeat(np.repeat(reward_matrices[i][j][:, :, np.newaxis, np.newaxis, np.newaxis], repeats=opts.num_channels, axis=2), repeats=32, axis=3), repeats=32, axis=4)
                    reward_image = np.reshape(reward_image.transpose(2, 0, 3, 1, 4), (1, opts.num_channels, opts.N*32, opts.M*32))/255.0

                    # Concatenate the true image, optimal view image and reward image for display
                    concatenated_images = np.concatenate([true_images[i][j], optimal_views_images, reward_image], axis=0)
                    x = vutils.make_grid(torch.Tensor(concatenated_images), padding=3, normalize=False, scale_each=False, nrow=1)
                    writer.add_image('Panorama #%5.3d image'%(images_count), x, 0)

        
        utility_matrices = np.concatenate(utility_matrices, axis=0)
        reward_matrices = np.concatenate(reward_matrices, axis=0)
        h5file.create_dataset('%s/utility_maps'%split, data=utility_matrices)
        h5file.create_dataset('%s/nms'%(split), data=reward_matrices)

    json.dump(vars(opts), open(opts.save_path_json, 'w'))
    writer.close()
    h5file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path', type=str, default='../data/SUN360/data.h5')
    parser.add_argument('--h5_path_unseen', type=str, default='')
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
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=True)
    parser.add_argument('--save_path_vis', type=str, default='')
    parser.add_argument('--save_path_h5', type=str, default='utility_maps.h5')
    parser.add_argument('--save_path_json', type=str, default='utility_maps.json')
    parser.add_argument('--mean_subtract', type=str2bool, default=True)
    parser.add_argument('--actorType', type=str, default='actor')
    parser.add_argument('--reward_scale', type=float, default=1e-2, help='scaling for rewards')
    parser.add_argument('--reward_scale_expert', type=float, default=1e-4, help='scaling for expert rewards if used')
    parser.add_argument('--baselineType', type=str, default='average', help='[ average | critic ]') 
    # Environment options
    parser.add_argument('--T', type=int, default=4, help='Number of allowed steps / views')
    parser.add_argument('--M', type=int, default=8, help='Number of azimuths')
    parser.add_argument('--N', type=int, default=4, help='Number of elevations')
    parser.add_argument('--delta_M', type=int, default=5, help='Number of movable azimuths')
    parser.add_argument('--delta_N', type=int, default=3, help='Number of movable elevations')
    parser.add_argument('--wrap_azimuth', type=str2bool, default=True, help='wrap around the azimuths when rotating?')
    parser.add_argument('--wrap_elevation', type=str2bool, default=False, help='wrap around the elevations when rotating?')

    # Other options
    parser.add_argument('--debug', type=str2bool, default=False, help='Turn on debug mode to activate asserts for correctness')  
    parser.add_argument('--threshold_maps', type=str2bool, default=False, help='Threshold the utility maps to get binary utilities')
    parser.add_argument('--nms_nbd', type=int, default=1)

    opts = parser.parse_args()
    opts.memorize_views = False
    opts.critic_full_obs = False
    opts.act_full_obs = False
    main(opts)
