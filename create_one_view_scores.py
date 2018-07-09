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
from State import *
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from DataLoader import DataLoaderSimple as DataLoader

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_rewards_one_view(loader, agent, split, opts):
    """
    One View reward function - evaluates the MSE error for each view of the panorama and
    returns scores for each view.
    Outputs:
        true_images    : list of BxNxMxCx32x32 arrays (each element of list corresponds to one batch)
        score_images   : list of BxNxMxCx32x32 arrays, each Cx32x32 image is a constant 
                         image containing score corresponding to each location in NxM panorama
        score_matrices : list of BxNxM arrays, contains scores corresponding to each
                         location in NxM panorama
    """
    depleted = False
    agent.policy.eval()
    true_images = []
    scores_images = []
    scores_matrices = []

    while not depleted:
        if split == 'train':
            pano, depleted = loader.next_batch_train()
        if split == 'val':
            pano, depleted = loader.next_batch_val()
        if split == 'test':
            pano, depleted = loader.next_batch_test()

        curr_err = 0
        batch_size = pano.shape[0]
        # Compute the performance with the initial state 
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)
        # Scores images are stored as BxNxMx3x32x32 images with all values in an image proportional to
        # the assigned score. 
        scores_image = np.zeros(pano.shape)
        # Scores matrices are stored as BxNxM matrices with one value corresponding to one view. 
        scores_matrix = np.zeros(pano.shape[0:3])

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, None, start_idx, opts)
                _, rec_errs, _, _,  _, _, _, _, _ = agent.gather_trajectory(state, {'greedy': opts.greedy, 'memorize_views': opts.memorize_views})
                # For some random initial state, print the decoded images at all time steps
                rec_errs = rec_errs[0].data.cpu()
                for k in range(batch_size):
                    reward = 1/(rec_errs[k]*1000)
                    scores_image[k, i, j] = reward
                    scores_matrix[k, i, j] = reward 

        # Rescale scores for each image in batch
        for i in range(batch_size):
            max_v = np.max(scores_image[i])
            min_v = np.min(scores_image[i])
            scores_image[i] -= min_v
            scores_image[i] /= (max_v - min_v + 1e-8)
            scores_image[i] *= 255
            scores_matrix[i] -= min_v
            scores_matrix[i] /= (max_v - min_v + 1e-8)
            scores_matrix[i] *= 255

        true_images.append(pano)
        scores_images.append(scores_image)
        scores_matrices.append(scores_matrix)

    return true_images, scores_images, scores_matrices 

def get_rewards_uniform(loader, agent, split, opts):
    """
    This is a baseline mechanism where rewards are spread uniformly randomly throughout the
    different views.
    Outputs:
        true_images    : list of BxNxMxCx32x32 arrays (each element of list corresponds to one batch)
        score_images   : list of BxNxMxCx32x32 arrays, each Cx32x32 image is a constant 
                         image containing score corresponding to each location in NxM panorama
        score_matrices : list of BxNxM arrays, contains scores corresponding to each
                         location in NxM panorama
    """
    depleted = False
    agent.policy.eval()
    true_images = []
    scores_images = []
    scores_matrices = []

    while not depleted:
        if split == 'train':
            pano, depleted = loader.next_batch_train()
        if split == 'val':
            pano, depleted = loader.next_batch_val()
        if split == 'test':
            pano, depleted = loader.next_batch_test()

        curr_err = 0
        batch_size = pano.shape[0]
        # Compute the performance with the initial state 
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)
        # Scores images are stored as BxNxMx3x32x32 images with all values in an image proportional to
        # the assigned score. 
        scores_image = np.zeros(pano.shape)
        # Scores matrices are stored as BxNxM matrices with one value corresponding to one view. 
        scores_matrix = np.zeros(pano.shape[0:3])
    
        # Randomly sample nms_iters reward locations
        random_azimuths = np.random.randint(0, opts.M, (batch_size, opts.nms_iters))
        random_elevations = np.random.randint(0, opts.N, (batch_size, opts.nms_iters))

        for i in range(batch_size):
            for j in range(opts.nms_iters):
                 for k1 in [value_1%opts.N for value_1 in range(random_elevations[i, j]-opts.nms_nbd, random_elevations[i, j]+opts.nms_nbd+1)]: 
                    for k2 in [value_2%opts.M for value_2 in range(random_azimuths[i, j]-opts.nms_nbd, random_azimuths[i, j]+opts.nms_nbd+1)]:
                        curr_value = 1.0/(4.0**(max(abs(random_azimuths[i, j]-k2), abs(random_elevations[i, j]-k1))))
                        scores_image[i, k1, k2] += curr_value 
                        scores_matrix[i, k1, k2] += curr_value

        scores_image = np.minimum(scores_image, 1)*255.0
        scores_matrix = np.minimum(scores_matrix, 1)*255.0
        true_images.append(pano)
        scores_images.append(scores_image)
        scores_matrices.append(scores_matrix)

    return true_images, scores_images, scores_matrices 

def greedy_nms_image(score_image, nms_iters, nms_nbd, score_type = 0):
    """
    Takes in a BxNxMx3x32x32 numpy array and performs NMS on
    each NxMx3x32x32 panorama score image. 
    Note: Each 3x32x32 consists of just one pixel value corresponding
    to the score assigned to that view of the panorama
    score_type: 0 - only nms , 1 - nms + smoothing, 2 = none
    """
    if score_type == 2 or score_type == 3:
        final_score_image = score_image
    else:
        shape = score_image.shape
        final_score_image = np.zeros_like(score_image)
        N = shape[1]
        M = shape[2]
        for i in range(shape[0]):
            pano = np.copy(score_image[i])
            iter_count = 0
            while iter_count < nms_iters:
                max_val = 0
                max_idx = (0, 0)
                for j in range(N):
                    for k in range(M):
                        if max_val <= pano[j, k, 0, 0, 0]:
                            max_val = pano[j, k, 0, 0, 0]
                            max_idx = (j, k)
                if score_type == 0:
                    final_score_image[i, max_idx[0], max_idx[1], :, :, :] = max_val
                elif score_type == 1:
                    # Adds +1 to the actual maxima location and 0.25 to the adjacent locations
                    for j in [value_j%N for value_j in range(max_idx[0]-nms_nbd, max_idx[0]+nms_nbd+1)]: 
                        for k in [value_k%M for value_k in range(max_idx[1]-1, max_idx[1]+1+1)]:
                            final_score_image[i, j, k] += min(max_val/(float(4**(max(abs(max_idx[0]-j), abs(max_idx[1]-k))))), max_val)

                # Eliminate the maxima and neighbours for next iteration
                for j in [value_j%N for value_j in range(max_idx[0]-nms_nbd, max_idx[0]+nms_nbd+1)]: 
                    for k in [value_k%M for value_k in range(max_idx[1]-nms_nbd, max_idx[1]+nms_nbd+1)]:
                        pano[j, k, :, :, :] = 0
                
                iter_count += 1
    
    return final_score_image

def greedy_nms_matrix(score_matrix, nms_iters, nms_nbd, score_type):
    """
    Takes in a BxNxM numpy array and performs NMS on
    each NxM matrix.
    Output: BxNxM numpy array 
    """
    if score_type == 2 or score_type == 3:
        final_score_matrix = score_matrix
    else:
        shape = score_matrix.shape
        final_score_matrix = np.zeros_like(score_matrix)
        N = shape[1]
        M = shape[2]
        for i in range(shape[0]):
            matrix_copy = np.copy(score_matrix[i])
            iter_count = 0
            while iter_count < nms_iters:
                max_val = 0
                max_idx = (0, 0)
                for j in range(N):
                    for k in range(M):
                        if max_val <= matrix_copy[j, k]:
                            max_val = matrix_copy[j, k]
                            max_idx = (j, k)
                if score_type == 0:
                    final_score_matrix[i, max_idx[0], max_idx[1]] = max_val
                elif score_type == 1:
                    # Adds +1 to the actual maxima location and 0.25 to the adjacent locations
                    for j in [value_j%N for value_j in range(max_idx[0]-nms_nbd, max_idx[0]+nms_nbd+1)]: 
                        for k in [value_k%M for value_k in range(max_idx[1]-1, max_idx[1]+1+1)]:
                            final_score_matrix[i, j, k] += min(max_val/(float(4**(max(abs(max_idx[0]-j), abs(max_idx[1]-k))))), max_val)

                # Eliminate the maxima and neighbours for next iteration
                for j in [value_j%N for value_j in range(max_idx[0]-nms_nbd, max_idx[0]+nms_nbd+1)]: 
                    for k in [value_k%M for value_k in range(max_idx[1]-nms_nbd, max_idx[1]+nms_nbd+1)]:
                        matrix_copy[j, k] = 0
                
                iter_count += 1
        
    return final_score_matrix

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
    if opts.score_type != 3:
        loaded_state = torch.load(opts.load_model)
        agent.policy.load_state_dict(loaded_state['state_dict'])

    h5file = h5py.File(opts.save_path_h5, 'w')
   
    for split in ['train', 'val']:
        if opts.score_type == 3:
            true_images, scores_images, scores_matrices = get_rewards_uniform(loader, agent, split, opts)
        else:
            true_images, scores_images, scores_matrices = get_rewards_one_view(loader, agent, split, opts)
        if opts.debug:
            num_batches = len(true_images)
            assert(len(scores_images) == num_batches)
            assert(len(scores_matrices) == num_batches)
            for i in range(num_batches):
                batch_size = true_images[i].shape[0]
                assert(scores_images[i].shape == (batch_size, opts.N, opts.M, opts.num_channels, 32, 32))
                assert(scores_matrices[i].shape == (batch_size, opts.N, opts.M))
                
        final_scores_matrices_nms = []
        for i in range(len(true_images)):
            scores_matrix = scores_matrices[i]
            scores_matrix_nms = greedy_nms_matrix(scores_matrix, opts.nms_iters, opts.nms_nbd, opts.score_type)
            if opts.debug:
                batch_size = scores_matrix.shape[0]
                assert(scores_matrix_nms.shape == (batch_size, opts.N, opts.M))
            final_scores_matrices_nms.append(scores_matrix_nms)
    
        if split == 'val':
            images_count = 0
            # Iterate through the different batches
            for i in range(len(true_images)):
                shape = true_images[i].shape
                true_images[i] = np.reshape(true_images[i], (shape[0], shape[1]*shape[2], shape[3], shape[4], shape[5]))/255.0
                scores_images_nms = np.reshape(greedy_nms_image(scores_images[i], opts.nms_iters, opts.nms_nbd, opts.score_type), (shape[0], shape[1]*shape[2], shape[3], shape[4], shape[5]))/255.0
                scores_images_normal = np.reshape(scores_images[i], (shape[0], shape[1]*shape[2], shape[3], shape[4], shape[5]))/255.0
                concatenated = torch.Tensor(np.concatenate([true_images[i], scores_images_normal, scores_images_nms], axis=1))
                for j in range(shape[0]):    
                    x = vutils.make_grid(concatenated[j], padding=True, normalize=False, scale_each=False, nrow=opts.M)
                    images_count += 1

                    writer.add_image('Panorama #%5.3d'%(images_count), x, 0)
       
        scores_matrices = np.concatenate(scores_matrices, axis=0)
        final_scores_matrices_nms = np.concatenate(final_scores_matrices_nms, axis=0)
        h5file.create_dataset('%s/normal'%split, data=scores_matrices)
        h5file.create_dataset('%s/nms'%split, data=final_scores_matrices_nms)
    
    json.dump(vars(opts), open(opts.save_path_json, 'w'))
    writer.close()
    h5file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path', type=str, default='../data/SUN360/data.h5')
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
    parser.add_argument('--memorize_views', type=str2bool, default=True)
    parser.add_argument('--save_path_vis', type=str, default='')
    parser.add_argument('--save_path_h5', type=str, default='rewards.h5')
    parser.add_argument('--save_path_json', type=str, default='rewards.json')
    parser.add_argument('--mean_subtract', type=str2bool, default=True)
    parser.add_argument('--actorType', type=str, default='actor')
    parser.add_argument('--reward_scale', type=float, default=1, help='scaling for rewards')
    parser.add_argument('--reward_scale_expert', type=float, default=1e-4, help='scaling for expert rewards if used')
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
    parser.add_argument('--nms_iters', type=int, default=4, help='number of maxima to select from nms')
    parser.add_argument('--nms_nbd', type=int, default=1, help='distance of neighbours to suppress')
    parser.add_argument('--score_type', type=int, default=0, help='[ 0 - perform nms only and extract maxima | \
                                                                     1 - perform nms and smoothen to spread out the reward \
                                                                         to neighbouring views \
                                                                     2 - do not perform nms \
                                                                     3 - uniformly spread out rewards (baseline)]')
    parser.add_argument('--expert_trajectories', type=str2bool, default=False, help='Get expert trajectories for supervised policy learning')
    parser.add_argument('--utility_h5_path', type=str, default='', help='Stored utility maps from one-view expert to obtain expert trajectories')
    parser.add_argument('--supervised_scale', type=float, default=1e-2)

    # Other options
    parser.add_argument('--debug', type=str2bool, default=False, help='Turn on debug mode to activate asserts for correctness')  
    opts = parser.parse_args()
    opts.h5_path_unseen = ''
    main(opts)
