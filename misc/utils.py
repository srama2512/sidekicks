import sys
import pdb
import math
import json
import torch
import random
import argparse
import numpy as np
import torchvision
import tensorboardX
import torch.optim as optim
import torchvision.utils as vutils

from State import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def set_random_seeds(seed):
    """
    Sets the random seeds for numpy, python, pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_module(agent, opts):
    """
    Given the agent, load a pre-trained model and other setup based on the
    training_setting
    """
    # ---- Load the pre-trained model ----
    load_state = torch.load(opts.load_model)
    # strict=False ensures that only the modules common to loaded_dict and agent.policy's state_dict are loaded. 
    # Could potentially lead to errors being masked. Tread carefully! 
    if opts.actorType == 'actor' and opts.act_full_obs:
        # Don't load the actor module, since the full obs actor architecture is different. 
        partial_state_dict = {k: v for k, v in load_state['state_dict'].items() if 'act' not in k}
        agent.policy.load_state_dict(partial_state_dict, strict=False)
    else:
        agent.policy.load_state_dict(load_state['state_dict'], strict=False) 
    
    # ---- Other settings ----
    epoch_start = 0
    best_val_error = 100000
    train_history = []
    val_history = []
    
    if opts.training_setting == 1:
        """
        Scenario: Model trained on one-view reconstruction. Needs to be 
        finetuned for multi-view reconstruction.
        """
        # (1) Must fix sense, fuse modules
        for parameter in agent.policy.sense_im.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.sense_pro.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.fuse.parameters():
            parameter.requires_grad = False
        # (2) Fix decode module if requested
        if opts.fix_decode:
            for parameter in agent.policy.decode.parameters():
                parameter.requires_grad = False
        # (3) Re-create the optimizer with the above settings
        agent.create_optimizer(opts.lr, opts.weight_decay, opts.training_setting, opts.fix_decode)

    elif opts.training_setting == 2:
        """
        Scenario: Model trained on one-view reconstruction. Needs to be
        further trained on the same setting.
        """
        # (1) Keep a copy of the new number of epochs to run for
        epoch_total = opts.epochs
        # (2) Load the rest of the opts from saved model
        opts = load_state['opts']
        opts.epochs = epoch_total
        train_history = load_state['train_history']
        val_history = load_state['val_history']
        best_val_error = load_state['best_val_error']
        epoch_start = load_state['epoch']+1
        # (3) Create optimizer based on the new parameter settings 
        agent.create_optimizer(opts.lr, opts.weight_decay, 2, opts.fix_decode)
        # (4) Load the optimizer state dict
        agent.optimizer.load_state_dict(load_state['optimizer'])

    elif opts.training_setting == 3:
        """
        Scenario: Model training on multi-view reconstruction. Needs to be 
        further trained on the same setting.
        """
        # (1) Load opts from saved model and replace LR
        opts_copy = load_state['opts']
        opts_copy.lr = opts.lr
        train_history = load_state['train_history']
        val_history = load_state['val_history']
        best_val_error = load_state['best_val_error']
        epoch_start = load_state['epoch']+1
        opts_copy.training_setting = opts.training_setting
        opts = opts_copy
        # (2) Fix sense, fuse and decode (optionally) modules  
        for parameter in agent.policy.sense_im.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.sense_pro.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.fuse.parameters():
            parameter.requires_grad = False
        if opts.fix_decode:
            for parameter in agent.policy.decode.parameters():
                parameter.requires_grad = False
        # (3) Re-create the optimizer with the above settings
        agent.create_optimizer(opts.lr, opts.weight_decay, 3, opts.fix_decode)
        # (4) Load the optimizer state dict
        agent.optimizer.load_state_dict(load_state['optimizer'])

    elif opts.training_setting == 4:
        """
        Scenario: Model trained on one-view reconstruction. Needs to be
        further trained on some other setting.
        """
        # (1) Load the train history, val history and best validation errors from the saved model.
        train_history = load_state['train_history']
        val_history = load_state['val_history']
        best_val_error = load_state['best_val_error']
        epoch_start = load_state['epoch']+1
        # (2) Create the optimizer according to the new settings
        agent.create_optimizer(opts.lr, opts.weight_decay, opts.training_setting, False)

    return best_val_error, train_history, val_history, epoch_start

def get_starts(N, M, batch_size, option):
    """
    Given the number of elevations(N), azimuths(M), batch size and the option (different types of starts),
    this function returns the start indices for the batch.
    start_idx: list of [start_elev, start_azim] for each panorama in the batch
    """
    if option == 0:
        start_idx = [[random.randint(0, N-1), random.randint(0, M-1)] for i in range(batch_size)]
    else:
        start_idx = [[N//2, M//2-1] for i in range(batch_size)]
    return start_idx

def utility_function(utility_matrix, selected_views, threshold):
    """
    Evaluates the quality of the selected views based on the utility_matrix
    utility_matrix : NxMxNxM array
    selected_views : list of (i, j) pairs indicating selected views
    """
    M = utility_matrix.shape[1]
    N = utility_matrix.shape[0]
    total_utility_map = np.zeros((N, M))

    for view in selected_views:
        total_utility_map += utility_matrix[view[0], view[1]]

    total_utility_map = np.minimum(total_utility_map, threshold)
    return total_utility_map.sum()

def utility_function_unique(utility_matrix, selected_views, threshold):
    """
    Evaluates the quality of the selected views based on the utility_matrix.
    This selects only uniques views for computation, to ensure that the
    same view does get selected multiple times.

    utility_matrix : NxMxNxM array
    selected_views : list of (i, j) pairs indicating selected views
    """
    M = utility_matrix.shape[1]
    N = utility_matrix.shape[0]
    total_utility_map = np.zeros((N, M))
    
    selected_views_set = set()
    for view in selected_views:
        selected_views_set.add((view[0], view[1]))

    for view in selected_views_set:
        total_utility_map += utility_matrix[view[0], view[1]]

    total_utility_map = np.minimum(total_utility_map, threshold)
    return total_utility_map.sum()

def get_submodular_views(utility_matrix, num_views):
    """
    Uses greedy maximization of submodular utility function to get close to optimal set of views
    utility_matrix : NxMxNxM array
    num_views      : number of views to select
    """
    M = utility_matrix.shape[1]
    N = utility_matrix.shape[0]
    sel_views = []

    total_utility = 0
    for n in range(num_views):
        max_idx = [0, 0]
        max_utility_gain = 0
        for i in range(N):
            for j in range(M):
                curr_utility_gain = utility_function(utility_matrix, sel_views + [[i, j]], 1) - total_utility
                if curr_utility_gain >= max_utility_gain:
                    max_utility_gain = curr_utility_gain
                    max_idx = [i, j]
        sel_views.append(max_idx)
        total_utility += max_utility_gain

    return sel_views, total_utility

def get_expert_trajectories(state, pano_maps_orig, selected_views, opts):
    """
    Get greedy trajectories based on utility for each panorama in batch
    opts must contain:
    T, delta_M, delta_N, wrap_elevation, wrap_azimuth, N, M
    """
    pano_maps = np.copy(pano_maps_orig)
    batch_size = pano_maps.shape[0]
    # Note: Assuming atleast one view has been selected initially
    t_start = len(selected_views[0])-1 # What t to start from, if some views have already been selected
    # Access pattern: selected_views[batch_size][time_step]
    selected_actions = np.zeros((batch_size, opts.T-t_start-1), np.int32)  # Access pattern: selected_actions[batch_size][time_step]
    for i in range(batch_size):
        curr_utility = utility_function_unique(pano_maps[i], selected_views[i], 1)
        # Given the first view, select T-1 more views
        t = t_start
        while t < opts.T-1:
            curr_pos = selected_views[i][t]
            max_gain = 0
            max_delta = None
            max_pos = None
            for delta_ele in range(-(opts.delta_N//2), opts.delta_N//2 + 1):
                for delta_azi in range(-(opts.delta_M//2), opts.delta_M//2 + 1):
                    if opts.wrap_elevation:
                        new_ele = (curr_pos[0] + delta_ele)%opts.N
                    else:
                        new_ele = max(min(curr_pos[0] + delta_ele, opts.N-1), 0)

                    if opts.wrap_azimuth:
                        new_azi = (curr_pos[1] + delta_azi)%opts.M
                    else:
                        new_azi = max(min(curr_pos[1] + delta_azi, opts.M-1), 0)
                    
                    new_pos = [new_ele, new_azi]
                    curr_gain = utility_function_unique(pano_maps[i], selected_views[i] + [new_pos], 1) - curr_utility
                    if curr_gain >= max_gain:
                        max_gain = curr_gain
                        max_delta = (delta_ele, delta_azi)
                        max_pos = new_pos

            curr_utility += max_gain
            selected_views[i].append(max_pos)
            selected_actions[i][t-t_start] = state.delta_to_act[max_delta]
            t += 1

    return selected_views, selected_actions

def evaluate(loader, agent, split, opts):
    """
    Evaluation function - evaluates the agent over fixed grid locations as
    starting points and returns the overall average reconstruction error.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    overall_err = 0
    overall_count = 0
    err_values = []
    decoded_images = []
    while not depleted:
        # ---- Sample batch of data ----
        if split == 'val':
            if opts.expert_rewards and opts.expert_trajectories:
                pano, pano_maps, pano_rewards, depleted = loader.next_batch_val()
            elif opts.expert_trajectories or opts.actorType == 'demo_sidekick':
                pano, pano_maps, depleted = loader.next_batch_val()
                pano_rewards = None
            elif opts.expert_rewards:
                pano, pano_rewards, depleted = loader.next_batch_val()
                pano_maps = None
            else:
                pano, depleted = loader.next_batch_val()
                pano_rewards = None
                pano_maps = None
        elif split == 'test':
            if opts.actorType == 'demo_sidekick':
                pano, pano_masks, pano_maps, depleted = loader.next_batch_test()
            else:
                pano, pano_masks, depleted = loader.next_batch_test()
            pano_rewards = None
        elif split == 'test_unseen':
            if opts.actorType == 'demo_sidekick':
                pano, pano_masks, pano_maps, depleted = loader.next_batch_test_unseen()
            else:
                pano, pano_masks, depleted = loader.next_batch_test_unseen()
            pano_rewards = None

        # Initial setup for evaluating over a grid of views
        curr_err = 0
        curr_count = 0
        curr_err_batch = 0
        batch_size = pano.shape[0]
	# Compute the performance with the initial state
        # starting at fixed grid locations
        if opts.start_view == 0:
            # Randomly sample one location from grid
            elevations = [random.randint(0, opts.N-1)]
            azimuths = [random.randint(0, opts.M-1)]
        elif opts.start_view == 1:
            # Sample only the center location from grid
            elevations = [opts.N//2]
            azimuths = [opts.M//2-1]
        else:
            # Sample all the locations from grid
            elevations = range(0, opts.N, 2)
            azimuths = range(0, opts.M, 2)

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                if split == 'test' or split == 'test_unseen':
                    state = State(pano, pano_rewards, start_idx, opts, pano_masks)
                else:
                    state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                if opts.actorType == 'demo_sidekick': # Not enabling demo_sidekick training for AgentSupervised (that's not needed, doesn't make sense)
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True}, pano_maps=pano_maps, opts=opts)
                else:
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
                # For some random initial state, print the decoded images at all time steps
                if curr_count == 0:
                    curr_decoded_plus_true = None
                    for dec_idx in range(len(decoded_all)):
                        decoded = decoded_all[dec_idx].data.cpu()
                        curr_decoded = decoded.numpy()
                        # Rotate it forward by the start index
                        # Shifting all the images by equal amount since the start idx is same for all
                        if not opts.knownAzim:
                            curr_decoded = np.roll(curr_decoded, start_idx[0][1], axis=2)
                        if not opts.knownElev:
                            curr_decoded = np.roll(curr_decoded, start_idx[0][0], axis=1)

                        # Fill in the true views here
                        for jdx, jdx_v in enumerate(visited_idxes):
                            if jdx > dec_idx:
                                break
                            for idx in range(batch_size):
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :] = state.views_prepro[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :] 
                        curr_decoded = curr_decoded * 255
                        for c in range(opts.num_channels):
                            #curr_decoded[:, :, :, , c, :, :] *= opts.std[c]
                            curr_decoded[:, :, :, c, :, :] += opts.mean[c]
                        
                        if opts.num_channels == 1:
                            curr_decoded_3chn = np.zeros((batch_size, opts.N, opts.M, 3, 32, 32))
                            for c in range(3):
                                curr_decoded_3chn[:, :, :, c, :, :] = curr_decoded[:, :, :, 0, :, :]
                            curr_decoded = curr_decoded_3chn
                        #for jdx, jdx_v in enumerate(visited_idxes):
                        #    if jdx > dec_idx:
                        #        break
                        jdx_v = visited_idxes[dec_idx]
                        #for idx in range(batch_size):
                            # Fill in some red margin
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, 0:3, :] = 0
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, -3:, :] = 0
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, 0:3] = 0
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, -3:] = 0
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], 0, 0:3, :] = 255
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], 0, -3:, :] = 255
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], 0, :, 0:3] = 255
                            #curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], 0, :, -3:] = 255

                        # Need to convert from B x N x M x C x 32 x 32 to B x 1 x C x N*32 x M*32
                        # Convert from B x N x M x C x 32 x 32 to B x C x N x 32 x M x 32 and then reshape
                        curr_decoded = curr_decoded.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, 3, opts.N*32, opts.M*32)
                        true_state = np.array(state.views)
                        start_idx = state.start_idx
                        if opts.num_channels == 1:
                            true_state_3chn = np.zeros((batch_size, opts.N, opts.M, 3, 32, 32))
                            for c in range(3):
                                true_state_3chn[:, :, :, c, :, :] = true_state[:, :, :, 0, :, :]
                            true_state = true_state_3chn

                        # Fill in red margin for starting states of each true panorama
                        #for idx in range(batch_size):
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], :, 0:3, :] = 0
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], :, -3:, :] = 0
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], :, :, 0:3] = 0
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], :, :, -3:] = 0
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, 0:3, :] = 255
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, -3:, :] = 255
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, :, 0:3] = 255
                        #    true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, :, -3:] = 255

                        true_state = true_state.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, 3, opts.N*32, opts.M*32)
                            
                        if curr_decoded_plus_true is None:
                            curr_decoded_plus_true = curr_decoded
                        else:
                            curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, curr_decoded], axis=1)

                    curr_decoded_plus_true = np.concatenate([true_state, curr_decoded_plus_true], axis=1)
                    if opts.expert_rewards:
                        reward_image = np.zeros_like(curr_decoded)
                        for iter_N in range(opts.N):
                            for iter_M in range(opts.M):
                                for bn in range(batch_size):
                                    reward_image[bn, :, :, (iter_N*32):((iter_N+1)*32), (iter_M*32):((iter_M+1)*32)] = pano_rewards[bn, iter_N, iter_M]/255.0
                        curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, reward_image], axis=1)

                    decoded_images.append(torch.Tensor(curr_decoded_plus_true/255.0))
                 
                # Add error from the last step
                curr_err += rec_errs[-1].data.sum()
                curr_count += 1 # Count for the views
                curr_err_batch += rec_errs[-1].data.cpu().numpy()

        curr_err /= curr_count
        curr_err_batch /= curr_count
        for i in range(curr_err_batch.shape[0]):
            err_values.append(float(curr_err_batch[i]))
        overall_err += curr_err
        overall_count += batch_size
    
    err_values = np.array(err_values)
    overall_mean = float(np.mean(err_values))
    overall_std = float(np.std(err_values, ddof=1))
    overall_std_err = float(overall_std/math.sqrt(err_values.shape[0]))

    agent.policy.train()

    return overall_mean, overall_std, overall_std_err, decoded_images

def evaluate_adversarial_fixed(loader, agent, split, opts):
    """
    Evaluation function - evaluates the agent over the hardest starting points for
    a one-view model
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    overall_err = 0
    overall_count = 0
    decoded_images = []
    start_views = json.load(open(opts.start_views_json))['%s_adversarial_views'%split]
    for i in range(len(start_views)):
        start_views[i][0] = int(start_views[i][0])
        start_views[i][1] = int(start_views[i][1])
    
    err_values = []
    while not depleted:
        # ---- Sample batch of data ----
        if split == 'test':
            pano, pano_masks, depleted = loader.next_batch_test()
            pano_rewards = None
            pano_maps = None
        elif split == 'test_unseen':
            pano, pano_masks, depleted = loader.next_batch_test_unseen()
            pano_rewards = None
            pano_maps = None

        # Initial setup for evaluating over a grid of views
        batch_size = pano.shape[0]
        # Get the adversarial start_idx
        start_idx = start_views[overall_count:(overall_count+batch_size)]

        state = State(pano, pano_rewards, start_idx, opts, pano_masks)
        # Enable view memorization for testing by default
        _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
        # For some random initial state, print the decoded images at all time steps
        curr_decoded_plus_true = None
        for dec_idx in range(len(decoded_all)):
            decoded = decoded_all[dec_idx].data.cpu()
            curr_decoded = decoded.numpy()
            # Rotate it forward by the start index
            # Shifting all the images by equal amount since the start idx is same for all
            if not opts.knownAzim:
                curr_decoded = np.roll(curr_decoded, start_idx[0][1], axis=2)
            if not opts.knownElev:
                curr_decoded = np.roll(curr_decoded, start_idx[0][0], axis=1)

            # Fill in the true views here
            for jdx, jdx_v in enumerate(visited_idxes):
                if jdx > dec_idx:
                    break
                for idx in range(batch_size):
                    curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :] = state.views_prepro[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :]
                    # Fill in some black margin
                    curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, 0:3, :] = 0
                    curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, -3:-1, :] = 0
                    curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, 0:3] = 0
                    curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, -3:-1] = 0

            # Need to convert from B x N x M x C x 32 x 32 to B x 1 x C x N*32 x M*32
            # Convert from B x N x M x C x 32 x 32 to B x C x N x 32 x M x 32 and then reshape
            curr_decoded = curr_decoded.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, opts.num_channels, opts.N*32, opts.M*32)*255.0
            true_state = state.views.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, opts.num_channels, opts.N*32, opts.M*32)
            for c in range(opts.num_channels):
                #curr_decoded[:, :, c, :, :] *= opts.std[c]
                curr_decoded[:, :, c, :, :] += opts.mean[c]
            
            if curr_decoded_plus_true is None:
                curr_decoded_plus_true = curr_decoded
            else:
                curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, curr_decoded], axis=1)

        curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, true_state], axis=1)
        if opts.expert_rewards:
            reward_image = np.zeros_like(curr_decoded)
            for iter_N in range(opts.N):
                for iter_M in range(opts.M):
                    for bn in range(batch_size):
                        reward_image[bn, :, :, (iter_N*32):((iter_N+1)*32), (iter_M*32):((iter_M+1)*32)] = pano_rewards[bn, iter_N, iter_M]/255.0
            curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, reward_image], axis=1)

        decoded_images.append(torch.Tensor(curr_decoded_plus_true/255.0))
        
        err_values.append(rec_errs[-1].data.cpu().numpy())
        overall_err += np.sum(rec_errs[-1].data.cpu().numpy())
        overall_count += batch_size

    err_values = np.concatenate(err_values, axis=0)
    overall_mean = np.mean(err_values)
    overall_std = np.std(err_values, ddof=1)
    overall_std_err = overall_std/math.sqrt(err_values.shape[0])
    agent.policy.train()

    return overall_mean, overall_std, overall_std_err,  decoded_images

def evaluate_adversarial(loader, agent, split, opts):
    """
    Evaluation function - evaluates the agent over all grid locations as
    starting points and returns the average of worst reconstruction error over different
    locations for the panoramas (average(max error over locations)).
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    overall_err = 0
    overall_count = 0
    decoded_images = []
    err_values = []
    while not depleted:
        # ---- Sample batch of data ----
        if split == 'val':
            if opts.expert_trajectories or opts.actorType == 'demo_sidekick':
                pano, pano_maps, depleted = loader.next_batch_val()
                pano_rewards = None
            elif opts.expert_rewards:
                pano, pano_rewards, depleted = loader.next_batch_val()
                pano_maps = None
            else:
                pano, depleted = loader.next_batch_val()
                pano_rewards = None
                pano_maps = None
        elif split == 'test':
            if opts.actorType == 'demo_sidekick':
                pano, pano_masks, pano_maps, depleted = loader.next_batch_test()
            else:
                pano, pano_masks, depleted = loader.next_batch_test()
            pano_rewards = None
        elif split == 'test_unseen':
            if opts.actorType == 'demo_sidekick':
                pano, pano_masks, pano_maps, depleted = loader.next_batch_test_unseen()
            else:
                pano, pano_masks, depleted = loader.next_batch_test_unseen()
            pano_rewards = None

        # Initial setup for evaluating over a grid of views
        batch_size = pano.shape[0]
	# Compute the performance with the initial state
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)
        
        errs_across_grid = np.zeros((batch_size, opts.N, opts.M))

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                if split == 'test' or split == 'test_unseen':
                    state = State(pano, pano_rewards, start_idx, opts, pano_masks)
                else:
                    state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                if opts.actorType == 'demo_sidekick': # Not enabling demo_sidekick training for AgentSupervised (that's not needed, doesn't make sense)
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True}, pano_maps=pano_maps, opts=opts)
                else:
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
                # For some random initial state, print the decoded images at all time steps
                if i == 0 and j == 0:
                    curr_decoded_plus_true = None
                    for dec_idx in range(len(decoded_all)):
                        decoded = decoded_all[dec_idx].data.cpu()
                        curr_decoded = decoded.numpy()
                        # Rotate it forward by the start index
                        # Shifting all the images by equal amount since the start idx is same for all
                        if not opts.knownAzim:
                            curr_decoded = np.roll(curr_decoded, start_idx[0][1], axis=2)
                        if not opts.knownElev:
                            curr_decoded = np.roll(curr_decoded, start_idx[0][0], axis=1)

                        # Fill in the true views here
                        for jdx, jdx_v in enumerate(visited_idxes):
                            if jdx > dec_idx:
                                break
                            for idx in range(batch_size):
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :] = state.views_prepro[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :]
                                # Fill in some black margin
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, 0:3, :] = 0
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, -3:-1, :] = 0
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, 0:3] = 0
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, -3:-1] = 0

                        # Need to convert from B x N x M x C x 32 x 32 to B x 1 x C x N*32 x M*32
                        # Convert from B x N x M x C x 32 x 32 to B x C x N x 32 x M x 32 and then reshape
                        curr_decoded = curr_decoded.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, opts.num_channels, opts.N*32, opts.M*32)*255.0
                        true_state = state.views.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, opts.num_channels, opts.N*32, opts.M*32)
                        for c in range(opts.num_channels):
                            #curr_decoded[:, :, c, :, :] *= opts.std[c]
                            curr_decoded[:, :, c, :, :] += opts.mean[c]
                        
                        if curr_decoded_plus_true is None:
                            curr_decoded_plus_true = curr_decoded
                        else:
                            curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, curr_decoded], axis=1)

                    curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, true_state], axis=1)
                    if opts.expert_rewards:
                        reward_image = np.zeros_like(curr_decoded)
                        for iter_N in range(opts.N):
                            for iter_M in range(opts.M):
                                for bn in range(batch_size):
                                    reward_image[bn, :, :, (iter_N*32):((iter_N+1)*32), (iter_M*32):((iter_M+1)*32)] = pano_rewards[bn, iter_N, iter_M]/255.0
                        curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, reward_image], axis=1)

                    decoded_images.append(torch.Tensor(curr_decoded_plus_true/255.0))
                # endif
                errs_across_grid[:, i, j] = rec_errs[-1].data.cpu().numpy()
        
        errs_across_grid = errs_across_grid.reshape(batch_size, -1)
        overall_err += np.sum(np.max(errs_across_grid, axis=1))
        overall_count += batch_size
        err_values.append(np.max(errs_across_grid, axis=1))

    err_values = np.concatenate(err_values, axis=0)
    overall_mean = np.mean(err_values)
    overall_std = np.std(err_values, ddof=1)
    overall_std_err = overall_std/math.sqrt(err_values.shape[0])
 
    agent.policy.train()

    return overall_mean, overall_std, overall_std_err, decoded_images 

def get_all_trajectories(loader, agent, split, opts):
    """
    Gathers trajectories from all starting positions and returns them.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    trajectories = {}
    # Sample all the locations from grid
    elevations = range(0, opts.N)
    azimuths = range(0, opts.M)
    for i in elevations:
        for j in azimuths:
            trajectories[(i, j)] = []

    while not depleted:
        # ---- Sample batch of data ----
        if split == 'train':
            pano, depleted = loader.next_batch_train()
            pano_rewards = None
            pano_maps = None
        if split == 'val':
            pano, depleted = loader.next_batch_val()
            pano_rewards = None
            pano_maps = None
        elif split == 'test':
            pano, pano_masks, depleted = loader.next_batch_test()
            pano_rewards = None
            pano_maps = None
        elif split == 'test_unseen':
            pano, pano_masks, depleted = loader.next_batch_test_unseen()
            pano_rewards = None
            pano_maps = None

        batch_size = pano.shape[0]
        # Gather agent trajectories from each starting location
        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                if split == 'test' or split == 'test_unseen':
                    state = State(pano, pano_rewards, start_idx, opts, pano_masks)
                else:
                    state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                _, _, _, _, _, _, _, _, actions_taken = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
                # actions_taken: B x T torch Tensor
                trajectories[(i, j)].append(actions_taken)
    
    for i in elevations:
        for j in azimuths:
            trajectories[(i, j)] = torch.cat(trajectories[(i, j)], dim=0)

    agent.policy.train()

    return trajectories 

def select_adversarial_views(loader, agent, split, opts):
    """
    Adversarial selection function - evaluates the agent over all grid locations as
    starting points and returns the indices of the worst reconstruction error over different
    locations for the panoramas.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    decoded_images = []
    adversarial_views = []
    while not depleted:
        # ---- Sample batch of data ----
        if split == 'val':
            if opts.expert_trajectories:
                pano, pano_maps, depleted = loader.next_batch_val()
                pano_rewards = None
            elif opts.expert_rewards:
                pano, pano_rewards, depleted = loader.next_batch_val()
                pano_maps = None
            else:
                pano, depleted = loader.next_batch_val()
                pano_rewards = None
                pano_maps = None
        elif split == 'test':
            pano, pano_masks, depleted = loader.next_batch_test()
            pano_rewards = None
            pano_maps = None
        elif split == 'test_unseen':
            pano, pano_masks, depleted = loader.next_batch_test_unseen()
            pano_rewards = None
            pano_maps = None

        # Initial setup for evaluating over a grid of views
        batch_size = pano.shape[0]
	# Compute the performance with the initial state
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)
        
        errs_across_grid = np.zeros((batch_size, opts.N, opts.M))

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                if split == 'test' or split == 'test_unseen':
                    state = State(pano, pano_rewards, start_idx, opts, pano_masks)
                else:
                    state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
                # For some random initial state, print the decoded images at all time steps
                if i == 0 and j == 0:
                    curr_decoded_plus_true = None
                    for dec_idx in range(len(decoded_all)):
                        decoded = decoded_all[dec_idx].data.cpu()
                        curr_decoded = decoded.numpy()
                        # Rotate it forward by the start index
                        # Shifting all the images by equal amount since the start idx is same for all
                        if not opts.knownAzim:
                            curr_decoded = np.roll(curr_decoded, start_idx[0][1], axis=2)
                        if not opts.knownElev:
                            curr_decoded = np.roll(curr_decoded, start_idx[0][0], axis=1)

                        # Fill in the true views here
                        for jdx, jdx_v in enumerate(visited_idxes):
                            if jdx > dec_idx:
                                break
                            for idx in range(batch_size):
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :] = state.views_prepro[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, :]
                                # Fill in some black margin
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, 0:3, :] = 0
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, -3:-1, :] = 0
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, 0:3] = 0
                                curr_decoded[idx, jdx_v[idx][0], jdx_v[idx][1], :, :, -3:-1] = 0

                        # Need to convert from B x N x M x C x 32 x 32 to B x 1 x C x N*32 x M*32
                        # Convert from B x N x M x C x 32 x 32 to B x C x N x 32 x M x 32 and then reshape
                        curr_decoded = curr_decoded.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, opts.num_channels, opts.N*32, opts.M*32)*255.0
                        true_state = state.views.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, opts.num_channels, opts.N*32, opts.M*32)
                        for c in range(opts.num_channels):
                            #curr_decoded[:, :, c, :, :] *= opts.std[c]
                            curr_decoded[:, :, c, :, :] += opts.mean[c]
                        
                        if curr_decoded_plus_true is None:
                            curr_decoded_plus_true = curr_decoded
                        else:
                            curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, curr_decoded], axis=1)

                    curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, true_state], axis=1)
                    if opts.expert_rewards:
                        reward_image = np.zeros_like(curr_decoded)
                        for iter_N in range(opts.N):
                            for iter_M in range(opts.M):
                                for bn in range(batch_size):
                                    reward_image[bn, :, :, (iter_N*32):((iter_N+1)*32), (iter_M*32):((iter_M+1)*32)] = pano_rewards[bn, iter_N, iter_M]/255.0
                        curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, reward_image], axis=1)

                    decoded_images.append(torch.Tensor(curr_decoded_plus_true/255.0))
                # endif
                errs_across_grid[:, i, j] = rec_errs[-1].data.cpu().numpy()
        
        errs_across_grid = errs_across_grid.reshape(batch_size, -1)
        adversarial_views.append(np.argmax(errs_across_grid, axis=1))

    # The indices are encoded in the row major format. Need to convert to (n, m) format.
    adversarial_views = np.concatenate(adversarial_views, axis=0)
    adversarial_views_n_m = np.zeros((adversarial_views.shape[0], 2))
    for i in range(adversarial_views.shape[0]):
        # adversarial_views[i] = n*M + m
        m = adversarial_views[i]%opts.M
        n = math.floor(adversarial_views[i]/opts.M)
        assert(n*opts.M + m == adversarial_views[i])
        adversarial_views_n_m[i][0] = n
        adversarial_views_n_m[i][1] = m
    return adversarial_views_n_m.tolist()

def iunf(input_layer, initunf=0.1):
    # If the layer is an LSTM
    if str(type(input_layer)) == "<class 'torch.nn.modules.rnn.LSTM'>":
        for i in range(input_layer.num_layers):
            nn.init.uniform(getattr(input_layer, 'weight_ih_l%d'%(i)), -initunf, initunf)
            nn.init.uniform(getattr(input_layer, 'weight_hh_l%d'%(i)), -initunf, initunf)
            nn.init.uniform(getattr(input_layer, 'bias_ih_l%d'%(i)), -initunf, initunf)
            nn.init.uniform(getattr(input_layer, 'bias_hh_l%d'%(i)), -initunf, initunf)
    # For all other layers except batch norm
    elif not (str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>" or str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"):
        if hasattr(input_layer, 'weight'):
            nn.init.uniform(input_layer.weight, -initunf, initunf);
        if hasattr(input_layer, 'bias'):
            nn.init.uniform(input_layer.bias, -initunf, initunf);
    return input_layer

def ixvr(input_layer, bias_val=0.01):
    # If the layer is an LSTM
    if str(type(input_layer)) == "<class 'torch.nn.modules.rnn.LSTM'>":
        for i in range(input_layer.num_layers):
            nn.init.xavier_normal(getattr(input_layer, 'weight_ih_l%d'%(i)))
            nn.init.xavier_normal(getattr(input_layer, 'weight_hh_l%d'%(i)))
            nn.init.constant(getattr(input_layer, 'bias_ih_l%d'%(i)), bias_val)
            nn.init.constant(getattr(input_layer, 'bias_hh_l%d'%(i)), bias_val)
    # For all other layers except batch norm
    elif not (str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>" or str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"):
        if hasattr(input_layer, 'weight'):
            nn.init.xavier_normal(input_layer.weight);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, bias_val);
    return input_layer

def inrml(input_layer, mean=0, std=0.001):
    # If the layer is an LSTM
    if str(type(input_layer)) == "<class 'torch.nn.modules.rnn.LSTM'>":
        for i in range(input_layer.num_layers):
            nn.init.normal(getattr(input_layer, 'weight_ih_l%d'%(i)), mean, std)
            nn.init.normal(getattr(input_layer, 'weight_hh_l%d'%(i)), mean, std)
            nn.init.constant(getattr(input_layer, 'bias_ih_l%d'%(i)), 0.01)
            nn.init.constant(getattr(input_layer, 'bias_hh_l%d'%(i)), 0.01)
    # For all other layers except batch norm
    elif not (str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>" or str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"):
        if hasattr(input_layer, 'weight'):
            nn.init.normal(input_layer.weight, mean, std);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, 0.01);
    return input_layer

def initialize_sequential(var_sequential, init_method):
    """
    Given a sequential module (var_sequential) and an initialization method 
    (init_method), this initializes var_sequential using init_method
    
    Note: The layers returned are different from the one inputted. 
    Not sure if this affects anything.
    """
    var_list = []
    for i in range(len(var_sequential)):
        var_list.append(init_method(var_sequential[i]))

    return nn.Sequential(*var_list)

class View(nn.Module):
    def __init__(self, *shape):
        # shape is a list
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)
