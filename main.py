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

sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), 'misc/'))

import utils
from utils import *
from State import *
from models import *
from tensorboardX import SummaryWriter

def train(opts):
    """
    Training function - trains an agent for a fixed number of epochs
    """
    # Set number of actions
    opts.A = opts.delta_M * opts.delta_N
    # Set random seeds
    set_random_seeds(opts.seed)  

    if opts.expert_rewards and opts.expert_trajectories:
        from DataLoader import DataLoaderExpertBoth as DataLoader  
    elif opts.expert_rewards:
        from DataLoader import DataLoaderExpert as DataLoader
    elif opts.expert_trajectories or opts.actorType == 'demo_sidekick':
        from DataLoader import DataLoaderExpertPolicy as DataLoader
    else:
        from DataLoader import DataLoaderSimple as DataLoader

    if opts.dataset == 0:
        opts.num_channels = 3
        if opts.mean_subtract:
            # R, G, B means and stds
            opts.mean = [119.16, 107.68, 95.12]
            opts.std = [61.88, 61.72, 67.24]
        else:
            opts.mean = [0, 0, 0]
            opts.std = [1, 1, 1]
    elif opts.dataset == 1:
        opts.num_channels = 1
        if opts.mean_subtract:
            # R, G, B means and stds
            opts.mean = [193.0162338615919]
            opts.std = [37.716024486312811]
        else:
            opts.mean = [0]
            opts.std = [1]
    else:
        raise ValueError('Dataset %d does not exist!'%(opts.dataset))

    if opts.expert_trajectories:        
        opts.T_sup = opts.T-1
    loader = DataLoader(opts)
    if opts.expert_trajectories:
        agent = AgentSupervised(opts)
    else:
        agent = Agent(opts)
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=opts.save_path)
    # Set networks to train
    agent.policy.train()
    # Initiate statistics storage variables
    if opts.load_model == '': 
        best_val_error = 100000
        train_history = []
        val_history = []
        epoch_start = 0
    else:
       best_val_error, train_history, val_history, epoch_start = load_module(agent, opts)

    # To handle job eviction and restarts
    if os.path.isfile(os.path.join(opts.save_path, 'model_latest.net')):
        print('====> Resuming training from previous checkpoint')
        # undo most of the loading done before
        loaded_model = torch.load(os.path.join(opts.save_path, 'model_latest.net'))
        opts = loaded_model['opts']
        epoch_start = loaded_model['epoch'] + 1

        loader = DataLoader(opts)
        if opts.expert_trajectories:
            agent = AgentSupervised(opts)
            agent.T_sup = loaded_model['T_sup']
        else:
            agent = Agent(opts) 

        agent.policy.load_state_dict(loaded_model['state_dict'])
        train_history = loaded_model['train_history']
        val_history = loaded_model['val_history']
        #agent.optimizer.load_state_dict(loaded_model['optimizer'])
        best_val_error = loaded_model['best_val_error']

    # Some random selection of images to display
    rng_choices = random.sample(range(400//opts.batch_size), 2) 
    # Start training
    for epoch in range(epoch_start, opts.epochs):
        # Initialize epoch specific variables
        depleted = False
        train_err = 0
        train_count = 0
        iter_count = 0
        
        while not depleted:
            # pano - BxNxMxCx32x32
            if opts.expert_rewards and opts.expert_trajectories:
                pano, pano_maps, pano_rewards, depleted = loader.next_batch_train()
            elif opts.expert_rewards:
                pano, pano_rewards, depleted = loader.next_batch_train()
                pano_maps = None
            elif opts.expert_trajectories or opts.actorType == 'demo_sidekick':
                pano, pano_maps, depleted = loader.next_batch_train()
                pano_rewards = None
            else:
                pano, depleted = loader.next_batch_train()
                pano_rewards = None
                pano_maps = None

            # Note: This batch size is the current batch size, not the global batch size. This varies
            # when you reach the boundary of the dataset.
            batch_size = pano.shape[0]
            start_idx = get_starts(opts.N, opts.M, batch_size, opts.start_view)
            state = State(pano, pano_rewards, start_idx, opts)
            if opts.expert_trajectories:
                if  opts.hybrid_train:
                    rec_errs = agent.train_agent_hybrid(state, pano_maps, opts)
                elif opts.hybrid_inv_train:
                    rec_errs = agent.train_agent_hybrid_inv(state, pano_maps, opts)
                else:
                    rec_errs = agent.train_agent(state, pano_maps, opts)
            else:
                # Forward pass
                log_probs, rec_errs, rewards, entropies, decoded, values, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts=None, pano_maps=pano_maps, opts=opts) 
                # Backward pass
                agent.update_policy(rewards, log_probs, rec_errs, entropies, values, visited_idxes, decoded_all) 
            
            # Accumulate statistics
            train_err += rec_errs[-1].data.sum()
            train_count += batch_size
            iter_count += 1
        
        train_err /= train_count
        
        # Evaluate the agent after every epoch
        val_err, _, _, decoded_images = evaluate(loader, agent, 'val', opts)
        
        # Write out statistics to tensorboard
        writer.add_scalar('data/train_error', train_err, epoch+1)
        writer.add_scalar('data/val_error', val_err, epoch+1)
       
        # Write out models and other statistics to torch format file
        train_history.append([epoch, train_err])
        val_history.append([epoch, val_err])
        if best_val_error > val_err:
            best_val_error = val_err
            save_state = {
                            'epoch': epoch,
                            'state_dict': agent.policy.state_dict(),
                            'optimizer': agent.optimizer.state_dict(),
                            'opts': opts, 
                            'best_val_error': best_val_error,
                            'train_history': train_history,
                            'val_history': val_history
                         }
            if opts.expert_trajectories:
                save_state['T_sup'] = agent.T_sup

            torch.save(save_state, os.path.join(opts.save_path, 'model_best.net'))
                
        save_state = {
                        'epoch': epoch,
                        'state_dict': agent.policy.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'opts': opts, 
                        'best_val_error': best_val_error,
                        'train_history': train_history,
                        'val_history': val_history
                     }
        if opts.expert_trajectories:
            save_state['T_sup'] = agent.T_sup
        torch.save(save_state, os.path.join(opts.save_path, 'model_latest.net'))

        print('Epoch %d : Train loss: %9.6f    Val loss: %9.6f'%(epoch+1, train_err, val_err))

        # Reduce supervision gradually
        if opts.expert_trajectories and (opts.hybrid_train or opts.hybrid_inv_train):
            if (epoch+1) % opts.hybrid_schedule == 0 and agent.T_sup > 0:
                agent.T_sup -= 1
            # Save the model after the first schedule is over
            if epoch+1 == opts.hybrid_schedule:
                torch.save(save_state, os.path.join(opts.save_path, 'model_after_one_schedule.net'))

        # Decay exploration factor
        if agent.policy.explorationFactor > 0:
            agent.policy.explorationFactor = max(agent.policy.explorationFactor * opts.explorationDecay, 0.0001) - 0.0001
        
        # Decay expert reward gradually
        if opts.expert_rewards and (epoch+1) % opts.expert_rewards_scale_decay == 0:
            agent.reward_scale_expert /= 10

        # Display three randomly selected batches of panoramas every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            for choice in rng_choices:
                for pano_count in range(decoded_images[choice].size(0)):
                    x = vutils.make_grid(decoded_images[choice][pano_count], padding=5, normalize=True, scale_each=True, nrow=opts.T//2+1) 
                    writer.add_image('Validation batch # : %d  image # : %d'%(choice, pano_count), x, 0) # Converting this to 0 to save disk space, should be epoch ideally
        
        # Early stopping criterion
        if epoch == 0:
            best_val_error_in_past = best_val_error
        if (epoch+1) % opts.check_stop == 0:
            if abs(best_val_error - best_val_error_in_past) <= 1e-8:
                break
            else:
                best_val_error_in_past = best_val_error
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path', type=str, default='data/sun360/sun360_processed.h5')
    parser.add_argument('--h5_path_unseen', type=str, default='')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--init', type=str, default='xavier', help='[ xavier | normal | uniform ]')
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help='Entropy regularization')
    parser.add_argument('--critic_coeff', type=float, default=1e-2, help="coefficient for critic's loss term")
    parser.add_argument('--fix_decode', type=str2bool, default=False) 
    parser.add_argument('--check_stop', type=int, default=1000, help='Checks if the validation error has improved every N epochs, stops otherwise')
    parser.add_argument('--training_setting', type=int, default=1, \
            help='[0 - training full model from scratch | \
                   1 - freeze sense and fuse, start finetuning other modules | \
                   2-  resume training of stopped one view model (set epochs, optimizer state, etc) | \
                   3 - resume training of stopped multi view model (set epochs, optimizer state, etc | \
                   4 - resume training of stopped one view model with different epochs, learning rates, etc'
                                   
    )

    # Agent options
    parser.add_argument('--dataset', type=int, default=0, help='[ 0: SUN360 | 1: ModelNet ]')
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--actOnElev', type=str2bool, default=True)
    parser.add_argument('--actOnAzim', type=str2bool, default=False)
    parser.add_argument('--actOnTime', type=str2bool, default=True)
    parser.add_argument('--knownElev', type=str2bool, default=True)
    parser.add_argument('--knownAzim', type=str2bool, default=False)
    parser.add_argument('--explorationBaseFactor', type=float, default=0)
    parser.add_argument('--explorationDecay', type=float, default=0)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=False, help='enable greedy action selection during validation?')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--memorize_views', type=str2bool, default=False)
    parser.add_argument('--mean_subtract', type=str2bool, default=True)
    parser.add_argument('--actorType', type=str, default='actor', help='[ actor | random | greedyLookAhead | demo_sidekick ]')
    parser.add_argument('--baselineType', type=str, default='average', help='[ average | critic ]')
    parser.add_argument('--use_gae', type=str2bool, default=False)
    parser.add_argument('--lambda_gae', type=float, default=0.98)
    parser.add_argument('--act_full_obs', type=str2bool, default=False, help='Full observability for actor?')
    parser.add_argument('--critic_full_obs', type=str2bool, default=False, help='Full observability for critic?')
    parser.add_argument('--expert_trajectories', type=str2bool, default=False, help='Get expert trajectories for supervised policy learning')
    parser.add_argument('--trajectories_type', type=str, default='utility_maps', help='[ utility_maps | expert_trajectories ]')
    parser.add_argument('--utility_h5_path', type=str, default='', help='Stored utility maps from one-view expert to obtain expert trajectories')
    parser.add_argument('--supervised_scale', type=float, default=1e-3)
    parser.add_argument('--hybrid_train', type=str2bool, default=False)
    parser.add_argument('--hybrid_inv_train', type=str2bool, default=False)
    parser.add_argument('--hybrid_schedule', type=int, default=50)
    # Environment options
    parser.add_argument('--T', type=int, default=4, help='Number of allowed steps / views')
    parser.add_argument('--M', type=int, default=8, help='Number of azimuths')
    parser.add_argument('--N', type=int, default=4, help='Number of elevations')
    parser.add_argument('--delta_M', type=int, default=5, help='Number of movable azimuths')
    parser.add_argument('--delta_N', type=int, default=3, help='Number of movable elevations')
    parser.add_argument('--wrap_azimuth', type=str2bool, default=True, help='wrap around the azimuths when rotating?')
    parser.add_argument('--wrap_elevation', type=str2bool, default=False, help='wrap around the elevations when rotating?')
    parser.add_argument('--reward_scale', type=float, default=1, help='scaling for rewards')
    parser.add_argument('--expert_rewards', type=str2bool, default=False, help='Use rewards from expert agent?')
    parser.add_argument('--rewards_h5_path', type=str, default='', help='Reward file from expert agent')
    parser.add_argument('--reward_scale_expert', type=float, default=1e-4, help='scaling for expert rewards if used')
    parser.add_argument('--expert_rewards_scale_decay', type=float, default=1000, help='Divide the expert reward scale by a factor of 10 every K epochs')
    parser.add_argument('--start_view', type=int, default=0, help='[0 - random starts, 1 - middle start]')
    parser.add_argument('--reward_estimator', type=int, default=0, help='[ 0 - proper average | 1 - decaying average ]')
    # Other options
    parser.add_argument('--debug', type=str2bool, default=False, help='Turn on debug mode to activate asserts for correctness')
    
    opts = parser.parse_args()
    assert not(opts.hybrid_train and opts.hybrid_inv_train), "Cannot be hybrid and hybrid_inv simultaneously!"

    train(opts)
