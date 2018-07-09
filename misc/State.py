from torch.autograd import Variable

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import copy
import pdb

def preprocess_views(pano, mean, std):
    """
    This function preprocesses the input views by subtracting the mean and dividing by the standard deviation
    pano: BxNxMxCxHxW numpy array
    """
    pano_float = pano.astype(np.float32)
    for c in range(len(mean)):
        pano_float[:, :, :, c, :, :] -= mean[c]
        # Ignore this because it affects the MSE computation later
        #pano_float[:, :, :, c, :, :] /= std[c]
    return pano_float/255.0 # Scale pixel values from [0, 255] to [0, 1] range

def preprocess(im, pro):
    """
        This function converts the numpy arrays or lists to tensors and returns it.
        Can be augmented with different operations in the future if needed (like augmentation).
        im: BxCxHxW images 
        pro: list of list of integers [delta_elev, delta_azim, elev (optional), azim (optional)]
    """
    return torch.Tensor(im), torch.Tensor(pro)

class State:
    """
    This class takes a batch of panoramas or multiple views and stores the data. It can return
    views based on the current state indices and update the views based on relative rotations.
    It can optionally take rewards computed by an expert agent. 
    """
    def __init__(self, views, views_rewards, start_idx, opts, masks=None):
        """
            N = # elevations
            M = # azimuths
            views: B x N x M x C x H x W array 
            views_rewards: B x N x M array
            start_idx: Initial views for B panoramas [..., [elevation_idx, azimuth_idx], ...]

            init sets up the settings, data in the state and preprocesses the views
            settings needed:
            Panorama navigation settings:
            (1) M, N, A, C (2) start_idx (3) idx (4) delta (5) actOn*, known*, wrap*
            (6) act_to_delta, delta_to_act
            Data settings:
            (1) batch_size (2) normalization (3) total_pixels (4) debug
        """
       
        # ---- Panorama navigation settings ----
        self.M = opts.M
        self.N = opts.N
        self.A = opts.delta_M * opts.delta_N
        self.C = opts.num_channels
        self.start_idx = start_idx # Stored for the purpose of computing the loss
        self.idx = copy.deepcopy(start_idx) # Current view of the state
        # Proprioception is [elevation, change in azimuth]
        # Whether elevation and azimuth are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        # Whether azimuth, elevation are known to the sensor or not
        self.knownElev = opts.knownElev
        self.knownAzim = opts.knownAzim
        # Whether to wrap around elevation and azimuths
        self.wrap_elevation = opts.wrap_elevation
        self.wrap_azimuth = opts.wrap_azimuth
        # delta_M is the number of azimuths available to rotate to, usually odd
        # delta_N is the number of elevations available to rotate to, usually odd
        if masks is None:
            self.hasmasks = False
        else:
            self.hasmasks = True
            self.masks = Variable(torch.Tensor(masks), requires_grad=False)
            if opts.iscuda:
                self.masks = self.masks.cuda()
            self.masks_sum = torch.sum(self.masks.view(views.shape[0], -1), dim=1)

        self.debug = opts.debug
        if self.debug: 
            # These are necessary for the next step
            assert(opts.delta_N % 2 == 1)
            assert(opts.delta_M % 2 == 1)
        # Decodes actions to the corresponding changes in elevation and azimuth
        self.act_to_delta = {}
        self.delta_to_act = {}
        count_act = 0
        for i in range(-(opts.delta_N//2), opts.delta_N//2+1):
            for j in range(-(opts.delta_M//2), opts.delta_M//2+1):
                self.act_to_delta[count_act] = (i, j)
                self.delta_to_act[(i, j)] = count_act
                count_act += 1
        
        # ---- Data settings ----
        self.batch_size = views.shape[0]
        self.delta = [[0, 0] for i in range(self.batch_size)] # Starts off with no change
        # Store mean and std to preprocess the views
        self.mean = opts.mean
        self.std = opts.std
        # total_pixels is useful for computing MSE
        self.total_pixels = self.M * self.N * views.shape[3] * views.shape[4] * views.shape[5] 
       
        # ---- Save panorama data to the state and preprocess ---- 
        self.views = views
        if views_rewards is not None:
            self.views_rewards = np.copy(views_rewards) # To create a copy
            self.has_rewards = True
        else:
            self.views_rewards = np.zeros((views.shape[0], views.shape[1], views.shape[2]))
            self.has_rewards = False

        # Compute preprocessed views
        if self.debug:
            # Ensure that the 4th idx is the channel
            assert(self.views.shape[3] == 1 or self.views.shape[3] == 3)
        self.views_prepro = preprocess_views(self.views, self.mean, self.std)
        # Shift each panorama in views_prepro according to corresponding start_idxes
        # Rotate the original views such that the 0th azimuth is the start azimuth (if azimuth
        # is not known to the agent) or the 0th elevation is the start elevation (if elevation 
        # is not known to the agent)
        self.views_prepro_shifted = np.zeros_like(self.views_prepro)
        for i in range(self.batch_size):
            if not (self.knownElev or self.knownAzim):
                # Shift the azimuth and elevation
                self.views_prepro_shifted[i] = np.roll(np.roll(self.views_prepro[i], -start_idx[i][1], axis=1), -start_idx[i][0], axis=0)
            elif not self.knownAzim:
                # Shift the azimuths
                self.views_prepro_shifted[i] = np.roll(self.views_prepro[i], -start_idx[i][1], axis=1)
            elif not self.knownElev:
                # Shift the elevations
                self.views_prepro_shifted[i] = np.roll(self.views_prepro[i], -start_idx[i][0], axis=0)

        # ---- Debug ----
        self.W = views.shape[5]
        self.H = views.shape[4]
        if self.debug:
            assert(self.views.shape == (self.batch_size, self.N, self.M, self.C, self.H, self.W))
            assert((self.views_prepro[0] == np.roll(self.views_prepro_shifted[0], start_idx[0][1], axis=1)).all())
            assert((self.views_prepro_shifted <= 1).all() and (self.views_prepro_shifted >= -1).all())
            assert(self.A == len(self.act_to_delta))
            assert(self.A == len(self.delta_to_act))
            for key in self.act_to_delta:
                assert((key < self.A) and (key >= 0)) 
            if self.hasmasks:
                assert(self.masks_sum.size() == torch.Size([self.batch_size]))

    def get_view(self, prepro=True):
        # Returns the current view and proprioception for each panorama
        # output view: BxCx32x32
        # output proprioception: list of [delta_elev, delta_azim, elev (optional), azim (optional)]
        
        pro_out = copy.deepcopy(self.delta)
        if self.knownElev or self.knownAzim:
            for i in range(len(pro_out)):
                if self.knownElev:
                    pro_out[i].append(self.idx[i][0])
                if self.knownAzim:
                    pro_out[i].append(self.idx[i][1])
 
        # Using python advanced indexing to get views for all panoramas simultaneously
        if prepro:
            return self.views_prepro[range(len(self.idx)), [i[0] for i in self.idx], [i[1] for i in self.idx]], \
                   pro_out
        else:
            return self.views[range(len(self.idx)), [i[0] for i in self.idx], [i[1] for i in self.idx]], \
                   pro_out 

    def rotate(self, act):
        # Rotates the state by delta corresponding to act. Returns the reward (intrinsic)
        # corresponding to this transition. 
        # act: tensor of integers between 0 to opts.delta_M * opts.delta_N
        # output reward: reward corresponding to visited view (optional)
        
        delta = [list(self.act_to_delta[act[i]]) for i in range(act.shape[0])]
        self.delta = delta
        if self.wrap_elevation and self.wrap_azimuth:
            self.idx = [[(self.idx[i][0] + delta[i][0])%self.N, (self.idx[i][1] + delta[i][1])%self.M] for i in range(self.batch_size)]
        elif self.wrap_elevation:
            self.idx = [[(self.idx[i][0] + delta[i][0])%self.N, max(min(self.idx[i][1] + delta[i][1], self.M-1), 0)] for i in range(self.batch_size)]
        elif self.wrap_azimuth:
            self.idx = [[max(min(self.idx[i][0] + delta[i][0], self.N-1), 0), (self.idx[i][1] + delta[i][1])%self.M] for i in range(self.batch_size)]
        else:
            self.idx = [[max(min(self.idx[i][0] + delta[i][0], self.N-1), 0), max(min(self.idx[i][1] + delta[i][1], self.M-1), 0)] for i in range(self.batch_size)]

        # After reaching the next state, return the reward for this transition
        # Collect rewards and then zero them out.
        rewards_copy = np.copy(self.views_rewards[range(len(self.idx)), [i[0] for i in self.idx], [i[1] for i in self.idx]])
        if self.has_rewards: # To save some compute time
            for i in range(len(self.idx)):
                for j in range(self.idx[i][0]-1, self.idx[i][0]+2):
                    for k in range(self.idx[i][1]-1, self.idx[i][1]+2):
                        self.views_rewards[i, j%self.N, k%self.M] = 0
        
        return rewards_copy

    def rec_loss(self, rec_views, iscuda):
        # Computes loss between self.views and rec_views with start_idx shift
        # rec_views: B x N x M x C x H x W torch Variable with preprocessed values
        # masks: B x N x M x C x H x W torch Variable 
        true_views = Variable(torch.Tensor(self.views_prepro_shifted))
        if iscuda:
            true_views = true_views.cuda()
        if not self.hasmasks: 
            return ((true_views - rec_views)**2).view(self.batch_size, -1).sum(dim=1)/(self.total_pixels)
        else:
            return (self.masks*(true_views - rec_views)**2).view(self.batch_size, -1).sum(dim=1)/self.masks_sum
