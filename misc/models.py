
from torch.autograd import Variable
from utils import *
from State import *

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import copy
import pdb

class Policy(nn.Module):
    # The overall policy class
    def __init__(self, opts):
        """
        Setup the settings for the policy network, debug, create the policy network
        components and initialize the network parameters:
        Panorama operation settings:
        (1) M, N, C, A (2) action input, agent operation parameters (known*, actOn*) 
        (3) Normalization
        Network settings:
        (1) iscuda (2) explorationFactor (3) lstm_hidden_size (4) baselineType (5) actorType
        """
        # ---- Settings for policy network ----
        super(Policy, self).__init__()
        # Panorama operation settings
        self.M = opts.M
        self.N = opts.N
        self.A = opts.A
        self.C = opts.num_channels
        # Whether elevation on azimuth are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        self.actOnTime = opts.actOnTime
        # Whether azimuth, elevation are known to the sensor or not
        self.knownElev = opts.knownElev
        self.knownAzim = opts.knownAzim
        # Normalization settings
        self.mean = opts.mean
        self.std = opts.std
        # Network settings
        self.iscuda = opts.iscuda
        self.explorationFactor = opts.explorationBaseFactor
        self.lstm_hidden_size = 256
        self.baselineType = opts.baselineType # Can be average or critic
        if not(opts.baselineType == 'critic' or opts.baselineType == 'average'):
            raise ValueError('baselineType %s does not exist!'%(opts.baselineType))
        self.act_full_obs = opts.act_full_obs
        self.critic_full_obs = opts.critic_full_obs
        self.actorType = opts.actorType
        # ---- Create the Policy Network ----
        # The input size for location embedding / proprioception stack
        input_size_loc = 2 # Relative camera position
        if self.knownAzim:
            input_size_loc += 1
        if self.knownElev:
            input_size_loc += 1
        
        # (1) Sense - image: Takes in BxCx32x32 image input and converts it to Bx256 matrix
        self.sense_im = nn.Sequential( # BxCx32x32
                            nn.Conv2d(self.C, 32, kernel_size=5, stride=1, padding=2), # Bx32x32x32
                            nn.MaxPool2d(kernel_size=3, stride=2), # Bx32x15x15
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2), # Bx32x15x15
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(kernel_size=3, stride=2), # Bx32x7x7
                            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # Bx64x7x7
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(kernel_size=3, stride=2), # Bx64x3x3
                            View(-1, 576),
                            nn.Linear(576, 256),
                            nn.ReLU(inplace=True)
                        )  
        
        # (2) Sense - proprioception stack: Converts proprioception inputs to 16-D vector
        self.sense_pro = nn.Sequential(
                            nn.Linear(input_size_loc, 16),
                            nn.ReLU(inplace=True)
                         )

        # (3) Fuse: Fusing the outputs of (1) and (2) to give 256-D vector per image
        self.fuse = nn.Sequential( # 256+16
                        nn.Linear(272, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 256), # Bx256
                        nn.BatchNorm1d(256) 
                    )
        
        # (4) Aggregator: View aggregating LSTM
        self.aggregate = nn.LSTM(input_size=256, hidden_size=self.lstm_hidden_size, num_layers=1)

        # (5) Act module: Takes in aggregator hidden state + other inputs to produce probability 
        #                 distribution over actions
        if self.actorType == 'actor':  
            if not self.act_full_obs:
                input_size_act = self.lstm_hidden_size + 2 # Add the relative positions
                # Optionally feed in elevation, azimuth
                if opts.actOnElev:
                    input_size_act += 1
                if opts.actOnAzim:
                    input_size_act += 1
                if opts.actOnTime:
                    input_size_act += 1
                self.act = nn.Sequential( # self.lstm_hidden_size
                                nn.Linear(input_size_act, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(128),
                                nn.Linear(128, self.A)
                           )
            else:
                # Fully observed actor
                input_size_act = self.lstm_hidden_size + 5 # delta_elev, delta_azim, elev, azim, time
                input_size_act += 256 # Panorama encoded
                self.act_fuse = nn.Sequential( # BNM x 256
                                            nn.Linear(256,128),
                                            nn.ReLU(inplace=True),
                                            View(-1, self.N*self.M*128),
                                            nn.Linear(self.N*self.M*128, 256),
                                            nn.BatchNorm1d(256),
                                      )

                self.act =  nn.Sequential( # self.lstm_hidden_size + 5 + 256
                                nn.Linear(input_size_act, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(128),
                                nn.Linear(128, self.A)
                           )
            # Assuming Critic not needed without an Actor
            # (5b) Critic module
            if opts.baselineType == 'critic':
                if not self.critic_full_obs:
                    # Partially observed critic
                    self.critic = nn.Sequential( # self.lstm_hidden_size
                                        nn.Linear(input_size_act, 128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 128),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(128),
                                        nn.Linear(128, 1)
                                   )
                else:
                    # Fully observed critic
                    input_size_critic = self.lstm_hidden_size + 5# delta_elev, delta_azim, elev, azim, time 
                    input_size_critic += 256 # Panorama encoded
                    self.critic_fuse = nn.Sequential( # BNM x 256
                                            nn.Linear(256,128),
                                            nn.ReLU(inplace=True),
                                            View(-1, self.N*self.M*128),
                                            nn.Linear(self.N*self.M*128, 256),
                                            nn.BatchNorm1d(256),
                                      )
                    self.critic = nn.Sequential( # self.lstm_hidden_size+5+256
                                        nn.Linear(input_size_critic, 128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 128),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(128),
                                        nn.Linear(128, 1)
                                   )

        # (6) Decode: Decodes aggregated views to panorama
        decode_layers = [
                            nn.BatchNorm1d(self.lstm_hidden_size),
                            nn.Linear(self.lstm_hidden_size, 1024),
                            nn.LeakyReLU(0.2, inplace=True),
                            View(-1, 64, 4, 4), # Bx64x4x4
                            nn.ConvTranspose2d(64, 256, kernel_size=5, stride=2, padding=2, output_padding=1), # Bx256x8x8
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), # Bx128x16x16
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.ConvTranspose2d(128, self.M*self.N*self.C, kernel_size=5, stride=2, padding=2, output_padding=1) # B x M*N*C x 32 x 32
                         ]

        if opts.mean_subtract: 
            # If mean subtraction is performed, output can be negative, hence LeakyRelu at the last layer
            decode_layers.append(nn.LeakyReLU(0.8, inplace=True))
        else:
            # If mean subtraction is not performed, output has to be positive, hence ReLU at the last layer
            decode_layers.append(nn.ReLU(inplace=True))
        self.decode = nn.Sequential(*decode_layers)

        # ---- Initialize parameters according to specified strategy ----
        if opts.init == 'xavier':
            init_strategy = ixvr
        elif opts.init == 'normal': 
            init_strategy = inrml
        else:
            init_strategy = iunf

        self.sense_im = initialize_sequential(self.sense_im, init_strategy)
        self.sense_pro = init_strategy(self.sense_pro)
        self.fuse = initialize_sequential(self.fuse, init_strategy)
        self.aggregate = init_strategy(self.aggregate)
        if self.actorType == 'actor':
            self.act = initialize_sequential(self.act, init_strategy)
            if self.act_full_obs:
                self.act_fuse = initialize_sequential(self.act_fuse, init_strategy)
            if self.baselineType == 'critic':
                self.critic = initialize_sequential(self.critic, init_strategy)
                if self.critic_full_obs:
                    self.critic_fuse = initialize_sequential(self.critic_fuse, init_strategy)

        self.decode = initialize_sequential(self.decode, init_strategy)
        
        # ---- Debug settings ----
        self.debug = opts.debug

    def forward(self, x, hidden=None):
        """
        x consists of the integer proprioception *pro* and the view *im*. It can optionally
        absolute elevation *elev*, absolute azimuth *azim* and time *time* as well. If the 
        critic is fully observed, the input must include the batch of full panoramas *pano*.
        
        It senses the image, proprioception and fuses them. The fused representation is
        aggregated to update the belief state about the panorama. This is used to predict
        the decoded panorama and take the next action. 
        """
        # ---- Setup the initial setup for forward propagation ----
        batch_size = x['im'].size(0)
        
        if hidden is None:
            hidden = [Variable(torch.zeros(1, batch_size, self.lstm_hidden_size)), # hidden state: num_layers x batch_size x hidden size
                      Variable(torch.zeros(1, batch_size, self.lstm_hidden_size))] # cell state  : num_layers x batch_size x hidden_size
            if self.iscuda:
                hidden[0] = hidden[0].cuda()
                hidden[1] = hidden[1].cuda()
        
        # ---- Sense the inputs ----
        xin = x
        x1 = self.sense_im(x['im'])
        x2 = self.sense_pro(x['pro'])
       
        if self.debug: 
            assert(x1.size() == torch.Size([batch_size, 256]))
            assert(x2.size() == torch.Size([batch_size, 16]))

        if self.actorType == 'actor':
            # ---- Create the inputs for the actor ----
            if self.actOnElev:
                xe = x['elev']
            if self.actOnAzim:
                xa = x['azim']
            if self.actOnTime:
                xt = x['time'] 

        x = torch.cat([x1, x2], dim=1)
        # ---- Fuse the representations ----
        x = self.fuse(x)
        
        if self.debug:
            assert(x.size() == torch.Size([batch_size, 256]))
        # ---- Update the belief state about the panorama ----
        # Note: input to aggregate lstm has to be seq_length x batch_size x input_dims
        # Since we are feeding in the inputs one by one, it is 1 x batch_size x 256
        x, hidden = self.aggregate(x.view(1, *x.size()), hidden)
        # Note: hidden[0] = h_n , hidden[1] = c_n
        if self.actorType == 'actor':
            if not self.act_full_obs:
                act_input = hidden[0].view(batch_size, -1)
                # Concatenate the relative change
                act_input = torch.cat([act_input, xin['pro'][:, :2]], dim=1)
                if self.actOnElev:
                    act_input = torch.cat([act_input, xe], dim=1)
                if self.actOnAzim:
                    act_input = torch.cat([act_input, xa], dim=1)
                if self.actOnTime:
                    act_input = torch.cat([act_input, xt], dim=1)
            else:
                # If fully observed actor
                act_input = hidden[0].view(batch_size, -1)
                # BxNxMx32x32 -> BNMx256 
                pano_encoded = self.sense_im(xin['pano'].view(-1, self.C, 32, 32))
                # BNMx256 -> Bx256
                pano_fused = self.act_fuse(pano_encoded)
                act_input = torch.cat([act_input, xin['pro'][:, :2], xin['elev'], \
                                       xin['azim'], xin['time'], pano_fused], dim=1)

            # ---- Predict the action propabilities ----
            probs = F.softmax(self.act(act_input)/(self.explorationFactor+1), dim=1)
            if self.debug:
                assert(probs.size() == torch.Size([batch_size, self.A]))

            # ---- Predict the value for the current state ----
            if self.baselineType == 'critic' and not self.critic_full_obs:
                values = self.critic(act_input).view(-1)
                if self.debug:
                    assert(values.size() == torch.Size([batch_size]))
            elif self.baselineType == 'critic' and self.critic_full_obs:
                critic_input = hidden[0].view(batch_size, -1)
                # BxNxMxCx32x32 -> BNMx256
                pano_encoded = self.sense_im(xin['pano'].view(-1, self.C, 32, 32))
                # BNMx256 ->  Bx256
                pano_fused = self.critic_fuse(pano_encoded)
                critic_input = torch.cat([critic_input, xin['pro'][:, :2], xin['elev'], \
                                          xin['azim'], xin['time'], pano_fused], dim=1)
                values = self.critic(critic_input).view(-1)
                if self.debug:
                    assert(values.size() == torch.Size([batch_size]))
            else:
                values = None
        else:
            probs = None
            act_input = None
            values = None

        # ---- Deocode the aggregated state ----
        # hidden[0] gives num_layer x batch_size x hidden_size , hence hidden[0][0] since 
        # only one hidden layer is used
        x = self.decode(F.normalize(hidden[0][0], p=1, dim=1))
        decoded = x.view(batch_size, self.N, self.M, self.C, 32, 32)
        
        return probs, decoded, hidden, values 

class Agent:
    """
    This agent implements the policy from Policy class and uses REINFORCE / Actor-Critic for policy improvement
    """
    def __init__(self, opts, mode='train'):
        """
        Creates the policy network, optimizer, initial settings, debug settings:
        Panorama operation settings:
        (1) T, C, M, N (2) action input, agent operation parameters (known*, actOn*)
        (3) memorize_views (4) normalization
        Optimization settings:
        (1) mode (2) baselineType (3) iscuda (4) average reward baselines (5) lr (6) weightDecay
        (7) scaling factors (critic_coeff, lambda_entropy, reward_scale, reward_scale_expert)
        """
        # ---- Create the policy network ----
        self.policy = Policy(opts) 
        # ---- Panorama operation settings ----
        self.C = opts.num_channels
        self.T = opts.T
        self.M = opts.M
        self.N = opts.N
        # Whether elevation, azimuth or time are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        self.actOnTime = opts.actOnTime
        # Whether azimuth, elevation are known to the sensor or not
        self.knownElev = opts.knownElev
        self.knownAzim = opts.knownAzim
        # Whether to memorize views or not
        self.memorize_views = opts.memorize_views
        # Preprocessing
        self.mean = opts.mean
        self.std = opts.std 
        # ---- Optimization settings ----
        self.mode = mode
        self.actorType = opts.actorType
        self.baselineType = opts.baselineType
        self.act_full_obs = opts.act_full_obs
        self.critic_full_obs = opts.critic_full_obs
        self.iscuda = opts.iscuda
        if opts.iscuda:
            self.policy = self.policy.cuda()
        # Average reward baselines
        if self.baselineType == 'average' and mode == 'train':
            # Set a baseline for REINFORCE
            self.R_avg = 0
            self.R_avg_expert = 0
            # Average counts maintained to update baselines
            self.avg_count = 0
            self.avg_count_expert = 0
        # Scaling factors
        if self.mode == 'train':
            self.critic_coeff = opts.critic_coeff
            self.lambda_entropy = opts.lambda_entropy # Entropy term coefficient
        self.reward_scale = opts.reward_scale
        self.reward_scale_expert = opts.reward_scale_expert
        # ---- Create the optimizer ----
        if self.mode == 'train':
            self.create_optimizer(opts.lr, opts.weight_decay)  
        # ---- Debug ----
        self.debug = opts.debug
        if self.debug:
            # Optimizer needed only for training
            if self.mode == 'train':
                assert hasattr(self, 'optimizer')
            else:
                assert not hasattr(self, 'optimizer')
            # Baselines needed only for training with REINFORCE
            if (self.mode == 'train' and self.baselineType == 'critic') or (self.mode == 'test'):
                assert not hasattr(self, 'R_avg')
                assert not hasattr(self, 'avg_count')
                assert not hasattr(self, 'R_avg_expert')
                assert not hasattr(self, 'avg_count_expert')

    def create_optimizer(self, lr, weight_decay, training_setting=0, fix_decode=False):
        # Can be used to create the optimizer
        # Refer main.py for training_setting
        if training_setting == 0 or training_setting == 2 or training_setting == 4:
            list_of_params = [{'params': self.policy.parameters()}]
        elif training_setting == 1 or training_setting == 3:
            list_of_params = [{'params': self.policy.aggregate.parameters()}]
            if not fix_decode: 
                list_of_params.append({'params': self.policy.decode.parameters()})
            if hasattr(self.policy, 'act'):
                list_of_params.append({'params': self.policy.act.parameters()})
                if self.act_full_obs:
                    list_of_params.append({'params': self.policy.act_fuse.parameters()})
                if hasattr(self.policy, 'critic'):
                    list_of_params.append({'params': self.policy.critic.parameters()})
                    if self.critic_full_obs:
                        list_of_params.append({'params': self.policy.critic_fuse.parameters()})
        self.optimizer = optim.Adam(list_of_params, lr=lr, weight_decay=weight_decay)

    def gather_trajectory(self, state_object, eval_opts=None, pano_maps=None, opts=None):
        """
        gather_trajectory gets an observation, updates it's belief of the state, decodes the
        panorama and takes the next action. This is done repeatedly for T time steps. 
        It returns the log probabilities, reconstruction errors, entropies, rewards, values
        and the visited views at every step of the trajectory. It also additionally returns
        the decoded images. 
        
        Note: eval_opts are provided only during testing, will not contribute
        to the training in any way
        Note: pano_maps, opts are provided only when the actor is demo_sidekick
        """
        # ---- Setup variables to store trajectory information ----
        rewards = []
        log_probs = []
        rec_errs = []
        entropies = []
        hidden = None
        visited_idxes = []
        batch_size = state_object.batch_size 
        decoded_all = []
        values = []
        actions_taken = torch.zeros(batch_size, self.T-1)

        if (self.baselineType == 'critic' and self.critic_full_obs) or (self.actorType == 'actor' and self.act_full_obs):
            pano_input = torch.Tensor(state_object.views_prepro)

        if self.actorType == 'demo_sidekick':
            start_idx = state_object.start_idx
            # ---- Get the expert planned trajectories ----
            selected_views = []
            for i in range(batch_size):
                selected_views.append([start_idx[i]])
            selected_views, target_actions = get_expert_trajectories(state_object, pano_maps, selected_views, opts)  
            if self.debug:
                assert((target_actions >= 0).all() and (target_actions < self.policy.A).all())
                assert(target_actions.shape == (batch_size, self.T-1))
                assert(len(selected_views) == batch_size)
                assert(len(selected_views[0]) == self.T)

        for t in range(self.T):
            # ---- Observe the panorama ----
            im, pro = state_object.get_view() 
            if self.debug:
                assert(len(im.shape) == 4)
                assert(im.shape == (batch_size, self.C, state_object.H, state_object.W))
                assert(len(pro) == batch_size)
                
            im, pro = preprocess(im, pro)
            if self.debug:
                assert(im.size() == torch.Size([batch_size, self.C, state_object.H, state_object.W]))
                assert(pro.size(0) == batch_size)

            # Keep track of visited locations
            visited_idxes.append(state_object.idx)
            # ---- Policy forward pass ----
            policy_input = {'im': im, 'pro': pro}
            # If critic or act have full observability, then elev, azim and time must be included in policy_input 
            # along with the batch of panoramas
            if (self.baselineType == 'critic' and self.critic_full_obs) or (self.actorType == 'actor' and self.act_full_obs):
                policy_input['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
                policy_input['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
                policy_input['time'] = torch.Tensor([[t] for i in range(batch_size)])
                policy_input['pano'] = pano_input 
            else:
                if self.actOnElev:
                    policy_input['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
                if self.actOnAzim:
                    policy_input['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
                if self.actOnTime:
                    policy_input['time'] = torch.Tensor([[t] for i in range(batch_size)])

            if self.iscuda:
                for var in policy_input:
                    policy_input[var] = policy_input[var].cuda()

            for var in policy_input:
                policy_input[var] = Variable(policy_input[var])
          
            # Note: decoded and hidden correspond to the previous transition, where a new state was visited
            # and the belief was updated. probs and value correspond to the new transition, where the value
            # and action probabilities of the current state are estimated for PG update. 
            probs, decoded, hidden, value = self.policy.forward(policy_input, hidden) 
            # ---- Memorize views ----
            # The 2nd condition allows to use memorize while validation only
            if self.memorize_views or (eval_opts is not None and eval_opts['memorize_views'] == True):
                # TODO: Using memorize during training may not work in this current version due to challenges in PyTorch
                # Have to fix this by updating gradients appropriately using well defined hooks. 
                for i in range(len(visited_idxes)):
                    for bno in range(batch_size):
                        # Shifting it to use appropriately for the decoded images and views_prepro_shifted
                        if not self.knownElev:
                            visited_ele = visited_idxes[i][bno][0] - state_object.start_idx[bno][0]
                        else:
                            visited_ele = visited_idxes[i][bno][0]
                        if not self.knownAzim: 
                            visited_azi = visited_idxes[i][bno][1] - state_object.start_idx[bno][1]
                        else:
                            visited_azi = visited_idxes[i][bno][1]

                        view_copied = Variable(torch.Tensor(state_object.views_prepro_shifted[bno][visited_ele][visited_azi]))
                        if self.iscuda:
                            view_copied = view_copied.cuda()
                        decoded[bno, visited_ele, visited_azi] = view_copied
                        
            decoded_all.append(decoded)
            
            # ---- Compute reconstruction loss (corresponding to the previous transition)----
            rec_err = state_object.rec_loss(decoded, self.iscuda)
            
            # Reconstruction reward is obtained only at the final step
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t < self.T-1 or t == 0:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            else:
                reward = -rec_err.data # Disconnects reward from future updates
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size
            if t > 0:
                rewards[t-1] += reward
            
            # There are self.T reconstruction errors as opposed to self.T-1 rewards
            rec_errs.append(rec_err)

            # ---- Sample action ----
            # except for the last time step when only the selected view from previous step is used in aggregate 
            if t < self.T - 1:
                if self.policy.actorType == 'actor': 
                    # Act based on the policy network
                    if eval_opts == None or eval_opts['greedy'] == False:
                        act = probs.multinomial(num_samples=1).data
                    else:
                        # This works only while evaluating, not while training
                        _, act = probs.max(dim=1)
                        act = act.data.view(-1, 1)
                    # Compute entropy
                    entropy = -(probs*((probs+1e-7).log())).sum(dim=1)
                    # Store log probabilities of selected actions (Advanced indexing)
                    log_prob = (probs[range(act.size(0)), act[:, 0]]+1e-7).log()
                
                elif self.policy.actorType == 'random':
                    # Act randomly
                    act = torch.Tensor(np.random.randint(0, self.policy.A, size=(batch_size, 1)))
                    log_prob = None
                    entropy = None
                
                elif self.policy.actorType == 'greedyLookAhead':
                    # Accumulate scores for each batch for every action
                    act_scores_ = torch.ones(batch_size, self.policy.A).fill_(10000)
                    # For each action, compute the next state, perform the policy forward pass and obtain
                    # reconstruction error. 
                    for a_iter in range(self.policy.A):
                        state_object_ = copy.deepcopy(state_object) 
                        _ = state_object_.rotate(torch.ones(batch_size).fill_(a_iter).int())
                        im_, pro_ = state_object_.get_view()
                        im_, pro_ = preprocess(im_, pro_)
                        policy_input_ = {'im': im_, 'pro': pro_}
                        # Assume no critic or proprioception, time inputs needed if greedyLookAhead is performed
                        
                        if self.iscuda:
                            for var in policy_input_:
                                policy_input_[var] = policy_input_[var].cuda()

                        for var in policy_input_:
                            policy_input_[var] = Variable(policy_input_[var])
                      
                        # Note: decoded and hidden correspond to the previous transition, where a new state was visited
                        # and the belief was updated. probs and value correspond to the new transition, where the value
                        # and action probabilities of the current state are estimated for PG update. 
                        _, decoded_, _, _ = self.policy.forward(policy_input_, hidden)

                        rec_err_ = state_object_.rec_loss(decoded_, self.iscuda)
                        act_scores_[:, a_iter] = rec_err_.data.cpu()
                    
                    _, act = torch.min(act_scores_, dim=1)
                    act = act.view(-1, 1)
                    log_prob = None
                    entropy = None

                elif self.policy.actorType == 'demo_sidekick':
                    act = torch.LongTensor(target_actions[:, t]).contiguous().view(-1, 1)
                    log_prob = None
                    entropy = None

                # ---- Rotate the view of the state and collect expert reward for this transition ----
                actions_taken[:, t] = act[:, 0]
                reward_expert = state_object.rotate(act[:, 0])
                if self.debug:
                    assert(reward_expert.shape == (batch_size,))

                reward_expert = torch.Tensor(reward_expert)
                if self.debug:
                    assert(reward_expert.size() == torch.Size([batch_size]))

                if self.iscuda: 
                    reward_expert = reward_expert.cuda()
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg_expert = (self.R_avg_expert * self.avg_count_expert + reward_expert.sum())/(self.avg_count_expert + batch_size)
                    self.avg_count_expert += batch_size
                
                # This is the intrinsic reward corresponding to the current action
                rewards.append(reward_expert*self.reward_scale_expert)
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
        
        if self.debug:
            assert(len(rec_errs) == self.T)
            assert(len(rewards) == self.T-1)
            assert(len(log_probs) == self.T-1)
            assert(len(entropies) == self.T-1)
            assert(len(values) == self.T-1)
            assert(len(decoded_all) == self.T)
            assert(len(visited_idxes) == self.T)

            for t in range(self.T):
                assert(rec_errs[t].size() == torch.Size([batch_size]))
                if t < self.T-1:
                    assert(rewards[t].size() == torch.Size([batch_size]))
                    if self.policy.actorType == 'actor':
                        assert(log_probs[t].size() == torch.Size([batch_size]))
                        assert(entropies[t].size() == torch.Size([batch_size]))
                        if self.baselineType == 'critic':
                            assert(values[t].size() == torch.Size([batch_size]))
                        else:
                            assert(values[t] is None)
                    if self.policy.actorType == 'random' or self.policy.actorType == 'greedyLookAhead' \
                    or self.policy.actorType == 'demo_sidekick':
                        assert(log_probs[t] is None)
                        assert(entropies[t] is None)
                        assert(values[t] is None)

        return log_probs, rec_errs, rewards, entropies, decoded, values, visited_idxes, decoded_all, actions_taken

    def update_policy(self, rewards, log_probs, rec_errs, entropies, values=None, start_idxes=None, decoded_all=None):
        """
        This function will take the rewards, log probabilities and reconstruction errors from
        the trajectory and perform the parameter updates for the policy using REINFORCE
        INPUTS: 
            rewards: list of T-1 Tensors containing reward for each batch at each time step
            log_probs: list of T-1 logprobs Variables of each transition of batch
            rec_errs: list of T reconstruction error Variables for each transition of batch
            entropies: list of T-1 entropy Variables for each transition of batch
            values: list of T-1 predicted values Variables for each transition of batch
            start_idxes: list of T start indices (needed for memorize)
            decoded_all: list of T decoded entries as Variables (needed for memorize)
        """
        #TODO: Implement gradient correction for memorize
        # ---- Setup initial values ----
        batch_size = rec_errs[0].size(0)
        R = torch.zeros(batch_size) # Reward accumulator
        B = 0 # Baseline accumulator - used primarily for the average baseline case
        loss = Variable(torch.Tensor([0]))
        if self.iscuda:
            loss = loss.cuda()
            R = R.cuda()
        # ---- Reconstruction error based loss computation
        for t in reversed(range(self.T)):
            loss = loss + rec_errs[t].sum()/batch_size

        # --- REINFORCE / Actor-Critic loss based on T-1 transitions ----
        # Note: This will automatically be ignored when self.T = 1
        for t in reversed(range(self.T-1)):
            if self.policy.actorType == 'actor':
                R = R + rewards[t] # A one sample MC estimate of Q[t]
                # Compute the advantage
                if self.baselineType == 'critic':
                    adv = R - values[t].data 
                else:
                    # B - an estimate of V[t] when no critic is present. Equivalent to subtracting the average
                    # rewards at each time step which was done in the previous versions of the code.
                    if t == self.T-2:
                        B += self.R_avg
                    B += self.R_avg_expert * self.reward_scale_expert
                    adv = R - B
                # PG loss
                loss_term_1 = - (log_probs[t]*self.reward_scale*Variable(adv, requires_grad=False)).sum()/batch_size 
                # Entropy loss, maximize entropy
                loss_term_2 = - self.lambda_entropy*entropies[t].sum()/batch_size
                # Critic prediction error
                if self.baselineType == 'critic':
                    loss_term_3 = self.critic_coeff*((Variable(R, requires_grad=False) - values[t])**2).sum()/batch_size
                else:
                    loss_term_3 = 0
                
                loss = loss + loss_term_1 + loss_term_2 + loss_term_3
                
        self.optimizer.zero_grad()
        loss.backward()
    
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()

class AgentSupervised:
    """
    This agent implements the policy from Policy class and uses Supervised Learning to learn the policy
    """
    def __init__(self, opts, mode='train'):
        """
        Creates the policy network, supervised criterion, optimizer, initial settings, debug settings:
        Panorama operation settings:
        (1) T, C, M, N (2) action input, agent operation parameters (known*, actOn*)
        (3) memorize_views (4) normalization
        Optimization settings:
        (1) mode (2) iscuda (3) lr (4) weightDecay (5) scaling factors
        """
        # ---- Create the policy network ----
        self.policy = Policy(opts) 
        
        # ---- NLL loss criterion for policy update ----
        self.criterion = nn.NLLLoss()
        
        # ---- Panorama operation settings ----
        self.T = opts.T # Max number of steps to take
        self.C = opts.num_channels
        self.M = opts.M
        self.N = opts.N
        # Whether elevation, azimuth or time are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        self.actOnTime = opts.actOnTime
        # Whether azimuth, elevation are known to the sensor or not
        self.knownElev = opts.knownElev
        self.knownAzim = opts.knownAzim
        # Whether to memorize views or not
        self.memorize_views = opts.memorize_views
        # Normalization
        self.mean = opts.mean
        self.std = opts.std
        
        # ---- Optimization settings ----
        self.mode = mode
        self.iscuda = opts.iscuda
        if opts.iscuda:
            self.policy = self.policy.cuda()
            self.criterion = self.criterion.cuda()
        #self.reward_estimator = opts.reward_estimator
        self.supervised_scale = opts.supervised_scale
        self.baselineType = opts.baselineType
        self.trajectories_type = opts.trajectories_type
        
        if self.baselineType == 'average' and mode == 'train':
	    # Set a baseline for REINFORCE
            self.R_avg = 0
            self.R_avg_expert = 0
            # Average counts maintained to update baselines
            self.avg_count = 0
            self.avg_count_expert = 0
	# Scaling factors
        if self.mode == 'train':
            self.critic_coeff = opts.critic_coeff
            self.lambda_entropy = opts.lambda_entropy # Entropy term coefficient
        self.T_sup = opts.T_sup
	self.reward_scale = opts.reward_scale
        self.reward_scale_expert = opts.reward_scale_expert 
        # ---- Create the optimizer ----
        if self.mode == 'train':
            self.create_optimizer(opts.lr, opts.weight_decay)  

        # ---- Debug ----
        self.debug = opts.debug
        if self.debug:
            self.C = opts.num_channels
            self.M = opts.M
            self.N = opts.N
            # Optimizer needed only for training
            if self.mode == 'train':
                assert hasattr(self, 'optimizer')
            else:
                assert not hasattr(self, 'optimizer')
            
            # Ensure that no critic is used
            assert self.baselineType == 'average'
	    # Average baselines needed only for training with REINFORCE
	    if (self.mode == 'train' and self.baselineType == 'critic') or (self.mode == 'test'):
                assert not hasattr(self, 'R_avg')
                assert not hasattr(self, 'avg_count')
                assert not hasattr(self, 'R_avg_expert')
                assert not hasattr(self, 'avg_count_expert')

    def create_optimizer(self, lr, weight_decay, training_setting=0, fix_decode=False):
        # Can be used to create the optimizer
        # Refer main.py for training_setting
        if training_setting == 0 or training_setting == 2 or training_setting == 4:
            list_of_params = [{'params': self.policy.parameters()}]
        elif training_setting == 1 or training_setting == 3:
            list_of_params = [{'params': self.policy.aggregate.parameters()}]
            if not fix_decode: 
                list_of_params.append({'params': self.policy.decode.parameters()})
            if hasattr(self.policy, 'act'):
                list_of_params.append({'params': self.policy.act.parameters()})
                if hasattr(self.policy, 'critic'):
                    list_of_params.append({'params': self.policy.critic.parameters()})
        
        self.optimizer = optim.Adam(list_of_params, lr=lr, weight_decay=weight_decay)

    def train_agent(self, state_object, pano_maps, opts, eval_opts=None):
        """
        train_agent gets a batch of panoramas and the optimal trajectories to take 
        for these observations. The function forward propagates the trajectory through the
        policy, gets the actions selected at each step and reconstruction losses. It further 
        updates the policy based on the loss. 
         
        Note: eval_opts are provided only during testing, will not contribute
        to the training in any way
        """
        batch_size = state_object.batch_size 
        start_idx = state_object.start_idx
        # ---- Get the expert planned trajectories ----
        if opts.trajectories_type == 'utility_maps':
            selected_views = []
            for i in range(batch_size):
                selected_views.append([start_idx[i]])
            selected_views, target_actions = get_expert_trajectories(state_object, pano_maps, selected_views, opts)  
            if self.debug:
                assert((target_actions >= 0).all() and (target_actions < self.policy.A).all())
                assert(target_actions.shape == (batch_size, self.T-1))
                assert(len(selected_views) == batch_size)
                assert(len(selected_views[0]) == self.T)
        else:
            target_actions = torch.cat([pano_maps[tuple(start_idx[i])][i, :].view(1, -1) for i in range(batch_size)], dim=0).numpy()

        # target_actions: B x T-1 array of integers between (0, self.policy.A-1) 
        # ---- Setup variables to store trajectory information ----
        probs_all = []
        rec_errs = []
        hidden = None
        decoded_all = []
        visited_idxes = []
        # ---- Forward propagate trajectories through the policy ----
        for t in range(self.T):
            # Observe the panorama
            im, pro = state_object.get_view() 
            if self.debug:
                assert(len(im.shape) == 4)
                assert(im.shape == (batch_size, self.C, state_object.H, state_object.W))
                assert(len(pro) == batch_size)
        
            im, pro = preprocess(im, pro)
            if self.debug:
                assert(im.size() == torch.Size([batch_size, self.C, state_object.H, state_object.W]))
                assert(pro.size(0) == batch_size)
            
            # Keep track of visited locations
            visited_idxes.append(state_object.idx)
            # Policy forward pass 
            policy_input = {'im': im, 'pro': pro}
            if self.actOnElev:
                policy_input['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
            if self.actOnAzim:
                policy_input['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
            if self.actOnTime:
                policy_input['time'] = torch.Tensor([[t] for i in range(batch_size)])

            if self.iscuda:
                for var in policy_input:
                    policy_input[var] = policy_input[var].cuda()

            for var in policy_input:
                policy_input[var] = Variable(policy_input[var])
           
            probs, decoded, hidden, value = self.policy.forward(policy_input, hidden) 
            # ---- Memorize views ----
            # The 2nd condition allows to use memorize while validation only
            if self.memorize_views or (eval_opts is not None and eval_opts['memorize_views'] == True):
                # TODO: Using memorize during training may not work in this current version due to challenges in PyTorch
                # Have to fix this by updating gradients appropriately using well defined hooks. 
                for i in range(len(visited_idxes)):
                    for bno in range(batch_size):
                        visited_ele = visited_idxes[i][bno][0]
                        # Shifting it to use appropriately for the decoded images and views_prepro_shifted
                        if not self.knownElev:
                            visited_ele = visited_idxes[i][bno][0] - state_object.start_idx[bno][0]
                        else:
                            visited_ele = visited_idxes[i][bno][0]
                        if not self.knownAzim: 
                            visited_azi = visited_idxes[i][bno][1] - state_object.start_idx[bno][1]
                        else:
                            visited_azi = visited_idxes[i][bno][1]

                        view_copied = Variable(torch.Tensor(state_object.views_prepro_shifted[bno][visited_ele][visited_azi]))
                        if self.iscuda:
                            view_copied = view_copied.cuda()
                        decoded[bno, visited_ele, visited_azi] = view_copied
                        
            decoded_all.append(decoded)

            # ---- Compute reconstruction loss ----
            rec_err = state_object.rec_loss(decoded, self.iscuda)

            # ---- Sample action, except in the last step ----
            if t < self.T-1:
                act = target_actions[:, t] 
                # ---- Rotate the view of the state ----
                _ = state_object.rotate(act)
                probs_all.append(probs) 
            
            rec_errs.append(rec_err)
        
        if self.debug:
            for t in range(self.T):
                assert(rec_errs[t].size() == torch.Size([batch_size]))
                if t < self.T-1:
                    assert(probs_all[t].size() == torch.Size([batch_size, self.policy.A]))
        # ---- Update the policy ----
        loss = 0
        for t in range(self.T):
            loss = loss + rec_errs[t].sum()/batch_size
            if t < self.T-1:
                targets = Variable(torch.LongTensor(target_actions[:, t]), requires_grad=False)
                if self.iscuda:
                    targets = targets.cuda()
                loss = loss + self.criterion((probs_all[t]+1e-8).log(), targets)*self.supervised_scale 
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()
        
        return rec_errs

    def train_agent_hybrid(self, state_object, pano_maps, opts, eval_opts=None):
        """
        train_agent_hybrid combines both supervised and unsupervised training schemes. It 
        uses the expert agent for first K time steps and then acts based on it's own policy.
        The rewards are used in PG updates for the actions based on the policy and 
        supervised updates are used for the actions taken based on the expert agent.

        Note: eval_opts are provided only during testing, will not contribute
        to the training in any way
        """
        batch_size = state_object.batch_size 
        start_idx = state_object.start_idx
        # ---- Get the expert planned trajectories ----
        if opts.trajectories_type == 'utility_maps':
            selected_views = []
            for i in range(batch_size):
                selected_views.append([start_idx[i]])
            selected_views, target_actions = get_expert_trajectories(state_object, pano_maps, selected_views, opts)  
            if self.debug:
                assert((target_actions >= 0).all() and (target_actions < self.policy.A).all())
                assert(target_actions.shape == (batch_size, self.T-1))
                assert(len(selected_views) == batch_size)
                assert(len(selected_views[0]) == self.T)
        else:
            target_actions = torch.cat([pano_maps[tuple(start_idx[i])][i, :].view(1, -1) for i in range(batch_size)], dim=0).numpy()

        # ---- Forward propagate the above trajectories through policy network and update the policy----
        # ---- Setup variables to store trajectory information ----
        probs_all = []
        log_probs_all = []
        rec_errs = []
        values = []
        hidden = None
        visited_idxes = []
        rewards = []
        entropies = []
        decoded_all = []

        T_sup = self.T_sup  

        # ---- Run the hybrid trajectory collection ----
        for t in range(self.T):
            # ---- Observe the panorama ----
            # Assuming actorType == 'actor' 
            im, pro = state_object.get_view()
            if self.debug:
                assert(len(im.shape) == 4)
                assert(im.shape == (batch_size, self.C, state_object.H, state_object.W))
                assert(len(pro) == batch_size) 
            im, pro = preprocess(im, pro)
            if self.debug:
                assert(im.size() == torch.Size([batch_size, self.C, state_object.H, state_object.W]))
                assert(pro.size(0) == batch_size)           
 
            # Store the visited locations
            visited_idxes.append(state_object.idx)
            
            # ---- Create the policy input ----
            input_policy = {'im': im, 'pro': pro}
            
            if self.actOnElev:
                input_policy['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
            if self.actOnAzim:
                input_policy['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
            if self.actOnTime:
                input_policy['time'] = torch.Tensor([[t] for i in range(batch_size)])

            if self.iscuda:
                for x in input_policy:
                    input_policy[x] = input_policy[x].cuda()
            
            for x in input_policy:
                input_policy[x] = Variable(input_policy[x])

            probs, decoded, hidden, value = self.policy.forward(input_policy, hidden)
            # ---- Memorize views ----
            if self.memorize_views:
                # TODO: Using memorize during training may not work in this current version due to challenges in PyTorch
                # Have to fix this by updating gradients appropriately using well defined hooks. 
                for i in range(len(visited_idxes)):
                    for bno in range(batch_size):
                        visited_ele = visited_idxes[i][bno][0]
                        # Shifting it to use appropriately for the decoded images and views_prepro_shifted
                        if not self.knownElev:
                            visited_ele = visited_idxes[i][bno][0] - state_object.start_idx[bno][0]
                        else:
                            visited_ele = visited_idxes[i][bno][0]
                        if not self.knownAzim: 
                            visited_azi = visited_idxes[i][bno][1] - state_object.start_idx[bno][1]
                        else:
                            visited_azi = visited_idxes[i][bno][1]

                        view_copied = Variable(torch.Tensor(state_object.views_prepro_shifted[bno][visited_ele][visited_azi]))
                        if self.iscuda:
                            view_copied = view_copied.cuda()
                        decoded[bno, visited_ele, visited_azi] = view_copied
                        
            decoded_all.append(decoded)
            # ---- Compute reconstruction loss (corresponding to previous transition) ----
            rec_err = state_object.rec_loss(decoded, self.iscuda)
                
            # Reconstruction reward is obtained only at the final step
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t < self.T-1 or t == 0:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            else:
                reward = -rec_err.data # Disconnect rewards from future updates
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size

            if t > 0:
                rewards[t-1] += reward

            # There are self.T reconstruction errors as opposed to self.T-1 rewards
            rec_errs.append(rec_err)

            # ---- Sample action ----
            # except for the last time step when only the selected view from previous step is used in aggregate
            if t < self.T - 1:
                if t < T_sup:
                    # Act according to the supervised agent
                    act = target_actions[:, t]
                    log_prob = None
                    entropy = None
                else:
                    # Act based on the policy network
                    act = probs.multinomial(num_samples=1).data[:, 0]
                    # Compute entropy
                    entropy = -(probs*((probs+1e-7).log())).sum(dim=1)
                    # Store log probabilities of selected actions (Advanced indexing_
                    log_prob = (probs[range(act.size(0)), act]+1e-7).log()
		
                # ---- Rotate the view of the state ----
                reward_expert = state_object.rotate(act)
                if self.debug:
                    assert(reward_expert.shape == (batch_size,))
		
                reward_expert = torch.Tensor(reward_expert)
                if self.iscuda:
                    reward_expert = reward_expert.cuda()
		
                if self.baselineType == 'average':	
                    self.R_avg_expert = (self.R_avg_expert * self.avg_count_expert + reward_expert.sum())/(self.avg_count_expert + batch_size)
                    self.avg_count_expert += batch_size
                
                # This is the intrinsic reward corresponding to the current action
                rewards.append(reward_expert*self.reward_scale_expert)
                log_probs_all.append(log_prob)
                entropies.append(entropy)
                probs_all.append(probs)
                values.append(value)
        
        if self.debug:
            assert(len(rec_errs) == self.T)
            assert(len(rewards) == self.T-1)
            assert(len(probs_all) == self.T-1)
            assert(len(log_probs_all) == self.T-1)
            assert(len(entropies) == self.T-1)
            assert(len(values) == self.T-1)
            assert(len(decoded_all) == self.T)

            for t in range(self.T):
                assert(rec_errs[t].size() == torch.Size([batch_size]))
                if t < self.T-1:
                    assert(rewards[t].size() == torch.Size([batch_size]))
                    assert(probs_all[t].size() == torch.Size([batch_size, self.policy.A]))
                    if t >= T_sup:
                        assert(log_probs_all[t].size() == torch.Size([batch_size]))
                        assert(entropies[t].size() == torch.Size([batch_size]))
                    if self.baselineType == 'critic':
                        assert(values[t].size() == torch.Size([batch_size]))
        
        # ---- Update the policy ----
        R = torch.zeros(batch_size) # Reward accumulator
        B = 0 # Baseline accumulator - used primarily for the average baseline case
        loss = 0
        if self.iscuda:
            R = R.cuda()
        
        # ---- Reconstruction error based loss computation ----
        for t in reversed(range(self.T)):
            loss = loss + rec_errs[t].sum()/batch_size

        # ---- REINFORCE / Actor-Critic / Supervised loss based on T-1 transitions
        for t in reversed(range(self.T-1)):
            if t < T_sup:
                targets = Variable(torch.LongTensor(target_actions[:, t]), requires_grad=False)
                if self.iscuda:
                    targets = targets.cuda()
                loss = loss + self.criterion((probs_all[t]+1e-7).log(), targets)*self.supervised_scale
            elif t < self.T-1:
                R = R + rewards[t] # A one sample MC estimate of Q[t]
                # Compute the advantage
                if self.baselineType == 'critic':
                    adv = R - values[t].data
                else:
                    # B - an estimate of V[t] when no critic is present. Equivalent to subtracting
                    # the average  rewards at each time.
                    if t == self.T-2:
                        B += self.R_avg
                    B += self.R_avg_expert * self.reward_scale_expert
                    adv = R - B

                loss_term_1 = -(log_probs_all[t]*self.reward_scale*Variable(adv, requires_grad=False)).sum()/batch_size # PG loss
                loss_term_2 = -self.lambda_entropy*entropies[t].sum()/batch_size # Entropy loss, maximize entropy
                # Critic prediction error
                if self.baselineType == 'critic':
                    loss_term_3 = self.critic_coeff*((Variable(R, requires_grad=False)-values[t])**2).sum()/batch_size
                else:
                    loss_term_3 = 0

                loss = loss + loss_term_1 + loss_term_2 + loss_term_3

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()

        return rec_errs

    def train_agent_hybrid_inv(self, state_object, pano_maps, opts, eval_opts=None):
        """
        train_agent_dagger combines both supervised and unsupervised training schemes. It 
        uses it's own policy for first K time steps and then acts based on expert. This is
        meant to resemble DAgger. 
        The rewards are used in PG updates for the actions based on the policy and 
        supervised updates are used for the actions taken based on the expert agent.
        Note: The PG updates kick in only when supervision stops / intrinsic rewards
        are provided in the middle. The reconstruction loss is still provided only at
        the final step. Till the agent actually uses the policy for the final step, 
        the agent is purely updated based on supervision or intrinsic rewards + supervision.
        """
       	# ---- Setup variables to store trajectory information ----
        batch_size = state_object.batch_size 
        probs_all = []
        log_probs_all = []
        rec_errs = []
        values = []
        hidden = None
        visited_idxes = []
        rewards = []
        entropies = []
        decoded_all = []
        target_actions = None
        T_sup = self.T_sup
        T_rl = self.T-T_sup-1

        for t in range(opts.T):
            im, pro = state_object.get_view ()
            if self.debug:
                assert(len(im.shape) == 4)
                assert(im.shape == (batch_size, self.C, state_object.H, state_object.W))
                assert(len(pro) == batch_size)
        
            im, pro = preprocess(im, pro)
            if self.debug:
                assert(im.size() == torch.Size([batch_size, self.C, state_object.H, state_object.W]))
                assert(pro.size(0) == batch_size)
	
            # Store the visited locations
            visited_idxes.append(state_object.idx)

            # ---- Create the policy input ----
            input_policy = {'im': im, 'pro': pro}

            if self.actOnElev:
                input_policy['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
            if self.actOnAzim:
                input_policy['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
            if self.actOnTime:
                input_policy['time'] = torch.Tensor([[t] for i in range(batch_size)])

            if self.iscuda:
                for x in input_policy:
                    input_policy[x] = input_policy[x].cuda()

            for x in input_policy:
                input_policy[x] = Variable(input_policy[x]) 
            
            probs, decoded, hidden, value = self.policy.forward(input_policy, hidden)
	    # ---- Memorize views ----
            if self.memorize_views:
                # TODO: Using memorize during training may not work in this current version due to challenges in PyTorch
                # Have to fix this by updating gradients appropriately using well defined hooks.
                for i in range(len(visited_idxes)):
                    for bno in range(batch_size):
                        visited_ele = visited_idxes[i][bno][0]
                        # Shifting it to use appropriately for the decoded images and views_prepro_shifted
                        if not self.knownElev:
                            visited_ele = visited_idxes[i][bno][0] - state_object.start_idx[bno][0]
                        else:
                            visited_ele = visited_idxes[i][bno][0]
                        if not self.knownAzim:
                            visited_azi = visited_idxes[i][bno][1] - state_object.start_idx[bno][1]
                        else:
                            visited_azi = visited_idxes[i][bno][1]

                        view_copied = Variable(torch.Tensor(state_object.views_prepro_shifted[bno][visited_ele][visited_azi]))
                        if self.iscuda:
                            view_copied = view_copied.cuda()
                        decoded[bno, visited_ele, visited_azi] = view_copied

            decoded_all.append(decoded)
            # ---- Compute reconstruction loss (corresponding to previous transition) ----
            rec_err = state_object.rec_loss(decoded, self.iscuda)

            # Reconstruction reward is obtained only at the final step of the policy
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t == T_rl and t > 0:
                reward = -rec_err.data # Disconnect rewards from future updates
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size
            else:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            
            if t > 0:
                rewards[t-1] += reward

            # There are self.T reconstruction errors as opposed to self.T-1 rewards
            rec_errs.append(rec_err)
            # ---- Sample action ----
            # except for the last time step when only the selected view from previous step is used in aggregate
            if t < self.T - 1:
                if t < T_rl:
                    # Act based on the policy network:
		    act = probs.multinomial(num_samples=1).data[:, 0]
                    # Compute entropy
                    entropy = -(probs*((probs+1e-7).log())).sum(dim=1)
                    # Store log probabilities of selected actions (Advanced indexing_
                    log_prob = (probs[range(act.size(0)), act]+1e-7).log() 
                else:
                    if t == T_rl:
                        selected_views = []
                        for i in range(batch_size):
                            selected_views.append([])
                            for j in range(len(visited_idxes)):
                                selected_views[i].append(visited_idxes[j][i])
                        _, target_actions = get_expert_trajectories(state_object, pano_maps, selected_views, opts) 
                        if self.debug:
                            assert(target_actions.shape == (batch_size, T_sup))
                    #pdb.set_trace()
                    act = target_actions[:, t-T_rl]
                    log_prob = None
                    entropy = None

		# ---- Rotate the view of the state ----
                reward_expert = state_object.rotate(act)
                if self.debug:
                    assert(reward_expert.shape == (batch_size,))

                reward_expert = torch.Tensor(reward_expert)
                if self.iscuda:
                    reward_expert = reward_expert.cuda()

                if self.baselineType == 'average':
                    self.R_avg_expert = (self.R_avg_expert * self.avg_count_expert + reward_expert.sum())/(self.avg_count_expert + batch_size)
                    self.avg_count_expert += batch_size

                # This is the intrinsic reward corresponding to the current action
                rewards.append(reward_expert*self.reward_scale_expert)
                log_probs_all.append(log_prob)
                entropies.append(entropy)
                probs_all.append(probs)
                values.append(value) 

        if self.debug:
            assert(len(rec_errs) == self.T)
            assert(len(rewards) == self.T-1)
            assert(len(probs_all) == self.T-1)
            assert(len(log_probs_all) == self.T-1)
            assert(len(entropies) == self.T-1)
            assert(len(values) == self.T-1)

            for t in range(self.T):
                assert(rec_errs[t].size() == torch.Size([batch_size]))
                if t < self.T-1:
                    assert(rewards[t].size() == torch.Size([batch_size]))
                    if self.policy.actorType == 'actor':
                        assert(probs_all[t].size() == torch.Size([batch_size, self.policy.A]))
                        if t < T_rl:
                            assert(log_probs_all[t].size() == torch.Size([batch_size]))
                            assert(entropies[t].size() == torch.Size([batch_size]))
                        if self.baselineType == 'critic':
                            assert(values[t].size() == torch.Size([batch_size]))    

        # ---- Update the policy ----
        R = torch.zeros(batch_size) # Reward accumulator
        B = 0 # Baseline accumulator - used primarily for the average baseline case
        loss = 0
        if self.iscuda:
            R = R.cuda()

        # ---- Reconstruction error based loss computation ----
        for t in reversed(range(self.T)):
            loss = loss + rec_errs[t].sum()/batch_size

        # ---- REINFORCE / Actor-Critic / Supervised loss based on T-1 transitions ----
        for t in reversed(range(self.T-1)):
            if t < T_rl:
                R = R + rewards[t] # A one sample MC estimate of Q[t]
                # Compute the advantage
                if self.baselineType == 'critic':
                    adv = R - values[t].data
                else:
                    # B - an estimate of V[t] when no critic is present. Equivalent to subtracting
                    # the average  rewards at each time.
                    if t == self.T-2:
                        B += self.R_avg
                    B += self.R_avg_expert * self.reward_scale_expert
                    adv = R - B

                loss_term_1 = -(log_probs_all[t]*self.reward_scale*Variable(adv, requires_grad=False)).sum()/batch_size # PG loss
                loss_term_2 = -self.lambda_entropy*entropies[t].sum()/batch_size # Entropy loss, maximize entropy
                # Critic prediction error
                if self.baselineType == 'critic':
                    loss_term_3 = self.critic_coeff*((Variable(R, requires_grad=False)-values[t])**2).sum()/batch_size
                else:
                    loss_term_3 = 0

                loss = loss + loss_term_1 + loss_term_2 + loss_term_3
 
            elif t < self.T-1:
                targets = Variable(torch.LongTensor(target_actions[:, t-T_rl]), requires_grad=False)

                if self.iscuda:
                    targets = targets.cuda()
                loss = loss + self.criterion((probs_all[t]+1e-7).log(), targets)*self.supervised_scale                
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()

        return rec_errs

    def gather_trajectory(self, state_object, eval_opts=None):
        """
        gather_trajectory gets an observation, updates it's model of the state, decodes the
        panorama and takes the next action. This is done repeatedly for T time steps. 
        It returns the log probabilities, reconstruction errors, entropies, rewards, values
        and the visited views at every step of the trajectory. It also additionally returns
        the decoded images. 
        
        Note: eval_opts are provided only during testing, will not contribute
        to the training in any way
        """
        # TODO: Need to incorporate handling fully observable critic and actor here.
        # ---- Setup variables to store trajectory information ----
        rewards = []
        log_probs = []
        rec_errs = []
        entropies = []
        hidden = None
        visited_idxes = []
        batch_size = state_object.batch_size 
        decoded_all = []
        values = []
        actions_taken = torch.zeros(batch_size, self.T-1)

        for t in range(self.T):
            # ---- Observe the panorama ----
            im, pro = state_object.get_view() 
            if self.debug:
                assert(len(im.shape) == 4)
                assert(im.shape == (batch_size, self.C, state_object.H, state_object.W))
                assert(len(pro) == batch_size)
                
            im, pro = preprocess(im, pro)
            if self.debug:
                assert(im.size() == torch.Size([batch_size, self.C, state_object.H, state_object.W]))
                assert(pro.size(0) == batch_size)

            # Keep track of visited locations
            visited_idxes.append(state_object.idx)
            # ---- Policy forward pass ----
            policy_input = {'im': im, 'pro': pro}
            if self.actOnElev:
                policy_input['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
            if self.actOnAzim:
                policy_input['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
            if self.actOnTime:
                policy_input['time'] = torch.Tensor([[t] for i in range(batch_size)])

            if self.iscuda:
                for var in policy_input:
                    policy_input[var] = policy_input[var].cuda()

            for var in policy_input:
                policy_input[var] = Variable(policy_input[var])
          
            # Note: decoded and hidden correspond to the previous transition, where a new state was visited
            # and the belief was updated. probs and value correspond to the new transition, where the value
            # and action probabilities of the current state are estimated for PG update. 
            probs, decoded, hidden, value = self.policy.forward(policy_input, hidden) 
            # ---- Memorize views ----
            # The 2nd condition allows to use memorize while validation only
            if self.memorize_views or (eval_opts is not None and eval_opts['memorize_views'] == True):
                # TODO: Using memorize during training may not work in this current version due to challenges in PyTorch
                # Have to fix this by updating gradients appropriately using well defined hooks. 
                for i in range(len(visited_idxes)):
                    for bno in range(batch_size):
                        # Shifting it to use appropriately for the decoded images and views_prepro_shifted
                        if not self.knownElev:
                            visited_ele = visited_idxes[i][bno][0] - state_object.start_idx[bno][0]
                        else:
                            visited_ele = visited_idxes[i][bno][0]
                        if not self.knownAzim: 
                            visited_azi = visited_idxes[i][bno][1] - state_object.start_idx[bno][1]
                        else:
                            visited_azi = visited_idxes[i][bno][1]

                        view_copied = Variable(torch.Tensor(state_object.views_prepro_shifted[bno][visited_ele][visited_azi]))
                        if self.iscuda:
                            view_copied = view_copied.cuda()
                        decoded[bno, visited_ele, visited_azi] = view_copied
                        
            decoded_all.append(decoded)
            
            # ---- Compute reconstruction loss (corresponding to the previous transition)----
            rec_err = state_object.rec_loss(decoded, self.iscuda)
            
            # Reconstruction reward is obtained only at the final step
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t < self.T-1 or t == 0:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            else:
                reward = -rec_err.data # Disconnects reward from future updates
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size
            if t > 0:
                rewards[t-1] += reward
            
            # There are self.T reconstruction errors as opposed to self.T-1 rewards
            rec_errs.append(rec_err)

            # ---- Sample action ----
            # except for the last time step when only the selected view from previous step is used in aggregate 
            if t < self.T - 1:
                if self.policy.actorType == 'actor': 
                    # Act based on the policy network
                    if eval_opts == None or eval_opts['greedy'] == False:
                        act = probs.multinomial(num_samples=1).data
                    else:
                        # This works only while evaluating, not while training
                        _, act = probs.max(dim=1)
                        act = act.data.view(-1, 1)
                    # Compute entropy
                    entropy = -(probs*((probs+1e-7).log())).sum(dim=1)
                    # Store log probabilities of selected actions (Advanced indexing)
                    log_prob = (probs[range(act.size(0)), act[:, 0]]+1e-7).log()
                
                elif self.policy.actorType == 'random':
                    # Act randomly
                    act = torch.Tensor(np.random.randint(0, self.policy.A, size=(batch_size, 1)))
                    log_prob = None
                    entropy = None
            
                # ---- Rotate the view of the state and collect expert reward for this transition ----
                actions_taken[:, t] = act[:, 0]
                reward_expert = state_object.rotate(act[:, 0])
                if self.debug:
                    assert(reward_expert.shape == (batch_size,))

                reward_expert = torch.Tensor(reward_expert)
                if self.debug:
                    assert(reward_expert.size() == torch.Size([batch_size]))

                if self.iscuda: 
                    reward_expert = reward_expert.cuda()
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg_expert = (self.R_avg_expert * self.avg_count_expert + reward_expert.sum())/(self.avg_count_expert + batch_size)
                    self.avg_count_expert += batch_size
                
                # This is the intrinsic reward corresponding to the current action
                rewards.append(reward_expert*self.reward_scale_expert)
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
        
        if self.debug:
            assert(len(rec_errs) == self.T)
            assert(len(rewards) == self.T-1)
            assert(len(log_probs) == self.T-1)
            assert(len(entropies) == self.T-1)
            assert(len(values) == self.T-1)
            assert(len(decoded_all) == self.T)
            assert(len(visited_idxes) == self.T)

            for t in range(self.T):
                assert(rec_errs[t].size() == torch.Size([batch_size]))
                if t < self.T-1:
                    assert(rewards[t].size() == torch.Size([batch_size]))
                    if self.policy.actorType == 'actor':
                        assert(log_probs[t].size() == torch.Size([batch_size]))
                        assert(entropies[t].size() == torch.Size([batch_size]))
                        if self.baselineType == 'critic':
                            assert(values[t].size() == torch.Size([batch_size]))
                        else:
                            assert(values[t] is None)

        return log_probs, rec_errs, rewards, entropies, decoded, values, visited_idxes, decoded_all, actions_taken

