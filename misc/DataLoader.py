import h5py
import random
import numpy as np
import pdb
import torch

class DataLoaderSimple(object):
    """
    DataLoader class for abstracting the reading, batching and shuffling operations
    Does not use expert rewards.
    """

    def __init__(self, opts):
        """
        Loads the dataset and saves settings needed:
        (1) dataset statistics (2) shuffle (3) debug statistics (4) iteration tracker  
        Opts required: seed, h5_path, shuffle, batch_size, h5_path_unseen (optional)
        mask_path (optional)
        """
        # ---- Load the dataset ----
        self.h5_file = h5py.File(opts.h5_path, 'r')
        self.data = {}
        self.data['train'] = np.array(self.h5_file['train'])
        self.data['val'] = np.array(self.h5_file['val'])
        self.data['test'] = np.array(self.h5_file['test']) 
        if 'val_highres' in self.h5_file.keys():
            self.data['val_highres'] = np.array(self.h5_file['val_highres'])
            self.data['test_highres'] = np.array(self.h5_file['test_highres'])

        # ---- Load the unseen classes ----
        if opts.h5_path_unseen != '':
            h5_file_unseen = h5py.File(opts.h5_path_unseen, 'r')
            self.data['test_unseen'] = np.array(h5_file_unseen['test'])
        # ---- Save settings needed for batching operations ----
        # Dataset statistics
        self.train_count = self.h5_file['train'].shape[0]
        self.val_count = self.h5_file['val'].shape[0]
        self.test_count = self.h5_file['test'].shape[0]
        if opts.h5_path_unseen != '':
            self.test_unseen_count = self.data['test_unseen'].shape[0]
        if hasattr(opts, 'mask_path') and opts.mask_path != '':
            mask_file = h5py.File(opts.mask_path, 'r')
            self.masks = {}
            self.masks['test'] = np.array(mask_file['test_mask'])
            if opts.h5_path_unseen != '':
                self.masks['test_unseen'] = np.array(mask_file['test_unseen_mask'])
            self.hasmasks = True
        else:
            self.hasmasks = False

        self.pano_shape = self.h5_file['train'].shape[1:]
        # Iteration tracker 
        self.train_idx = 0
        self.val_idx = 0
        self.test_idx = 0 
        if opts.h5_path_unseen != '':
            self.test_unseen_idx = 0

        self.batch_size = opts.batch_size
        # Shuffle the training data indices and access them in the shuffled order
        self.shuffle = opts.shuffle
        self.shuffled_idx = list(range(self.h5_file['train'].shape[0]))
        if self.shuffle:
            random.shuffle(self.shuffled_idx)
        # Debug mode
        self.debug = opts.debug
        self.N = self.data['train'].shape[1]
        self.M = self.data['train'].shape[2]
        self.C = self.data['train'].shape[3]
        self.H = self.data['train'].shape[4]
        self.W = self.data['train'].shape[5]
        if 'val_highres' in self.data:
            self.H_highres = self.data['val_highres'].shape[4]
            self.W_highres = self.data['test_highres'].shape[5]

    def next_batch_train(self):
        """
        Returns the next training batch (indexed by self.shuffled_idx and starting at self.train_idx)
        out: BxNxMxCx32x32
        depleted: is the epoch over?
        """
        batch_size = min(self.batch_size, self.train_count - self.train_idx)

        out = np.array(self.data['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)], :, :, :, :, :]) 
        
        if self.debug: 
            assert((batch_size == self.batch_size) or (self.train_idx + batch_size == self.train_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))

        if self.train_idx + batch_size == self.train_count:
            depleted = True
            self.train_idx = 0
        else:
            depleted = False
            self.train_idx = self.train_idx + batch_size
        
        return out, depleted
    
    def next_batch_val(self, highres=False):
        """
        Returns the next validation batch
        out: BxNxMxCx32x32
        out_highres: BxNxMxCx448x448 (optional)
        depleted: is the epoch over?
        """
        batch_size = min(self.batch_size, self.val_count - self.val_idx)
        out = np.array(self.data['val'][self.val_idx:(self.val_idx+batch_size), :, :, :, :, :])
        if highres:
            out_highres = np.array(self.data['val_highres'][self.val_idx:(self.val_idx+batch_size), :, :, :, :, :])

        if self.debug:
            assert((batch_size == self.batch_size) or (self.val_idx + batch_size == self.val_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            if highres:
                assert(out_highres.shape == (batch_size, self.N, self.M, self.C, self.H_highres, self.W_highres))

        if self.val_idx + batch_size == self.val_count:
            depleted = True
            self.val_idx = 0
        else:
            depleted = False
            self.val_idx = self.val_idx + batch_size

        if not highres:
            return out, depleted
        else:
            return out, out_highres, depleted
    
    def next_batch_test(self, highres=False):
        """
        Returns the next testing batch
        out: BxNxMxCx32x32
        out_highres: BxNxMxCx448x448 (optional)
        depleted: is the epoch over?
        """
        batch_size = min(self.batch_size, self.test_count - self.test_idx)
        out = np.array(self.data['test'][self.test_idx:(self.test_idx+batch_size), :, :, :, :, :])
        if highres:
            out_highres = np.array(self.data['test_highres'][self.test_idx:(self.test_idx+batch_size), :, :, :, :, :])
            
        if self.hasmasks:
            out_masks = self.masks['test'][self.test_idx:(self.test_idx+batch_size), :, :, :, :, :]
        else:
            out_masks = None

        if self.debug:
            assert((batch_size == self.batch_size) or (self.test_idx + batch_size == self.test_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            if highres:
                assert(out_highres.shape == (batch_size, self.N, self.M, self.C, self.H_highres, self.W_highres))
            if self.hasmasks:
                assert(out_masks.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))

        if self.test_idx + batch_size == self.test_count:
            depleted = True
            self.test_idx = 0
        else:
            depleted = False
            self.test_idx = self.test_idx + batch_size
        
        if not highres:
            return out, out_masks, depleted
        else:
            return out, out_highres, out_masks, depleted

    def next_batch_test_unseen(self):
        """
        Returns the next unseen classes testing batch
        out: BxNxMxCx32x32
        """
        batch_size = min(self.batch_size, self.test_unseen_count - self.test_unseen_idx)
        out = np.array(self.data['test_unseen'][self.test_unseen_idx:(self.test_unseen_idx+batch_size), :, :, :, :, :])
        if self.hasmasks:
            out_masks = self.masks['test_unseen'][self.test_unseen_idx:(self.test_unseen_idx+batch_size), :, :, :, :, :]
        else:
            out_masks = None

        if self.debug:
            assert((batch_size == self.batch_size) or (self.test_unseen_idx + batch_size == self.test_unseen_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))

        if self.test_unseen_idx + batch_size == self.test_unseen_count:
            depleted = True
            self.test_unseen_idx = 0
        else:
            depleted = False
            self.test_unseen_idx = self.test_unseen_idx + batch_size

        return out, out_masks, depleted

class DataLoaderExpert(DataLoaderSimple):
    """
    DataLoader class for abstracting the reading, batching and shuffling operations
    Uses expert rewards.
    """
    def __init__(self, opts):
        """
        Loads the dataset, rewards and saves settings needed:
        (1) dataset statistics (2) shuffle (3) debug statistics (4) iteration tracker  
        Opts required: seed, h5_path, shuffle, batch_size, rewards_h5_path
        """
        # ---- Load the dataset, save settings ----
        super(DataLoaderExpert, self).__init__(opts)
        # ---- Load the rewards ----
        rewards_file = h5py.File(opts.rewards_h5_path)
        self.rewards = {}
        # These are KxNxM arrays containing rewards corresponding to each views of
        # all panoramas in the train and val splits
        self.rewards['train'] = np.array(rewards_file['train/nms'])
        self.rewards['val'] = np.array(rewards_file['val/nms'])

    def next_batch_train(self):
        """
        Returns the next training batch (indexed by self.shuffled_idx and starting at self.train_idx)
        out: BxNxMxCx32x32
        out_rewards: BxNxM
        """
        batch_size = min(self.batch_size, self.train_count - self.train_idx)
        out = np.array(self.data['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)], :, :, :, :, :]) 
        
        out_rewards = self.rewards['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)], :, :]
       
        if self.debug:
            assert((batch_size == self.batch_size) or (self.train_idx + batch_size == self.train_count)) 
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            assert(out_rewards.shape == (batch_size, self.N, self.M))

        if self.train_idx + batch_size == self.train_count:
            depleted = True
            self.train_idx = 0
        else:
            depleted = False
            self.train_idx = self.train_idx + batch_size
        
        return out, out_rewards, depleted
            
    def next_batch_val(self):
        """
        Returns the next validation batch
        out: BxNxMxCx32x32
        out_rewards: BxNxM
        """ 
        batch_size = min(self.batch_size, self.val_count - self.val_idx)

        out = self.data['val'][self.val_idx:(self.val_idx+batch_size), :, :, :, :, :]
        out_rewards = self.rewards['val'][self.val_idx:(self.val_idx+batch_size), :, :]

        if self.debug:
            assert((batch_size == self.batch_size) or (self.val_idx + batch_size == self.val_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            assert(out_rewards.shape == (batch_size, self.N, self.M))

        if self.val_idx + batch_size == self.val_count:
            depleted = True
            self.val_idx = 0
        else:
            depleted = False
            self.val_idx = self.val_idx + batch_size

        return out, out_rewards, depleted

class DataLoaderExpertPolicy(DataLoaderSimple):
    """
    DataLoader class for abstracting the reading, batching and shuffling operations
    Uses expert trajectories.
    """
    def __init__(self, opts):
        """
        Loads the dataset, utility maps and saves settings needed:
        (1) dataset statistics (2) shuffle (3) debug statistics (4) iteration tracker  
        Opts required: seed, h5_path, shuffle, batch_size, utility_h5_path, h5_path_unseen, debug
        """
        # ---- Load the dataset, save the settings ----
        super(DataLoaderExpertPolicy, self).__init__(opts)
        self.trajectories_type = opts.trajectories_type
        if opts.trajectories_type == 'utility_maps':
            # ---- Load the utility maps ----
            utility_file = h5py.File(opts.utility_h5_path)
            self.utility_maps = {}
            # These are KxNxMxNxM arrays 
            for split in utility_file.keys():
                self.utility_maps[split] = np.array(utility_file[split]['utility_maps'])
            
        elif opts.trajectories_type == 'expert_trajectories':
            # ---- Load the trajectories ----
            # {'train': #train_samples x T-1 numpy array, 'val': #val_samples x T-1 numpy array}
            self.trajectories = torch.load(opts.utility_h5_path)
        else:
            raise ValueError('Wrong trajectories_type!')
         
    def next_batch_train(self):
        """
        Returns the next training batch (indexed by self.shuffled_idx and starting at self.train_idx)
        out: BxNxMxCx32x32
        out_maps: BxNxMxNxM
        """
        batch_size = min(self.batch_size, self.train_count - self.train_idx)
        out = np.array(self.data['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)], :, :, :, :, :]) 
       
        if self.trajectories_type == 'utility_maps':
            out_maps = self.utility_maps['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)]]
        else:
            out_maps = {}
            for i in range(self.N):
                for j in range(self.M):
                    out_maps[(i, j)] = self.trajectories['train'][(i, j)][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)], :]

        if self.debug:
            assert((batch_size == self.batch_size) or (self.train_idx + batch_size == self.train_count)) 
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            if self.trajectories_type == 'utility_maps':
                assert(out_maps.shape == (batch_size, self.N, self.M, self.N, self.M))
            else:
                assert(len(out_maps.keys()) == self.M * self.N)
                assert(out_maps[(0, 0)].shape[0] == batch_size)

        if self.train_idx + batch_size == self.train_count:
            depleted = True
            self.train_idx = 0
        else:
            depleted = False
            self.train_idx = self.train_idx + batch_size
        
        return out, out_maps, depleted
            
    def next_batch_val(self):
        """
        Returns the next validation batch
        out: BxNxMxCx32x32
        out_maps: BxNxMxNxM
        """ 
        batch_size = min(self.batch_size, self.val_count - self.val_idx)

        out = self.data['val'][self.val_idx:(self.val_idx+batch_size), :, :, :, :, :]
        if self.trajectories_type == 'utility_maps':
            out_maps = self.utility_maps['val'][self.val_idx:(self.val_idx+batch_size)]
        else:
            out_maps = {}
            for i in range(self.N):
                for j in range(self.M):
                    out_maps[(i, j)] = self.trajectories['val'][(i, j)][self.val_idx:(self.val_idx+batch_size), :]

        if self.debug:
            assert((batch_size == self.batch_size) or (self.val_idx + batch_size == self.val_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            if self.trajectories_type == 'utility_maps':
                assert(out_maps.shape == (batch_size, self.N, self.M, self.N, self.M))
            else:
                assert(len(out_maps.keys()) == self.M * self.N)
                assert(out_maps[(0, 0)].shape[0] == batch_size)

        if self.val_idx + batch_size == self.val_count:
            depleted = True
            self.val_idx = 0
        else:
            depleted = False
            self.val_idx = self.val_idx + batch_size

        return out, out_maps, depleted

    def next_batch_test(self, highres=False):
        """
        Returns the next testing batch
        out: BxNxMxCx32x32
        out_masks: ???
        out_maps: BxNxMxNxM
        out_highres: BxNxMxCx448x448 (optional)
        depleted: is the epoch over?
        """
        batch_size = min(self.batch_size, self.test_count - self.test_idx)
        out = np.array(self.data['test'][self.test_idx:(self.test_idx+batch_size), :, :, :, :, :])
        if highres:
            out_highres = np.array(self.data['test_highres'][self.test_idx:(self.test_idx+batch_size), :, :, :, :, :])
            
        if self.hasmasks:
            out_masks = self.masks['test'][self.test_idx:(self.test_idx+batch_size), :, :, :, :, :]
        else:
            out_masks = None

        if self.trajectories_type == 'utility_maps':
            out_maps = self.utility_maps['test'][self.test_idx:(self.test_idx+batch_size)]
        else:
            out_maps = {}
            for i in range(self.N):
                for j in range(self.M):
                    out_maps[(i, j)] = self.trajectories['test'][(i, j)][self.test_idx:(self.test_idx+batch_size), :]

        if self.debug:
            assert((batch_size == self.batch_size) or (self.test_idx + batch_size == self.test_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            if highres:
                assert(out_highres.shape == (batch_size, self.N, self.M, self.C, self.H_highres, self.W_highres))
            if self.hasmasks:
                assert(out_masks.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))

            if self.trajectories_type == 'utility_maps':
                assert(out_maps.shape == (batch_size, self.N, self.M, self.N, self.M))
            else:
                assert(len(out_maps.keys()) == self.M * self.N)
                assert(out_maps[(0, 0)].shape[0] == batch_size)

        if self.test_idx + batch_size == self.test_count:
            depleted = True
            self.test_idx = 0
        else:
            depleted = False
            self.test_idx = self.test_idx + batch_size
        
        if not highres:
            return out, out_masks, out_maps, depleted
        else:
            return out, out_highres, out_masks, out_maps, depleted

    def next_batch_test_unseen(self):
        """
        Returns the next unseen classes testing batch
        out: BxNxMxCx32x32
        out_maps: BxNxMxNxM
        out_masks: ??? 
        depleted: is the epoch over?
        """
        batch_size = min(self.batch_size, self.test_unseen_count - self.test_unseen_idx)
        out = np.array(self.data['test_unseen'][self.test_unseen_idx:(self.test_unseen_idx+batch_size), :, :, :, :, :])
        if self.hasmasks:
            out_masks = self.masks['test_unseen'][self.test_unseen_idx:(self.test_unseen_idx+batch_size), :, :, :, :, :]
        else:
            out_masks = None
        if self.trajectories_type == 'utility_maps':
            out_maps = self.utility_maps['test_unseen'][self.test_unseen_idx:(self.test_unseen_idx + batch_size)]
        else:
            out_maps = {}
            for i in range(self.N):
                for j in range(self.M):
                    out_maps[(i, j)] = self.trajectories['test_unseen'][self.test_unseen_idx:(self.test_unseen_idx+batch_size), :]

        if self.debug:
            assert((batch_size == self.batch_size) or (self.test_unseen_idx + batch_size == self.test_unseen_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            if self.trajectories_type == 'utility_maps':
                assert(out_maps.shape == (batch_size, self.N, self.M, self.N, self.M))
            else:
                assert(len(out_maps.keys()) == self.M * self.N)
                assert(out_maps[(0, 0)].shape[0] == batch_size)

        if self.test_unseen_idx + batch_size == self.test_unseen_count:
            depleted = True
            self.test_unseen_idx = 0
        else:
            depleted = False
            self.test_unseen_idx = self.test_unseen_idx + batch_size

        return out, out_masks, out_maps, depleted

class DataLoaderExpertBoth(DataLoaderSimple):
    # TODO: Need to update trajectories_type here
    # TODO: Add next_batch_test with expert trajectories option here
    """
    DataLoader class for abstracting the reading, batching and shuffling operations
    Uses expert trajectories and rewards.
    """
    def __init__(self, opts):
        """
        Loads the dataset, utility maps and saves settings needed:
        (1) dataset statistics (2) shuffle (3) debug statistics (4) iteration tracker  
        Opts required: seed, h5_path, shuffle, batch_size, utility_h5_path, rewards_h5_path, h5_path_unseen, debug
        """
        # ---- Load the dataset, save the settings ----
        super(DataLoaderExpertBoth, self).__init__(opts)
        # ---- Load the utility maps and rewards ----
        utility_file = h5py.File(opts.utility_h5_path)
        rewards_file = h5py.File(opts.rewards_h5_path)
        self.rewards = {}
        self.utility_maps = {}
        # These are KxNxMxNxM arrays 
        self.utility_maps['train'] = np.array(utility_file['train/utility_maps'])
        self.utility_maps['val'] = np.array(utility_file['val/utility_maps'])
        # These are KxNxM arrays containing rewards corresponding to each views of 
        # all panoramas in the train and val splits
        self.rewards['train'] = np.array(rewards_file['train/nms'])
        self.rewards['val'] = np.array(rewards_file['val/nms'])

    def next_batch_train(self):
        """
        Returns the next training batch (indexed by self.shuffled_idx and starting at self.train_idx)
        out: BxNxMxCx32x32
        out_maps: BxNxMxNxM
        out_rewards: BxNxM
        """
        batch_size = min(self.batch_size, self.train_count - self.train_idx)
        out = np.array(self.data['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)], :, :, :, :, :]) 
        
        out_maps = self.utility_maps['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)]]
        out_rewards = self.rewards['train'][self.shuffled_idx[self.train_idx:(self.train_idx+batch_size)], :, :]

        if self.debug:
            assert((batch_size == self.batch_size) or (self.train_idx + batch_size == self.train_count)) 
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            assert(out_maps.shape == (batch_size, self.N, self.M, self.N, self.M))
            assert(out_rewards.shape == (batch_size, self.N, self.M))

        if self.train_idx + batch_size == self.train_count:
            depleted = True
            self.train_idx = 0
        else:
            depleted = False
            self.train_idx = self.train_idx + batch_size
        
        return out, out_maps, out_rewards, depleted
            
    def next_batch_val(self):
        """
        Returns the next validation batch
        out: BxNxMxCx32x32
        out_maps: BxNxMxNxM
        out_rewards: BxNxM
        """ 
        batch_size = min(self.batch_size, self.val_count - self.val_idx)

        out = self.data['val'][self.val_idx:(self.val_idx+batch_size), :, :, :, :, :]
        out_maps = self.utility_maps['val'][self.val_idx:(self.val_idx+batch_size)]
        out_rewards = self.rewards['val'][self.val_idx:(self.val_idx+batch_size)]

        if self.debug:
            assert((batch_size == self.batch_size) or (self.val_idx + batch_size == self.val_count))
            assert(out.shape == (batch_size, self.N, self.M, self.C, self.H, self.W))
            assert(out_maps.shape == (batch_size, self.N, self.M, self.N, self.M))
            assert(out_rewards.shape == (batch_size, self.N, self.M))

        if self.val_idx + batch_size == self.val_count:
            depleted = True
            self.val_idx = 0
        else:
            depleted = False
            self.val_idx = self.val_idx + batch_size

        return out, out_maps, out_rewards, depleted
