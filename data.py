import math
import numpy as np
import torch

# Matrix
class tensor:
    '''
        input_size: order x k
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, input_size, input_path, sample_ratio, device):                
        # Load data
        self.src_tensor = np.load(input_path + ".npy")               
        self.sample_idx = np.load(input_path + f'_sample_{sample_ratio}.npy')                        
        self.num_sample = self.sample_idx.shape[0]
        self.num_train = self.num_sample
        print(self.src_tensor.shape)
        
        # Set base        
        if self.src_tensor.dtype != "float64":
            self.src_tensor = self.src_tensor.astype(np.float64)
        self.src_dims = list(np.shape(self.src_tensor))                            
        self.order = len(self.src_dims)
        self.dims = [int(np.prod(np.array(input_size[i]))) for i in range(self.order)]
        
        self.src_vals = self.src_tensor.flatten()    
        self.real_num_entries = len(self.src_vals)        
        self.norm = math.sqrt(np.square(self.src_vals).sum())                
        self.src_base, self._base = [], []
        temp_base = 1
        for i in range(self.order-1, -1, -1):
            self.src_base.insert(0, temp_base)
            temp_base *= self.src_dims[i]
                    
        temp_base = 1
        for i in range(self.order-1, -1, -1):
            self._base.insert(0, temp_base)
            temp_base *= self.dims[i]
            
        # change idx to 1d
        self.src_sample_idx = np.sum(self.sample_idx * np.array(self.src_base), axis=1)
        self.src_train_idx = self.src_sample_idx
        #self.src_valid_idx = self.src_sample_idx[self.num_train:]
        temp_set = set(self.src_sample_idx.tolist())
        self.src_test_idx = np.array([i for i in range(self.src_vals.size) if i not in temp_set])        
        del temp_set
        
        # Load to gpu
        device = torch.device("cuda:" + str(device))
        self.src_base = torch.tensor(self.src_base, dtype=torch.long, device=device)
        self.src_dims_gpu = torch.tensor(self.src_dims, dtype=torch.long, device=device)        
        self.dims = torch.tensor(self.dims, dtype=torch.long, device=device)        
        self._base = torch.tensor(self._base, dtype=torch.long, device=device)
        print(f'norm of the tensor: {self.norm}')