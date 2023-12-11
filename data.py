import math
import numpy as np
import torch

# Matrix
class tensor:
    '''
        input_size: order x k
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, dims, input_size, input_path, device):                
        # Load data        
        train_set = np.load(input_path + "_train.npy")
        val_set = np.load(input_path + "_valid.npy")
        test_set = np.load(input_path + "_test.npy")
        
        self.num_train, self.num_val, self.num_test = train_set.shape[0], val_set.shape[0], test_set.shape[0]        
        self.src_dims = dims          
        self.order = len(self.src_dims)        
        self.src_train_idx, self.src_val_idx, self.src_test_idx = train_set[:, :self.order].astype(int), val_set[:, :self.order].astype(int), test_set[:, :self.order].astype(int)
        self.src_train_vals, self.src_val_vals, self.src_test_vals = train_set[:, self.order].astype(np.float64), val_set[:, self.order].astype(np.float64), test_set[:, self.order].astype(int)
        self.train_norm = math.sqrt(np.square(self.src_train_vals).sum())                
        self.val_norm = math.sqrt(np.square(self.src_val_vals).sum())
        self.test_norm = math.sqrt(np.square(self.src_test_vals).sum())                
        
        # Set base                        
        self.dims = [int(np.prod(np.array(input_size[i]))) for i in range(self.order)]                
        self.src_base = []
        temp_base = 1
        for i in range(self.order-1, -1, -1):
            self.src_base.insert(0, temp_base)
            temp_base *= self.src_dims[i]
                            
        # Load to gpu
        device = torch.device("cuda:" + str(device))
        self.src_base = torch.tensor(self.src_base, dtype=torch.long, device=device)
        self.src_dims_gpu = torch.tensor(self.src_dims, dtype=torch.long, device=device)        
        self.dims = torch.tensor(self.dims, dtype=torch.long, device=device)                