import math
import numpy as np
import torch
from tqdm import tqdm

# Matrix
class tensor:
    '''
        input_size: order x k
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, input_size, input_path, device):                
        # Load training and validation set
        input_tensor = np.load(input_path, allow_pickle=True)
        num_tensor = len(input_tensor)
        first_dim = np.array([input_tensor[_i].shape[0] for _i in range(num_tensor)])
        max_first = np.max(first_dim)
                
        self.src_dims = [max_first, input_tensor[0].shape[1], num_tensor]
        self.order = 3        
        self.src_train_idx = np.zeros((np.sum(first_dim)*input_tensor[0].shape[1], self.order), dtype=int)
        temp_row = 0
        for i in tqdm(range(num_tensor)):
            num_entry = first_dim[i]*input_tensor[0].shape[1]
            row_idx, col_idx = np.indices(input_tensor[i].shape, dtype=int)
            row_idx, col_idx = row_idx.flatten(), col_idx.flatten()
            self.src_train_idx[temp_row:(temp_row + num_entry), 0] = row_idx
            self.src_train_idx[temp_row:(temp_row + num_entry), 1] = col_idx
            self.src_train_idx[temp_row:(temp_row + num_entry), 2] = i
            temp_row += num_entry
        
        self.src_train_vals = np.concatenate([input_tensor[i].flatten() for i in range(num_tensor)], axis = 0)
        self.train_norm = math.sqrt(np.square(self.src_train_vals).sum())  
        self.num_train = self.src_train_idx.shape[0]
        
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
        #print(f'saved: {np.sum(self.src_train_vals)}, real: {np.sum(input_tensor[self.src_train_idx[:, 0], self.src_train_idx[:, 1], self.src_train_idx[: ,2]])}')
