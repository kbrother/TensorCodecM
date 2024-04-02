import math
import numpy as np
import torch
from tqdm import tqdm
import pickle 

# Matrix
class tensor:
    '''
        input_size: order x k
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, input_size, input_path, device):                
        with open(input_path, 'rb') as f:
            raw_dict = pickle.load(f)
        _indices = raw_dict['idx']
        num_nnz = len(_indices[0])
        _values = np.array(raw_dict['val'])
        self.order = len(_indices)

        for m in range(self.order):
            _indices[m] = np.array(_indices[m], dtype=int)
        
        idx2newidx = np.argsort(_indices[-1])
        for m in range(self.order):
            _indices[m] = _indices[m][idx2newidx]
        _values = _values[idx2newidx]      

        # Save tensor stat
        max_first = max(_indices[0]) + 1
        num_tensor = max(_indices[-1]) + 1
        middle_dim = []        
        for m in range(1, self.order-1):
            middle_dim.append(max(_indices[m]) + 1)       
        
        tidx2start = [0]
        for i in range(num_nnz):
            if _indices[-1][tidx2start[-1]] != _indices[-1][i]:
                tidx2start.append(i)
        tidx2start.append(num_nnz)        
        
        # Set first dim
        first_dim = []
        for i in range(num_tensor):
            first_dim.append(max(_indices[0][tidx2start[i]:tidx2start[i+1]]) + 1)
        first_dim = np.array(first_dim)

        # Required
        self.src_dims = [max_first] + middle_dim + [num_tensor]
        self.src_train_idx = np.zeros((np.sum(first_dim)*np.prod(middle_dim), self.order), dtype=int)
        temp_row = 0
        self.src_train_vals = []
        for i in tqdm(range(num_tensor)):
            input_tensor = np.zeros([first_dim[i]] + middle_dim)        
            temp_idx = [_indices[j][tidx2start[i]:tidx2start[i+1]] for j in range(self.order-1)]            
            if self.order == 3:
                input_tensor[temp_idx[0], temp_idx[1]] = _values[tidx2start[i]:tidx2start[i+1]]            
            else:
                input_tensor[temp_idx[0], temp_idx[1], temp_idx[2]] = _values[tidx2start[i]:tidx2start[i+1]]            
            curr_indices = np.indices(input_tensor.shape, dtype=int)
            num_entry = np.prod(input_tensor.shape)
            
            for j in range(self.order-1):
                self.src_train_idx[temp_row:(temp_row + num_entry), j] = curr_indices[j].flatten()
            self.src_train_idx[temp_row:(temp_row + num_entry), -1] = i
            temp_row += num_entry
            self.src_train_vals.append(input_tensor.flatten())
        
        self.src_train_vals = np.concatenate(self.src_train_vals, axis=0)                
        self.train_norm = math.sqrt(np.square(_values).sum())  
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
