import math
import numpy as np
import torch

# Matrix
class tensor:
    '''
        input_size: order x k
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, input_size, input_path, device, known_entry):                
        # Load training and validation set
        input_tensor = np.load(input_path + "_orig.npy", allow_pickle=True)
        test_set = np.load(input_path + f"_{known_entry}_orig_test.npy", allow_pickle=True)
        val_set = np.load(input_path + f"_{known_entry}_orig_valid.npy", allow_pickle=True)        
        
        self.num_test, self.num_val = test_set.shape[0], val_set.shape[0]
        self.num_train = input_tensor.size - self.num_test - self.num_val
        
        self.src_dims = input_tensor.shape
        self.order = len(self.src_dims)        
        self.src_test_idx, self.src_val_idx = test_set[:, :self.order].astype(int), val_set[:, :self.order].astype(int)
        self.src_test_vals, self.src_val_vals = test_set[:, self.order].astype(np.float64), val_set[:, self.order].astype(np.float64)

        #self.train_avg = np.mean(self.src_train_vals)
        self.test_norm = math.sqrt(np.square(self.src_test_vals).sum())                
        self.val_norm = math.sqrt(np.square(self.src_val_vals).sum())    

        # load test set
        train_tensor = np.ones(self.src_dims)
        train_tensor[self.src_test_idx[:, 0], self.src_test_idx[:, 1], self.src_test_idx[: ,2]] = 0
        train_tensor[self.src_val_idx[:, 0], self.src_val_idx[:, 1], self.src_val_idx[:, 2]] = 0        
        self.src_train_idx = np.transpose(np.nonzero(train_tensor))
        self.src_train_vals = input_tensor[self.src_train_idx[:, 0], self.src_train_idx[:, 1], self.src_train_idx[:, 2]]        
        self.train_norm = math.sqrt(np.square(self.src_train_vals).sum())                
        
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
        print(f'saved: {np.sum(self.src_train_vals)}, real: {np.sum(input_tensor[self.src_train_idx[:, 0], self.src_train_idx[:, 1], self.src_train_idx[: ,2]])}')
