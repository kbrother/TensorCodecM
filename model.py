from torch import nn, Tensor
import torch
from tqdm import tqdm
import math
import numpy as np
import random
import copy
import pickle

# Model
class rnn_model(torch.nn.Module):
    '''
        input_size: list of list that saves the size of inputs of all levels for each mode
        order x k
    '''
    def __init__(self, rank, input_size, hidden_size):
        super(rnn_model, self).__init__()
        self.rank = rank
        self.k = len(input_size[0])
        self.layer_first = nn.Linear(hidden_size, rank)
        self.layer_middle = nn.Linear(hidden_size, rank*rank)
        self.layer_final = nn.Linear(hidden_size, rank)        
        
        self.rnn = nn.LSTM(hidden_size, hidden_size)                    
        self.hidden_size = hidden_size        
        self.order = len(input_size)
        #self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        input_set = set()
        for i in range(self.k):
            curr_input = [input_size[j][i] for j in range(self.order)]            
            input_set.add(tuple(curr_input))            
        num_emb = 0
        for _input in input_set:
            curr_num_emb = 1
            for i in range(self.order): curr_num_emb *= _input[i]
            num_emb += curr_num_emb
        self.emb = nn.Embedding(num_embeddings=num_emb, embedding_dim = hidden_size)        
        
    '''
        _input: batch size x seq len        
       -----------------------------------
       preds: batch size 
    '''
    def forward(self, _input):
        _input = _input.transpose(0, 1)
        seq_len, batch_size = _input.size()
        _input = self.emb(_input)   # seq len x batch size x hidden dim                
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(_input)   # seq len x batch size x hidden dim 
        
        #rnn_output = torch.reshape(rnn_output, (batch_size, self.hidden_size, seq_len))
        #rnn_output = self.batch_norm(rnn_output)
        #rnn_output = torch.reshape(rnn_output, (seq_len, batch_size, self.hidden_size))
        
        first_mat = self.layer_first(rnn_output[0,:,:])   
        final_mat = self.layer_final(rnn_output[-1,:,:])   # batch size x R
        first_mat, final_mat = first_mat.unsqueeze(1), final_mat.unsqueeze(-1)   # batch size x 1 x R, batch size x R x 1
        middle_mat = self.layer_middle(rnn_output[1:-1,:,:])  # seq len -2 x batch size x R^2
        
        middle_mat = middle_mat.view(self.k-2, batch_size, self.rank, self.rank)  # seq len - 2  x batch size x R x R
        preds = torch.matmul(first_mat, middle_mat[0, :, :, :])
        for j in range(1, self.k-2):
            preds = torch.matmul(preds, middle_mat[j, :, :, :])
        preds = torch.matmul(preds, final_mat).squeeze()  # batch size 
        return preds
        
    
class TensorCodec:
    '''
        input_size: list of list that saves the size of inputs of all levels for each mode,
        order x k 
    '''
    def __init__(self, input_mat, rank, input_size, hidden_size, device, args):
        # Intialize parameters
        self.input_mat = input_mat
        self.input_size = input_size
        self.k = len(self.input_size[0])
        self.order = len(self.input_size)
        self.hidden_size = hidden_size
        self.device = device
        self.i_device = torch.device("cuda:" + str(self.device[0]))
        self.model = rnn_model(rank, input_size, hidden_size)
        self.model.double()     
        if len(self.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids = self.device)                        
        self.model = self.model.to(self.i_device)        
        
        # Build bases, order x k
        self.bases_list = [[] for _ in range(self.order)]        
        for i in range(self.order):
            _base = 1
            for j in range(self.k-1, -1, -1):
                self.bases_list[i].insert(0, _base)
                _base *= self.input_size[i][j]            
                
        # Build add term 
        input_size_dict = {}        
        _temp = 0
        for i in range(self.k):
            curr_input = [input_size[j][i] for j in range(self.order)]
            curr_input_size = np.prod(np.array(curr_input))
            curr_input = tuple(curr_input)            
            
            if curr_input not in input_size_dict:
                input_size_dict[curr_input] = _temp
                _temp += curr_input_size

        self._add = []
        for i in range(self.k):
            curr_input = tuple([input_size[j][i] for j in range(self.order)])
            self._add.append(input_size_dict[curr_input])
        #print(self._add)    
        # move to gpu
        for i in range(self.order):
            self.input_size[i] = torch.tensor(self.input_size[i], dtype=torch.long, device=self.i_device)  # order x k    
            self.bases_list[i] = torch.tensor(self.bases_list[i], dtype=torch.long, device=self.i_device)  # order x k
        self._add = torch.tensor(self._add, dtype=torch.long, device=self.i_device).unsqueeze(0)                
        self.comp_size =  sum(p.numel() for p in self.model.parameters() if p.requires_grad) * 8
        for i in range(self.order):
            self.comp_size += math.ceil(self.input_mat.src_dims[i] * math.ceil(math.log(self.input_mat.src_dims[i], 2)) / 8)
        print(f"Compressed size:{self.comp_size} bytes")

        
        # model -> tensor
        self.perm_list = [torch.tensor(list(range(self.input_mat.dims[i])), dtype=torch.long, device=self.i_device) for i in range(self.order)]
        with open(f'TensorCodec_completion/mapping/{args.dataset}_{args.known_entry}_model2tens.pickle', 'rb') as f:
            mappings = pickle.load(f)
        for mode in range(self.order):
            self.perm_list[mode][range(len(mappings[mode]))] = torch.tensor(mappings[mode], dtype=torch.long, device=self.i_device)
        
        # tensor -> model
        self.inv_perm_list = [torch.tensor(list(range(self.input_mat.dims[i])), dtype=torch.long, device=self.i_device) for i in range(self.order)]
        for mode in range(self.order):
            self.inv_perm_list[mode][self.perm_list[mode]] = torch.arange(self.input_mat.dims[mode], dtype=torch.long, device=self.i_device)
            print(f'mode:{mode}, perm: {torch.mean(self.perm_list[mode].to(torch.float))}, inv perm: {torch.mean(self.inv_perm_list[mode].to(torch.float))}') 
    
    # Given a model indices output predictions
    # model_idx: batch size x order
    # output: batch size
    def predict(self, model_idx):
        batch_size = model_idx.shape[0]
        model_input = torch.zeros((batch_size, self.k), dtype=torch.long, device=self.i_device)  # batch size x k   
        for i in range(self.order):
            curr_idx = model_idx[:, i].unsqueeze(-1) # batch_size
            curr_idx = curr_idx // self.bases_list[i] % self.input_size[i]  # batch size x k
            model_input = model_input * self.input_size[i].unsqueeze(0) + curr_idx
        
        model_input = model_input + self._add
        #self.model = self.model.to(torch.device("cpu"))
        #model_input = model_input.cpu()               
        return self.model(model_input)

    
    # minibatch L2 loss
    # samples: sample idx, batch size
    def L2_loss(self, need_bp, set_type, batch_size, samples):
        return_loss, minibatch_norm = 0., 0.
        num_sample = samples.shape[0]
        # Indices of sampled matrix entries        
        for i in range(0, num_sample, batch_size):
            with torch.no_grad():
                curr_batch_size = min(batch_size, num_sample - i)

                if set_type == "train":
                    curr_ten_idx = self.input_mat.src_train_idx[samples[i:i+curr_batch_size]]
                    vals = torch.tensor(self.input_mat.src_train_vals[samples[i:i+curr_batch_size]], device=self.i_device)
                elif set_type == "val":
                    curr_ten_idx = self.input_mat.src_val_idx[samples[i:i+curr_batch_size]]
                    vals = torch.tensor(self.input_mat.src_val_vals[samples[i:i+curr_batch_size]], device=self.i_device)
                elif set_type == "test":
                    curr_ten_idx = self.input_mat.src_test_idx[samples[i:i+curr_batch_size]]
                    vals = torch.tensor(self.input_mat.src_test_vals[samples[i:i+curr_batch_size]], device=self.i_device)
                else:
                    raise ValueError("wrong input for the set type")
                curr_ten_idx = torch.tensor(curr_ten_idx, device=self.i_device)                
                
                curr_model_idx = curr_ten_idx.clone()                
                for j in range(self.order):
                    curr_model_idx[:, j] = self.inv_perm_list[j][curr_ten_idx[:, j]]
                                                
            preds = self.predict(curr_model_idx) # + self.input_mat.train_avg
            curr_loss = torch.square(preds - vals).sum()
            return_loss += curr_loss.item()
            minibatch_norm += torch.square(vals).sum().item()
            
            if need_bp: curr_loss.backward()
        return return_loss, minibatch_norm

    # minibatch L2 loss
    # samples: indices of sampled matrix entries
    def L2_loss_orig(self, is_train, batch_size, samples):
        return_loss, minibatch_norm = 0., 0.
        num_sample = samples.shape[0]
        # Indices of sampled matrix entries        
        for i in range(0, num_sample, batch_size):
            with torch.no_grad():
                curr_batch_size = min(batch_size, num_sample - i)
                curr_ten_idx = samples[i:i+curr_batch_size]
                vals = torch.tensor(self.input_mat.src_fake_vals[curr_ten_idx], device=self.i_device)
                
                curr_ten_idx = torch.tensor(curr_ten_idx, device=self.i_device).unsqueeze(-1)
                curr_ten_idx = curr_ten_idx // self.input_mat.src_base % self.input_mat.src_dims_gpu # batch size x self.order
                curr_model_idx = curr_ten_idx.clone()                
                for j in range(self.order):
                    curr_model_idx[:, j] = self.inv_perm_list[j][curr_ten_idx[:, j]]
                                                
            preds = self.predict(curr_model_idx)
            curr_loss = torch.square(preds - vals).sum()
            return_loss += curr_loss.item()
            minibatch_norm += torch.square(vals).sum().item()
            
            if is_train: curr_loss.backward()
        return math.sqrt(return_loss), math.sqrt(minibatch_norm)
        
    '''
        Input
            curr_order: 0 for row and 1 for col
            model_idx: indices of model
            num_bucket: number of bucket
            
        Output
            bucket_idx: buckets correspond to model indices
    '''
    def hashing_euclid(self, curr_order, model_idx, num_bucket, batch_size):
        num_idx = model_idx.shape[0]
        slice_size = 1
        for i in range(self.order):
            if i ==  curr_order: continue
            slice_size *= self.input_mat.src_dims[i]
        
        curr_line = 5 * (np.random.rand(slice_size) - 0.5)
        curr_line = curr_line / np.linalg.norm(curr_line)
        curr_lines = np.tile(curr_line, num_idx)
        proj_pts = [0 for _ in range(num_idx)]
        
        # Build repeated long vector for the current line
        slices = np.zeros(slice_size*num_idx)        
        for i in range(num_idx):
            curr_slice = self.input_mat.extract_slice(curr_order, self.perm_list[curr_order][model_idx[i]].item())            
            slices[slice_size*i: slice_size*(i+1)] = curr_slice
        
        proj_pts = torch.zeros(num_idx, dtype=torch.double).to(self.i_device)
        with torch.no_grad():
            for i in range(0, num_idx * slice_size, batch_size):
                if num_idx*slice_size - i < batch_size: curr_batch_size = num_idx*slice_size - i
                else: curr_batch_size = batch_size
                temp_vec1 = torch.tensor(curr_lines[i:i+curr_batch_size]).to(self.i_device)
                temp_vec2 = torch.tensor(slices[i:i+curr_batch_size]).to(self.i_device)
                dot_prod = temp_vec1 * temp_vec2
                curr_idx = torch.arange(i,i+curr_batch_size, device=self.i_device)
                proj_pts.scatter_(0, curr_idx//slice_size, dot_prod, reduce='add')
        
        proj_pts = proj_pts.cpu().numpy()
        min_point, max_point = min(proj_pts), max(proj_pts)
                
        seg_len = (max_point - min_point) / (num_bucket - 1)
        start_point = random.uniform(min_point - seg_len, min_point)        
        bucket_idx = proj_pts.copy()
        #print(f'min: {min_point}, max: {max_point}, seg_len: {seg_len}, start point: {start_point}, max point: {(max_point-start_point) // seg_len}')
        
        if (max_point - min_point) < 1e-12:
            for i in range(num_idx):
                bucket_idx[i] = 0
        else:
            for i in range(num_idx):
                bucket_idx[i] = int((proj_pts[i] - start_point) // seg_len)            
        return bucket_idx.astype(int)
    
    
    '''
        Use euclid hashing 
        curr_order: 0 for row and 1 for col
    '''
    def change_permutation(self, batch_size, curr_order):
        # Hashing
        _matrix = self.input_mat
        curr_dim = _matrix.src_dims[curr_order]
        
        num_pair = curr_dim//2
        _temp = (curr_dim-2)//2 + 1
        model_idx = 2*np.arange(_temp) + np.random.randint(2, size=_temp)
        num_bucket = curr_dim // 8
        if num_bucket <= 1: 
            num_bucket = 1
            bucket_idx = [0 for _ in range(_temp)]
        else:
            bucket_idx = self.hashing_euclid(curr_order, model_idx, num_bucket, batch_size)
                
        # Build bucket
        buckets = [[] for _ in range(num_bucket)]
        #print(model_idx.size)
        #print(bucket_idx.size)
        #print(num_pair)
        for i in range(num_pair):            
            if bucket_idx[i] >= len(buckets):
                print(f'{bucket_idx[i]}, {len(buckets)}')
            buckets[bucket_idx[i]].append(model_idx[i])
        if curr_dim % 2 == 1: remains = [curr_dim - 1]
        else: remains = []
            
        # Build pairs within buckets
        first_elem, second_elem = [], []
        for i in range(num_bucket):
            random.shuffle(buckets[i])
            if len(buckets[i]) % 2 == 1:
                rem_part = buckets[i].pop(-1)
                remains.append(rem_part)
                remains.append(rem_part^1)
            
            first_elem = first_elem + buckets[i]
            second_elem_temp = [0 for _ in range(len(buckets[i]))]
            second_elem_temp[0::2] = [elem^1 for elem in buckets[i][1::2]]
            second_elem_temp[1::2] = [elem^1 for elem in buckets[i][0::2]]
            second_elem = second_elem + second_elem_temp
        
        # Build pairs within remains
        random.shuffle(remains)
        if len(remains) % 2 == 1: 
            poped=remains.pop(-1)
            poped = self.perm_list[curr_order][poped].item()
        first_elem = first_elem + remains[0::2]
        second_elem = second_elem + remains[1::2]
        first_elem, second_elem = torch.tensor(first_elem, dtype=torch.long, device=self.i_device), torch.tensor(second_elem, dtype=torch.long, device=self.i_device) 
        first_tidx, second_tidx = self.perm_list[curr_order][first_elem], self.perm_list[curr_order][second_elem]        
        
        # Initialize variables
        num_slice_entry = 1
        for i in range(self.order):
            if i == curr_order: continue
            num_slice_entry *= self.input_mat.src_dims[i]
        loss_list = torch.zeros(num_pair, device=self.i_device, dtype=torch.double)
       
        # Compute the loss change
        self.model.eval()
        num_total_entry = num_pair * num_slice_entry
        delta_loss, curr_idx = 0., 0
        
        # chech whether the loss decreases         
        with torch.no_grad():
            # random pairing  
            '''
            num_pair = math.ceil(self.input_mat.src_dims[mode]/2)
            tidx = torch.randperm(self.input_mat.src_dims[mode], device=self.i_device)
            first_tidx, second_tidx = tidx[:num_pair], tidx[num_pair:]
            if self.input_mat.src_dims[mode] % 2 == 1:
                second_tidx = torch.cat([second_tidx, torch.tensor([first_tidx[-1].item()], device=self.i_device)])             
            '''
            
            #preprocess            
            tidx2pidx = torch.arange(curr_dim, device=self.i_device)
            tidx2pidx[first_tidx] = torch.arange(num_pair, device=self.i_device)
            tidx2pidx[second_tidx] = torch.arange(num_pair, device=self.i_device)        
            if curr_dim % 2 == 1:
                tidx2pidx[poped] = 0
            
            tidx2new_tidx = torch.arange(curr_dim, device=self.i_device)
            tidx2new_tidx[first_tidx] = second_tidx.clone()
            tidx2new_tidx[second_tidx] = first_tidx.clone()

            # check the loss before switching
            pair2loss = torch.zeros(num_pair, device=self.i_device, dtype=torch.double)           
            curr_idx = 0
            while curr_idx < self.input_mat.num_train:
                # Compute thre current
                curr_batch_size = min(self.input_mat.num_train - curr_idx, batch_size)
                curr_ten_idx = self.input_mat.src_train_idx[curr_idx:curr_idx+curr_batch_size]
                curr_ten_idx = torch.tensor(curr_ten_idx, device=self.i_device)                
                curr_model_idx = curr_ten_idx.clone()                
                for j in range(self.order):
                    curr_model_idx[:, j] = self.inv_perm_list[j][curr_ten_idx[:, j]]

                curr_preds = self.predict(curr_model_idx) # + self.input_mat.train_avg
                curr_vals = torch.tensor(self.input_mat.src_train_vals[curr_idx:curr_idx+curr_batch_size], device=self.i_device)                
                curr_loss = torch.square(curr_preds - curr_vals)
                pair_idx = tidx2pidx[curr_ten_idx[:, curr_order]]
                pair2loss = pair2loss.scatter_reduce(0, pair_idx, -curr_loss, reduce="sum")                

                # Compute the future loss
                new_tidx = tidx2new_tidx[curr_ten_idx[:, curr_order]]
                curr_model_idx[:, curr_order] = self.inv_perm_list[curr_order][new_tidx]
                #print(f"mean: {torch.max(curr_model_idx, dim=0)}")
                curr_preds = self.predict(curr_model_idx)  # + self.input_mat.train_avg
                curr_loss = torch.square(curr_preds - curr_vals)
                pair_idx = tidx2pidx[new_tidx]
                pair2loss = pair2loss.scatter_reduce(0, pair_idx, curr_loss, reduce="sum")                
                curr_idx += curr_batch_size

            #print(pair2loss)
            valid_pair = pair2loss < 0
            delta_loss = torch.sum(pair2loss[valid_pair]).item()
            first_tidx, second_tidx = first_tidx[valid_pair], second_tidx[valid_pair]
            prev_inv_perm = self.inv_perm_list[curr_order].clone()
            self.inv_perm_list[curr_order][first_tidx] = prev_inv_perm[second_tidx]
            self.inv_perm_list[curr_order][second_tidx] = prev_inv_perm[first_tidx]
      
        return delta_loss