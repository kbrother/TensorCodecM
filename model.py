from torch import nn, Tensor
import torch
from tqdm import tqdm
import math
import numpy as np
import random
import copy

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
    def __init__(self, input_mat, rank, input_size, hidden_size, device):
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
        # tensor -> model
        self.inv_perm_list = [torch.tensor(list(range(self.input_mat.dims[i])), dtype=torch.long, device=self.i_device) for i in range(self.order)]
    
    
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
                                                
            preds = self.predict(curr_model_idx)
            curr_loss = torch.square(preds - vals).sum()
            return_loss += curr_loss.item()
            minibatch_norm += torch.square(vals).sum().item()
            
            if need_bp: curr_loss.backward()
        return return_loss, minibatch_norm


    def reordering(self, mode, batch_size):        
        # tensor -> model need to be changed
        # chech whether the loss decreases         
        with torch.no_grad():
            #preprocess
            num_pair = math.ceil(self.input_mat.src_dims[mode]/2)
            tidx = torch.randperm(self.input_mat.src_dims[mode], device=self.i_device)
            first_tidx, second_tidx = tidx[:num_pair], tidx[num_pair:]
            if self.input_mat.src_dims[mode] % 2 == 1:
                second_idx = torch.cat([second_tidx, first_tidx[-1].clone()]) 
            
            tidx2pidx = torch.arange(self.input_mat.src_dims[mode], device=self.i_device)
            tidx2pidx[first_tidx] = torch.arange(num_pair, device=self.i_device)
            tidx2pidx[second_tidx] = torch.arange(num_pair, device=self.i_device)            
            tidx2new_tidx = torch.arange(self.input_mat.src_dims[mode], device=self.i_device)
            tidx2new_tidx[first_tidx] = second_tidx.clone()
            tidx2new_tidx[second_tidx] = first_tidx.clone()

            # check the loss before switching
            pair2loss = torch.zeros(num_pair, device=self.i_device)           
            curr_idx = 0
            while tqdm(curr_idx < self.input_mat.num_train):
                # Compute thre current
                curr_batch_size = min(self.input_mat.num_train - curr_idx, batch_size)
                curr_ten_idx = self.input_mat.src_train_idx[curr_idx:curr_idx+curr_batch_size]      
                curr_ten_idx = torch.tensor(curr_ten_idx, device=self.i_device)
                curr_model_idx = curr_ten_idx.clone()
                for j in range(self.order):
                    curr_model_idx[:, j] = self.inv_perm_list[j][curr_ten_idx[:, j]]

                curr_preds = self.predict(curr_model_idx)
                curr_vals = torch.tensor(self.input_mat.src_train_vals[curr_idx:curr_idx+curr_batch_size], device=self.i_device)                
                curr_loss = torch.square(curr_preds - curr_vals)
                pair_idx = tidx2pidx[curr_ten_idx[:, mode]]
                print(curr_loss)
                print(pair_idx)
                #curr_loss = curr_loss.cpu()
                #pair_idx = pair_idx.cpu()
                pair2loss = torch.scatter_reduce(-curr_loss, 0, pair_idx,  reduce="sum")                

                # Compute the future loss
                new_tidx = tidx2new_tidx[curr_ten_idx[:, mode]]
                curr_model_idx[:, mode] = self.inv_perm_list[mode][new_tidx]
                curr_preds = self.predict(curr_model_idx)
                curr_loss = torch.square(curr_preds - curr_vals)
                pair_idx = tidx2pidx[new_tidx]
                curr_loss = curr_loss.cpu()
                pair_idx = pair_idx.cpu()
                pair2loss = torch.scatter_reduce(curr_loss, 0, pair_idx, reduce="sum")                
                curr_idx += curr_batch_size

            valid_pair = pair2loss < 0
            delta_loss = torch.sum(pair2loss[valid_pair]).item()
            first_tidx, second_tidx = first_tidx[valid_pair], second_tidx[valid_pair]
            prev_inv_perm = self.inv_perm_list[mode].clone()
            self.inv_perm_list[mode][first_tidx] = prev_inv_perm[second_tidx]
            self.inv_perm_list[mode][second_tidx] = prev_inv_perm[first_tidx]
        
        return delta_loss