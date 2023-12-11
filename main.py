import numpy as np
import argparse
import torch
from tqdm import tqdm
import math
import sys
from model import TensorCodec
from data import tensor
import copy
import time

def test_perm(n_model, args):
    with torch.no_grad():
        n_model.model.eval()
        curr_loss = n_model.L2_loss(False, args.batch_size)
        print(f'initial loss: {curr_loss}')        
        for i in range(1):
            for j in range(4):
                delta_loss = n_model.change_permutation(args.batch_size, j)
                curr_loss += delta_loss
                print(delta_loss)
                
        print(f'our loss: {curr_loss}, real loss:{n_model.L2_loss(False, args.batch_size)}')
        
        
def train_model(n_model, args):
    device = torch.device("cuda:" + str(args.device[0]))   
    max_fit = -sys.float_info.max
    prev_fit = -1
    n_model.model.train()
    minibatch_size = n_model.input_mat.num_train // args.num_batch
    
    with open(args.save_path + ".txt", 'a') as lossfile:
        lossfile.write(f'compressed size: {n_model.comp_size} bytes\n')    
        
    tol_count = 0
    start_time = time.time()
    for epoch in range(args.epoch): 
        optimizer = torch.optim.Adam(n_model.model.parameters(), lr=args.lr/args.num_batch) 
        n_model.model.train()               
        curr_order = np.random.permutation(n_model.input_mat.num_train)            
        train_loss, train_norm = 0, 0
        for i in tqdm(range(0, n_model.input_mat.num_train, minibatch_size)):
            curr_batch_size = min(n_model.input_mat.num_train - i, minibatch_size)            
            #samples = n_model.input_mat.src_train_idx[curr_order[i:i+curr_batch_size]]            
            samples = curr_order[i:i+curr_batch_size]
            optimizer.zero_grad()
            sub_train_loss, sub_train_norm = n_model.L2_loss("train", args.batch_size, samples)
            train_loss += sub_train_loss
            train_norm += sub_train_norm
            optimizer.step() 

        train_fit = 1 - math.sqrt(train_loss) / math.sqrt(train_norm)
        with torch.no_grad():
            n_model.model.eval()
            val_loss, val_norm = n_model.L2_loss("val", args.batch_size, np.arange(n_model.input_mat.num_val))
                        
            val_fit = 1 - math.sqrt(val_loss) / math.sqrt(val_norm)                          
            if max_fit < val_fit:
                max_fit = val_fit     
                max_epoch = epoch
                prev_model = copy.deepcopy(n_model.model.state_dict())
          
        with open(args.save_path + ".txt", 'a') as lossfile:
            lossfile.write(f'epoch:{epoch}, train loss: {train_fit}, valid loss: {val_fit}\n')    
            print(f'epoch:{epoch}, train loss: {train_fit}, valid loss: {val_fit}')     
                   
        #if tol_count >= args.tol: break
    
    n_model.model.load_state_dict(prev_model)
    with torch.no_grad():
        test_loss, test_norm = n_model.L2_loss("test", args.batch_size, np.arange(n_model.input_mat.num_test))
        
        test_fit = 1 - math.sqrt(test_loss) / math.sqrt(test_norm)
        with open(args.save_path + ".txt", 'a') as lossfile:
            lossfile.write(f'epoch: {epoch}, test loss: {test_fit}\n')    
            print(f'epoch: {epoch}, test loss: {test_fit}\n')
    
    torch.save({
        'model_state_dict': prev_model,
        'loss': max_fit,
    }, args.save_path + ".pt")
            
    end_time = time.time()
    with open(args.save_path + ".txt", 'a') as lossfile:
        lossfile.write(f'running time: {end_time - start_time}\n')    
    print(f'running time: {end_time - start_time}')
    
    
def test(n_model, args):
    _device = torch.device("cuda:" + str(args.device[0]))     
    checkpoint = torch.load(args.load_path , map_location = _device)
    n_model.model.load_state_dict(checkpoint['model_state_dict'])          
    n_model.perm_list = checkpoint['perm']     
    for i in range(n_model.order):
        n_model.inv_perm_list[i][n_model.perm_list[i]] = torch.arange(n_model.input_mat.dims[i], device=_device)
    
    n_model.model.eval()
    with torch.no_grad():
        curr_loss = n_model.L2_loss(False, args.batch_size)
        print(f"saved loss: {checkpoint['loss']}, computed loss: {1 - math.sqrt(curr_loss) / n_model.input_mat.norm}")
    
            
# python main.py train -d ml -ip results/ml -de 0 1 2 3 -di 6040 3952 -rk 8 -hs 8 -sp results/ml_r8_h8 -lr 0.1 -e 50
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train')
    parser.add_argument("-d", "--dataset", type=str)   
    parser.add_argument("-ip", "--input_path", type=str)
    
    parser.add_argument(
        "-de", "--device",
        action="store", nargs='+', type=int
    )    

    parser.add_argument(
        "-di", "--dims",
        action="store", nargs='+', type=int
    )
    
    parser.add_argument(
        "-rk", "--rank",
        action="store", default=12, type=int
    )
    
    parser.add_argument(
        "-lr", "--lr",
        action="store", default=1e-3, type=float
    )
    
    parser.add_argument(
        "-e", "--epoch",
        action="store", default=500, type=int
    )
    
    parser.add_argument(
        "-b", "--batch_size",
        action="store", default=2**22, type=int
    )
    
    parser.add_argument(
        "-nb", "--num_batch",
        action="store", default=100, type=int
    )
    
    parser.add_argument(
        "-sp", "--save_path",
        action="store", default="./params/", type=str
    )
    
    parser.add_argument(
        "-lp", "--load_path",
        action="store", default="./params/", type=str
    )
        
    parser.add_argument(
        "-hs", "--hidden_size",
        action="store", default=11, type=int
    )
    
        
    args = parser.parse_args()      
    # decompsress m_list and n_list
    with open("input_size/" + args.dataset + ".txt") as f:
        lines = f.read().split("\n")
        input_size = [[int(word) for word in line.split()] for line in lines if line]        
     
    input_mat = tensor(args.dims, input_size, args.input_path, args.device[0])        
    print("load finish")
    
    if args.action == "train":
        n_model = TensorCodec(input_mat, args.rank, input_size, args.hidden_size, args.device)
        train_model(n_model, args)    
    else:
        assert(False)