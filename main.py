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
    n_model.model.eval()
    with torch.no_grad():
        samples = np.arange(n_model.input_mat.num_train)
        curr_loss, _ = n_model.L2_loss(False, "train", args.batch_size, samples)

    print(f'initial loss: {curr_loss}')
    for j in range(n_model.input_mat.order):
        delta_loss = n_model.reordering(j, args.batch_size)
        curr_loss += delta_loss

    with torch.no_grad():
        real_loss, _ = n_model.L2_loss(False, "train", args.batch_size, samples)
        print(f'our loss: {curr_loss}, real loss:{real_loss}')
        
        
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
        optimizer = torch.optim.Adam(n_model.model.parameters(), lr=args.lr/args.num_batch, weight_decay=0) 
        n_model.model.train()               
        curr_order = np.random.permutation(n_model.input_mat.num_train)            
        
        # Gradietn descent
        train_loss, train_norm = 0, 0
        for i in tqdm(range(0, n_model.input_mat.num_train, minibatch_size)):
            curr_batch_size = min(n_model.input_mat.num_train - i, minibatch_size)            
            #samples = n_model.input_mat.src_train_idx[curr_order[i:i+curr_batch_size]]            
            samples = curr_order[i:i+curr_batch_size]
            optimizer.zero_grad()
            sub_train_loss, sub_train_norm = n_model.L2_loss(True, "train", args.batch_size, samples)
            train_loss += sub_train_loss
            train_norm += sub_train_norm
            optimizer.step() 

        
        # Reordering
        n_model.model.eval()
        for j in range(n_model.input_mat.order):
            delta_loss = n_model.reordering(j, args.batch_size)
            train_loss += delta_loss
        
        train_fit = 1 - math.sqrt(train_loss) / math.sqrt(train_norm)
        with torch.no_grad():
            val_loss, val_norm = n_model.L2_loss(False, "val", args.batch_size, np.arange(n_model.input_mat.num_val))
            test_loss, test_norm = n_model.L2_loss(False, "test", args.batch_size, np.arange(n_model.input_mat.num_test))
            test_fit = 1 - math.sqrt(test_loss)/math.sqrt(test_norm)
            val_fit = 1 - math.sqrt(val_loss)/math.sqrt(val_norm) 
            
            if max_fit < 0.5*(train_fit + val_fit):
                max_fit = 0.5*(train_fit + val_fit)
                max_epoch = epoch
                prev_model = copy.deepcopy(n_model.model.state_dict())
          
        with open(args.save_path + ".txt", 'a') as lossfile:
            lossfile.write(f'epoch:{epoch}, train loss: {train_fit}, valid loss: {val_fit}, test loss: {test_fit}\n')    
            print(f'epoch:{epoch}, train loss: {train_fit}, valid loss: {val_fit}, test loss: {test_fit}')     
                   
        #if tol_count >= args.tol: break
    
    n_model.model.load_state_dict(prev_model)
    with torch.no_grad():
        train_loss, train_norm = n_model.L2_loss(False, "train", args.batch_size, np.arange(n_model.input_mat.num_train))
        val_loss, val_norm = n_model.L2_loss(False, "val", args.batch_size, np.arange(n_model.input_mat.num_val))
        test_loss, test_norm = n_model.L2_loss(False, "test", args.batch_size, np.arange(n_model.input_mat.num_test))
        
        train_fit = 1 - math.sqrt(train_loss) / math.sqrt(train_norm)
        val_fit = 1 - math.sqrt(val_loss)/math.sqrt(val_norm) 
        test_fit = 1 - math.sqrt(test_loss) / math.sqrt(test_norm)
        with open(args.save_path + ".txt", 'a') as lossfile:
            lossfile.write(f'epoch: {max_epoch}, train loss: {train_fit}, valid loss: {val_fit}, test loss: {test_fit}\n')    
            print(f'epoch: {max_epoch}, train loss: {train_fit}, valid loss: {val_fit}, test loss: {test_fit}\n')
    
    torch.save({
        'model_state_dict': prev_model,
        'loss': max_fit,
    }, args.save_path + ".pt")
            
    end_time = time.time()
    with open(args.save_path + ".txt", 'a') as lossfile:
        lossfile.write(f'running time: {end_time - start_time}\n')    
    print(f'running time: {end_time - start_time}')
    

# python main.py test_perm -d uber -ip ../data/uber -k 5 -de 0 1 2 3 -hs 12 -r 12 -lr 0.1 -sp results/uber
# python main.py train -d uber -ip ../data/uber -k 5 -de 0 1 2 3 -hs 12 -r 12 -lr 0.1 -sp results/uber
# python main.py train -d airquality -ip ../data/airquality -k 5 -de 0 1 2 3 -hs 8 -r 8 -lr 0.1 -sp results/airquality
# python main.py train -d action -ip ../data/action -k 5 -de 0 1 2 3 -hs 8 -r 8 -lr 0.1 -sp results/action
# python main.py train -d pems -ip ../data/pems -k 5 -de 0 1 2 3 -hs 8 -r 8 -lr 0.1 -sp results/pems
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train')
    parser.add_argument("-d", "--dataset", type=str)   
    parser.add_argument("-ip", "--input_path", type=str)
    parser.add_argument("-k", "--known_entry", type=int)
    
    parser.add_argument(
        "-de", "--device",
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
        action="store", default=2**20, type=int
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
     
    input_mat = tensor(input_size, args.input_path, args.device[0], args.known_entry)        
    print("load finish")

    n_model = TensorCodec(input_mat, args.rank, input_size, 
                          args.hidden_size, args.device, args)
    if args.action == "train":        
        train_model(n_model, args)    
    elif args.action == "test_perm":
        test_perm(n_model, args)
    else:
        assert(False)