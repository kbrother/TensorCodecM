import sys
import numpy as np
import torch
import time
import argparse
from tqdm import tqdm

# z: j_i x j_1*...*j_d
def loss_fn(z, batch_size=-1):
    rets = []
    total_avg = torch.sum(z) / torch.count_nonzero(z)
    nnz_cnt_row = torch.sum(z!=0, dim=1)
    nnz_cnt_col = torch.sum(z!=0, dim=0)
    mat_avg_row = torch.sum(z, dim=1) / nnz_cnt_row  # j_i
    mat_avg_col = torch.sum(z, dim=0) / nnz_cnt_col   # j1*...*j_d
    mat_avg_row[nnz_cnt_row == 0] = total_avg
    mat_avg_col[nnz_cnt_col == 0] = total_avg
    for j in range(0, z.shape[0]-1, batch_size):
        bsize = min(batch_size, z.shape[0]-1-j)
        _from, _to = z[j:j+bsize], z[j+1:j+bsize+1]

        # fill in the values in from
        empty_entry = torch.logical_and(_from == 0, _to != 0)   # bsize x j1*...*j_d
        if torch.sum(empty_entry) > 0:
            empty_idx = torch.nonzero(empty_entry)   # nnz x 2
            curr_row = j + empty_idx[:,0]
            fill_in_val = 0.5 * (mat_avg_row[curr_row] + mat_avg_col[empty_idx[:, 1]])
            _from[empty_idx[:, 0], empty_idx[:, 1]] = fill_in_val

        # fill in the value in to
        empty_entry = torch.logical_and(_from != 0, _to == 0)
        if torch.sum(empty_entry) > 0:
            empty_idx = torch.nonzero(empty_entry)
            curr_row = j + empty_idx[:,0]
            fill_in_val = 0.5 * (mat_avg_row[curr_row] + mat_avg_col[empty_idx[:, 1]])
            _to[empty_idx[:, 0], empty_idx[:, 1]] = fill_in_val

        num_cup = torch.sum(torch.logical_or(_from != 0, _to != 0), dim=-1)
        _dist = (_from - _to).pow(2).sum(-1).pow(0.5) * torch.sqrt(z.shape[1]/num_cup) 
        _dist[_dist.isnan()] = 0
        
        rets.append(_dist)
    return torch.cat(rets, dim=-1)


def in_order_search(_tree_dict, curr_node):
    return_list = [curr_node]
    if curr_node not in _tree_dict: return return_list
    for child in _tree_dict[curr_node]:
        return_list = return_list + in_order_search(_tree_dict, child)
    return return_list


def reorder(idx, vals, dims, device):
    with torch.no_grad():    
        order = len(dims)
        input_tensor = torch.zeros(*dims, dtype=torch.double)   
        vals = torch.tensor(vals)
        if order == 2:
            input_tensor[idx[:, 0], idx[:, 1]] = vals
        elif order == 3:
            input_tensor[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
        elif order == 4:
            input_tensor[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = vals
        else:
            raise ValueError("need an additional implementation for a higher order tensor")    

        mxsz = (1 << 25)
        total_avg = torch.sum(input_tensor) / torch.count_nonzero(input_tensor)
        change_order, final_orders = [], []
        for i in range(order):
            mat = input_tensor.to(device).permute(i, *[j for j in range(order) if j != i]).contiguous().view(input_tensor.shape[i], -1)            
            nnz_cnt_row = torch.sum(mat != 0, dim=1)
            nnz_cnt_col = torch.sum(mat != 0, dim=0)
            mat_avg_row = torch.sum(mat, dim=1) / nnz_cnt_row  # j_i
            mat_avg_col = torch.sum(mat, dim=0) / nnz_cnt_col   # j1*...*j_d
            mat_avg_row[nnz_cnt_row == 0] = total_avg
            mat_avg_col[nnz_cnt_col == 0] = total_avg
            
            batch_size = (mxsz // mat.shape[-1])           
            adj = torch.zeros(mat.shape[0] * mat.shape[0]).to(device).double()
            for j in tqdm(range(0, mat.shape[0] * mat.shape[0], batch_size)):
                bsize = min(batch_size, mat.shape[0] * mat.shape[0] - j)
                _idx = j + torch.arange(bsize, dtype=torch.long)                
                _from, _to = mat[_idx // mat.shape[0]], mat[_idx % mat.shape[0]]  # bsize x j1*...*j_d

                # fill in the values in from
                empty_entry = torch.logical_and(_from == 0, _to != 0)   # bsize x j1*...*j_d
                if torch.sum(empty_entry) > 0:
                    empty_idx = torch.nonzero(empty_entry)   # nnz x 2
                    curr_row = (j + empty_idx[:,0]) // mat.shape[0]
                    fill_in_val = 0.5 * (mat_avg_row[curr_row] + mat_avg_col[empty_idx[:, 1]])
                    _from[empty_idx[:, 0], empty_idx[:, 1]] = fill_in_val

                # fill in the value in to
                empty_entry = torch.logical_and(_from != 0, _to == 0)
                if torch.sum(empty_entry) > 0:
                    empty_idx = torch.nonzero(empty_entry)
                    curr_row = (j + empty_idx[:,0]) % mat.shape[0]
                    fill_in_val = 0.5 * (mat_avg_row[curr_row] + mat_avg_col[empty_idx[:, 1]])
                    _to[empty_idx[:, 0], empty_idx[:, 1]] = fill_in_val

                num_cup = torch.sum(torch.logical_or(_from != 0, _to != 0), dim=-1)
                #assert(torch.count_nonzero(num_cup) == bsize)
                _dist = (_from - _to).pow(2).sum(-1).pow(0.5)         
                adj[_idx] = _dist * torch.sqrt(mat.shape[1]/num_cup) 
                adj[adj.isnan()] = 0
            
            inf = adj.max() * 2 + 1
            adj = adj.view(mat.shape[0], mat.shape[0])
            dist = torch.ones(mat.shape[0], dtype=torch.double).to(device) * inf
            argdist = -torch.ones(mat.shape[0], dtype=torch.long).to(device)
            dist[0] = 0
            mask = torch.zeros(mat.shape[0], dtype=torch.bool).to(device)            
            tree_dict = {}
            for j in range(mat.shape[0]):
                temp_dist = dist.clone()
                temp_dist[mask] = inf
                new_node = torch.argmin(temp_dist).item()
                parent_node = argdist[new_node].item()
                if parent_node >= 0:                    
                    if parent_node not in tree_dict:
                        tree_dict[parent_node] = []
                    tree_dict[parent_node].append(new_node)

                mask[new_node] = True
                new_vals = adj[new_node]
                need_update = (dist > new_vals)
                dist[need_update] = new_vals[need_update]
                argdist[need_update] = new_node

            prev_length_sum = loss_fn(mat, batch_size=batch_size).sum().item()
            y = in_order_search(tree_dict, 0)
            y.append(0)
            
            new_mat = mat[y]
            lengths = loss_fn(new_mat, batch_size=batch_size)
            cut_idx = lengths.argmax(-1)
            final_orders.append(y[cut_idx+1:-1] + y[:cut_idx+1])
            
            new_length_sum = loss_fn(mat[final_orders[-1]], batch_size=batch_size).sum().item()
            change_order.append(prev_length_sum > new_length_sum)
            print(f'order: {i}, loss before: {prev_length_sum}, loss after: {new_length_sum}')
            del mat
            
    return change_order, final_orders

# python TensorCodec_completion/init_order.py -di 6040 3952 -de 0 -lp data/TensorCodec_journal/ml -sp TensorCodec_completion/results/ml
# python TensorCodec_completion/init_order.py -di 5600 362 6 -de 0 -lp data/airquality_orig -sp TensorCodec_completion/results/airquality
# python TensorCodec_completion/init_order.py -di 963 144 440 -de 0 -lp data/pems_orig -sp TensorCodec_completion/results/pems
# python TensorCodec_completion/init_order.py -di 192 288 30 120 -de 0 -lp data/absorb_orig -sp TensorCodec_completion/results/absorb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-di", "--dims",
        action="store", nargs="+", type=int
    )
    
    parser.add_argument("-de", "--device", default=0, type=int)
    parser.add_argument('-lp', '--load_path', type=str)
    parser.add_argument('-sp', '--save_path', type=str)    
    
    args = parser.parse_args()
    train_set = np.load(args.load_path + "_train.npy")
    valid_set = np.load(args.load_path + "_valid.npy")
    test_set = np.load(args.load_path + "_test.npy")
    known_entry = np.concatenate((train_set, valid_set), axis=0)

    order = known_entry.shape[1] - 1
    device = torch.device(f"cuda:{args.device}")
    idx, val = known_entry[:, :-1], known_entry[:, -1]
    
    start_time = time.time()    
    change_order, final_orders = reorder(idx, val, args.dims, device)
    # final_orders: reordered tensor -> original tensor

    inv_final_orders = []
    for i in range(order):
        final_orders[i] = np.array(final_orders[i])
        print(f'order: {i}, mean: {np.mean(final_orders[i])}')
        inv_final_orders.append(np.copy(final_orders[i]))
    for i in range(order):
        inv_final_orders[i][final_orders[i]] = np.arange(args.dims[i])
    # inv_final_orders: origianl tensor -> reordered tensor

    for i in range(order):
        if change_order[i]:
            train_set[:, i] = inv_final_orders[i][train_set[:, i].astype(int)]
            valid_set[:, i] = inv_final_orders[i][valid_set[:, i].astype(int)]
            test_set[:, i] = inv_final_orders[i][test_set[:, i].astype(int)]
        
    print("Total elapsed time:", time.time() - start_time)
    np.save(args.save_path + "_train.npy", train_set)
    np.save(args.save_path + "_valid.npy", valid_set)
    np.save(args.save_path + "_test.npy", test_set)