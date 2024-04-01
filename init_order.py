import sys
import numpy as np
import torch
import time
import argparse
import pickle
import sys

def loss_fn(z, batch_size=-1):
    rets = []
    for j in range(0, z.shape[0]-1, batch_size):
        bsize = min(batch_size, z.shape[0]-1-j)
        rets.append((z[j:j+bsize] - z[j+1:j+bsize+1]).pow(2).sum(-1).pow(0.5))
    return torch.cat(rets, dim=-1)


def in_order_search(_tree_dict, curr_node):
    return_list = [curr_node]
    if curr_node not in _tree_dict: return return_list
    for child in _tree_dict[curr_node]:
        return_list = return_list + in_order_search(_tree_dict, child)
    return return_list


def reorder(load_path, order, device):
    with torch.no_grad():                    
        mxsz = (1 << 25)
        change_order, model2tens = [], []
        for mode in range(order):
            mat = np.load(args.load_path + f"{mode + 1}.npy")            
            mat = torch.from_numpy(mat).to(device)            
            batch_size = (mxsz // mat.shape[-1])           

            adj = torch.zeros(mat.shape[0] * mat.shape[0]).to(device).double()
            for j in range(0, mat.shape[0] * mat.shape[0], batch_size):
                bsize = min(batch_size, mat.shape[0] * mat.shape[0] - j)
                _idx = j + torch.arange(bsize, dtype=torch.long)
                _from, _to = mat[_idx // mat.shape[0]], mat[_idx % mat.shape[0]]
                adj[_idx] = (_from - _to).pow(2).sum(-1).pow(0.5)

            inf = adj.max() * 2 + 1
            adj = adj.view(mat.shape[0], mat.shape[0])
            dist = torch.ones(mat.shape[0], dtype=torch.double).to(device) * inf
            argdist = -torch.ones(mat.shape[0], dtype=torch.long).to(device)
            dist[0] = 0
            mask = torch.zeros(mat.shape[0], dtype=torch.long).to(device)
            edges = []
            mst_size = 0.
            tree_dict = {}
            for j in range(mat.shape[0]):
                new_node = torch.argmin(dist + (mask * inf * 2), dim=-1).item()
                parent_node = argdist[new_node].item()
                if parent_node >= 0:
                    edges.append((new_node, parent_node))
                    # mst_size += adj[new_node, parent_node].item()
                    if parent_node not in tree_dict:
                        tree_dict[parent_node] = []
                    tree_dict[parent_node].append(new_node)

                mask[new_node] += 1
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
            model2tens.append(y[cut_idx+1:-1] + y[:cut_idx+1])
            #counts = torch.bincount(torch.LongTensor(final_orders[-1]))
            new_length_sum = loss_fn(mat[model2tens[-1]], batch_size=batch_size).sum().item()
            change_order.append(((prev_length_sum - new_length_sum) > 0.1*prev_length_sum))
            print(f'order: {mode}, loss before: {prev_length_sum}, loss after: {new_length_sum}')
            del mat
    
    return change_order, model2tens

# python init_order.py -lp features/pems_80_factor -sp mapping/pems_80_model2tens -de 0 -di 963 144 440
# python init_order.py -lp features/kstock_factor -sp mapping/kstock_model2tens -de 0 -di 5270 88 1000
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--load_path', type=str)
    parser.add_argument('-sp', '--save_path', type=str)    
    parser.add_argument(
        "-de", "--device",
        action="store", default=0, type=int
    )

    parser.add_argument(
        "-di", "--dims",
        action="store", nargs='+', type=int
    )        
    args = parser.parse_args()    
    order = len(args.dims)

    start_time = time.time()
    device = torch.device(f'cuda:{args.device}')
    sys.setrecursionlimit(max(args.dims) + 1)
    change_order, model2tens = reorder(args.load_path, order, device)

    for i in range(order):
        if not change_order[i]: model2tens[i] = list(range(args.dims[i]))

    with open(args.save_path + ".pickle", mode="wb") as f:
        pickle.dump(model2tens, f)
    print("Total elapsed time:", time.time() - start_time)