import sys
import numpy as np
import torch
import time
import argparse
import pickle


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


def reorder(input_tensor):
    with torch.no_grad():    
        input_tensor = torch.from_numpy(input_tensor)
        dim = len(input_tensor.shape)
        mxsz = (1 << 25)
        change_order, final_orders = [], []
        for i in range(dim):
            mat = input_tensor.to(0).permute(i, *[j for j in range(dim) if j != i]).contiguous().view(input_tensor.shape[i], -1)
            batch_size = (mxsz // mat.shape[-1])           

            adj = torch.zeros(mat.shape[0] * mat.shape[0]).to(0).double()
            for j in range(0, mat.shape[0] * mat.shape[0], batch_size):
                bsize = min(batch_size, mat.shape[0] * mat.shape[0] - j)
                _idx = j + torch.arange(bsize, dtype=torch.long)
                _from, _to = mat[_idx // mat.shape[0]], mat[_idx % mat.shape[0]]
                adj[_idx] = (_from - _to).pow(2).sum(-1).pow(0.5)

            inf = adj.max() * 2 + 1
            adj = adj.view(mat.shape[0], mat.shape[0])
            dist = torch.ones(mat.shape[0], dtype=torch.double).to(0) * inf
            argdist = -torch.ones(mat.shape[0], dtype=torch.long).to(0)
            dist[0] = 0
            mask = torch.zeros(mat.shape[0], dtype=torch.long).to(0)
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
            final_orders.append(y[cut_idx+1:-1] + y[:cut_idx+1])
            #counts = torch.bincount(torch.LongTensor(final_orders[-1]))
            new_length_sum = loss_fn(mat[final_orders[-1]], batch_size=batch_size).sum().item()
            change_order.append(((prev_length_sum - new_length_sum) > 0.1*prev_length_sum))
            print(f'order: {i}, loss before: {prev_length_sum}, loss after: {new_length_sum}')
            del mat
    
    return change_order, final_orders

# python init_order.py -lp ../data/uber -sp mapping/uber_80_model2tens -de 0 -k 80
# python init_order.py -lp ../data/airquality -sp mapping/airquality_80_model2tens -de 0 -k 80
# python init_order.py -lp ../data/action -sp mapping/action_80_model2tens -de 0 -k 80
# python init_order.py -lp ../data/pems -sp mapping/pems_80_model2tens -de 0 -k 80
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--load_path', type=str)
    parser.add_argument('-sp', '--save_path', type=str)    
    parser.add_argument(
        "-de", "--device",
        action="store", default=0, type=int
    )

    parser.add_argument('-k', '--known_entry', type=str)
    
    args = parser.parse_args()   

    start_time = time.time()
    device = torch.device(f'cuda:{args.device}')

    # Load tensor, test set, valid set
    input_tensor = np.load(args.load_path + "_orig.npy")
    dims = input_tensor.shape
    order = len(dims)
    test_set = np.load(args.load_path + f"_{args.known_entry}_orig_test.npy")
    val_set = np.load(args.load_path + f"_{args.known_entry}_orig_valid.npy")        
    test_idx, val_idx = test_set[:, :order].astype(int), val_set[:, :order].astype(int)
    test_vals, val_vals = test_set[:, order].astype(np.float64), val_set[:, order].astype(np.float64)

    # load test set
    train_tensor = np.ones(input_tensor.shape)
    train_tensor[test_idx[:, 0], test_idx[:, 1], test_idx[: ,2]] = 0
    train_tensor[val_idx[:, 0], val_idx[:, 1], val_idx[:, 2]] = 0        
    train_idx = np.transpose(np.nonzero(train_tensor))
    train_mean = np.mean(input_tensor[train_idx[:, 0], train_idx[:, 1], train_idx[:, 2]])   
    
    # Filter input tensor
    input_tensor[test_idx[:, 0], test_idx[:, 1], test_idx[: ,2]] = train_mean
    input_tensor[val_idx[:, 0], val_idx[:, 1], val_idx[:, 2]] = train_mean        
    
    change_order, model2tens = reorder(input_tensor)
    for i in range(order):
        if not change_order[i]: model2tens[i] = list(range(dims[i]))

    with open(args.save_path + ".pickle", mode="wb") as f:
        pickle.dump(model2tens, f)
    print("Total elapsed time:", time.time() - start_time)