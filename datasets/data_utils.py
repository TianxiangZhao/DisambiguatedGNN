import torch
import numpy as np
import ipdb

from scipy.sparse import coo_matrix

def class_size_label(y: torch.Tensor, split=5):
    cs_label = y.new(y.shape).fill_(-1)
    class_numbers = []

    labels, count = y.unique(return_counts=True)


    max_c = count.max()
    min_c = count.min()
    for iter in range(split):
        bound = (max_c-min_c)/(split-iter)
        
        sel_label_idx = count >=(max_c-bound)
        sel_label = labels[sel_label_idx]

        for i, label in enumerate(y):
            if (label == sel_label).any():
                cs_label[i] = iter
        class_numbers.append(sel_label.shape[0])

        count[sel_label_idx] = -1
        max_c = count.max()

    assert not (cs_label==-1).any(), "cs label not assigned"

    print("class size group: {}".format(class_numbers))

    return cs_label

def tensor2coo(x):
        """ converts tensor x to scipy coo matrix """

        node_num = x.shape[0]
        if not x.is_sparse:
            indices = torch.nonzero(x)
            indices = indices.t()
            values = x[list(indices[i] for i in range(indices.shape[0]))].cpu().numpy()
        else:
            indices = x.coalesce().indices()  
            values = x.coalesce().values().cpu().numpy()
        if len(indices.shape) == 0:  # if all elements are zeros
            return coo_matrix((node_num, node_num), dtype=np.int8)
        
        row = indices[0,:].cpu().numpy()
        column = indices[1,:].cpu().numpy()
        

        return coo_matrix((values,(row,column)), shape=(node_num, node_num)) 

def boundary_label(y: torch.Tensor, edge_index: torch.Tensor, cs_label: torch.Tensor, cs_group_num=5):
    # based on most neighbors
    # need to be in the cpu
    bound_label = y.new(y.shape).fill_(-1)

    adj_coo = coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0],edge_index[1])),shape=(y.shape[0],y.shape[0]))
    adj_csr = adj_coo.tocsr()

    for i, label in enumerate(y):

        neighbors = adj_csr[i].nonzero()[-1]
        if neighbors.shape[0] == 0:
            bound_label[i] = 9
            continue

        neighbor_class_group = cs_label[neighbors]
        neighbor_y = y[neighbors]

        neighbor_class, neighbor_class_count = neighbor_y.unique(return_counts=True)
        neighbor_class_max = neighbor_class[neighbor_class_count.argmax()]
        max_index = (neighbor_y==neighbor_class_max).nonzero()[0][0]
        max_class_group = neighbor_class_group[max_index]

        if cs_label[i]<= int(cs_group_num/2)-1: #majority classes
            if neighbor_class_max == y[i]:
                bound_label[i] = 0
            elif max_class_group >= cs_group_num/4*3: # most neighbors are tail classes
                bound_label[i] = 1
            else:
                bound_label[i] = 2
        elif cs_label[i]>= int(cs_group_num/4*3): #last several classes
            if neighbor_class_max == cs_label[i]:
                bound_label[i] = 3
            elif max_class_group >= cs_group_num/4*3: #
                bound_label[i] = 4
            else:
                bound_label[i] = 5
        else:
            if neighbor_class_max == cs_label[i]:
                bound_label[i] = 6
            elif max_class_group >= cs_group_num/4*3: #
                bound_label[i] = 7
            else:
                bound_label[i] = 8


    assert not (bound_label==-1).any(), "boundary label not assigned"

    return bound_label

def boundary_label2(y: torch.Tensor, edge_index: torch.Tensor, cs_label: torch.Tensor, cs_group_num=5):
    # based on whether direct neighbor exist

    bound_label = y.new(y.shape).fill_(-1)

    adj_coo = coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0],edge_index[1])),shape=(y.shape[0],y.shape[0]))
    adj_csr = adj_coo.tocsr()

    for i, label in enumerate(y):

        neighbors = adj_csr[i].nonzero()[-1]
        if neighbors.shape[0] == 0:
            bound_label[i] = 4

            continue
        neighbor_y = y[neighbors]
        neighbor_class_group = cs_label[neighbors]

        if (neighbor_class_group >= cs_group_num-2).any():
            if cs_label[i] >= cs_group_num-2 and (neighbor_y == y[i]).any():
                bound_label[i] = 2
            else:
                bound_label[i] = 3 #adjacent to minority class, not self
        else:
            if (neighbor_y == y[i]).any():
                bound_label[i] = 0 
            else:
                bound_label[i] = 1 # not adjacent to self majority class

    assert not (bound_label==-1).any(), "boundary label not assigned"

    return bound_label