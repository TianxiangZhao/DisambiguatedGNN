import torch
import os
import argparse
import json

from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F
import ipdb
import numpy as np
from sklearn.cluster import SpectralClustering, DBSCAN
import random
from scipy.spatial.distance import pdist,squareform
import torch_geometric.utils as tg_utils 
from torch_geometric.data import Data
from torch.autograd import grad
from scipy.sparse import coo_matrix

from sklearn.metrics import confusion_matrix
# -------------------------------
# args, model save&load
# -------------------------------

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--no-sparse', action='store_false', default=True,
                    help='whether use sparse adj matrix')
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--task', type=str, default='pretrain', choices=['pretrain', 'joint'])
    parser.add_argument('--method', type=str, default='DGI', 
        choices=['DGI', 'GRACE','MVGRL', 'SUP', 'SUPBAL', 'BOUND']) 
    parser.add_argument('--model', type=str, default='gcn', 
        choices=['sage','gcn', 'gin', 'sgc' ])
    parser.add_argument('--loss', type=str, default='jsd', 
        choices=['jsd', 'infonce', ]) 
    parser.add_argument('--SUPBAL', action='store_true', default=False,
                    help='whether use SUPBAL as an auxiliary to existing methods')
    parser.add_argument('--normalize', action='store_true', default=False,
                    help='whether normalize features for SUPBAL')
    parser.add_argument('--do-map', action='store_true', default=False,
                    help='whether update the prototypes with mapping')
    parser.add_argument('--BAL_weights', nargs='+', type=float,
                    help='weights of BAL_related losses')
    parser.add_argument('--debias', action='store_true', default=False,
                    help='whether use debias the gradient in BAL contrastive')
    parser.add_argument('--usefilter', action='store_true', default=False,
                    help='whether filter out abnormality in contrastive')
    parser.add_argument('--filterrho', type=float, default=0.997,
                    help='rho for filter weighting')
    parser.add_argument('--classwise', action='store_true', default=False,
                    help='whether show classwise performance')
    parser.add_argument('--groupnumber', type=float, default=2.0,
                    help='group number as a ratio of class numbers')
    
    
    
    parser.add_argument('--Bound', action='store_true', default=False,
                    help='whether use BOUND as an auxiliary to existing methods')
    parser.add_argument('--Boundrho', type=float, default=0.8,
                    help='rho for BOUND for update ambiguity')
    parser.add_argument('--Boundthresh', type=float, default=0.5,
                    help='thresh for BOUND for selecting ambiguous nodes')
    parser.add_argument('--Posthresh', type=float, default=0.6,
                    help='thresh for BOUND for selecting pos neighbors')
    parser.add_argument('--Negthresh', type=float, default=0.4,
                    help='thresh for BOUND for selecting neg neighbors')
    
    parser.add_argument('--Boundweight', type=float, default=1.0,
                    help='weight for BOUND')
    parser.add_argument('--boundtype', type=str, default='prob', 
        choices=['prob','loss', 'emb' ])
    parser.add_argument('--aug_remote', action='store_true', default=False,
                    help='whether augment deambiguous with remote ones for contrasts')
    parser.add_argument('--remote_K', type=int, default=5,
                    help='number of remote similar nodes as positive')
    parser.add_argument('--boun_aug', action='store_true', default=False,
                    help='whether augment deambiguous with boundary node augmentation')
    
    
    
    # parser.add_argument('--scale', type=str, default='small', choices=['small', 'large'])
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora_full','cora','computer', 'chameleon', 'squirrel','reddit','blogcatalog','flickr', 'photo','pubmed','actor' ]) 
    parser.add_argument('--nlayer', type=int, default=2)#intermediate feature dimension
    parser.add_argument('--nhid', type=int, default=64)#intermediate feature dimension
    parser.add_argument('--nclass', type=int, default=5)#number of labels
    parser.add_argument('--nfeat', type=int, default=64) # input feature dimension
    parser.add_argument('--res', action='store_true', default=False) # whether use residual in GNN
    parser.add_argument('--epochs', type=int, default=2001,
                help='Number of epochs to train.')
    
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=1024,
                help='Number of batches inside an epoch.')

    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    #parser.add_argument('--test_ratio', type=float, default=0.6)

    parser.add_argument('--save', action='store_true', default=False,help='whether save checkpoints')
    parser.add_argument('--log', action='store_true', default=False,
                    help='whether creat tensorboard logs')
    
    parser.add_argument('--load', type=str, default=None) #To indicate pre-train in other folders. Like "./checkpoint/SpuriousMotif_0.3/best".
    parser.add_argument('--load_config', action='store_true', default=False, help='whether load training configurations')

    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()

    if args.load_config:
        config_path='./configs/{}/{}/{}'.format(args.task,args.dataset,args.model)
        with open(config_path) as f:
            args.__dict__ = json.load(f)

    return args

def save_args(args):
    config_path='./configs/{}/{}/'.format(args.task,args.dataset)

    if not os.path.exists(config_path):
        os.makedirs(config_path)

    with open(config_path+args.model,'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    return

def save_model(args, model, epoch=None, name='model'):
    saved_content = {}
    saved_content[name] = model.state_dict()

    path = './checkpoint/{}/{}'.format(args.dataset, args.model)
    if not os.path.exists(path):
        os.makedirs(path)

    #torch.save(saved_content, 'checkpoint/{}/{}_epoch{}_edge{}_{}.pth'.format(args.dataset,args.model,epoch, args.used_edge, args.method))
    if epoch is not None:
        torch.save(saved_content, os.path.join(path,'{}_{}.pth'.format(name, epoch)))
        print("successfully saved: {}".format(epoch))
    else:
        torch.save(saved_content, os.path.join(path,'{}.pth'.format(name)))
        print("successfully saved: {}".format(name))

    return

def load_model(args, model, name='model_500'):
    
    loaded_content = torch.load('./checkpoint/{}/{}/{}.pth'.format(args.dataset, args.model,name), map_location=lambda storage, loc: storage)

    model.load_state_dict(loaded_content['best'])

    print("successfully loaded: {}.pth".format(name))

    return model



# -------------------------------
# evaluation
# -------------------------------
def accuracy(logits, labels):
    preds = logits.argmax(-1)
    correct = (preds==labels).sum()
    return correct / len(labels)

def grouped_accuracy(logits, labels, group_labels):
    # all inputs should be stored in numpy array

    preds = logits.argmax(1)
    group_ac_list = []
    for group in set(group_labels):
        group_idx = group_labels==group
        group_ac = (preds[group_idx]==labels[group_idx]).sum()/(group_idx.sum()+0.00000001)
        group_ac_list.append(group_ac)

    return group_ac_list

def balance_precision(logits, labels, sigma=0.5):
    
    preds = logits.argmax(-1)
    matrix = confusion_matrix(labels, preds)
    class_prec = matrix.diagonal()/matrix.sum(axis=1)
    total_score = 0
    for c1 in set(labels):
        for c2 in set(labels):
            cur_score = np.exp(-np.square(class_prec[c1] - class_prec[c2])/sigma)
            total_score += cur_score

    total_score/=np.square(len(set(labels)))

    return total_score




def grouped_measure(measures, group_labels):
    # all inputs should be stored in numpy array
    # measures: calculated measures on each input

    group_measure_dict={}
    group_measure_list=[]
    for group in set(group_labels):
        group_idx = group_labels==group
        group_measure = (measures[group_idx]).sum(0)/(group_idx.sum()+0.00000001)
        group_measure_dict[group] = group_measure
        
        group_measure_list.append(group_measure)

    return group_measure_list

def group_consistency(group_labels, labels):
    # label consistency inside each group

    micro_consis = meters()
    macro_consis = meters()

    group_cons_dict={}
    for group in set(group_labels):
        group_idx = group_labels==group
        label_count = np.bincount(labels[group_idx])
        group_cons_dict[group] = label_count.max()/group_idx.sum() # ratio of 
        
        micro_consis.update(group_cons_dict[group],group_idx.sum()) #
        macro_consis.update(group_cons_dict[group],1)

    return group_cons_dict, micro_consis.avg(), macro_consis.avg()

def grided_similarity(similarity, group_label, class_label, adj_csr):
    data_lists = [[],[],[], []]
    name_lists = ['same_class_neigh', 'largest_neigh', 'avg neigh', 'same_class']

    for i_group, group in enumerate(set(group_label)):
        data_meters = [meters() for i in range(len(name_lists))]

        sel_idx = (group_label==group).nonzero()[0]

        for sel_id in sel_idx:
            neighbor_idx = adj_csr[sel_id].nonzero()[-1]

            if len(neighbor_idx) > 0:
                neibhor_cls = class_label[neighbor_idx]
                sel_cls = class_label[sel_id]

                # same_class_neighbor:
                if sel_cls in set(neibhor_cls):
                    sel_neibhors = neighbor_idx[neibhor_cls==sel_cls]
                    data_meters[0].update(similarity[sel_id][sel_neibhors].mean())
                
                # largest_neigh:
                sel_similarity = similarity[sel_id][neighbor_idx]
                sel_classes = class_label[neighbor_idx]
                class_wei_max = 0
                for sel_class in set(sel_classes):
                    class_wei = sel_similarity[sel_classes==sel_class].mean()
                    if class_wei > class_wei_max:
                        class_wei_max = class_wei
                data_meters[1].update(class_wei_max)

                # avg neibhbor
                data_meters[2].update(sel_similarity.mean())

                # same class
                data_meters[3].update(similarity[sel_id][class_label==sel_cls].mean())


        for i in range(len(name_lists)):
            data_lists[i].append(data_meters[i].avg())

    data_lists_np = [np.array(data_list) for data_list in data_lists]

    return data_lists_np, name_lists


def print_class_acc(logits, labels, pre='valid'):
    #print class-wise performance
    
    for i in range(labels.max()+1):
        index_pos = labels==i
        cur_tpr = accuracy(logits[index_pos], labels[index_pos])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_neg = labels != i
        labels_neg = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(logits[index_neg,:], labels_neg[index_neg])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))
    

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return

def Roc_F(logits, labels, pre='valid'):
    #print class-wise performance
    '''
    pre_num = 0
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(logits[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(logits[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    if labels.max() > 1:#require set(labels) to be the same as columns of logits 
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return auc_score, macro_F

def groupewise_perform(logits, labels, group_label, pre='valid'):
    #print class-wise performance
    
    accs = meters()
    aucs = meters()
    Fs = meters()

    for i in range(group_label.max()+1):
        index_pos = group_label==i
        cur_acc = accuracy(logits[index_pos], labels[index_pos]).item()
        #cur_auc = roc_auc_score(labels[index_pos].detach().cpu(), F.softmax(logits[index_pos], dim=-1).detach().cpu(), average='macro', multi_class='ovr')
        cur_auc = 0 #not computed for now, if a group does not have all classes
        cur_F = f1_score(labels[index_pos].detach().cpu(), torch.argmax(logits[index_pos], dim=-1).detach().cpu(), average='macro')
        
        accs.update(cur_acc)
        aucs.update(cur_auc)
        Fs.update(cur_F)

    return accs.avg(), aucs.avg(), Fs.avg()

class meters:
    '''
    collects the results at each inference batch, and return the result in total
    param orders: the order in updating values
    '''
    def __init__(self, orders=1):
        self.avg_value = 0
        self.tot_weight = 0
        self.orders = orders
        
    def update(self, value, weight=1.0):
        value = float(value)

        if self.orders == 1:
            update_step = self.tot_weight/(self.tot_weight+weight)
            self.avg_value = self.avg_value*update_step + value*(1-update_step)
            self.tot_weight += weight
        

    def avg(self):

        return self.avg_value
    


# -------------------------------
# dataset split
# -------------------------------

def balanced_split(data, train_num: int = 13, valid_num=1):
    train_idx = []
    val_idx = []
    test_idx = []
    
    labels = data.y.cpu().numpy()
    label_set = np.unique(labels)
    for label_i in label_set:
        c_idx = (labels==label_i).nonzero()[0]
        assert len(c_idx) > train_num + valid_num, "class {} size {} smaller than training and validation size".format(label_i, len(c_idx))

        train_idx = train_idx + c_idx[:train_num].tolist()
        val_idx = val_idx + c_idx[train_num:train_num+valid_num].tolist()
        test_idx = test_idx + c_idx[train_num+valid_num:].tolist()

    train_mask = torch.LongTensor(train_idx)
    val_mask = torch.LongTensor(val_idx)
    test_mask = torch.LongTensor(test_idx)

    
    return {
        'train': train_mask,
        'valid': val_mask,
        'test': test_mask
    }
    

