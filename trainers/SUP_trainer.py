import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import copy

from sklearn.metrics import f1_score, roc_auc_score
import ipdb
import torch
from torch_geometric.nn.inits import uniform
from plots import plot_chart, plot_bar, plot_tsne_anchored, plot_pca_anchored
import numpy as np
from scipy.sparse import coo_matrix
import utils
    
class Trainer_SUP(object):
    def __init__(self, args, model, device, loss, is_sup=True):
        if args.res:
            hidden_dim = args.nhid * args.nlayer
        else:
            hidden_dim = args.nhid

        self.model = model
        self.is_sup = is_sup

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        losses={}

        if self.is_sup:
            y_logit = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            loss_pred = F.nll_loss(y_logit[data.train_mask], data.y[data.train_mask])
            loss = loss_pred
            losses['pred loss']= loss_pred.item()

            loss.backward()
            self.optimizer.step()
        
        return losses
    
    def test(self, data, vis=False):
        self.model.eval()
        losses={}

        with torch.no_grad():
            y_logit = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            
        y_test = data.y[data.test_mask].detach().cpu().numpy()
        y_logit = y_logit[data.test_mask].detach()
        y_pred = y_logit.argmax(-1).detach().cpu().numpy()
        
        microF = f1_score(y_test, y_pred, average='micro')
        macroF = f1_score(y_test, y_pred, average='macro')
        acc = utils.accuracy(y_logit.cpu().numpy(), y_test)
            
        if y_test.max() > 1:
            auc_score = roc_auc_score(y_test, F.softmax(y_logit, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
        else:
            auc_score = roc_auc_score(y_test, F.softmax(y_logit, dim=-1)[:,1].detach().cpu(), average='macro')

        losses = {
                'micro_f1': microF,
                'macro_f1': macroF,
                'accuracy': acc,
                'auc': auc_score,
            }
        
        # group-wise perform
        logits = y_logit.cpu().numpy()
        y = data.y[data.test_mask].cpu().numpy()

        if vis:
            group1 = data.group_label[data.test_mask].cpu().numpy()
            group2 = data.group_label2[data.test_mask].cpu().numpy()

            group_acc1 = utils.grouped_accuracy(logits, y,group1)

            fig1 = plot_chart([np.array(group_acc1)], name_list=['group main accuracy'], x_start=1, x_name='group main', y_name='acc')
            losses['fig_1'] = fig1

            group_acc2 = utils.grouped_accuracy(logits, y,group2)
            fig2 = plot_chart([np.array(group_acc2)], name_list=['group margin accuracy'], x_start=1, x_name='group margin', y_name='acc')
            losses['fig_2'] = fig2
        
        return losses
