# identify nodes in the boundary, and contrast over them.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import copy

import ipdb
import torch
from torch_geometric.nn.inits import uniform
import GCL.augmentors as A
import GCL.losses as L
from contrasts import DualBranchContrast_ex, SingleBranchContrast_ex
from plots import plot_chart,plot_dist1D
import utils
from sklearn.metrics import f1_score, roc_auc_score

class Encoder_aug(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim=32):
        super(Encoder_aug, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        z = self.encoder.embedding(x, edge_index, edge_weight)
        z1 = self.encoder.embedding(x1, edge_index1, edge_weight1)
        return z, z1

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
class Trainer_BoundCont(object):
    def __init__(self, args, model, device, loss, is_sup=False, rho=0.8, thresh=0.7, type='prob', weight=1.0, vis_freq=600):
        # identify boundary nodes in the graph
        # can be used only for loss computation, and worked under other trainers
        # rho: control slow-update of the prob_summary
        # thresh: if None, use soft selection with sampling. else, within [0,1], to control which nodes identified as boundary
        print("not support sampled subgraph yet")
        self.args=args
        if args.res:
            hidden_dim = args.nhid * args.nlayer
        else:
            hidden_dim = args.nhid

        
        aug1 = A.Compose([A.EdgeRemoving(pe=0.3)])
        self.encoder = model
        self.encoder_model = Encoder_aug(encoder=model, augmentor=(aug1,), hidden_dim=hidden_dim).to(device)
        self.contrast_model = DualBranchContrast_ex(loss=loss, mode='L2L', intraview_negs=True).to(device)

        self.loss = loss
        self.model = model
        self.is_sup = is_sup

        self.rho = rho
        self.thresh = thresh
        self.pos_thresh = args.Posthresh
        self.neg_thresh = args.Negthresh
        self.weight = weight
        self.type = type
        if type == 'prob':
            self.update_weight = self._update_prob
            self.prob_summary = torch.zeros((args.nsample,args.nclass), requires_grad=False, device=device)
        elif type =='loss':
            self.update_weight = self._update_loss
            self.prob_summary = torch.zeros((args.nsample,), requires_grad=False, device=device)
        elif type =='emb':
            self.update_weight = self._update_emb
            self.prob_summary = torch.zeros((args.nsample,hidden_dim), requires_grad=False, device=device)
        else:
            ipdb.set_trace()

        self.aug_remote = args.aug_remote
        self.remote_K = args.remote_K
        self.boun_aug = args.boun_aug

        self.boundary_weight =  torch.zeros((args.nsample,), requires_grad=False, device=device)
        self.call_count = 0
        self.vis_freq = vis_freq

        self.neighbor_idx = {}


        if self.boun_aug:
            self.optimizer = optim.Adam(self.encoder_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.Adam(self.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    def _update_prob(self, logits,):
        # update metric of each node for being in the boundary, based on classification boundary
        prob = torch.softmax(logits, dim=-1)
        self.prob_summary = self.prob_summary*self.rho + prob*(1-self.rho)
        self.boundary_weight = -torch.mul(self.prob_summary, torch.log(self.prob_summary)).sum(dim=-1)

        # find quantile of self.boundary_weight
        quantile = torch.quantile(self.boundary_weight, 0.25)
        self.boundary_weight[self.boundary_weight<quantile] = quantile


        self.boundary_weight -= self.boundary_weight.min()
        self.boundary_weight /= self.boundary_weight.max()
        
        return self.boundary_weight 
    
    def _update_loss(self, loss,):
        # update metric of each node for being in the boundary, based on losses, like contrastive learning
        self.prob_summary = self.prob_summary*self.rho + loss*(1-self.rho)
        
        self.boundary_weight -= self.boundary_weight.min()
        self.boundary_weight /= self.boundary_weight.max()

        return self.boundary_weight
    
    def _update_emb(self, z,):
        z = z.detach()

        self.boundary_weight = torch.einsum('bij, bjk->bik', self.prob_summary.unsqueeze(1), z.unsqueeze(2)).squeeze() / (torch.sqrt(torch.norm(self.prob_summary,dim=-1))*torch.sqrt(torch.norm(z,dim=-1))) # 
        
        self.prob_summary = self.prob_summary*self.rho + z*(1-self.rho)
        
        self.boundary_weight -= self.boundary_weight.min()
        self.boundary_weight /= self.boundary_weight.max()

        return self.boundary_weight
    
    def _neighbor_contrast_loss(self, embeddings, edge_index):
        # 
        sel_idx = (self.boundary_weight>self.thresh).nonzero().flatten()

        #pos_mask = torch.eye(embeddings.shape[0],device=embeddings.device)
        if len(sel_idx) == 0:
            print("not identify any boundary nodes")
            pos_mask = torch.eye(embeddings.shape[0],device=embeddings.device)
            neg_mask = torch.eye(embeddings.shape[0],device=embeddings.device)
        else:
            pos_mask = torch.zeros((embeddings.shape[0],embeddings.shape[0]),device=embeddings.device)
            neg_mask = torch.zeros((embeddings.shape[0],embeddings.shape[0]),device=embeddings.device)

        embeddings = F.normalize(embeddings, dim=-1)
        #
        with torch.no_grad():
            sel_similarity = embeddings[sel_idx].detach() @ embeddings.detach().t()
            sel_similarity.fill_diagonal_(0)

        for sel_id, node_id in enumerate(sel_idx):
            if node_id in self.neighbor_idx.keys():
                neighbor_idx = self.neighbor_idx[node_id]
            else:
                neighbor_idx = edge_index[1,:][(edge_index[0,:]==node_id).nonzero().flatten()]
                neighbor_idx2 = edge_index[0,:][(edge_index[1,:]==node_id).nonzero().flatten()]
                neighbor_idx = torch.cat([neighbor_idx, neighbor_idx2], dim=-1)
                self.neighbor_idx[node_id] = neighbor_idx


            if len(neighbor_idx) !=0:
                with torch.no_grad():
                    distances = embeddings[node_id] @ embeddings[neighbor_idx].T
                    pos_idx = neighbor_idx[distances>=distances.max()*self.pos_thresh]
                    neg_idx = neighbor_idx[distances<=distances.max()*self.neg_thresh]

                    pos_mask[node_id,:][pos_idx] = 1
                    neg_mask[node_id,:][neg_idx] = 1

            # add distant but similar nodes as positive
            if self.aug_remote:
                _, topk_distant = sel_similarity[sel_id].topk(k=self.remote_K)
                pos_mask[node_id,:][topk_distant] = 1

        loss = self.loss(anchor=embeddings, sample=embeddings, pos_mask=pos_mask, neg_mask=neg_mask)
        if loss.isnan():
            ipdb.set_trace()

        return loss


    def loss_summary(self, embeddings, edge_index, group_labels=None):
        # compute the overall boundary-aware loss

        loss_bound = self._neighbor_contrast_loss(embeddings, edge_index)
        loss_dict = {"boundary loss": loss_bound.item(), }



        if self.call_count%self.vis_freq==0:
            for i, group_label in enumerate(group_labels):
                grouped_weights = utils.grouped_measure(self.boundary_weight.cpu().numpy(), group_labels=group_label.cpu().numpy())
                boundary_weight_dist = plot_chart([np.array(grouped_weights)], name_list=['group boundary weight'], x_start=1, x_name='group main', y_name='weight')
                loss_dict["boundary weight dist{}".format(i)] = boundary_weight_dist

            ins_weight_dist = plot_dist1D(self.boundary_weight.cpu().numpy(),label=group_labels[0].cpu().numpy())
            loss_dict["instance boundary weight dist{}".format(i)] = ins_weight_dist

        self.call_count = (self.call_count+1)%self.vis_freq

        loss_all = loss_bound

        return loss_all, loss_dict

    def train(self, data):
        self.optimizer.zero_grad()

        if self.boun_aug:
            self.encoder_model.train()
            embeddings, z2 = self.encoder_model(data.x, data.edge_index, data.edge_attr)
            h1, h2 = [self.encoder_model.project(x) for x in [embeddings, z2]]
        else:
            self.encoder.train()
            embeddings = self.encoder.embedding(data.x, data.edge_index, data.edge_attr)

        if self.args.return_data:
            loss_bal, losses = self.loss_summary(embeddings, data.edge_index, [data.group_label, data.group_label2])
        else:
            loss_bal, losses = self.loss_summary(embeddings, data.edge_index, [])

        loss= loss_bal

        if self.boun_aug:
            extra_pos_mask = torch.zeros((h1.shape[0],h1.shape[0]*2),device=h1.device)
            sel_idx = (self.boundary_weight>self.thresh).nonzero().flatten()
            
            #extra_mask_value = torch.zeros((h1.shape[0],),device=h1.device)
            #extra_mask_value[sel_idx] = 3
            #extra_pos_mask.diagonal_scatter_(extra_mask_value)

            for idx in sel_idx:
                extra_pos_mask[idx][idx] =3

            loss_ssl = self.contrast_model(h1, h2, extra_pos_mask=extra_pos_mask)
            losses['boun_aug loss'] = loss_ssl.item()
            loss += loss_ssl

        if self.is_sup:
            y_logit = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            loss_pred = F.nll_loss(y_logit[data.train_mask], data.y[data.train_mask])
            loss = loss*self.weight + loss_pred
            losses['pred loss']= loss_pred.item()
            
        else:
            loss = loss

        if self.type == 'prob':
            self.update_weight(y_logit.detach())
        elif self.type == 'loss':
            ipdb.set_trace()
            self.update_weight(loss_pred.detach()) # cannot use classification loss here, 1. reduction should set to 'none'; 2. no labels of all nodes
        elif self.type == 'emb':
            self.update_weight(embeddings.detach())

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
