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
from GCL.models import DualBranchContrast, SingleBranchContrast
    
    
class Trainer_BALANCE(object):
    def __init__(self, args, model, device, loss, is_sup=False, proto_num=40, normalize=True, weights_list=[1,1,1], do_map=True):
        # use prototypes to regularize the embedding space.
        # can be used only for loss computation, and worked under other trainers
        # normalize: whether normalize embeddings before computing losses
        # weights_list: [proto_sel, proto_balance, proto_class] respectively

        if args.res:
            hidden_dim = args.nhid * args.nlayer
        else:
            hidden_dim = args.nhid

        self.proto_num = int(args.nclass * args.groupnumber)
        proto_num = self.proto_num
        self.loss = loss
        self.model = model
        self.is_sup = is_sup
        self.normalize = normalize
        if args.BAL_weights is not None:
            self.weights_list = args.BAL_weights
        else:
            self.weights_list = weights_list
        self.do_map = do_map

        if not do_map:        
            self.proto_memory = torch.nn.Parameter(torch.rand((proto_num,hidden_dim), requires_grad=True, device=device))
            self.prot_optimizer = optim.Adam([self.proto_memory], lr=args.lr, weight_decay=args.weight_decay)
            self.prot_optimizer.zero_grad()
        else:
            self.proto_memory = torch.rand((proto_num,hidden_dim), requires_grad=False, device=device)

        self.proto_density = torch.ones((proto_num,), requires_grad=False, device=device)
        self.density_rho = 0.9
        self.debias = args.debias

        self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _proto_sel_loss(self,embeddings, s_matrix):
        # compute the distance between input instances and prototypical embeddings
        with torch.no_grad():
            sel_index = s_matrix.argmax(-1)
            pos_mask = torch.zeros((embeddings.shape[0], self.proto_num)).to(embeddings.device)
            pos_mask.scatter_add_(dim=-1,index=sel_index.reshape(-1,1), src=torch.ones((sel_index.shape[0],1),device=sel_index.device))
            #neg_mask = torch.ones((embeddings.shape[0], self.proto_num)).to(embeddings.device) # for jsd, should I use this form?
            neg_mask = 1. - pos_mask 

        # weight:
        if self.debias:
            with torch.no_grad():
                proto_weight = F.normalize(1/(self.proto_density+0.001), dim=0)
                proto_weight *= 1/(proto_weight.max())
                sel_weight = proto_weight[sel_index]
            pos_weight = torch.ones(pos_mask.shape).to(embeddings.device) * sel_weight.unsqueeze(-1)
            neg_weight = torch.ones(pos_mask.shape).to(embeddings.device) * sel_weight.unsqueeze(-1)
        else:
            pos_weight = None
            neg_weight = None
        loss = self.loss(anchor=embeddings, sample=self.proto_memory, pos_mask=pos_mask, neg_mask=neg_mask, pos_weight=pos_weight, neg_weight=neg_weight)

        return loss
    
    def _sel_ent_loss(self, s_matrix):
        # encourage each proto to be selected

        s_matrix = s_matrix-s_matrix.min().item()
        s_matrix = s_matrix / (s_matrix.sum(dim=-1)+0.000001).unsqueeze(dim=-1)
        loss = torch.mul(s_matrix.mean(dim=0), torch.log(s_matrix.mean(dim=0)+0.000001)).sum()

        return loss

    def _proto_class_loss(self, embeddings, labels, s_matrix):
        # encourage each proto to match with only one class
        
        sel_index = s_matrix.argmax(-1)

        pos_mask = torch.eye(embeddings.shape[0],device=embeddings.device)

        pos_mask = torch.logical_and(labels.reshape(-1,1) == labels.reshape(1,-1), sel_index.reshape(-1,1)==sel_index.reshape(1,-1))
        neg_mask = torch.logical_and(labels.reshape(-1,1)!=labels.reshape(1,-1), sel_index.reshape(-1,1) == sel_index.reshape(1,-1))

        loss = self.loss(anchor=embeddings, sample=embeddings, pos_mask=pos_mask, neg_mask=neg_mask)

        return loss

    def balance_loss(self, embeddings, train_ind=None, labels=None, update_density=True):
        # compute the overall balanceness-aware loss

        if self.normalize:
            embeddings = F.normalize(embeddings)
            proto_memory = F.normalize(self.proto_memory)
        else:
            proto_memory = self.proto_memory

        s_matrix = embeddings @ proto_memory.t()

        if update_density:
            self.proto_density = self.proto_density*(self.density_rho) + (1-self.density_rho)* s_matrix.detach().mean(dim=0).squeeze()
            
        loss_sel = self._proto_sel_loss(embeddings, s_matrix)
        loss_ent = self._sel_ent_loss(s_matrix)
        loss_dict = {"selection loss": loss_sel.item(), "entropy loss": loss_ent.item()}
        
        if loss_ent.isnan():
            ipdb.set_trace()

        if self.is_sup:
            loss_class = self._proto_class_loss(embeddings[train_ind], labels[train_ind], s_matrix[train_ind])
            loss_dict["class-aware loss"] = loss_class.item()
            losses_all = loss_sel*self.weights_list[0] + loss_ent*self.weights_list[1] + loss_class*self.weights_list[2]
        else:
            losses_all = loss_sel*self.weights_list[0] + loss_ent*self.weights_list[1]


        return losses_all, loss_dict

    def map_proto(self, embeddings=None, s_matrix=None):
        # 

        if self.normalize:
            embeddings = F.normalize(embeddings)
            proto_memory = F.normalize(self.proto_memory)
        else:
            proto_memory = self.proto_memory

        if s_matrix is None:
            with torch.no_grad():
                s_matrix = embeddings @ proto_memory.t()
        sel_ind = s_matrix.argmax(dim=-1)
        for i in range(self.proto_memory.shape[0]):
            if (sel_ind==i).sum() !=0:
                proto_memory[i] = proto_memory[i]*0.8+ torch.mean(embeddings[sel_ind].reshape(-1, embeddings.shape[1]),dim=0)*0.2
    
        self.proto_memory = proto_memory.detach()

        return
    
    def update_proto(self, embeddings):
        
        if not self.do_map:
            self.prot_optimizer.step()        
            self.prot_optimizer.zero_grad()
        else:
            self.map_proto(embeddings.detach())

        return

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        if not self.do_map:        
            self.prot_optimizer.zero_grad()

        embeddings = self.model.embedding(data.x, data.edge_index, edge_weight=data.edge_attr)
        if embeddings.isnan().any():
            ipdb.set_trace()

        loss_bal, losses = self.balance_loss(embeddings, data.train_mask, data.y)

        if self.is_sup:
            y_logit = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            loss_pred = F.nll_loss(y_logit[data.train_mask], data.y[data.train_mask])
            loss = loss_bal + loss_pred
            losses['pred loss']= loss_pred.item()
        else:
            loss = loss_bal

        loss.backward()
        self.optimizer.step()
        self.update_proto(embeddings)

        if self.proto_memory.isnan().any():
            ipdb.set_trace()
        
        return losses