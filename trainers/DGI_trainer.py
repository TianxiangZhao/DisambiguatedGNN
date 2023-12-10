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
from contrasts import DualBranchContrast_ex, SingleBranchContrast_ex
from .SupBalance_trainer import Trainer_BALANCE
from .BoundCont_trainer import Trainer_BoundCont


class Encoder_DGI(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder_DGI, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder.embedding(x, edge_index, edge_weight)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder.embedding(*self.corruption(x, edge_index),edge_weight)
        return z, g, zn

class Trainer_DGI(object):
    def __init__(self, args, model, device, loss, is_sup=False):
        self.args = args
        if args.res:
            hidden_dim = args.nhid * args.nlayer
        else:
            hidden_dim = args.nhid
        self.model = model
        self.encoder_model = Encoder_DGI(encoder=model, hidden_dim=hidden_dim).to(device)
        self.contrast_model = SingleBranchContrast_ex(loss=loss, mode='G2L').to(device)
        self.is_sup = is_sup
        self.optimizer = optim.Adam(self.encoder_model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

        self.use_bal = False
        if args.SUPBAL:
            self.use_bal=True
            self.bal = Trainer_BALANCE(args, model, device, loss, is_sup, proto_num=args.nclass*2, normalize=args.normalize,do_map=args.do_map)

        self.use_boun = False
        if args.Bound:
            self.use_boun=True
            self.boun = Trainer_BoundCont(args, model, device, loss, is_sup=is_sup, type=args.boundtype, rho=args.Boundrho, thresh=args.Boundthresh, weight=args.Boundweight)



    def train(self, data):
        self.encoder_model.train()
        self.optimizer.zero_grad()
        z, g, zn = self.encoder_model(data.x, data.edge_index, data.edge_attr)

        if self.use_bal and self.args.usefilter:
            anchors = self.bal.proto_memory.detach().clone()
        else:
            anchors = None

        if z.isnan().any() or zn.isnan().any():
            ipdb.set_trace()

        loss_ssl = self.contrast_model(h=z, g=g, hn=zn, anchors = anchors)

        losses = {'ssl loss': loss_ssl.item()}

        if self.is_sup:
            y_logit = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            loss_pred = F.nll_loss(y_logit[data.train_mask], data.y[data.train_mask])
            loss = loss_ssl + loss_pred
            losses['pred loss']= loss_pred.item()
        else:
            loss = loss_ssl
        
        if self.use_bal:
            bal_loss, loss_dict = self.bal.balance_loss(z, data.train_mask, data.y)
            loss = loss + bal_loss
            losses.update(loss_dict)

        if self.use_boun:
            if self.args.return_data:
                loss_bal, loss_dict = self.boun.loss_summary(z, data.edge_index, [data.group_label, data.group_label2])
            else:
                loss_bal, loss_dict = self.boun.loss_summary(z, data.edge_index, [])
            loss = loss + loss_bal*self.boun.weight
            losses.update(loss_dict)

            if self.boun.type == 'prob':
                self.boun.update_weight(y_logit.detach())
            elif self.boun.type == 'loss':
                ipdb.set_trace()
                self.boun.update_weight(loss_pred.detach())
            elif self.boun.type == 'emb':
                self.boun.update_weight(z.detach())

        loss.backward()
        self.optimizer.step()

        if self.use_bal:
            self.bal.update_proto(z.detach())
            if self.bal.proto_memory.isnan().any():
                ipdb.set_trace()
        
        return losses
        

