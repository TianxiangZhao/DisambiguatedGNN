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
from .SupBalance_trainer import Trainer_BALANCE
from .BoundCont_trainer import Trainer_BoundCont

class Encoder_MVGRL(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder_MVGRL, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1.embedding(x1, edge_index1, edge_weight1)
        z2 = self.encoder2.embedding(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1.embedding(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2.embedding(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n
    
class Trainer_MVGRL(object):
    def __init__(self, args, model, device, loss, is_sup=False):
        self.args = args
        if args.res:
            hidden_dim = args.nhid * args.nlayer
        else:
            hidden_dim = args.nhid

        aug1 = A.Identity()
        aug2 = A.PPRDiffusion(alpha=0.2)

        model2 = copy.deepcopy(model).to(device)

        self.model = model
        self.encoder_model = Encoder_MVGRL(encoder1=model, encoder2=model2, augmentor=(aug1,aug2), hidden_dim=hidden_dim).to(device)
        self.contrast_model = DualBranchContrast_ex(loss=loss, mode='G2L').to(device)
        self.is_sup = is_sup

        self.optimizer = optim.Adam(self.encoder_model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        self.filter_weight = None

        self.use_bal = False
        if args.SUPBAL:
            self.use_bal=True
            self.bal = Trainer_BALANCE(args, model, device, loss, is_sup, proto_num=args.nclass*2, normalize=args.normalize,do_map=args.do_map)
            self.filter_weight = 1.0
            self.filter_rho = args.filterrho

        self.use_boun = False
        if args.Bound:
            self.use_boun=True
            self.boun = Trainer_BoundCont(args, model, device, loss, is_sup=is_sup, type=args.boundtype, rho=args.Boundrho, thresh=args.Boundthresh, weight=args.Boundweight)


    def train(self, data):
        self.encoder_model.train()
        self.optimizer.zero_grad()

        z1, z2, g1, g2, z1n, z2n = self.encoder_model(data.x, data.edge_index, data.edge_attr)
        
        if self.use_bal and self.args.usefilter:
            anchors = self.bal.proto_memory.detach().clone()
        else:
            anchors = None

        #loss_ssl = self.contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n,anchors = anchors, reweight_rho=self.filter_weight)
        loss_ssl = self.contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
        if self.filter_weight is not None and self.filter_weight>=0.3:
            self.filter_weight *= self.filter_rho
        losses = {'ssl loss': loss_ssl.item()}

        if self.is_sup:
            y_logit = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            loss_pred = F.nll_loss(y_logit[data.train_mask], data.y[data.train_mask])
            loss = loss_ssl + loss_pred
            losses['pred loss']= loss_pred.item()
        else:
            loss = loss_ssl

        if self.use_bal:
            bal_loss, loss_dict = self.bal.balance_loss(z1, data.train_mask, data.y)
            loss = loss + bal_loss
            losses.update(loss_dict)

        if self.use_boun:
            if self.args.return_data:
                loss_bal, loss_dict = self.boun.loss_summary(z1, data.edge_index, [data.group_label, data.group_label2])
            else:
                loss_bal, loss_dict = self.boun.loss_summary(z1, data.edge_index, [])
            loss = loss + loss_bal*self.boun.weight
            losses.update(loss_dict)

            if self.boun.type == 'prob':
                self.boun.update_weight(y_logit.detach())
            elif self.boun.type == 'loss':
                ipdb.set_trace()
                self.boun.update_weight(loss_pred.detach())
            elif self.boun.type == 'emb':
                self.boun.update_weight(z1.detach())

        loss.backward()
        self.optimizer.step()

        if self.use_bal:
            self.bal.update_proto(z1.detach())
        
        return losses