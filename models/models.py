import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GlobalAttention, SGConv
import ipdb
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool


class GCN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(GCNConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
            
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            
            xs.append(x)

        if return_list:
            return xs
        elif self.res:
            x = torch.cat(xs, dim=-1)
        else:
            return x


class SAGE(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(SAGEConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            xs.append(x)

        if True:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)

        return x

class GIN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            mlp = MLP(nlast, nhid, nhid, is_cls=False)
            self.convs.append(GINConv(mlp))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            xs.append(x)

        if True:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)

        return x
    
class MLP(torch.nn.Module):
    def __init__(self, in_feat, hidden_size, out_size, layers=2, dropout=0.1, is_cls=False):
        super(MLP, self).__init__()

        modules = []
        in_size = in_feat
        for layer in range(layers-1):
            modules.append(torch.nn.Linear(in_size, hidden_size))
            in_size = hidden_size
            modules.append(torch.nn.LeakyReLU(0.1))
        modules.append(torch.nn.Linear(in_size, out_size))
        self.model = torch.nn.Sequential(*modules)

        self.is_cls = is_cls

    def forward(self, features):
        output = self.model(features)

        if self.is_cls:
            return F.log_softmax(output, dim=1)
        else:
            return output

class SGC(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(SGConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
            
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            
            xs.append(x)

        if return_list:
            return xs
        elif self.res:
            x = torch.cat(xs, dim=-1)
        else:
            return x
    
