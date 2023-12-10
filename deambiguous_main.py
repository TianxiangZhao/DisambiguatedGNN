import torch
import numpy as np
import random
import os
import datasets as datasets
from torch_geometric.loader import DataLoader,RandomNodeSampler, NeighborLoader
import models.models as models

import utils
import math
import torch_geometric.datasets as tg_dataset
import torch_geometric.transforms as T
import GCL.losses as L
import losses as L_ex
import trainers
import tester
import datasets
from GCL.eval import get_split

from tensorboardX import SummaryWriter
import ipdb
import matplotlib.pyplot as plt

#seeds = [42, 43, 4, 22]
seeds=[4]
###configure arguments
args = utils.get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
args.datatype = 'node'
args.return_data = False

for seed in seeds:
    #args.seed = seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.log:
        tb_path = './tensorboard_deambiguous_ap1/{}/{}_{}Bound{}/model{}loss{}seed{}/Boundrho{}thresh{}quntile_pos{}neg{}weight{}type{}remote{}K{}bounaug{}'.format(args.dataset,args.task, args.method,args.Bound, args.model,args.loss, args.seed,
                                                                                              args.Boundrho, args.Boundthresh, args.Posthresh,args.Negthresh, args.Boundweight, args.boundtype,
                                                                                              args.aug_remote, args.remote_K, args.boun_aug)
        if args.BAL_weights is not None:    
            weight_seq = [str(i) for i in args.BAL_weights]
            tb_path += '_'.join(weight_seq)
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        writer = SummaryWriter(tb_path)

    # ------------------
    # load dataset
    # ------------------
    if args.dataset == 'cora':
        num_sample = 2708
        args.train_ratio=0.03
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.PlanetoidNew(root='./datasets_new/', name='Cora', transform=trans)
        args.return_data = True

    elif args.dataset == 'blogcatalog':
        num_sample = 5196
        args.train_ratio=0.03
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.AttributedGraphDatasetNew(root='./datasets_new/', name='BlogCatalog', transform=trans)
        args.return_data = True

    elif args.dataset == 'flickr':
        num_sample = 7575
        trans=T.Compose([T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.AttributedGraphDatasetNew(root='./datasets_new/', name='Flickr', transform=trans)
        args.return_data = True

    elif args.dataset == 'cora_full':
        #dataset = tg_dataset.CitationFull(root='./datasets/', name='Cora', transform=T.NormalizeFeatures())
        num_sample = 19793
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.CitationFullNew(root='./datasets_new/', name='Cora_full', transform=trans)
        args.return_data = True
        
    elif args.dataset == 'computer':
        num_sample = 13752
        args.train_ratio=0.03
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.AmazonNew(root='./datasets_new/',name='Computers', transform=trans)
        args.return_data = True

    elif args.dataset == 'chameleon':
        num_sample = 2277
        args.train_ratio=0.03
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.WikipediaNetworkNew(root='./datasets_new/', name=args.dataset, transform=trans)
        args.return_data = True

    elif args.dataset == 'squirrel':
        num_sample = 5201
        args.train_ratio=0.03
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.WikipediaNetworkNew(root='./datasets_new/', name=args.dataset, transform=trans)
        args.return_data = True

    elif args.dataset == 'actor':
        num_sample = 7600
        args.train_ratio=0.01
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.ActorNew(root='./datasets_new/actor/', transform=trans)
        args.return_data = True

    elif args.dataset == 'reddit':
        #dataset = tg_dataset.Reddit2(root='./datasets/Reddit2/', transform=T.NormalizeFeatures())
        num_sample = 232965
        trans=T.Compose([T.NormalizeFeatures(),T.RandomNodeSplit(split='train_rest', num_val=int(args.val_ratio*num_sample), num_test=int(num_sample*(1-args.val_ratio-args.train_ratio)))])
        dataset = datasets.Reddit2New(root='./datasets/Reddit2/', transform=trans)
        args.return_data = True

    else:
        ipdb.set_trace()
        print('error, unrecognized node classification dataset, {}'.format(args.dataset))
    # examine data split

    args.nfeat = dataset[0].x.shape[-1]
    args.nclass = len(set(dataset[0].y.tolist()))
    args.nsample = dataset[0].x.shape[0]

    if dataset[0].x.shape[0] > 100000:
        dataloader = RandomNodeSampler(dataset[0], num_parts=10, shuffle=True)
        # dataloader = NeighborLoader(dataset[0], num_neighbors=[10,10], batch_size=args.batch_size, input_nodes=dataset[0].train_mask)
        testloader =  DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        ipdb.set_trace()
        # not implemented yet
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    label, count = dataset[0].y.unique(return_counts=True)
    print("label set of dataset: {}".format(label))
    print("counts of labels: {}".format(count))

    # ---------------------
    # initialize GNN model
    # ---------------------
    if args.datatype == 'node':
        if args.model == 'gcn':
            model = models.GCN(args, nfeat=args.nfeat, 
                    nhid=args.nhid, 
                    nclass=args.nclass, 
                    dropout=args.dropout,
                    nlayer=args.nlayer, res=args.res)
        elif args.model == 'sage':
            model = models.SAGE(args, nfeat=args.nfeat, 
                    nhid=args.nhid, 
                    nclass=args.nclass, 
                    dropout=args.dropout,
                    nlayer=args.nlayer, res=args.res)
        elif args.model == 'gin':
            model = models.GIN(args, nfeat=args.nfeat, 
                    nhid=args.nhid, 
                    nclass=args.nclass, 
                    dropout=args.dropout,
                    nlayer=args.nlayer, res=args.res)
        elif args.model =='sgc':
            model = models.SGC(args, nfeat=args.nfeat, 
                    nhid=args.nhid, 
                    nclass=args.nclass, 
                    dropout=args.dropout,
                    nlayer=args.nlayer, res=args.res)

    if args.load is not None:
        model = utils.load_model(args, model, name='model_{}'.format(args.load))
    model = model.to(device)

    # ---------------------
    # configure ssl
    # ---------------------
    if args.loss == 'jsd':
        loss = L_ex.JSD_ex()
    elif args.loss == 'infonce':
        loss = L_ex.InfoNCE_ex(tau=0.2)
    elif args.loss == 'triplet':
        loss = L.tripletMargin()
        ipdb.set_trace()
    is_sup = True

    if args.method == 'DGI':
        ssl_trainer = trainers.Trainer_DGI(args, model, device, loss, is_sup=is_sup)
    elif args.method == 'GRACE':
        ssl_trainer = trainers.Trainer_GRACE(args, model, device, loss, is_sup=is_sup)
    elif args.method == 'MVGRL':
        ssl_trainer = trainers.Trainer_MVGRL(args, model, device, loss, is_sup=is_sup)
    elif args.method == 'SUP': # use supervised training alone, without SSL tasks
        ssl_trainer = trainers.Trainer_SUP(args, model, device, loss, is_sup=True)
    elif args.method == 'BOUND': # use supervised training alone, without SSL tasks
        ssl_trainer = trainers.Trainer_BoundCont(args, model, device, loss, is_sup=True, type=args.boundtype, rho=args.Boundrho, thresh=args.Boundthresh, weight=args.Boundweight)
    
    # ---------------------
    # train
    # ---------------------
    for epoch in range(args.epochs):
        if epoch%50 == 0:
            if epoch%400 == 0:
                vis = True
            else:
                vis = False
            # test
            for data in testloader:
                result = ssl_trainer.test(data.to(device), vis=vis)
            if args.log:
                for key in result:
                    if np.isscalar(result[key]):
                        writer.add_scalar('test {}'.format(key), result[key], epoch)
                    else: #figure
                        writer.add_figure('test {}'.format(key), result[key], epoch)
            plt.close('all')
            
        for batch, data in enumerate(dataloader):
            losses = ssl_trainer.train(data.to(device))
            if args.log:
                for key in losses:
                    if np.isscalar(losses[key]):
                        writer.add_scalar('train {}'.format(key), losses[key], (epoch-1)*len(dataloader)+batch)
                    else: #figure
                        writer.add_figure('train {}'.format(key), losses[key], (epoch-1)*len(dataloader)+batch)
            plt.close('all')
            
                
    # ---------------------
    # log
    # ---------------------
    if args.log:
        writer.close()


