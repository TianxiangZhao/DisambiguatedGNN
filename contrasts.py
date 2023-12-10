import torch

from GCL.losses import Loss
from GCL.models import get_sampler
import ipdb

def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        #pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
        pos_mask = (pos_mask+extra_pos_mask).float()

    if extra_neg_mask is not None:
        # neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
        neg_mask = (neg_mask.bool+extra_neg_mask).float()
    #else:
    #    neg_mask = 1. - pos_mask
    
    return pos_mask, neg_mask


class SingleBranchContrast_ex(torch.nn.Module):
    # extend with anchors for reweighting
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(SingleBranchContrast_ex, self).__init__()
        assert mode == 'G2L'  # only global-local pairs allowed in single-branch contrastive learning
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h, g, batch=None, hn=None, extra_pos_mask=None, extra_neg_mask=None, anchors=None):
        if batch is None:  # for single-graph datasets
            assert hn is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, neg_sample=hn)
        else:  # for multi-graph datasets
            assert batch is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, batch=batch)

        pos_mask, neg_mask = add_extra_mask(pos_mask, neg_mask, extra_pos_mask, extra_neg_mask)

        # need not to do the filter, hence noot using anchors

        loss = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask, **self.kwargs)
        return loss


class DualBranchContrast_ex(torch.nn.Module):
    # extend with anchors for reweighting
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast_ex, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None, anchors=None, reweight_rho=0.9):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
            if anchors is not None:
                # reweight neg_samples, those similar should be down-weighted
                with torch.no_grad():
                    anc_temple = (anchor1 @ anchors.t()).argmax(dim=-1)
                    samp_temple = (sample1 @ anchors.t()).argmax(dim=-1)
                    same_temple = anc_temple.unsqueeze(dim=1) == samp_temple.unsqueeze(dim=0)
                    dif_temple = torch.logical_not(same_temple)
                    neg_mask1[same_temple] *= reweight_rho
                    pos_mask1[dif_temple] *= reweight_rho

                    
                    anc_temple = (anchor2 @ anchors.t()).argmax(dim=-1)
                    samp_temple = (sample2 @ anchors.t()).argmax(dim=-1)
                    same_temple = anc_temple.unsqueeze(dim=1) == samp_temple.unsqueeze(dim=0)
                    dif_temple = torch.logical_not(same_temple)
                    neg_mask2[same_temple] *= reweight_rho
                    pos_mask2[dif_temple] *= reweight_rho


        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
            if anchors is not None:        
                ipdb.set_trace()
                # not implemented yet


        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
                if anchors is not None:        
                    ipdb.set_trace()
                    # not needed yet



            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)
                if anchors is not None:        
                    ipdb.set_trace()
                    # not implemented yet

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5