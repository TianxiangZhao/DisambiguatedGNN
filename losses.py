import torch
import numpy as np
import torch.nn.functional as F
from GCL.losses.losses import Loss
import ipdb

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE_ex(Loss):
    # extended with weights
    def __init__(self, tau):
        super(InfoNCE_ex, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask,  pos_weight=None, neg_weight=None, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        if pos_weight is not None:
            pos_mask *= pos_weight
            neg_mask *= neg_weight

        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / (pos_mask.sum(dim=1)+0.0001)


        return -loss.mean()

class JSD_ex(Loss):
    # extended with weights
    def __init__(self, discriminator=lambda x, y: x @ y.t()):
        super(JSD_ex, self).__init__()
        self.discriminator = discriminator

    def compute(self, anchor, sample, pos_mask, neg_mask, pos_weight=None, neg_weight=None, *args, **kwargs):
        
        if pos_weight is not None:
            pos_mask *= pos_weight
            neg_mask *= neg_weight
        num_neg = neg_mask.sum()
        num_pos = pos_mask.sum()



        similarity = self.discriminator(anchor, sample)

        E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
        E_pos /= (num_pos+0.001)

        neg_sim = similarity * neg_mask
        E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
        E_neg /= (num_neg+0.001)



        return E_neg - E_pos