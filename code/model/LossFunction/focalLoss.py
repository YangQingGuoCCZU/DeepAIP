import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#used
class FocalLoss_v2(nn.Module):
    def __init__(self, num_class=2, gamma=2, alpha=None):

        super(FocalLoss_v2, self).__init__()
        self.gamma = gamma
        self.num_class = num_class
        if alpha == None:
            self.alpha = torch.ones(num_class)
        else:
            self.alpha=alpha

    def forward(self, logit, target):

        target = target.view(-1)

        alpha = self.alpha[target.cpu().long()]

        logpt = - F.cross_entropy(logit, target, reduction='none')
        pt = torch.exp(logpt)
        focal_loss = -(alpha * (1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()
