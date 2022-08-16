# --------------------------------------------------------
# Partly based on
# https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def smooth_labels(target, smoothing):
    n = target.size(-1)
    max_probs, idxs = torch.max(target, dim=-1)

    residuals = max_probs * smoothing
    norm_residuals = residuals / n

    target[range(len(idxs)), idxs] = target[range(len(idxs)), idxs] - residuals
    target = target + norm_residuals.unsqueeze(dim=-1).repeat(1, n)
    
    # Commenting this out replicates troch's implementation
    # target[range(len(idxs)), idxs] = target[range(len(idxs)), idxs] - norm_residuals
    
    return target

class CustomLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        """
        Custom cross entropy with label smoothing, which allows
        for label smoothing to be applied to a ground truth distribution
        IMPORTANT: Does not get same loss as torch w/ one-hot labels 
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        if len(target.size()) < 2: F.one_hot(target)
        
        target = smooth_labels(target, self.epsilon)
        
        return F.cross_entropy(preds, target)

class DeepClusteringLoss(nn.Module):
    def __init__(self):
        super(DeepClusteringLoss, self).__init__()

    def forward(self, output, target):
        if len(target.shape) < 3: target = F.one_hot(target.to(torch.int64)).type_as(output)
        num_frames = output.size()[0]
        vt_v = torch.norm(torch.matmul(torch.transpose(output, 0, 1), output), p=2) ** 2
        vt_y = torch.norm(torch.matmul(torch.transpose(output, 0, 1), target), p=2) ** 2
        yt_y = torch.norm(torch.matmul(torch.transpose(target, 0, 1), target), p=2) ** 2
        DC_loss = vt_v - 2 * vt_y + yt_y
        return DC_loss/(num_frames**2)

class AngularMarginLoss(nn.Module):

    def __init__(self, in_features, out_features, eps=1e-7, s=None, m=None):
        '''
        ArcFace: https://arxiv.org/abs/1801.07698
        '''

        super(AngularMarginLoss, self).__init__()

        # Defaults used in paper
        self.s = 64.0 if not s else s
        self.m = 0.5 if not m else m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)

        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)

        return -torch.mean(L)


