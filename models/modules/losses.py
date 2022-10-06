import torch
import torch.nn as nn
import torch.nn.functional as F

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



