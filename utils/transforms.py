import torch
import torch.nn.functional as F

def interp_targets(targets, max_length):
    targets_ = targets.reshape(targets.size(0), 1, 1, targets.size(1)).to(torch.float32)
    interp_legnths = (1, int(max_length))
    ds_targets = F.interpolate(targets_, interp_legnths).squeeze()
    return ds_targets.to(torch.long)