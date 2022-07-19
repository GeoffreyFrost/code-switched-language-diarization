import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import random

spec_transform  = torchaudio.transforms.Spectrogram(power=None)
time_stretch = torchaudio.transforms.TimeStretch()
freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
invspec_transform = torchaudio.transforms.InverseSpectrogram()

def interp_targets(targets, max_length):
    if len(targets.size()) > 1:
        targets_ = targets.reshape(targets.size(0), 1, 1, targets.size(1)).to(torch.float32)
    else: 
        targets_ = targets.reshape(1, 1, 1, targets.size(0)).to(torch.float32)
    interp_legnths = (1, int(max_length))
    ds_targets = torch.round(F.interpolate(targets_, interp_legnths).squeeze())
    return ds_targets.to(torch.long)

class SpecAugment():
    def __init__(self, feature_masking_percentage=0.05, time_masking_percentage=0.05, n_feature_masks=2, n_time_masks=2):
        self.feature_masking_percentage = feature_masking_percentage
        self.time_masking_percentage = time_masking_percentage
        self.n_feature_masks = n_feature_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        freq_mask_param = int(x.size(-1)*self.feature_masking_percentage)
        time_mask_param = int(x.size(-2)*self.time_masking_percentage)
        freq_masking_transform = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        time_masking_transform = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        for i in range(self.n_feature_masks): x = freq_masking_transform(x)
        for i in range(self.n_time_masks): x = time_masking_transform(x)
        return x
    
def wav_specaugment(x, x_l, y=None):
    time_mask_param = torch.round(torch.tensor(x.size(-1)/180 * 0.05)) # 5% time mask
    time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    x_s = spec_transform(x)
    # Time stretch, max 10%
    #time_stretch_factor = (0.9 - 1.1)*random.uniform(0, 1) + 1.1
    #x_s = time_stretch(x_s, time_stretch_factor)
    # Freq masking
    # Real
    seed = torch.seed()
    x_r = freq_masking(x_s.real)
    #x_r = freq_masking(x_r)
    x_r = time_masking(x_r)
    #x_r = time_masking(x_r)

    # Imag
    torch.manual_seed(seed)
    x_i = freq_masking(x_s.imag)
    #x_i = freq_masking(x_i)
    x_i = time_masking(x_i)
    #x_i = time_masking(x_i)

    # Combine & convert to audio
    x_s = torch.complex(x_r, x_i)
    x_prime = invspec_transform(x_s)

    x_l = torch.floor(x_prime.size(-1)/x.size(-1)*x_l)
    diff = x_l.max() - x_prime.size(-1)
    x_l = x_l - diff
    if y != None:
        y = interp_targets(y, x_prime.size(-1))
        return x_prime.cuda(), x_l.cuda(), y.cuda()
    return x_prime.cuda(), x_l.cuda()