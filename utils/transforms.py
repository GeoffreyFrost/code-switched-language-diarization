import torch
import torch.nn.functional as F
import torchaudio

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

def powspace(start, stop, power, num):
    start = torch.pow(torch.tensor(start), 1/float(power))
    stop = torch.pow(torch.tensor(stop), 1/float(power))
    return torch.pow( torch.linspace(start, stop, num), power)

class AudioTransforms():
    def __init__(self, speed_min=0.9, speed_max=1.1, p_phone=0.5):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.p_phone = p_phone

    def forward(self, x):
        effects = []
        if torch.rand(1) > self.p_phone: 
            effects.append(["lowpass", "4000"])
            effects.append([       
                        "compand",
                        "0.02,0.05",
                        "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                        "-8",
                        "-7",
                        "0.05",
                        ])
            speed = (self.speed_max - self.speed_min) * torch.rand(1) + self.speed_max
            effects.append(["speed", f'{float(speed)}'])
            effects.append(["rate", "16000"])

        old_length = x.size(-1)
        x, sr = torchaudio.sox_effects.apply_effects_tensor(x, 16000, effects)
        new_length = x.size(-1)



        return x, float(new_length)/float(old_length)

class MixUp():
    def __init__(self, mixup_prob=0.25, mixup_size=0.2, beta_min=0.0, beta_max=0.5):
        self.mixup_size = mixup_size
        self.mixup_prob = mixup_prob
        self.beta_max = beta_max
        self.beta_min = beta_min

    def forward(self, x, lengths, y):
        
        time_param = int(lengths.min()*self.mixup_size)
        if time_param < 1: return x, y
        self.beta = ((self.beta_min - self.beta_max)*torch.rand(1) + self.beta_max).type_as(x).to(x.dtype)
        value = torch.rand(1) * time_param

        x, y = self.mixup(x, y, lengths, value)

        return x, y

    def mixup(self, x, y, lengths, value):

        samples_to_apply_mixup = (torch.rand(x.size(0)) < self.mixup_prob).to(x.device)
        if not samples_to_apply_mixup.any(): return x, y

        indexs = torch.arange(0, x.size(0)).type_as(x).long()
        indexs_rolled = indexs.roll(1, 0)
        x_rolled = x.roll(1, 0)
        y_rolled = y.roll(1, 0)
        y_rolled = y_rolled.float()
        y = y.float()

        for i in indexs:
            if samples_to_apply_mixup[i]:

                min_value = torch.rand(1) * (int(lengths[i]) - value)
                mask_start = (min_value.long()).squeeze()
                mask_end = (min_value.long() + value.long()).squeeze()

                min_value = torch.rand(1) * (int(lengths[indexs_rolled[i]]) - value)
                mask_start_r = (min_value.long()).squeeze()
                mask_end_r = (min_value.long() + value.long()).squeeze()

                # I don't have time to debug why this randomly fails once half way through training
                try:
                    x[i, mask_start:mask_end] = (1-self.beta)*x[i, mask_start:mask_end] + self.beta*x_rolled[i, mask_start_r:mask_end_r]
                    y[i, mask_start:mask_end] = (1-self.beta)*y[i, mask_start:mask_end] + self.beta*y_rolled[i, mask_start_r:mask_end_r]

                except: pass 
        return x, y

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
        if freq_mask_param > 0:
            for i in range(self.n_feature_masks): x = freq_masking_transform(x)
        if time_mask_param > 0: 
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
        return x_prime.type_as(x), x_l.type_as(x), y.type_as(x)
    return x_prime.type_as(x), x_l.type_as(x)