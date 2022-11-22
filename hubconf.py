
dependencies = ['torch']

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from models.lit_cs_detector import get_padding_masks_from_length
from models.WavLM import WavLM, WavLMConfig

# helpers
class Model(nn.Module):
    def __init__(self, diarzation_config, backbone):
        super().__init__()

        if diarzation_config == 0: n_classes = 2
        if diarzation_config == 1: n_classes = 3
        if diarzation_config == 2: n_classes = 5

        if backbone == "wavlm-large":
            checkpoint = torch.load('/home/gfrost/projects/penguin/models/weights/WavLM-Large-cfg.pt')
            cfg = WavLMConfig(checkpoint['cfg'])
            self.backbone = WavLM(cfg)
            embed_dim = 1024

        if backbone == "wavlm-base":
            checkpoint = torch.load('/home/gfrost/projects/penguin/models/weights/WavLM-Base-cfg.pt')
            cfg = WavLMConfig(checkpoint['cfg'])
            self.backbone = WavLM(cfg)
            embed_dim = 768

        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.xavier_uniform_(self.head.weight, gain=1 / math.sqrt(2))

    def forward(self, x, l):

        padding_masks = get_padding_masks_from_length(x, l)
        x, padding_masks, lengths = self.backbone.custom_feature_extractor(x, padding_masks)
        x, lengths = self.backbone.transformer_encoder(x, padding_mask=padding_masks, ret_lengths=True)
        x = self.head(x)

        return x, lengths

def load_wavlm(diarzation_config, device='cuda') -> Model:
    """Load fine-tuned wavlm for LD for diarzation config"""

    if diarzation_config == 0: ckpt_file_path = ''
    if diarzation_config == 1: ckpt_file_path = ''
    if diarzation_config == 2: ckpt_file_path = ''

    model = model(diarzation_config)

    model = model.load_from_checkpoint(ckpt_file_path).eval()
    device = torch.device(device)

    return model.to(device)