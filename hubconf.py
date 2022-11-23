
dependencies = ['torch']

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from models.WavLM import WavLM, WavLMConfig

# helpers
def get_padding_masks_from_length(source, lengths):
    indexs = torch.arange(source.size(-1)).unsqueeze(dim=0).repeat(lengths.size(0), 1).type_as(lengths)
    lengths = lengths.unsqueeze(dim=-1).repeat(1, indexs.size(-1))
    padding_mask = indexs > lengths
    return padding_mask

class Model(nn.Module):
    def __init__(self, diarization_config, backbone):
        super().__init__()

        if diarization_config == 0: n_classes = 2
        if diarization_config == 1: n_classes = 3
        if diarization_config == 2: n_classes = 5

        if backbone == "wavlm-large":
            checkpoint = torch.hub.load_state_dict_from_url("https://github.com/GeoffreyFrost/code-switched-language-diarization/releases/download/v1.0.0/WavLM-Large-cfg.pt")
            cfg = WavLMConfig(checkpoint['cfg'])
            self.backbone = WavLM(cfg)
            embed_dim = 1024

        else: raise NotImplementedError

        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.xavier_uniform_(self.head.weight, gain=1 / math.sqrt(2))

    def forward(self, x, l):

        padding_masks = get_padding_masks_from_length(x, l)
        x, padding_masks, lengths = self.backbone.custom_feature_extractor(x, padding_masks)
        x, lengths = self.backbone.transformer_encoder(x, padding_mask=padding_masks, ret_lengths=True)
        x = self.head(x)

        return x, lengths

def wavlm_for_ld(diarization_config=0, backbone='wavlm-large', pretrained=True, progress=True, device='cuda'):
    """Load a fine-tuned wavlm model for LD for a diarzation config
        0 - eng/other
        1 - eng/nguni/sesotho-tswana
        2 - eng/zulu/xhosa/sesotho/setswana
    """

    if diarization_config == 0: ckpt_url = 'https://github.com/GeoffreyFrost/code-switched-language-diarization/releases/download/v1.0.0/WavLM-Large-ld-0.pt'
    if diarization_config == 1: ckpt_url = 'https://github.com/GeoffreyFrost/code-switched-language-diarization/releases/download/v1.0.0/WavLM-Large-ld-1.pt'
    if diarization_config == 2: ckpt_url = 'https://github.com/GeoffreyFrost/code-switched-language-diarization/releases/download/v1.0.0/WavLM-Large-ld-2.pt'

    model = Model(diarization_config, backbone)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(ckpt_url, progress=progress)
        model.load_state_dict(state_dict)
    
    device = torch.device(device)

    return model.to(device)