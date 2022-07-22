import gc
import logging
import torch
import torch.nn as nn
from torch import autocast
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from utils.transforms import interp_targets, SpecAugment
from models.WavLM import WavLM, WavLMConfig
from dataclasses import dataclass

@dataclass
class ModelConfig():
    backbone:str='base'
    freeze_feature_extractor:bool=True
    label_smoothing:float=0.0
    specaugment:bool=False
    time_masking_percentage:float=0.02
    feature_masking_percentage:float=0.02
    n_feature_masks:int=2
    n_time_masks:int=2
    n_classes:int=2
    combine_intermediate:bool=False
    cross_attention:bool=True
    
class LitCSDetector(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, model_config=None):
        super().__init__()
        self.save_hyperparameters()

        if model_config==None: self.model_config = ModelConfig()
        else: self.model_config = model_config
        self.label_smoothing = model_config.label_smoothing
        self.specaugment = model_config.specaugment
        self.combine_intermediate = model_config.combine_intermediate
        self.cross_attention = model_config.cross_attention
        
        if self.combine_intermediate: factor = 2
        else: factor = 1

        if self.specaugment: self.spec_augmenter=SpecAugment(
                                    feature_masking_percentage=model_config.feature_masking_percentage,
                                    time_masking_percentage=model_config.time_masking_percentage,
                                    n_feature_masks=model_config.n_feature_masks,
                                    n_time_masks=model_config.n_time_masks
                                    )

        assert model_config.backbone in ["base","large", "xlsr"], f'model: {model_config.backbone} not supported.'
        
        if model_config.backbone == "base":
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.backbone = bundle.get_model()
            embed_dim = 768

        if model_config.backbone == "large":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE
            self.backbone = bundle.get_model()
            embed_dim = 1024

        if model_config.backbone == "xlsr":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
            self.backbone = bundle.get_model()
            embed_dim = 1024
        if self.cross_attention and self.combine_intermediate:
            self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=16, dropout=0.1, batch_first=True)
            self.head = nn.Linear(embed_dim*(factor-1), model_config.n_classes)
        else: self.head = nn.Linear(embed_dim*factor, model_config.n_classes)

        if model_config.freeze_feature_extractor:
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x, l):
        
        x, lengths = self.backbone.feature_extractor(x, l)
        if self.specaugment: x = self.spec_augmenter.forward(x)
        # Include intermediate layer for more syntactic infomation (wav2vec-U)
        if self.combine_intermediate: 
            x = self.backbone.encoder.extract_features(x, lengths)
            if self.cross_attention: 
                att_masks = get_attention_masks(lengths, x[-1].size(-2), 16)
                x, _ = self.cross_attention(x[len(x)//2+4], x[-1], x[-1], attn_mask=att_masks)
            else: x = torch.cat((x[len(x)//2+4], x[-1]), dim=-1)

        else: x = self.backbone.encoder(x, lengths)
        x = self.head(x)

        return x, lengths

    def training_step(self, batch, batch_idx):
        x, x_l, y, y_l = batch
        y_hat, lengths = self.forward(x, x_l)
        loss, y = aggregate_bce_loss(y_hat, y, lengths, self.model_config.n_classes, self.label_smoothing)
        self.log('train/loss', loss)
        return {'loss':loss, 'y_hat':y_hat, 'y':y, 'lengths':lengths}
    
    def training_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'].view(-1, self.model_config.n_classes)[get_unpadded_idxs(x['lengths'])] for x in out])
        y = torch.cat([x['y'].view(-1)[get_unpadded_idxs(x['lengths'])]for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y.argmax(dim=-1)).sum().float() / float(y.size(0))
        self.log("train/train_acc", accuracy, on_epoch=True)
        gc.collect()

    def validation_step(self, batch, batch_idx):
        x, x_l, y, y_l = batch
        y_hat, lengths = self.forward(x, x_l)
        loss, y = aggregate_bce_loss(y_hat, y, lengths, self.model_config.n_classes, self.label_smoothing)
        self.log('val/loss', loss)
        return {'y_hat':y_hat, 'y':y, 'lengths':lengths}

    def validation_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'].view(-1, self.model_config.n_classes)[get_unpadded_idxs(x['lengths'])] for x in out])
        y = torch.cat([x['y'].view(-1)[get_unpadded_idxs(x['lengths'])]for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y.argmax(dim=-1)).sum().float() / float(y.size(0))
        self.log("val/val_acc", accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                    lr=self.hparams.learning_rate, 
                                    weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler}

def get_attention_masks(lengths, max_len, nheads):
    masks1d = (torch.arange(max_len).cuda()[None, :] > lengths[:, None]).repeat(nheads, 1).unsqueeze(dim=-1)
    masks1d = masks1d.float()
    masks2d = torch.matmul(masks1d,masks1d.transpose(-2, -1))
    return masks2d.bool()
    
def get_unpadded_idxs(lengths):
    max_len = torch.max(lengths)
    return torch.cat([torch.arange(max_len*i, max_len*i+l) for i, l in enumerate(lengths)]).to(torch.long)

def aggregate_bce_loss(y_hat, y, lengths, n_classes, label_smoothing=0.0):
    y = interp_targets(y, torch.max(lengths))
    idxs = get_unpadded_idxs(lengths)
    y = fuzzy_cs_labels(y, lengths, n_classes)
    # loss = F.cross_entropy(y_hat.view(-1, n_classes)[idxs], y.view(-1)[idxs], label_smoothing=label_smoothing)
    loss = F.cross_entropy(y_hat.view(-1, n_classes)[idxs], y.view(-1, n_classes)[idxs], label_smoothing=label_smoothing)
    return loss, y

def fuzzy_cs_labels(targets, lengths, num_classes):

    grad = torch.gradient(targets, dim=-1)[0]
    switches = torch.where(grad!=0)

    l_s = targets[switches[0][1::2], switches[1][0::2]].long()
    r_s = targets[switches[0][1::2], switches[1][1::2]].long()

    l_index = switches[1][0::2].long()
    r_index = switches[1][1::2].long()
    b_index = switches[0][1::2].long()
    switch_from_pad = r_index+1 <= lengths[b_index]
    b_index = b_index[switch_from_pad]
    l_index = l_index[switch_from_pad]
    l_s = l_s[switch_from_pad]
    r_s = r_s[switch_from_pad]

    one_hot_labels = F.one_hot(targets.to(torch.long), num_classes=num_classes)
    inter_probs =  torch.linspace(0, 1, 5)
    if len(b_index):
        for i, prob in enumerate(inter_probs):
            idex = torch.tensor(len(inter_probs)-i).long()
            half_way = len(inter_probs)//2 + 1
            assert(b_index.max() <= lengths.size(0)), f'{b_index.max()} > bs {lengths.size(0)}'
            assert((l_index+half_way-idex).max() < one_hot_labels.size(-2)), f'{(l_index+half_way-idex).max()} < max seq len'
            one_hot_labels[b_index, l_index+half_way-idex, l_s] = torch.tensor(1.0 - float(prob)).long().cuda()
            one_hot_labels[b_index, l_index+half_way-idex, r_s] = torch.tensor(float(prob)).long().cuda()

    return one_hot_labels.float()