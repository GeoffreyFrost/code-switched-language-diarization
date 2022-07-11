from typing import OrderedDict
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.transforms import interp_targets
from models.WavLM import WavLM, WavLMConfig

class LitCSDetector(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        assert backbone in ["base", "wavlm"], f'model: {backbone} not supported.'

        if backbone == "base":
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.wav2vec2 = bundle.get_model()
            self.head = nn.Sequential(OrderedDict([('linear1', nn.Linear(768, 4096)), ('linear2', nn.Linear(4096, 2))]))

        # if backbone == "wavlm":
        #     checkpoint = torch.load('/home/geoff/Documents/penguin/models/WavLM-Base.pt')
        #     cfg = WavLMConfig(checkpoint['cfg'])
        #     self.feature_extractor = WavLM(cfg)
        #     self.feature_extractor.load_state_dict(checkpoint['model'])
            
        #     self.head = nn.Sequential(OrderedDict([('linear1', nn.Linear(768, 4096)), ('linear2', nn.Linear(4096, 2))]))

    def forward(self, x, l):
        x, lengths = self.wav2vec2(x, l)
        x = self.head(x)
        return x, lengths

    def training_step(self, batch, batch_idx):
        x, x_l, y, y_l = batch
        y_hat, lengths = self.forward(x, x_l)
        loss, y = aggregate_bce_loss(y_hat, y, lengths)
        self.log('train/loss', loss)
        return {'loss':loss, 'y_hat':y_hat, 'y':y, 'lengths':lengths}
    
    def training_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'].view(-1, 2)[get_unpadded_idxs(x['lengths'])] for x in out])
        y = torch.cat([x['y'].view(-1)[get_unpadded_idxs(x['lengths'])]for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float( y.size(0))
        self.log("train/train_acc", accuracy, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, x_l, y, y_l = batch
        y_hat, lengths = self.forward(x, x_l)
        loss, y = aggregate_bce_loss(y_hat, y, lengths)
        self.log('va;/loss', loss)
        return {'y_hat':y_hat, 'y':y, 'lengths':lengths}

    def validation_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'].view(-1, 2)[get_unpadded_idxs(x['lengths'])] for x in out])
        y = torch.cat([x['y'].view(-1)[get_unpadded_idxs(x['lengths'])]for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float( y.size(0))
        self.log("val/train_acc", accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler}

def get_unpadded_idxs(lengths):
    max_len = torch.max(lengths)
    return torch.cat([torch.arange(max_len*i, max_len*i+l) for i, l in enumerate(lengths)]).to(torch.long)

def aggregate_bce_loss(y_hat, y, lengths):
    y = interp_targets(y, torch.max(lengths))
    idxs = get_unpadded_idxs(lengths)
    loss = F.cross_entropy(y_hat.view(-1, 2)[idxs], y.view(-1)[idxs])
    return loss, y