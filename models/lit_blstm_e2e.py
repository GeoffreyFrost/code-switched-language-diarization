import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.baseline_models import BLSTM_E2E_LID
import torch.nn.utils.rnn as rnn_util
from utils.transforms import interp_targets
from models.modules.losses import DeepClusteringLoss
import gc

loss_func_DCL = DeepClusteringLoss().to('cuda')
loss_func_CRE = nn.CrossEntropyLoss().to('cuda')
alpha = 1

class LitBLSTME2E(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config
        
        self.encoder = BLSTM_E2E_LID(
                            input_dim=23, # 23 dim mel-spectrograms
                            n_lang=model_config.n_classes,
                            dropout=0.25,
                            hidden_size=256,
                            num_emb_layer=2,
                            num_lstm_layer=3,
                            emb_dim=256
                            )

    def forward(self, x):
        x, embeddings = self.encoder(x)
        return x, embeddings

    def training_step(self, batch, batch_idx):

        x, x_l, y, y_l = batch
        y = interp_targets(y, x.size(-2))
        
        # copy pasta
        x_ = rnn_util.pack_padded_sequence(x, x_l.cpu(), batch_first=True, enforce_sorted=False)
        y_ = rnn_util.pack_padded_sequence(y, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)

        y_hat, embeddings = self.forward(x_)

        loss_DCL = loss_func_DCL(embeddings, y_)
        loss_CRE = loss_func_CRE(y_hat, y_)

        loss = alpha * loss_CRE + (1 - alpha) * loss_DCL

        self.log('train/joint_loss', loss, sync_dist=True)
        return {'loss':loss, 'y_hat':y_hat.detach(), 'y':y_.detach(), 'lengths':x_l.detach()}

    def training_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'] for x in out])
        y = torch.cat([x['y'] for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float(y.size(0))
        self.log("train/train_acc", accuracy, on_epoch=True, sync_dist=True)
        gc.collect()


    def validation_step(self, batch, batch_idx):

        x, x_l, y, y_l = batch
        y = interp_targets(y, x.size(-2))

        # copy pasta
        x_ = rnn_util.pack_padded_sequence(x, x_l.cpu(), batch_first=True, enforce_sorted=False)
        y_ = rnn_util.pack_padded_sequence(y, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)

        y_hat, embeddings = self.forward(x_)

        loss_DCL = loss_func_DCL(embeddings, y_)
        loss_CRE = loss_func_CRE(y_hat, y_)

        loss = alpha * loss_CRE + (1 - alpha) * loss_DCL
        self.log('val/loss', loss.detach().item(), sync_dist=True)
        
        return {'y_hat':y_hat.detach(), 'y':y_.detach(), 'lengths':x_l.detach()}

    def validation_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'] for x in out])
        y = torch.cat([x['y'] for x in out])
        print(y.max())
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float(y.size(0))
        self.log("val/val_acc", accuracy, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

        return {'optimizer':optimizer, 'lr_scheduler':{'scheduler':lr_scheduler, 'interval':'epoch'}}

# def get_unpadded_idxs(lengths):
#     max_len = torch.max(lengths)
#     return torch.cat([torch.arange(max_len*i, max_len*i+l) for i, l in enumerate(lengths)]).to(torch.long)