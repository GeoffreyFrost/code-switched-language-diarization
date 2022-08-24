import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.baseline_models import X_Transformer_E2E_LID
import torch.nn.utils.rnn as rnn_util
from utils.transforms import interp_targets
from models.modules.losses import DeepClusteringLoss
import gc

loss_func_CRE = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=255).to('cuda')
loss_func_xv = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=255).to('cuda') # this is important since 255 is for zero paddings

alpha = 0.5

class LitXSAE2E(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        
        self.encoder = X_Transformer_E2E_LID(n_lang=model_config.n_classes,
                                dropout=0.1,
                                input_dim=23,
                                feat_dim=256,
                                n_heads=4,
                                d_k=256,
                                d_v=256,
                                d_ff=2048,
                                max_seq_len=200,
                            )

    def forward(self, x, x_l):
        
        atten_mask = get_atten_mask(x_l, x.size(0)).to(x.device)
        outputs, cnn_outputs = self.encoder(x, x_l, atten_mask)
        return outputs, cnn_outputs

    def training_step(self, batch, batch_idx):

        x, x_l, y, y_l = batch

        y_hat, cnn_y_hat = self.forward(x, x_l)

        # y = interp_targets(y, y_hat.size(1))

        y_ = rnn_util.pack_padded_sequence(y, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)
        y_hat_ = rnn_util.pack_padded_sequence(y_hat, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)
        # cnn_y_hat_ = rnn_util.pack_padded_sequence(cnn_y_hat, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)

        loss_trans = loss_func_CRE(y_hat_, y_)
        loss_xv = loss_func_xv(cnn_y_hat, y.view(-1))
        # loss = loss_trans
        loss = alpha*loss_trans + (1-alpha)*loss_xv

        self.log('train/joint_loss', loss, sync_dist=True)
        return {'loss':loss, 'y_hat':y_hat_.detach(), 'y':y_.detach(), 'lengths':x_l.detach()}

    def training_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'] for x in out])
        y = torch.cat([x['y'] for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float(y.size(0))
        self.log("train/train_acc", accuracy, on_epoch=True, sync_dist=True)
        gc.collect()


    def validation_step(self, batch, batch_idx):

        x, x_l, y, y_l = batch
        y_hat = self.forward(x, x_l)

        y_hat, cnn_y_hat = self.forward(x, x_l)

        # y = interp_targets(y, y_hat.size(1))

        y_ = rnn_util.pack_padded_sequence(y, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)
        y_hat_ = rnn_util.pack_padded_sequence(y_hat, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)
        
        loss_CRE = loss_func_CRE(y_hat_, y_)
        loss = loss_CRE

        self.log('val/loss', loss.detach().item(), sync_dist=True)
        
        return {'y_hat':y_hat_.detach(), 'y':y_.detach(), 'lengths':x_l.detach()}

    def validation_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'] for x in out])
        y = torch.cat([x['y'] for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float(y.size(0))
        self.log("val/val_acc", accuracy, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):

        x, x_l, y, y_l = batch
        y = replace_label_pad_token(y)
        self.eval()
        with torch.no_grad():
            y_hat, _ = self.forward(x, x_l)

        return {'y_hat':y_hat.detach(), 'y':y.detach(), 'lengths':x_l.detach()}

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

        return {'optimizer':optimizer, 'lr_scheduler':{'scheduler':lr_scheduler, 'interval':'epoch'}}

def replace_label_pad_token(y):
    y[y == 255] = 0
    return y

def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_

def get_atten_mask(seq_lens, batch_size):
    max_len = int(seq_lens.max())
    batch_size = int(batch_size)
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = int(seq_lens[i])
        atten_mask[i, :length,:length] = 0
    return atten_mask.bool()