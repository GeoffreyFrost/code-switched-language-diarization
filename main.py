
import argparse
from dataclasses import dataclass
from utils.datasets import create_dataloaders, read_pickle_df
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.lit_cs_detector import LitCSDetector, BackboneFinetuning
import os

@dataclass
class Config:
    batch_size=1
    max_epochs=16
    label_smoothing=0.0
    grad_clip=0.5
    precision=32
    learning_rate=1e-4
    backbone_warmup=True
    unfreeze_at_epoch=1
    backbone='base'

class CSDetection():
    def __init__(self, config):
        self.config = config

    def run_experiment(self, cs_pair, data_dir):
        assert cs_pair in ["engzul"], f"cs pair {cs_pair} not implemented. Try one of these: {data_dir}."

        data_df_root_dir = os.path.join(data_dir, "soapies_balanced_corpora/cs_engzul_balanced/lang_targs_mult/")
        df_trn = read_pickle_df(os.path.join(data_df_root_dir, f"cs_{cs_pair}_trn.pkl"))
        df_dev = read_pickle_df(os.path.join(data_df_root_dir, f"cs_{cs_pair}_dev.pkl"))

        train_dataloader, dev_dataloader = create_dataloaders(df_trn, df_dev, bs=1)
        callbacks = self.get_callbacks()
        model = LitCSDetector(self.config.backbone, 
                        learning_rate=self.config.learning_rate, 
                        label_smoothing=self.config.label_smoothing)
        self.train(model, callbacks, train_dataloader, dev_dataloader)

    def train(self, model, callbacks, train_dataloader, dev_dataloader):
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
        trainer = pl.Trainer(logger=tb_logger, 
                        callbacks=callbacks, 
                        max_epochs=16, 
                        gpus=1, 
                        gradient_clip_val=0.5, 
                        accumulate_grad_batches=32, 
                        log_every_n_steps=100, 
                        precision=32)
        trainer.fit(model, train_dataloader, dev_dataloader)

    def get_callbacks(self):
        learning_rate_callback = LearningRateMonitor(logging_interval='step') 
        checkpoint_callback = ModelCheckpoint(monitor='val/val_acc',
                                                filename='{epoch}-{val/val_loss:.2f}-{val/val_auc:.2f}',
                                                save_on_train_epoch_end=False,
                                                auto_insert_metric_name=False,
                                                save_last=True,
                                                mode='max'
                                                )
        callbacks = [learning_rate_callback, checkpoint_callback]
        if self.config.backbone_warmup==True: 
            callbacks.append(BackboneFinetuning(self.config.unfreeze_at_epoch, lambda epoch: 2))
        return callbacks

    def compute_metrics():
        pass

def set_config(args):
    config = Config()
    config.batch_size=args.batch_size
    config.max_epochs=args.max_epochs
    config.label_smoothing=args.label_smoothing
    config.grad_clip=args.grad_clip
    config.precision=args.precision
    config.learning_rate=args.learning_rate
    config.backbone_warmup=args.backbone_warmup
    config.backbone=args.backbone
    config.unfreeze_at_epoch=args.unfreeze_at_epoch
    return config

def arg_paser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1)
    parser.add_argument('--max-epochs', default=16)
    parser.add_argument('--label-smoothing', default=0.0)
    parser.add_argument('--grad-clip', default=0.5)
    parser.add_argument('--precision', default=16)
    parser.add_argument('--learning-rate', default=1e-4)
    parser.add_argument('--backbone-warmup', default=True)
    parser.add_argument('--unfreeze-at-epoch', default=1)
    parser.add_argument('--backbone', default='base')
    return parser

if __name__ == '__main__':
    parser = arg_paser()
    args = parser.parse_args()
    config = set_config(args)