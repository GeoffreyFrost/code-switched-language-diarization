from dataclasses import dataclass
from utils.datasets import create_dataloaders, load_dfs
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.lit_cs_detector import LitCSDetector
from utils.lit_callbacks import BackboneFinetuning
from utils.modules import config_logger
import pandas as pd
import os

@dataclass
class TrainerConfig():
    batch_size:int=8
    max_epochs:int=16
    grad_clip_val:float=0.5
    precision:int=32
    backbone_warmup:bool=False
    unfreeze_at_epoch:int=1
    learning_rate:float=1e-4
    accumulate_grad_batches:int=8

@dataclass
class ExperimentConfig():
    cs_pair:str='all'
    routine:str='semi-supervised'
    n_refinement_stages:int=5
    unlabeled_ratio:float=0.2
    data_dir:str="/home/gfrost/datasets"

class Trainer():
    def __init__(self, model_config, trainer_config, experimental_config):
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.experimental_config = experimental_config
        self.supported_cs_pairs = ["engzul", "engxho", "engtsn", "engsot"]
        config_logger(model_config, trainer_config, experimental_config)

    def get_dfs(self):
        if self.experimental_config.cs_pair != 'all':
            data_df_root_dir = os.path.join(self.experimental_config.data_dir, \
                f"soapies_balanced_corpora/cs_{self.experimental_config.cs_pair}_balanced/lang_targs_mult/")
            df_trn, df_dev = load_dfs(data_df_root_dir, self.experimental_config.cs_pair)
        else:
            dfs_trn = []
            dfs_dev = []
            for cs_pair in self.supported_cs_pairs:
                data_df_root_dir = os.path.join(self.experimental_config.data_dir, \
                    f"soapies_balanced_corpora/cs_{cs_pair}_balanced/lang_targs_mult/")
                df_trn, df_dev = load_dfs(data_df_root_dir, cs_pair)
                dfs_trn.append(df_trn)
                dfs_dev.append(df_dev)
            df_trn = pd.concat(dfs_trn)
            df_dev = pd.concat(dfs_dev)
            self.model_config.n_classes = len(self.supported_cs_pairs) + 1

        d = pd.Series(df_dev.audio_fpath)
        t = pd.Series(df_trn.audio_fpath)
        assert d.isin(t).any() == False # Sanity check that splits are indeed clean
        return df_trn, df_dev

    def run_experiment(self):
        df_trn, df_dev = self.get_dfs()
        train_dataloader, dev_dataloader = create_dataloaders(df_trn, df_dev, bs=self.trainer_config.batch_size)
        self.callbacks = self.get_callbacks()
        self.model = LitCSDetector(learning_rate=self.trainer_config.learning_rate, 
                        model_config=self.model_config)
        self.train(train_dataloader, dev_dataloader)

    def train(self, train_dataloader, dev_dataloader):
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
        trainer = pl.Trainer(logger=tb_logger, 
                        callbacks=self.callbacks, 
                        max_epochs=self.trainer_config.max_epochs, 
                        gpus=1, 
                        gradient_clip_val=self.trainer_config.grad_clip_val, 
                        accumulate_grad_batches=self.trainer_config.accumulate_grad_batches, 
                        log_every_n_steps=100, 
                        precision=self.trainer_config.precision)

        trainer.fit(self.model, train_dataloader, dev_dataloader)

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
        if self.trainer_config.backbone_warmup==True: 
            callbacks.append(BackboneFinetuning(self.trainer_config.unfreeze_at_epoch, lambda epoch: 2))
        return callbacks

    def compute_metrics():
        pass