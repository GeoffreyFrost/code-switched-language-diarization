from dataclasses import dataclass
from gc import freeze, unfreeze
from utils.datasets import create_dataloaders, filter_mono_eng, load_dfs, filter_code_for_switched_only
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.lit_cs_detector import LitCSDetector
from models.lit_blstm_e2e import LitBLSTME2E
from utils.lit_callbacks import BackboneFinetuning, FeatureExtractorFreezeUnfreeze
from utils.modules import config_logger
import pandas as pd
import torch
import torch.nn as nn
import math
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
    resume_from_checkpoint:str=None

@dataclass
class ExperimentConfig():
    no_mono_eng:bool=False
    filter_cs:bool=False
    baseline:str=None
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
        pl.seed_everything(42)

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
                df_trn, df_dev = load_dfs(data_df_root_dir, cs_pair, True)
                dfs_trn.append(df_trn)
                dfs_dev.append(df_dev)
            df_trn = pd.concat(dfs_trn)
            df_dev = pd.concat(dfs_dev)

            self.model_config.n_classes = len(self.supported_cs_pairs) + 1

        d = pd.Series(df_dev.audio_fpath)
        t = pd.Series(df_trn.audio_fpath)
        assert d.isin(t).any() == False # Sanity check that splits are indeed clean
        
        print(f'Number of training samples: {len(df_trn)}')

        if self.experimental_config.no_mono_eng:
            df_trn = filter_mono_eng(df_trn)

        if self.experimental_config.filter_cs:
            df_trn = filter_code_for_switched_only(df_trn)
        
        print(f'Number of filtered training samples: {len(df_trn)}')

        return df_trn, df_dev

    def run_experiment(self):
        df_trn, df_dev = self.get_dfs()
        
        if self.experimental_config.baseline != None: melspecs=True # Baselines use melspecs
        else: melspecs=False

        train_dataloader, dev_dataloader = create_dataloaders(df_trn, df_dev, melspecs=melspecs, bs=self.trainer_config.batch_size, num_workers=4)
        # train_dataloader, dev_dataloader = create_dataloaders(df_trn, df_dev, melspecs=melspecs, bs=self.trainer_config.batch_size, num_workers=1)
        self.callbacks = self.get_callbacks()
        self.load_model()
        self.train(train_dataloader, dev_dataloader)

    def load_model(self):

        if self.experimental_config.baseline != None:
            assert self.experimental_config.baseline in ['blstm']
            if self.experimental_config.baseline == 'blstm':
                self.model = LitBLSTME2E(self.model_config)

        else: 
            self.model = LitCSDetector(learning_rate=self.trainer_config.learning_rate, 
                model_config=self.model_config)

            ckpt = torch.load('logs/lightning_logs/version_66/checkpoints/15-0.00-0.00.ckpt')
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.head = nn.Linear(1024, self.model_config.n_classes) # re-initialize head
            nn.init.xavier_uniform_(self.head.weight, gain=1 / math.sqrt(2))
            

    def train(self, train_dataloader, dev_dataloader):
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
        trainer = pl.Trainer(logger=tb_logger,
                        callbacks=self.callbacks, 
                        max_epochs=self.trainer_config.max_epochs, 
                        # gpus=list(range(torch.cuda.device_count())), # Assumes we start at cuda 0
                        gpus=[0],
                        gradient_clip_val=self.trainer_config.grad_clip_val,
                        accumulate_grad_batches = self.trainer_config.accumulate_grad_batches,
                        #accumulate_grad_batches=int(self.trainer_config.accumulate_grad_batches / torch.cuda.device_count()), 
                        log_every_n_steps=50, 
                        precision=self.trainer_config.precision,
                        #strategy="ddp",
                        # strategy='deepspeed_stage_3',
                        #profiler="advanced"
        )

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
        if self.model_config.soft_units:
            callbacks.append(FeatureExtractorFreezeUnfreeze(freeze_at_epoch=self.model_config.soft_unit_layer_train, 
                                                        unfreeze_at_epoch=self.model_config.soft_unit_layer_train+1))

        return callbacks

    def compute_metrics():
        pass