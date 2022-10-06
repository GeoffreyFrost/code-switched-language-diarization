from dataclasses import dataclass
from gc import freeze, unfreeze
from utils.datasets import create_dataloaders, filter_mono_eng, load_dfs, filter_code_for_switched_only
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.lit_cs_detector import LitCSDetector
from models.lit_blstm_e2e import LitBLSTME2E
from models.lit_x_transformer_e2e import LitXSAE2E
from utils.lit_callbacks import BackboneFinetuning, FeatureExtractorFreezeUnfreeze, GradNormCallback
from utils.modules import config_logger
import pandas as pd
import torch
import torch.nn as nn
import math
import os
import dataclasses

@dataclass
class TrainerConfig():
    gpus:list=dataclasses.field(default_factory=list)
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
    final:bool=False
    flatten_melspecs:bool=False
    no_mono_eng:bool=False
    filter_cs:bool=False
    eng_other:bool=False
    lang_fams:bool=False
    pretrained_lang_fams:bool=False
    pretrained_eng_other:bool=True
    baseline:str=None
    cs_pair:str='all'
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

                df_trn, df_dev = load_dfs(data_df_root_dir, cs_pair, all_cs_pairs=True, 
                                        eng_other=self.experimental_config.eng_other,
                                        lang_fams=self.experimental_config.lang_fams)

                dfs_trn.append(df_trn)
                dfs_dev.append(df_dev)
            df_trn = pd.concat(dfs_trn)
            df_dev = pd.concat(dfs_dev)
            
            # if self.experimental_config.lang_fams: self.model_config.n_classes = 3
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
        
        if self.experimental_config.baseline == 'blstm': melspecs, stack_frames = True, True
        elif self.experimental_config.baseline == 'xsa': melspecs, stack_frames = True, False
        else: melspecs, stack_frames = False, False
        
        train_dataloader, dev_dataloader = create_dataloaders(df_trn, df_dev, 
                                                            melspecs=melspecs, 
                                                            stack_frames=stack_frames,
                                                            flatten_melspecs=self.experimental_config.flatten_melspecs,
                                                            bs=self.trainer_config.batch_size, 
                                                            num_workers=4)
        self.callbacks = self.get_callbacks()
        self.load_model()
        self.train(train_dataloader, dev_dataloader)

    def load_model(self):
        if self.experimental_config.baseline != None:
            self.model_config.backbone = self.experimental_config.baseline
            assert self.experimental_config.baseline in ['blstm', 'xsa']

            if self.experimental_config.baseline == 'blstm':
                self.model = LitBLSTME2E(self.model_config, self.experimental_config)
            if self.experimental_config.baseline == 'xsa':
                self.model = LitXSAE2E(self.model_config)

        else: 
            self.model = LitCSDetector(learning_rate=self.trainer_config.learning_rate, 
                model_config=self.model_config)

            pretrained = self.experimental_config.pretrained_eng_other or self.experimental_config.pretrained_lang_fams
            self.load_pt(pretrained)


    def load_pt(self, pretrained):
        if pretrained:
            log_path = self.get_pt_log_path()
            ckpt = torch.load(get_checkpoint_path(log_path))
            self.model = load_pretrained_weights(self.model, ckpt, self.model_config)

    def get_pt_log_path(self):

        if self.experimental_config.pretrained_eng_other:
            log_path  = f'logs/final/{self.get_log_model_name()}/lightning_logs/version_0/'

        if self.experimental_config.pretrained_lang_fams:
            log_path  = f'logs/final/{self.get_log_model_name()}/lightning_logs/version_1/'

        return log_path

    def get_log_model_name(self):
        if self.experimental_config.final:
            log_model_name = self.model_config.backbone
        else: log_model_name = self.model_config.backbone
        return log_model_name

    def train(self, train_dataloader, dev_dataloader):
        
        if self.experimental_config.final: 
            log_dir = f"logs/final/{self.get_log_model_name()}"
        else: log_dir = "logs/"

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
        trainer = pl.Trainer(logger=tb_logger,
                        callbacks=self.callbacks, 
                        max_epochs=self.trainer_config.max_epochs,
                        gradient_clip_val=self.trainer_config.grad_clip_val,
                        accumulate_grad_batches=self.trainer_config.accumulate_grad_batches, 
                        log_every_n_steps=50,
                        precision=self.trainer_config.precision,
                        accelerator='gpu',
                        devices=self.trainer_config.gpus,
                        strategy="ddp" if len(self.trainer_config.gpus) > 1 else None
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
        gradnorm_callback  = GradNormCallback()
        callbacks = [learning_rate_callback, checkpoint_callback, gradnorm_callback]
        if self.trainer_config.backbone_warmup==True: 
            callbacks.append(BackboneFinetuning(self.trainer_config.unfreeze_at_epoch, lambda epoch: 2))
        if self.model_config.soft_units:
            callbacks.append(FeatureExtractorFreezeUnfreeze(freeze_at_epoch=self.model_config.soft_unit_layer_train, 
                                                        unfreeze_at_epoch=self.model_config.soft_unit_layer_train+1))

        return callbacks

def get_checkpoint_path(root):
    for root, dirs, files in os.walk(root, topdown=False):
        for file in files:
            if file.split('.')[-1] == 'ckpt' and file.split('.')[0] != 'last' and file.split('.')[0] != 'kek':
                return os.path.join(root, file)

def load_pretrained_weights(model, ckpt, model_config):
    model.load_state_dict(ckpt['state_dict'])

    if model_config.backbone == 'blstm': 
        model.encoder.head = nn.Linear(512, model_config.n_classes) # re-initialize head
        nn.init.xavier_uniform_(model.encoder.head.weight, gain=1 / math.sqrt(2))

    elif model_config.backbone == 'xsa': 
        model.encoder.head = nn.Linear(256, model_config.n_classes) 
        nn.init.xavier_uniform_(model.encoder.head.weight, gain=1 / math.sqrt(2))

    else: 
        emb_dim = list(model.backbone.encoder.parameters())[-1].shape[0]
        model.head = nn.Linear(emb_dim, model_config.n_classes)
        nn.init.xavier_uniform_(model.head.weight, gain=1 / math.sqrt(2))

    return model

def tracked_gradient_global_only():
    def remove_per_weight_norms(func):
        def f(*args):
            norms = func(*args)
            norms= dict(filter(lambda elem: '_total' in elem[0], norms.items()))
            return norms
        return f
    pl.core.grads.GradInformation.grad_norm = remove_per_weight_norms(pl.core.grads.GradInformation.grad_norm)
